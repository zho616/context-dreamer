"""Exchange must retain target physics/action and use a foreign episode context."""
import unittest
import torch
from context import context_swap_loss, auxiliary_loss, STATE_SCALE, SWAP_PROFILE_KEYS, HistoryEncoder, ResidualDynamics


def batch():
    b, t = 2, 4
    data = {
        "ctx_ego_state": torch.zeros(b, t, 4),
        "ctx_prev_action": torch.zeros(b, t, 2),
        "ctx_history": torch.randn(b, t, 5, 6),
        "ctx_mask": torch.ones(b, t, 5, dtype=torch.bool),
        "ctx_episode_uid": torch.tensor([[0, 1], [1, 1]])[:, None].expand(b, t, 2).clone(),
    }
    for key in ("is_first", "is_last", "is_terminal", "ctx_collision"):
        data[key] = torch.zeros(b, t, 1, dtype=torch.bool)
    for key in SWAP_PROFILE_KEYS:
        data[key] = torch.ones(b, t, 1)
    return data


class SwapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_target_action_and_context_exchange(self):
        data = batch()
        # Foreign contexts add 3 and 1 respectively; target actions add 2 and 4.
        # Both true sequences increase by 5. Shifted RSSM action is a decoy.
        data["ctx_ego_state"][..., 0] = torch.arange(4) * 5
        data["ctx_prev_action"][0, :, 0] = .2
        data["ctx_prev_action"][1, :, 0] = .4
        data["action"] = torch.full((2, 4, 2), -1.)
        ctx = torch.zeros(2, 4, 8, requires_grad=True)
        with torch.no_grad():
            ctx[0, :, 0] = 1
            ctx[1, :, 0] = 3

        class Predictor(torch.nn.Module):
            def forward(self, c, s, a):
                return torch.nn.functional.pad(c[..., :1] + 10 * a[..., :1], (0, 3))

        loss, m = context_swap_loss(Predictor(), data, ctx)
        self.assertEqual(m["context/swap_pairs"].item(), 6)
        self.assertAlmostEqual(loss.item(), 0)
        self.assertGreater(m["context/swap_own_l1"].item(), 0)
        data["ctx_ego_state"][..., 0] = 0
        loss, _ = context_swap_loss(Predictor(), data, ctx)
        loss.backward()
        self.assertGreater(ctx.grad[0].abs().sum().item(), 0)
        self.assertGreater(ctx.grad[1].abs().sum().item(), 0)

    def test_no_pairs_for_same_episode_or_different_parameters(self):
        for reason in ("episode", *SWAP_PROFILE_KEYS, "partial"):
            data = batch()
            if reason == "episode":
                data["ctx_episode_uid"][1] = data["ctx_episode_uid"][0]
            elif reason == "partial":
                data["ctx_mask"][:, :, 0] = False
            else:
                data[reason][1] = 2
            ctx = torch.randn(2, 4, 8, requires_grad=True)
            loss, m = context_swap_loss(ResidualDynamics(), data, ctx)
            self.assertEqual(m["context/swap_pairs"].item(), 0, reason)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(ctx.grad).all())
            self.assertEqual(ctx.grad.abs().sum().item(), 0)

    def test_boundaries_and_nonfinite(self):
        for key in ("is_first", "is_last", "is_terminal", "ctx_collision"):
            data = batch()
            data[key][:, 2] = True
            loss, m = context_swap_loss(ResidualDynamics(), data, torch.zeros(2, 4, 8))
            self.assertEqual(m["context/swap_pairs"].item(), 2, key)
            self.assertTrue(torch.isfinite(loss))
        data = batch()
        data["ctx_ego_state"][:] = float("nan")
        loss, m = context_swap_loss(ResidualDynamics(), data, torch.zeros(2, 4, 8))
        self.assertEqual(m["context/swap_pairs"].item(), 0)
        self.assertEqual(loss.item(), 0)

    def test_encoder_gradient_amp(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = {k: v.to(device) for k, v in batch().items()}
        encoder, predictor = HistoryEncoder().to(device), ResidualDynamics().to(device)
        with torch.autocast(device_type=device, enabled=device == "cuda", dtype=torch.float16):
            ctx = encoder(data["ctx_history"], data["ctx_mask"])
            loss, m = context_swap_loss(predictor, data, ctx)
        loss.backward()
        self.assertEqual(m["context/swap_pairs"].item(), 6)
        for module in (encoder, predictor):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_physical_metrics_reconstruct_normalized_loss(self):
        data = batch()
        ctx = torch.zeros(2, 4, 8)
        loss, count, metrics = auxiliary_loss(ResidualDynamics(), data, ctx, return_metrics=True)
        names = ('accel_mps2', 'yaw_rate_radps', 'speed_mps', 'wheel_angle_rad')
        reconstructed = sum(metrics['context/mae_' + name] / scale
                            for name, scale in zip(names, STATE_SCALE))
        torch.testing.assert_close(loss.detach(), reconstructed)
        self.assertGreater(count.item(), 0)


if __name__ == "__main__":
    unittest.main()
