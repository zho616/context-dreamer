import pathlib
import tempfile
import unittest
from unittest.mock import Mock

import torch

from trainer import OnlineTrainer


class BestCheckpointTests(unittest.TestCase):
    def test_latest_crosses_20k_boundary_and_replaces_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = OnlineTrainer.__new__(OnlineTrainer)
            trainer.logdir = pathlib.Path(directory)
            trainer.save_every = 20000
            trainer._last_checkpoint_step = 0
            trainer._best_eval_key = (0.7, 0.8, 100.0)
            agent = torch.nn.Linear(1, 1)
            agent._optimizer = torch.optim.Adam(agent.parameters(), lr=4e-5)
            agent._scheduler = torch.optim.lr_scheduler.LambdaLR(agent._optimizer, lambda _: 1.0)
            agent(torch.ones(1, 1)).sum().backward()
            agent._optimizer.step()
            agent._scheduler.step()
            trainer._maybe_save_latest(agent, 19999, 10)
            self.assertFalse((trainer.logdir / 'latest.pt').exists())
            # Batched environments can step past, rather than land on, 20K.
            trainer._maybe_save_latest(agent, 20003, 11)
            first = torch.load(trainer.logdir / 'latest.pt', weights_only=False)
            self.assertEqual(first['step'], 20003)
            trainer._maybe_save_latest(agent, 39999, 12)
            self.assertEqual(torch.load(trainer.logdir / 'latest.pt', weights_only=False)['step'], 20003)
            with torch.no_grad():
                agent.weight.add_(1)
            trainer._maybe_save_latest(agent, 40002, 13)
            latest = torch.load(trainer.logdir / 'latest.pt', weights_only=False)
            self.assertEqual(latest['step'], 40002)
            self.assertEqual(latest['update_count'], 13)
            self.assertEqual(latest['best_eval_key'], trainer._best_eval_key)
            self.assertEqual(latest['scheduler_state_dict'], agent._scheduler.state_dict())
            self.assertTrue(latest['optims_state_dict']['_optimizer']['state'])
            torch.testing.assert_close(latest['agent_state_dict']['weight'], agent.weight)
            self.assertFalse((trainer.logdir / 'latest.pt.tmp').exists())
            self.assertFalse((trainer.logdir / 'best.pt').exists())
            # Completion also saves below the next periodic boundary.
            trainer._save_latest(agent, 40500, 14)
            self.assertEqual(torch.load(trainer.logdir / 'latest.pt', weights_only=False)['step'], 40500)

    def test_selection_order_and_saved_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = OnlineTrainer.__new__(OnlineTrainer)
            trainer.logdir = pathlib.Path(directory)
            trainer.logger = Mock()
            trainer._best_eval_key = None
            agent = torch.nn.Linear(1, 1)
            agent._optimizer = torch.optim.Adam(agent.parameters())
            agent(torch.ones(1, 1)).sum().backward()
            agent._optimizer.step()

            def save(step, success, route, score):
                return trainer._save_best(
                    agent,
                    step,
                    {"episode/eval_success": success, "episode/eval_route_completion": route,
                     "episode/eval_score": score},
                )

            self.assertTrue(save(10, 0.5, 0.6, 10))
            self.assertFalse(save(20, 0.4, 1.0, 1000))
            self.assertTrue(save(30, 0.6, 0.1, -10))
            self.assertTrue(save(40, 0.6, 0.2, -20))
            self.assertTrue(save(50, 0.6, 0.2, -15))
            self.assertFalse(save(60, 0.6, 0.2, -15))
            self.assertFalse(save(70, float("nan"), 1, 1000))
            checkpoint = torch.load(trainer.logdir / "best.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 50)
            self.assertEqual(checkpoint["eval_metrics"]["episode/eval_success"], 0.6)
            self.assertTrue(checkpoint["optims_state_dict"]["_optimizer"]["state"])
            for key, value in agent.state_dict().items():
                torch.testing.assert_close(checkpoint["agent_state_dict"][key], value)
            self.assertFalse((trainer.logdir / "best.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
