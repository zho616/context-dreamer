"""Evaluation episode accounting with fewer workers than episodes."""

import unittest
from unittest.mock import Mock

import torch
from tensordict import TensorDict

from trainer import OnlineTrainer


class UnequalLengthEnvs:
    env_num = 2

    def __init__(self):
        self.lengths = torch.zeros(2, dtype=torch.int32)

    def step(self, action, done):
        self.lengths = torch.where(done, 0, self.lengths + 1)
        ended = ~done & (self.lengths >= torch.tensor([1, 3]))
        return TensorDict(
            {
                "reward": (torch.tensor([1.0, 2.0]) * ~done).unsqueeze(-1),
                # A success signal repeated on each step must be clipped per episode.
                "log_success": (torch.tensor([0.0, 1.0]) * ~done).unsqueeze(-1),
                "log_cost": (0.5 * ~done).unsqueeze(-1),
                "log_end": ended.to(torch.float32).unsqueeze(-1),
                "log_chassis_mass": torch.tensor([[1100.0], [1200.0]]),
            },
            batch_size=(2,),
        ), ended


class EvaluationTests(unittest.TestCase):
    def check_episodes(self, episodes):
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.eval_envs = UnequalLengthEnvs()
        trainer.eval_episode_num = episodes
        trainer.batch_length = 8
        trainer.video_pred_log = False
        trainer.logger = Mock()
        trainer.save_best = False
        agent = Mock()
        agent.device = torch.device("cpu")
        agent.get_initial_state.return_value = {"prev_action": torch.zeros(2, 2)}
        agent.act.side_effect = lambda trans, state, eval: (torch.zeros(2, 2), state)
        trainer.eval(agent, 123)
        metrics = {call.args[0]: float(call.args[1]) for call in trainer.logger.scalar.call_args_list}
        first, second = (episodes + 1) // 2, episodes // 2
        self.assertAlmostEqual(metrics["episode/eval_score"], (first + 6 * second) / episodes, places=6)
        self.assertAlmostEqual(metrics["episode/eval_length"], (first + 3 * second) / episodes, places=6)
        self.assertAlmostEqual(metrics["episode/eval_success"], second / episodes, places=6)
        self.assertAlmostEqual(metrics["episode/eval_cost"], (0.5 * first + 1.5 * second) / episodes, places=6)
        self.assertEqual(metrics["episode/eval_end"], 1.0)
        self.assertAlmostEqual(metrics["episode/eval_chassis_mass"], (1100 * first + 1200 * second) / episodes, places=3)
        self.assertEqual(metrics["episode/eval_episodes"], episodes)
        trainer.logger.write.assert_called_once_with(123)
        agent.train.assert_called_once()

    def test_twenty_episodes_two_workers(self):
        self.check_episodes(20)

    def test_uneven_episode_quota(self):
        self.check_episodes(5)

    def test_one_episode_per_worker_compatibility(self):
        self.check_episodes(2)


if __name__ == "__main__":
    unittest.main()
