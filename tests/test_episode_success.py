"""Training success is emitted once per completed episode, including the last step."""
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from trainer import OnlineTrainer

class Episodes:
    env_num = 1
    def __init__(self):
        self.episode = -1
    def step(self, action, done):
        if done.item():
            self.episode += 1
        terminal = ~done
        return TensorDict({'reward': terminal.float().reshape(1, 1),
                           'log_success': (terminal.float() * (self.episode % 2 == 0)).reshape(1, 1)},
                          batch_size=[1]), terminal

class SuccessTests(unittest.TestCase):
    def test_success_failure_success_and_final_flush(self):
        config = SimpleNamespace(steps=3, pretrain=0, eval_every=100, eval_episode_num=0,
            video_pred_log=False, params_hist_log=False, batch_length=8, batch_size=1,
            train_ratio=1, update_log_every=100, action_repeat=1)
        replay = Mock()
        replay.count.return_value = 0
        logger = Mock()
        values = []
        logger.scalar.side_effect = lambda name, value: values.append((name, float(value)))
        trainer = OnlineTrainer(config, replay, logger, '.', Episodes(), None)
        trainer._save_latest = Mock()
        agent = Mock(device=torch.device('cpu'))
        agent.get_initial_state.return_value = {k: torch.zeros(1, 2) for k in ('prev_action','stoch','deter')}
        agent.act.side_effect = lambda trans, state, eval: (torch.zeros(1, 2), state)
        trainer.begin(agent)
        for name in ('episode/success', 'episode/sucess'):
            self.assertEqual([v for k, v in values if k == name], [1., 0., 1.])
        self.assertEqual([v for k,v in values if k == 'episode/length'], [1., 1., 1.])

if __name__ == '__main__':
    unittest.main()
