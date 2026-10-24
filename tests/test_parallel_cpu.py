"""Pure CPU rollouts must work with CUDA hidden, including reset transitions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from tensordict import TensorDict
from envs.parallel import ParallelEnv

class ParallelCpuTests(unittest.TestCase):
 def test_reset_and_step_do_not_pin_cpu_tensors(self):
  obs={'state':np.ones(3,np.float32),'is_first':True}
  worker=SimpleNamespace(reset=lambda:lambda:obs,step=lambda a:lambda:(dict(obs,is_first=False),2.,True,{}))
  env=ParallelEnv.__new__(ParallelEnv);env.envs=[worker];env.device='cpu'
  with patch.object(TensorDict,'pin_memory',side_effect=AssertionError('CPU requested pinned memory')):
   td,done=env.step(torch.zeros(1,2),torch.tensor([True]));self.assertFalse(done.item());self.assertEqual(td['reward'].item(),0.)
   td,done=env.step(torch.zeros(1,2),torch.tensor([False]));self.assertTrue(done.item());self.assertEqual(td['reward'].item(),2.)
   self.assertEqual(td['state'].device.type,'cpu')
if __name__=='__main__':unittest.main()
