"""One-way WM -> context gradient routing regression tests."""
import unittest
import copy
import torch
import gymnasium as gym
import numpy as np
from test_context import config, inputs
from dreamer import Dreamer
from context import SWAP_PROFILE_KEYS

class RoutingTests(unittest.TestCase):
    def test_routing_and_scale(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        data, obs, act = inputs()
        data['ctx_episode_uid'] = torch.tensor([[0,1],[0,2]])[:,None].expand(2,7,2)
        for key in SWAP_PROFILE_KEYS: data[key] = torch.ones(2,7,1)
        obs=gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,shape=tuple(v.shape[2:]),dtype=np.float32) for k,v in data.items() if k not in ('action','reward')})
        c = config(); c.model.context.injection='dali_shallow'; c.model.context.swap_weight=1.
        snapshots=[]
        for alpha in (0., .1, 1.):
            torch.manual_seed(31)
            c.model.context.wm_grad_scale=alpha
            model=Dreamer(copy.deepcopy(c.model),obs,act)
            model._scaler=torch.amp.GradScaler(enabled=False)
            captured={}
            original=model._context_wm_surrogate
            def check(losses, context, leaf):
                params=list(model._named_params.items())
                others=[p for n,p in params if not n.startswith(('context_encoder.','context_dynamics.'))]
                for key in ('context_aux','context_swap'):
                    gs=torch.autograd.grad(losses[key],others,allow_unused=True,retain_graph=True)
                    self.assertTrue(all(g is None or torch.count_nonzero(g)==0 for g in gs))
                self.assertGreater(losses['context_swap'].item(),0)
                surrogate,mets=original(losses,context,leaf)
                ep=list(model.context_encoder.parameters())
                captured['routed']=torch.cat([g.flatten() for g in torch.autograd.grad(surrogate,ep,retain_graph=True)])
                # repval is excluded even if its coefficient changes enormously.
                old=model._loss_scales['repval']; model._loss_scales['repval']=1e6
                other,_=original(losses,context,leaf); model._loss_scales['repval']=old
                gg=torch.cat([g.flatten() for g in torch.autograd.grad(other,ep,retain_graph=True)])
                torch.testing.assert_close(gg,captured['routed'],rtol=0,atol=0)
                return surrogate,mets
            model._context_wm_surrogate=check
            torch.manual_seed(71)
            model._cal_grad(data,model.rssm.initial(2))
            snapshots.append(({n:p.grad.detach().clone() for n,p in model._named_params.items() if p.grad is not None},captured))
        base=snapshots[0][0]
        for grads,captured in snapshots[1:]:
            for n,g in base.items():
                if not n.startswith('context_encoder.'):
                    torch.testing.assert_close(grads[n],g,rtol=0,atol=0)
            diff=torch.cat([(grads[n]-g).flatten() for n,g in base.items() if n.startswith('context_encoder.')])
            self.assertGreater(diff.norm().item(),0)
            torch.testing.assert_close(diff,captured['routed'],rtol=1e-4,atol=1e-7)
        torch.testing.assert_close(snapshots[1][1]['routed']*10,snapshots[2][1]['routed'],rtol=1e-4,atol=1e-7)

if __name__=='__main__': unittest.main()
