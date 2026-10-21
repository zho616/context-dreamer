"""Context alignment, boundary/gradient isolation and both learning paths."""
import unittest
from types import SimpleNamespace
import hydra
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from context import HistoryEncoder, ResidualDynamics, auxiliary_loss, canonical_action, predict_next_state
from dreamer import Dreamer
from envs.ego_history import EgoHistory


def config(method='r2dreamer', enabled=True, device='cpu'):
    with hydra.initialize(config_path='../configs', version_base=None):
        c = hydra.compose(config_name='metadrive_context', overrides=[f'device={device}',
            'model.compile=false', f'model.rep_loss={method}', f'model.context.enabled={str(enabled).lower()}',
            'model.deter=32', 'model.hidden=16', 'model.units=16', 'model.discrete=4',
            'model.rssm.stoch=4', 'model.rssm.blocks=4', 'model.imag_horizon=3'])
    return c


def inputs(device='cpu', state_dim=4):
    b,t=2,7
    data=TensorDict(dict(state=torch.randn(b,t,9), action=torch.rand(b,t,2)*2-1,
        reward=torch.randn(b,t,1), is_first=torch.zeros(b,t,1,dtype=torch.bool),
        is_last=torch.zeros(b,t,1,dtype=torch.bool),is_terminal=torch.zeros(b,t,1,dtype=torch.bool),
        ctx_ego_state=torch.randn(b,t,state_dim),ctx_prev_action=torch.rand(b,t,2)*2-1,
        ctx_history=torch.randn(b,t,5,state_dim+2),ctx_mask=torch.ones(b,t,5,dtype=torch.bool),
        ctx_collision=torch.zeros(b,t,1,dtype=torch.bool)),batch_size=(b,t)).to(device)
    data['is_first'][:,0]=True
    data['ctx_mask'][:,0]=False
    spaces=gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,shape=tuple(v.shape[2:]),dtype=np.float32)
                           for k,v in data.items() if k not in ('action','reward')})
    return data,spaces,gym.spaces.Box(-1,1,(2,),dtype=np.float32)


class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_mask_and_physics(self):
        e=HistoryEncoder()
        x=torch.randn(2,5,6); m=torch.tensor([[0,0,1,1,1],[0,0,0,0,0]],dtype=torch.bool)
        y=e(x,m); x[~m]=float('nan')
        torch.testing.assert_close(e(x,m),y)
        self.assertTrue((y[1]==0).all())
        body=SimpleNamespace(getAngularVelocity=lambda: [0,0,.2])
        system=SimpleNamespace(getSteeringValue=lambda i: 10.+4*i)
        v=SimpleNamespace(velocity=np.array([1.,0.]),heading_theta=0.,body=body,system=system)
        h=EgoHistory(.1); first=h.observe(v)
        self.assertFalse(first['ctx_mask'].any())
        v.velocity=np.array([1.2,-.1]); body.getAngularVelocity=lambda:[0,0,.3]
        obs=h.observe(v,np.array([.2,.7]))
        np.testing.assert_allclose(obs['ctx_ego_state'],[2,-.3,np.hypot(1.2,.1),-np.deg2rad(12)],rtol=1e-6)
        np.testing.assert_array_equal(obs['ctx_history'][-1,:4],obs['ctx_ego_state'])
        np.testing.assert_allclose(obs['ctx_history'][-1,-2:],[.7,-.2])
        self.assertEqual(obs['ctx_mask'].sum(),1)
        self.assertFalse(h.observe(v)['ctx_mask'].any())

    def test_aux_alignment_and_boundaries(self):
        data,_,_=inputs()
        # Constant action yields exactly unit longitudinal acceleration increments.
        data['ctx_ego_state'].zero_(); data['ctx_prev_action'].zero_()
        data['ctx_prev_action'][...,0]=1
        data['ctx_ego_state'][...,0]=torch.arange(7)
        class Predictor(torch.nn.Module):
            def forward(self,c,s,a):
                return torch.nn.functional.pad(a[...,:1],(0,s.shape[-1]-1))
        loss,count=auxiliary_loss(Predictor(),data,torch.zeros(2,7,8))
        self.assertEqual(loss.item(),0); self.assertEqual(count.item(),40)
        data['is_first'][:,3]=True
        _,count=auxiliary_loss(Predictor(),data,torch.zeros(2,7,8))
        self.assertEqual(count.item(),18) # 3 + 6 pairs per batch
        data['is_terminal'][:]=True
        loss,count=auxiliary_loss(ResidualDynamics(),data,torch.zeros(2,7,8))
        self.assertEqual(count.item(),0); self.assertEqual(loss.item(),0)
        loss.backward()

    def test_aux_stepwise_detach(self):
        data,_,_=inputs()
        data=data[:1,:3]
        data['ctx_ego_state']=torch.zeros(1,3,2)
        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p=torch.nn.Parameter(torch.ones(()))
            def forward(self,c,s,a):
                return self.p * s.new_tensor([10,1]).expand_as(s)
        predictor=Predictor()
        loss,count=auxiliary_loss(predictor,data,torch.zeros(1,3,8))
        self.assertEqual(count.item(),3)
        self.assertAlmostEqual(loss.item(),8/3,places=5)
        loss.backward()
        self.assertAlmostEqual(predictor.p.grad.item(),2,places=5)

    def test_imagined_history_endpoint_and_mask(self):
        from context import refresh_context
        encoder=HistoryEncoder(); predictor=ResidualDynamics()
        state=torch.randn(2,4); action=torch.tensor([[.2,.7],[.1,.3]])
        c=torch.randn(2,8); history=torch.randn(2,5,6); mask=torch.ones(2,5,dtype=torch.bool)
        expected=predict_next_state(predictor,c,state,canonical_action(action))
        nxt,ctx,h,m=refresh_context(encoder,predictor,state,action,c,history,mask,torch.tensor([True,False]))
        torch.testing.assert_close(nxt[0],expected[0])
        torch.testing.assert_close(h[0,-1,:4],nxt[0])
        torch.testing.assert_close(h[0,-1,4:],canonical_action(action)[0])
        self.assertGreaterEqual(nxt[0,2].item(),0)
        self.assertLessEqual(nxt[0,3].abs().item(),np.pi/3+1e-6)
        self.assertFalse(m[1].any()); self.assertTrue((ctx[1]==0).all())
        self.assertFalse(ctx.requires_grad)

    def test_legacy_layout_and_checkpoint_remain_supported(self):
        data,obs,act=inputs(state_dim=2)
        c=config()
        model=Dreamer(c.model,obs,act)
        self.assertEqual(model.context_encoder.input.in_features,4)
        _,metrics=model._cal_grad(data,model.rssm.initial(2))
        self.assertTrue(torch.isfinite(metrics['loss/context_aux']))
        model.load_state_dict(model.state_dict(),strict=True)
        v=SimpleNamespace(velocity=np.array([1.,0.]),heading_theta=0.,
                          body=SimpleNamespace(getAngularVelocity=lambda:[0,0,0]))
        history=EgoHistory(.1,include_kinematics=False)
        self.assertEqual(history.observe(v)['ctx_history'].shape,(5,4))

    def test_baseline_weights_independent_of_context_layout(self):
        for method in ('r2dreamer','dreamer'):
            _,old_obs,act=inputs(state_dim=2);_,new_obs,_=inputs(state_dim=4)
            torch.manual_seed(41);old=Dreamer(config(method,False).model,old_obs,act)
            torch.manual_seed(41);new=Dreamer(config(method,False).model,new_obs,act)
            for key,value in old.state_dict().items():
                torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)

    def test_modulation_identity_conditioning_and_gradients(self):
        import copy
        from context_rssm import ContextRSSM, ModulatedDeter
        from rssm import RSSM
        c=config().model.rssm
        base=RSSM(c,16,2)
        mod=ModulatedDeter(copy.deepcopy(base._deter_net))
        z=torch.randn(3,4,4); h=torch.randn(3,32); a=torch.randn(3,2)
        ctx=torch.randn(3,8,requires_grad=True)
        torch.testing.assert_close(mod(z,h,a,ctx),base._deter_net(z,h,a),rtol=0,atol=0)
        mod(z,h,a,ctx).square().sum().backward()
        self.assertIsNone(ctx.grad)
        self.assertTrue(mod.modulations[0].weight.grad.abs().sum()>0)
        rssm=ContextRSSM(c,16,2,injection="adarms_deter")
        with torch.no_grad():
            rssm._deter_net.modulations[0].weight.normal_(0,.1)
        embed=torch.randn(3,16); reset=torch.zeros(3,dtype=torch.bool)
        _,h1,q1=rssm.obs_step(z,h,a,embed,reset,ctx)
        _,h2,q2=rssm.obs_step(z,h,a,embed,reset,ctx+1)
        _,p1=rssm.prior(h1,ctx); _,p2=rssm.prior(h2,ctx+1)
        self.assertFalse(torch.allclose(q1,q2))
        self.assertFalse(torch.allclose(p1,p2))
        _,imag_h=rssm.img_step(z,h,a,ctx)
        torch.testing.assert_close(imag_h,h1)
        # Episode resets erase the previous context and recurrent state.
        _,reset_h1,_=rssm.obs_step(z,h,a,embed,~reset,ctx)
        _,reset_h2,_=rssm.obs_step(z+1,h+1,a+1,embed,~reset,ctx+1)
        torch.testing.assert_close(reset_h1,reset_h2)

    def test_recurrent_context_controls_transition_and_resets(self):
        from context_rssm import ContextRSSM
        rssm=ContextRSSM(config().model.rssm,16,2)
        self.assertEqual(rssm.injection,"recurrent_concat")
        z=torch.randn(3,4,4); h=torch.randn(3,32); a=torch.randn(3,2)
        ctx=torch.randn(3,8,requires_grad=True)
        embed=torch.randn(3,16); reset=torch.zeros(3,dtype=torch.bool)
        _,h1,q1=rssm.obs_step(z,h,a,embed,reset,ctx)
        _,h2,q2=rssm.obs_step(z,h,a,embed,reset,ctx+1)
        _,p1=rssm.prior(h1,ctx); _,p2=rssm.prior(h2,ctx+1)
        self.assertFalse(torch.allclose(h1,h2))
        self.assertFalse(torch.allclose(q1,q2))
        self.assertFalse(torch.allclose(p1,p2))
        _,imag_h=rssm.img_step(z,h,a,ctx)
        torch.testing.assert_close(imag_h,h1)
        (h1.square().mean()+q1.square().mean()+p1.square().mean()).backward()
        self.assertIsNone(ctx.grad)
        # The world-model loss must train the newly added context columns.
        grad=rssm._deter_net._dyn_in1[0].weight.grad[:,-8:]
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum()>0)
        _,reset_h1,_=rssm.obs_step(z,h,a,embed,~reset,ctx)
        _,reset_h2,_=rssm.obs_step(z+1,h+1,a+1,embed,~reset,ctx+1)
        torch.testing.assert_close(reset_h1,reset_h2)

    def test_dali_joint_input_order_gradients_and_reset(self):
        from context_rssm import ContextRSSM
        rssm = ContextRSSM(config().model.rssm, 16, 2, injection='dali_deep_joint')
        z, h, a = torch.randn(3, 4, 4), torch.randn(3, 32), torch.randn(3, 2)
        ctx = torch.randn(3, 8, requires_grad=True)
        seen = []
        hook = rssm._deter_net.img_in.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        embed, reset = torch.randn(3, 16), torch.zeros(3, dtype=torch.bool)
        _, h1, q1 = rssm.obs_step(z, h, a, embed, reset, ctx)
        hook.remove()
        expected = torch.cat((z.flatten(1), a / a.abs().clamp_min(1), ctx.detach()), -1)
        torch.testing.assert_close(seen[0], expected)
        self.assertFalse(hasattr(rssm._deter_net, '_dyn_in1'))
        (h1.square().mean() + q1.square().mean()).backward()
        self.assertIsNone(ctx.grad)
        weight_grad = rssm._deter_net.img_in[0].weight.grad
        for columns in (slice(0, 16), slice(16, 18), slice(18, 26)):
            self.assertGreater(weight_grad[:, columns].abs().sum().item(), 0)
        _, h2, _ = rssm.obs_step(z, h, a, embed, reset, ctx + 1)
        self.assertFalse(torch.allclose(h1, h2))
        _, r1, _ = rssm.obs_step(z, h, a, embed, ~reset, ctx)
        _, r2, _ = rssm.obs_step(z + 1, h + 1, a + 1, embed, ~reset, ctx + 1)
        torch.testing.assert_close(r1, r2)
        _, p1 = rssm.prior(h1.detach(), ctx)
        _, p2 = rssm.prior(h1.detach(), ctx + 1)
        torch.testing.assert_close(p1, p2)

    def test_both_methods_update_and_isolation(self):
        for method in ('r2dreamer','dreamer'):
            for injection in (None,"recurrent_concat","prior_posterior","adarms_deter","recurrent_actor_critic","dali_deep","dali_deep_joint","dali_shallow"):
                enabled = injection is not None
                with self.subTest(method=method,injection=injection):
                    c=config(method,enabled); data,obs,act=inputs()
                    if enabled:
                        c.model.context.injection="recurrent_concat" if injection=="recurrent_actor_critic" else injection
                        c.model.context.actor_critic=injection=="recurrent_actor_critic"
                    model=Dreamer(c.model,obs,act)
                    initial=model.rssm.initial(2)
                    if enabled:
                        ctx=model.context_encoder(data['ctx_history'],data['ctx_mask'])
                        _,_,logits=model.rssm.observe(model.encoder(data),data['action'],initial,data['is_first'],ctx)
                        logits.square().mean().backward()
                        self.assertTrue(all(p.grad is None for p in model.context_encoder.parameters()))
                        model.zero_grad(set_to_none=True)
                    _,metrics=model._cal_grad(data,initial)
                    self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v)))
                    if enabled:
                        for module in (model.context_encoder,model.context_dynamics):
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
                        self.assertTrue(all(not k.startswith('ctx_') for k in model.encoder.mlp_shapes))
                        if method=='dreamer': self.assertTrue(all(not k.startswith('ctx_') for k in model.decoder.all_keys))
                    model._scaler.unscale_(model._optimizer)
                    model._scaler.step(model._optimizer); model._scaler.update()
                    model.load_state_dict(model.state_dict(),strict=True)
                    stoch,deter=initial
                    action,_=model.act(data[:,0],TensorDict(dict(stoch=stoch,deter=deter,prev_action=torch.zeros(2,2)),batch_size=(2,)))
                    self.assertTrue(torch.isfinite(action).all())

    def test_behavior_context_alignment_and_gradient_isolation(self):
        c=config(); c.model.context.actor_critic=True
        data,obs,act=inputs(); model=Dreamer(c.model,obs,act)
        feat=torch.randn(2,7,model.rssm.feat_size,requires_grad=True)
        ctx=torch.randn(2,7,8,requires_grad=True)
        combined=model._behavior_feat(feat,ctx)
        torch.testing.assert_close(combined[...,-8:],ctx)
        combined.sum().backward()
        self.assertIsNone(ctx.grad); self.assertIsNotNone(feat.grad)
        with self.assertRaises(ValueError):model._behavior_feat(feat,ctx[:,0])
        with torch.no_grad():
            context=model.context_encoder(data['ctx_history'],data['ctx_mask'])
            context[:,0]=0
            captured=[]
            hook=model._frozen_actor.register_forward_pre_hook(lambda m,args:captured.append(args[0].clone()))
            initial=model.rssm.initial(14)
            feats,actions,contexts=model._imagine_context(initial,3,data,context)
            hook.remove()
            torch.testing.assert_close(contexts[:,0],context.reshape(-1,8))
            for t in range(3):
                torch.testing.assert_close(captured[t][...,-8:],contexts[:,t])
                torch.testing.assert_close(captured[t][...,:-8],feats[:,t])
            # Reset must also clear direct actor context even for a stale history.
            reset_data=data[:,0].clone(); reset_data['ctx_mask'].fill_(True)
            captured=[]
            hook=model._frozen_actor.register_forward_pre_hook(lambda m,args:captured.append(args[0].clone()))
            model.act(reset_data,model.get_initial_state(2)); hook.remove()
            self.assertTrue((captured[0][...,-8:]==0).all())

    def test_deep_heads_swap_update_and_isolation(self):
        from context import SWAP_PROFILE_KEYS
        c = config()
        c.model.context.injection = 'dali_deep_joint'
        c.model.context.swap_weight = 1.0
        data, _, act = inputs()
        data['ctx_episode_uid'] = torch.tensor([[0, 1], [1, 1]])[:, None].expand(2, 7, 2).clone()
        for key in SWAP_PROFILE_KEYS:
            data[key] = torch.ones(2, 7, 1)
        obs = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, shape=tuple(v.shape[2:]), dtype=np.float32)
                              for k, v in data.items() if k not in ('action', 'reward')})
        model = Dreamer(c.model, obs, act)
        self.assertTrue(model.context_actor_critic)
        self.assertEqual(model.rssm._img_net[0].in_features, model.rssm._deter)
        self.assertEqual(model.rssm._obs_net[0].in_features, model.rssm._deter + model.embed_size)
        # Every Deep prediction head receives aligned detached context.
        ctx = model.context_encoder(data['ctx_history'], data['ctx_mask'])
        feat = torch.randn(2, 7, model.rssm.feat_size, requires_grad=True)
        combined = model._prediction_feat(feat, ctx)
        torch.testing.assert_close(combined[..., -8:], ctx)
        model.reward(combined).log_prob(data['reward']).mean().backward()
        self.assertTrue(all(p.grad is None for p in model.context_encoder.parameters()))
        model.zero_grad(set_to_none=True)
        _, metrics = model._cal_grad(data, model.rssm.initial(2))
        self.assertGreater(metrics['context/swap_pairs'].item(), 0)
        self.assertEqual(model._loss_scales['context_swap'], 1.0)
        self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v)))
        # Prediction readouts start at zero, so hidden/input columns receive
        # gradients only after the first optimizer step updates the readout.
        model._scaler.unscale_(model._optimizer)
        model._scaler.step(model._optimizer)
        model._scaler.update()
        model.zero_grad(set_to_none=True)
        model._cal_grad(data, model.rssm.initial(2))
        for head in (model.reward, model.cont, model.prj, model.actor, model.value):
            first = next(m for m in head.modules() if isinstance(m, torch.nn.Linear))
            self.assertEqual(first.in_features, model.rssm.feat_size + 8)
            self.assertGreater(first.weight.grad[:, -8:].abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.context_encoder.parameters()))

    def test_shallow_only_conditions_posterior(self):
        from context_rssm import ContextRSSM
        c = config()
        c.model.context.injection = 'dali_shallow'
        data, obs, act = inputs()
        model = Dreamer(c.model, obs, act)
        rssm = model.rssm
        self.assertFalse(model.context_actor_critic)
        self.assertFalse(model.context_deep)
        self.assertEqual(rssm._obs_net[0].in_features, rssm._deter + model.embed_size + 8)
        self.assertEqual(rssm._img_net[0].in_features, rssm._deter)
        self.assertEqual(rssm._deter_net._dyn_in1[0].in_features, rssm.flat_stoch)
        for module in (model.actor, model.value, model.reward, model.cont, model.prj):
            first = next(m for m in module.modules() if isinstance(m, torch.nn.Linear))
            self.assertEqual(first.in_features, rssm.feat_size)
        z, h = rssm.initial(3)
        a, embed = torch.randn(3, 2), torch.randn(3, model.embed_size)
        ctx = torch.randn(3, 8, requires_grad=True)
        reset = torch.zeros(3, dtype=torch.bool)
        _, h1, q1 = rssm.obs_step(z, h, a, embed, reset, ctx)
        _, h2, q2 = rssm.obs_step(z, h, a, embed, reset, ctx + 1)
        torch.testing.assert_close(h1, h2)
        self.assertFalse(torch.allclose(q1, q2))
        _, p1 = rssm.prior(h1, ctx)
        _, p2 = rssm.prior(h1, ctx + 1)
        torch.testing.assert_close(p1, p2)
        _, ih1 = rssm.img_step(z, h, a, ctx)
        _, ih2 = rssm.img_step(z, h, a, ctx + 1)
        torch.testing.assert_close(ih1, ih2)
        q1.square().mean().backward()
        self.assertIsNone(ctx.grad)
        self.assertGreater(rssm._obs_net[0].weight.grad[:, -8:].abs().sum().item(), 0)
        _, rh1, rq1 = rssm.obs_step(z, h, a, embed, ~reset, ctx)
        _, rh2, rq2 = rssm.obs_step(z + 1, h + 1, a + 1, embed, ~reset, ctx + 1)
        torch.testing.assert_close(rh1, rh2)
        torch.testing.assert_close(rq1, rq2)

if __name__=='__main__': unittest.main()
