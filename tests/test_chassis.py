"""Physical application, integer grids, reset/reuse and independent seed streams."""
import unittest

import hydra
import numpy as np

from envs.metadrive import MetaDrive
from envs.chassis import sample_chassis, validate_spec
from metadrive.component.vehicle.vehicle_type import vehicle_type


class ChassisTests(unittest.TestCase):
    def config(self, **overrides):
        with hydra.initialize(config_path='../configs',version_base=None):
            return hydra.compose(config_name='metadrive_context_chassis',overrides=[
                'env.traffic_density=0',*[f'env.{k}={v}' for k,v in overrides.items()]]).env

    def test_integer_grid_and_validation(self):
        spec=dict(models=['default','m','l'],mass_scales=[.8,.9,1,1.1,1.2],steering_scales=[.8,.9,1,1.1,1.2])
        validate_spec(spec)
        seen=set()
        for seed in range(100):
            sample=sample_chassis(spec,seed)
            self.assertEqual(sample,sample_chassis(spec,seed))
            seen.add(sample['model'])
            self.assertIsInstance(sample['mass'],int)
            self.assertIsInstance(sample['max_steering'],int)
            self.assertTrue(.8<=sample['mass_scale']<=1.2)
            self.assertIn(sample['sampled_mass_scale'],spec['mass_scales'])
        self.assertEqual(seen,set(spec['models']))
        for broken in (dict(spec,models=['wrong']),dict(spec,mass_scales=[float('nan')]),
                       dict(spec,steering_scales=[2]),dict(spec,mass_scales=[0])):
            with self.assertRaises(ValueError): validate_spec(broken)

    def test_three_native_models_integer_physics_and_episode_constancy(self):
        env=MetaDrive(self.config())
        try:
            models=set(); masses={}; body_ids={}
            for seed in range(15):
                obs=env.reset(seed=seed)
                p=env.chassis_parameters; v=env._env.vehicle
                models.add(p['model'])
                self.assertIs(type(v),vehicle_type[p['model']])
                self.assertAlmostEqual(v.body.getMass(),p['mass'],places=3)
                self.assertEqual(v.max_steering,p['max_steering'])
                self.assertTrue(env.observation_space.contains(obs))
                self.assertFalse(obs['ctx_mask'].any())
                # Metadata is separate from state/context. No privileged labels.
                self.assertEqual(obs['ctx_history'].shape,(5,6))
                self.assertEqual(obs['state'].shape,(259,))
                for _ in range(2):
                    obs,_,_,info=env.step(np.array([.1,.2],np.float32))
                    self.assertEqual(info['chassis'],p)
                    self.assertAlmostEqual(v.body.getMass(),p['mass'],places=3)
                    self.assertAlmostEqual(v.system.getSteeringValue(0),.1*p['max_steering'],places=4)
                    self.assertAlmostEqual(float(obs['ctx_ego_state'][2]),np.linalg.norm(v.velocity),places=5)
                    self.assertAlmostEqual(float(obs['ctx_ego_state'][3]),-np.deg2rad(v.system.getSteeringValue(0)),places=6)
                masses.setdefault(p['model'],set()).add(p['mass'])
                body_ids.setdefault(p['model'],set()).add(id(v))
            self.assertEqual(models,{'default','m','l'})
            self.assertTrue(all(len(x)>1 for x in masses.values()))
            # Repeated explicit scenario has identical chassis and spawn/obs.
            first=env.reset(seed=3); params=env.chassis_parameters
            second=env.reset(seed=3)
            self.assertEqual(env.chassis_parameters,params)
            np.testing.assert_allclose(first['state'],second['state'])
        finally: env.close()

    def test_scaled_parameters_change_physical_response(self):
        accelerations=[]
        yaw_rates=[]
        for mode in ('mass','steering'):
            for scale in (.8,1.2):
                overrides={'chassis.models':'[default]', 'chassis.mass_scales':'[1.0]',
                           'chassis.steering_scales':'[1.0]'}
                overrides[f'chassis.{mode}_scales']=f'[{scale}]'
                env=MetaDrive(self.config(**overrides))
                try:
                    env.reset(seed=3); vehicle=env._env.vehicle
                    if mode=='mass':
                        speed0=np.linalg.norm(vehicle.velocity)
                        for _ in range(10): env.step(np.array([0,.3],np.float32))
                        accelerations.append(float(np.linalg.norm(vehicle.velocity)-speed0))
                    else:
                        vehicle.set_velocity(vehicle.heading,5.)
                        rates=[]
                        for _ in range(5):
                            env.step(np.array([.15,0],np.float32))
                            rates.append(abs(float(vehicle.body.getAngularVelocity()[2])))
                        yaw_rates.append(float(np.mean(rates)))
                finally: env.close()
        self.assertGreater(accelerations[0],accelerations[1]*1.2)
        self.assertGreater(yaw_rates[1],yaw_rates[0]*1.2)

    def test_training_draws_do_not_depend_on_map_count(self):
        sequences=[]
        for count in (1,7):
            env=MetaDrive(self.config(num_scenarios=count))
            try:
                sequence=[]
                for _ in range(5):
                    env.reset()
                    sequence.append(env.chassis_parameters)
                sequences.append(sequence)
            finally: env.close()
        self.assertEqual(*sequences)
        self.assertGreater(len({(p['model'],p['mass'],p['max_steering']) for p in sequences[0]}),1)

    def test_held_out_evaluation_is_repeatable(self):
        env=MetaDrive(self.config(**{'chassis.eval_models':'[s,xl]',
                                    'chassis.eval_mass_scales':'[1.0]',
                                    'chassis.eval_steering_scales':'[1.0]'}),evaluation=True)
        try:
            params=[]
            for _ in range(2):
                env.begin_evaluation()
                current=[]
                for _ in range(8):
                    env.reset(); current.append(env.chassis_parameters)
                params.append(current)
            self.assertEqual(*params)
            self.assertEqual({p['model'] for p in params[0]},{'s','xl'})
            self.assertTrue(all(p['mass_scale']==p['steering_scale']==1 for p in params[0]))
        finally: env.close()

if __name__=='__main__': unittest.main()
