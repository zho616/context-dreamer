"""Reward parity, replay-horizon truncation and nuPlan configuration contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, PropertyMock
import hydra
from envs.nuplan import NuPlanPaperEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.constants import TerminationState

class NuPlanAdapterTests(unittest.TestCase):
    def test_paper_reward_adds_terminal_and_uses_route_progress(self):
        vehicle=SimpleNamespace(navigation=SimpleNamespace(current_longitude=3.,last_longitude=2.,route_completion=.4),speed_km_h=25.,max_speed_km_h=50.,crash_vehicle=False,crash_object=False,crash_building=False,crash_human=False,crash_sidewalk=False)
        fake=SimpleNamespace(agents={'ego':vehicle},config={'driving_reward':1.,'speed_reward':.1,'success_reward':10.,'crash_vehicle_penalty':20.,'out_of_road_penalty':10.},_is_arrive_destination=lambda v:False,_is_out_of_road=lambda v:False)
        reward,info=NuPlanPaperEnv.reward_function(fake,'ego');self.assertAlmostEqual(reward,1.05)
        vehicle.crash_vehicle=True;reward,_=NuPlanPaperEnv.reward_function(fake,'ego');self.assertAlmostEqual(reward,-18.95)
        fake._is_arrive_destination=lambda v:True;reward,_=NuPlanPaperEnv.reward_function(fake,'ego');self.assertAlmostEqual(reward,11.05)

    def test_recording_end_is_truncation_not_natural_termination(self):
        fake=NuPlanPaperEnv.__new__(NuPlanPaperEnv);fake.episode_lengths={'ego':4}
        with patch.object(ScenarioEnv,'done_function',return_value=(False,{})):
            # Properties are backed by the native engine singleton; patch them
            # rather than constructing a second physics engine in a unit test.
            with patch.object(NuPlanPaperEnv,'engine',new_callable=PropertyMock)as engine,patch.object(NuPlanPaperEnv,'agents',new_callable=PropertyMock,return_value={'ego':SimpleNamespace(on_lane=True,on_yellow_continuous_line=False,on_white_continuous_line=False)}):
                engine.return_value=SimpleNamespace(data_manager=SimpleNamespace(current_scenario_length=5))
                done,info=fake.done_function('ego');self.assertFalse(done);self.assertTrue(info[TerminationState.MAX_STEP])

    def test_shallow_matches_old11_without_wm_feedback(self):
        with hydra.initialize(config_path='../configs',version_base=None):c=hydra.compose(config_name='metadrive_nuplan_shallow')
        self.assertEqual(c.model.context.injection,'dali_shallow');self.assertEqual(c.model.context.wm_grad_scale,0)
        self.assertEqual(c.model.context.aux_weight,1);self.assertEqual(c.model.context.swap_weight,1)
        self.assertNotEqual(c.env.scenario_train_dir,c.env.scenario_eval_dir)

if __name__=='__main__':unittest.main()
