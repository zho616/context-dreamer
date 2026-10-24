"""ScenarioNet nuPlan scenes with physical ego control and the state4 context API."""
from pathlib import Path
import gymnasium as gym
import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.constants import TerminationState
from metadrive.scenario.utils import read_dataset_summary
from envs.metadrive import MetaDrive

class NuPlanPaperEnv(ScenarioEnv):
    @classmethod
    def default_config(cls):
        config=super().default_config()
        config.update(dict(speed_reward=.1,spawn_random_seed=0,chassis_enabled=False,
                           chassis_random_seed=0,chassis_spec=dict(models=['default'],mass_scales=[1.],steering_scales=[1.],friction_scales=[1.])))
        config['vehicle_config'].update(dict(max_speed_km_h=50.))
        return config

    def setup_engine(self):
        super().setup_engine()
        if self.config['chassis_enabled']:
            from envs.chassis import ChassisManager
            # ScenarioMapManager fills initial pose/velocity at priority0;
            # chassis priority6 then selects the physical preset before ego creation.
            self.engine.register_manager('chassis_manager',ChassisManager())

    def reward_function(self,vehicle_id):
        vehicle=self.agents[vehicle_id];nav=vehicle.navigation
        progress=self.config['driving_reward']*(nav.current_longitude-nav.last_longitude)
        speed=self.config['speed_reward']*vehicle.speed_km_h/vehicle.max_speed_km_h
        collision=any(getattr(vehicle,k,False)for k in ['crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk'])
        terminal=(self.config['success_reward']if self._is_arrive_destination(vehicle)else
                  -self.config['crash_vehicle_penalty']if collision else
                  -self.config['out_of_road_penalty']if self._is_out_of_road(vehicle)else 0.)
        reward=float(progress+speed+terminal)
        return reward,dict(step_reward=reward,reward_progress=float(progress),reward_speed=float(speed),reward_lateral=0.,reward_terminal=float(terminal),route_completion=float(nav.route_completion))

    def done_function(self,vehicle_id):
        done,info=super().done_function(vehicle_id)
        # A replay scene cannot supply valid traffic beyond its recorded duration.
        # End at its last frame as a time-limit truncation, preserving bootstrap.
        if self.episode_lengths[vehicle_id]>=self.engine.data_manager.current_scenario_length-1:
            info[TerminationState.MAX_STEP]=True
        vehicle=self.agents[vehicle_id]
        info.update(off_lane=not vehicle.on_lane,yellow_line=vehicle.on_yellow_continuous_line,white_line=vehicle.on_white_continuous_line)
        return done,info

class NuPlanDrive(MetaDrive):
    """Reuse the tested state/action/history/physical-chassis boundary from PG tasks."""
    def __init__(self,config,env_id=0,evaluation=False,*,use_render=False):
        if config.task!='metadrive_nuplan':raise ValueError('Expected metadrive_nuplan')
        if str(config.reward.mode)!='paper':raise ValueError('nuPlan adapter currently implements paper reward only')
        self._action_repeat=int(config.action_repeat)
        if self._action_repeat!=1:raise ValueError('nuPlan replay alignment requires action_repeat=1')
        folder=Path(str(config.scenario_eval_dir if evaluation else config.scenario_train_dir)).expanduser().resolve()
        if not(folder/'dataset_summary.pkl').exists():raise FileNotFoundError(f'Missing converted scene summary: {folder}')
        _,scenario_ids,_=read_dataset_summary(str(folder))
        requested=int(config.eval_num_scenarios if evaluation else config.num_scenarios)
        self._num_scenarios=len(scenario_ids)if requested<0 else min(requested,len(scenario_ids))
        if self._num_scenarios<1:raise ValueError('Empty scenario split')
        self._start_seed=0 # Train/eval use physically separate, log-disjoint databases.
        self._rng=np.random.default_rng(int(config.seed)+env_id+(1000000 if evaluation else 0))
        self._evaluation=evaluation;self._env_id=env_id;self._context_episode_counter=0
        self._eval_stride=min(int(config.eval_env_num),int(config.eval_episode_num));self._eval_next_index=env_id
        self._expected_dataset=str(config.scenario_dataset)
        chassis=config.chassis;self._chassis_enabled=bool(chassis.enabled);spec=None
        if self._chassis_enabled:
            from envs.chassis import validate_spec
            prefix='eval_'if evaluation else ''
            spec={k:list(getattr(chassis,prefix+k))for k in ['models','mass_scales','steering_scales','friction_scales']}
            validate_spec(spec);self._chassis_seed=int(config.seed)
            self._chassis_rng=np.random.default_rng(np.random.SeedSequence([self._chassis_seed,env_id,72109]))
        self._env=NuPlanPaperEnv(dict(data_directory=str(folder),start_scenario_index=0,num_scenarios=self._num_scenarios,
            store_map=False,store_data=False,use_render=use_render,image_observation=False,force_destroy=True,
            decision_repeat=int(config.decision_repeat),horizon=int(config.time_limit),truncate_as_terminate=False,
            reactive_traffic=bool(config.reactive_traffic),set_static=False,use_bounding_box=False,
            crash_vehicle_done=True,crash_object_done=True,crash_human_done=True,
            driving_reward=float(config.reward.progress),speed_reward=float(config.reward.speed),
            success_reward=float(config.reward.success),crash_vehicle_penalty=float(config.reward.crash),
            out_of_road_penalty=float(config.reward.out_of_road),chassis_enabled=self._chassis_enabled,
            chassis_spec=spec or dict(models=['default'],mass_scales=[1.],steering_scales=[1.],friction_scales=[1.]),
            chassis_random_seed=0,vehicle_config=dict(max_speed_km_h=float(config.reward.max_speed_km_h),
              lidar=dict(num_lasers=int(config.lidar.num_lasers),distance=float(config.lidar.distance),num_others=int(config.lidar.num_others)),
              side_detector=dict(num_lasers=0,distance=50),lane_line_detector=dict(num_lasers=0,distance=50)),log_level=40))
        self.action_space=self._env.action_space
        self._context_history=None
        spaces={'state':self._env.observation_space}
        sample=self._obs(np.zeros(self._env.observation_space.shape,np.float32))
        for key,value in sample.items():
            if key=='state':continue
            spaces[key]=gym.spaces.Box(0,1,shape=(),dtype=np.bool_)if isinstance(value,bool)else gym.spaces.Box(-np.inf,np.inf,shape=(),dtype=np.float32)
        if self._chassis_enabled:
            from envs.chassis import LOG_KEYS
            for key in LOG_KEYS:spaces['log_chassis_'+key]=gym.spaces.Box(-np.inf,np.inf,shape=(),dtype=np.float32)
        if config.context_enabled:
            from envs.ego_history import EgoHistory
            self._context_history=EgoHistory(float(config.decision_repeat)*.02,include_kinematics=True)
            for key,shape in [('ctx_ego_state',(4,)),('ctx_prev_action',(2,)),('ctx_history',(5,6))]:spaces[key]=gym.spaces.Box(-np.inf,np.inf,shape=shape,dtype=np.float32)
            spaces['ctx_mask']=gym.spaces.Box(0,1,(5,),dtype=np.bool_);spaces['ctx_collision']=gym.spaces.Box(0,1,(),dtype=np.bool_)
            spaces['ctx_episode_uid']=gym.spaces.Box(0,np.iinfo(np.int64).max,(2,),dtype=np.int64)
        self.observation_space=gym.spaces.Dict(spaces)

    def reset(self,*,seed=None,options=None):
        obs=super().reset(seed=seed,options=options)
        scenario=self._env.engine.data_manager.current_scenario
        dataset=scenario['metadata']['dataset']
        if dataset!=self._expected_dataset:raise ValueError(f'Expected {self._expected_dataset}, found {dataset}')
        timestamps=np.asarray(scenario['metadata']['ts'])
        if len(timestamps)>1 and not np.allclose(np.diff(timestamps),self._env.config['decision_repeat']*.02,atol=1e-4):
            raise ValueError('Scenario timestamps do not match the10Hz controller')
        # Ego must remain under physical action control, not a trajectory teleport.
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        policy=self._env.engine.get_policy(self._env.vehicle.id)
        if isinstance(policy,ReplayEgoCarPolicy):raise RuntimeError('Replay ego policy cannot train action response')
        return obs
