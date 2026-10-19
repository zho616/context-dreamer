"""Adapter contract tests, including the real roundabout simulator (no rendering)."""

import unittest
from unittest.mock import Mock

import gymnasium as gym
import hydra
import numpy as np

from envs import make_env
from envs.metadrive import MetaDrive


class MetaDriveTests(unittest.TestCase):
    def test_random_entrances_and_fixed_evaluation_routes(self):
        config = self.config(traffic_density=0.2)
        env = MetaDrive(config, evaluation=True)
        try:
            entries = set()
            for seed in range(10000, 10020):
                env.reset(seed=seed)
                raw = env._env
                spawn = raw.engine.roundabout_spawn_manager.last_spawn
                entries.add(spawn["entry"])
                self.assertNotEqual(spawn["entry"], spawn["exit"])
                self.assertTrue(5 <= spawn["longitude"] <= 25)
                self.assertTrue(abs(spawn["lateral"]) <= 0.5)
                self.assertFalse(raw._is_out_of_road(raw.vehicle))
                self.assertFalse(raw._is_arrive_destination(raw.vehicle))
                self.assertGreater(len(raw.vehicle.navigation.checkpoints), 2)
                traffic = raw.engine.traffic_manager
                self.assertEqual(len(traffic.block_triggered_vehicles), 0)
                self.assertGreater(len(traffic._traffic_vehicles), 0)
                self.assertTrue(all(np.linalg.norm(vehicle.position - raw.vehicle.position) >= 8.0
                                    for vehicle in traffic._traffic_vehicles))
            self.assertEqual(entries, {0, 1, 2, 3})
            first = env.reset(seed=10000)["state"].copy()
            spawn = env._env.engine.roundabout_spawn_manager.last_spawn.copy()
            second = env.reset(seed=10000)["state"]
            self.assertEqual(spawn, env._env.engine.roundabout_spawn_manager.last_spawn)
            np.testing.assert_allclose(first, second)
        finally:
            env.close()

    def test_random_spawn_does_not_overlap_native_traffic(self):
        # Worker 3/map 81 used to overlap a traffic car and crash in step one.
        env = MetaDrive(self.config(), env_id=3)
        try:
            env.reset()
            self.assertEqual(env._env.current_seed, 81)
            traffic = env._env.engine.traffic_manager._traffic_vehicles
            self.assertTrue(all(np.linalg.norm(v.position - env._env.vehicle.position) >= 8 for v in traffic))
            _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertFalse(done or info["crash"])
        finally:
            env.close()
        # Reuse one training map: routes/positions still change each episode.
        config = self.config()
        config.num_scenarios = 1
        env = MetaDrive(config)
        try:
            starts = set()
            for _ in range(5):
                env.reset()
                spawn = env._env.engine.roundabout_spawn_manager.last_spawn
                starts.add((spawn["entry"], spawn["longitude"], spawn["lateral"]))
            self.assertEqual(len(starts), 5)
        finally:
            env.close()

    def config(self, **overrides):
        with hydra.initialize(version_base=None, config_path="../configs"):
            return hydra.compose(
                config_name="metadrive", overrides=[f"env.{key}={value}" for key, value in overrides.items()]
            ).env

    def test_real_roundabout_lidar_and_seed_splits(self):
        config = self.config(time_limit=5)
        for evaluation, start, count in ((False, 0, 100), (True, 10000, 20)):
            env = make_env(config, 0, evaluation=evaluation)
            try:
                obs = env.reset()
                raw = env.unwrapped._env
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(obs["state"].shape, (259,))
                self.assertNotIn("image", obs)
                self.assertTrue(start <= raw.current_seed < start + count)
                self.assertEqual([block.ID for block in raw.current_map.blocks], ["I", "O"])
                self.assertEqual(raw.config["map_config"]["lane_num"], 2)
                self.assertFalse(raw.config["use_render"])
                lidar_obs = raw.observations["default_agent"]
                np.testing.assert_allclose(obs["state"][-240:], lidar_obs.cloud_points)
                for _ in range(5):
                    obs, reward, done, _ = env.step(np.array([0, 0.2], dtype=np.float32))
                    self.assertTrue(np.isfinite(reward))
                    self.assertTrue(np.isfinite(obs["state"]).all())
                self.assertTrue(done)
                self.assertTrue(obs["is_last"])
                self.assertFalse(obs["is_terminal"])
                with self.assertRaises(ValueError):
                    env.unwrapped.reset(seed=start + count)
                first = env.unwrapped.reset(seed=start)["state"].copy()
                second = env.unwrapped.reset(seed=start)["state"]
                np.testing.assert_allclose(first, second)
            finally:
                env.close()

    def test_real_out_of_road_is_terminal(self):
        env = make_env(self.config(), 0)
        try:
            env.reset()
            env.unwrapped._env.vehicle.set_position([10000, 10000])
            obs, _, done, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertTrue(info["out_of_road"])
            self.assertTrue(done and obs["is_last"] and obs["is_terminal"])
            self.assertEqual(obs["log_out_of_road"], 1)
        finally:
            env.close()

    def test_real_parallel_lane_change_does_not_terminate(self):
        env = make_env(self.config(traffic_density=0), 0)
        try:
            env.reset()
            raw = env.unwrapped._env
            lanes = raw.vehicle.navigation.current_ref_lanes
            self.assertEqual(len(lanes), 2)
            centers = [lane.position(10, 0) for lane in lanes]
            for position in ((centers[0] + centers[1]) / 2, centers[1]):
                raw.vehicle.set_position(position)
                raw.vehicle.set_heading_theta(lanes[1].heading_theta_at(10))
                obs, _, done, info = env.step(np.zeros(2, dtype=np.float32))
                self.assertTrue(raw.vehicle.on_lane)
                self.assertFalse(info["out_of_road"])
                self.assertFalse(done or obs["is_terminal"])
        finally:
            env.close()

    def test_evaluation_seed_sequence_repeats_and_covers_all_scenarios(self):
        seeds = []
        for env_id in (0, 1):
            env = MetaDrive(self.config(), env_id, evaluation=True)
            try:
                env._env.reset = Mock(return_value=(np.zeros(259, dtype=np.float32), {}))
                for _ in range(10):
                    env.reset()
                    seeds.append(env._env.reset.call_args.kwargs["seed"])
                env.begin_evaluation()
                env.reset()
                self.assertEqual(env._env.reset.call_args.kwargs["seed"], 10000 + env_id)
            finally:
                env.close()
        self.assertEqual(sorted(seeds), list(range(10000, 10020)))

    def test_native_reward_overwrites_dense_terms_at_terminal(self):
        env = make_env(self.config(**{"reward.mode": "metadrive", "reward.lateral": 0,
                                      "reward.out_of_road": 5, "traffic_density": 0}), 0)
        try:
            for success in (True, False):
                env.reset()
                raw = env.unwrapped._env
                if success:
                    lane = raw.vehicle.navigation.final_lane
                    raw.vehicle.set_position(lane.position(lane.length - 2, 0))
                    raw.vehicle.set_heading_theta(lane.heading_theta_at(lane.length - 2))
                else:
                    raw.vehicle.set_position([10000, 10000])
                obs, reward, done, info = env.step(np.zeros(2, dtype=np.float32))
                self.assertTrue(done)
                self.assertEqual(bool(info["arrive_dest"]), success)
                self.assertEqual(reward, 10 if success else -5)
                self.assertEqual(obs["log_reward_lateral"], 0)
                self.assertEqual(obs["log_reward_progress"], 0)
                self.assertEqual(obs["log_reward_speed"], 0)
                self.assertEqual(obs["log_reward_terminal"], reward)
        finally:
            env.close()

    def test_real_horizon_stops_inside_action_repeat(self):
        env = make_env(self.config(action_repeat=3, time_limit=5), 0)
        try:
            env.reset()
            _, _, done, _ = env.step(np.array([0, 0.2], dtype=np.float32))
            self.assertFalse(done)
            obs, _, done, info = env.step(np.array([0, 0.2], dtype=np.float32))
            self.assertTrue(done and obs["is_last"] and not obs["is_terminal"])
            self.assertTrue(info["max_step"])
            self.assertEqual(env.unwrapped._env.episode_step, 5)
        finally:
            env.close()

    def test_action_repeat_stops_at_terminal_and_sums_reward_cost(self):
        env = MetaDrive.__new__(MetaDrive)
        env.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        env._action_repeat = 3
        env._env = Mock()
        state = np.zeros(259, dtype=np.float32)
        env._env.step.side_effect = [
            (state, 1.0, False, False, {"cost": 0.5}),
            (state, 2.0, True, False, {"cost": 1.0, "arrive_dest": True, "route_completion": 1.0}),
        ]
        obs, reward, done, _ = env.step(np.array([2.0, -2.0]))
        self.assertEqual(env._env.step.call_count, 2)
        np.testing.assert_array_equal(env._env.step.call_args.args[0], [1.0, -1.0])
        self.assertEqual(reward, 3.0)
        self.assertTrue(done and obs["is_terminal"])
        self.assertEqual(obs["log_success"], 1.0)
        self.assertEqual(obs["log_route_completion"], 1.0)
        self.assertEqual(obs["log_cost"], 1.5)
        for action in ([np.nan, 0], [0]):
            with self.assertRaises(ValueError):
                env.step(action)


if __name__ == "__main__":
    unittest.main()
