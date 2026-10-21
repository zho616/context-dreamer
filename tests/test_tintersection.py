"""Native T junction routes and backwards-compatible roundabout configuration."""
import unittest

import hydra
import numpy as np
from omegaconf import open_dict

from envs.metadrive import MetaDrive


class TIntersectionTests(unittest.TestCase):
    def config(self, name):
        with hydra.initialize(version_base=None, config_path="../configs"):
            return hydra.compose(config_name=name).env

    def test_three_entrances_reachable_routes_and_repeatable_physics(self):
        env = MetaDrive(self.config("metadrive_tintersection_shallow_swap1"), evaluation=True)
        try:
            entries = set()
            for seed in range(10000, 10020):
                obs = env.reset(seed=seed)
                raw = env._env
                self.assertEqual([block.ID for block in raw.current_map.blocks], ["I", "T"])
                spawn = raw.engine.roundabout_spawn_manager.last_spawn
                entries.add(spawn["entry"])
                self.assertIn(spawn["exit"], {0, 1, 2} - {spawn["entry"]})
                self.assertGreater(len(raw.vehicle.navigation.checkpoints), 2)
                self.assertFalse(raw._is_out_of_road(raw.vehicle))
                self.assertFalse(raw._is_arrive_destination(raw.vehicle))
                self.assertEqual(obs["ctx_history"].shape, (5, 6))
                self.assertEqual(obs["state"].shape, (259,))
                self.assertAlmostEqual(raw.vehicle.body.getMass(), env.chassis_parameters["mass"], places=3)
                traffic = raw.engine.traffic_manager._traffic_vehicles
                self.assertGreater(len(traffic), 0)
                self.assertTrue(all(np.linalg.norm(v.position - raw.vehicle.position) >= 8 for v in traffic))
                self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(entries, {0, 1, 2})
            first = env.reset(seed=10000)
            profile = env.chassis_parameters
            second = env.reset(seed=10000)
            self.assertEqual(profile, env.chassis_parameters)
            np.testing.assert_allclose(first["state"], second["state"])
            for _ in range(5):
                obs, _, done, _ = env.step(np.array([.05, .2], np.float32))
                self.assertFalse(done)
            self.assertTrue(obs["ctx_mask"].all())
            self.assertEqual(obs["ctx_ego_state"].shape, (4,))
        finally:
            env.close()

    def test_legacy_config_without_map_block_still_uses_roundabout(self):
        config = self.config("metadrive")
        with open_dict(config):
            del config.map_block
        env = MetaDrive(config, evaluation=True)
        try:
            env.reset(seed=10000)
            self.assertEqual([block.ID for block in env._env.current_map.blocks], ["I", "O"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
