"""Mass/friction randomization must reach all physical wheels and swap metadata."""
import unittest

import hydra
import numpy as np

from envs.chassis import sample_chassis, validate_spec
from envs.metadrive import MetaDrive


class FrictionTests(unittest.TestCase):
    def test_validation_and_fixed_steering(self):
        scales = [round(.6 + i * .1, 1) for i in range(9)]
        spec = dict(models=["default", "m", "l"], mass_scales=scales,
                    steering_scales=[1.0], friction_scales=scales)
        validate_spec(spec)
        seen = set()
        for seed in range(500):
            p = sample_chassis(spec, seed)
            self.assertEqual(p, sample_chassis(spec, seed))
            self.assertEqual(p["max_steering"], p["nominal_max_steering"])
            self.assertAlmostEqual(p["wheel_friction"], p["nominal_wheel_friction"] * p["friction_scale"])
            seen.add(p["friction_scale"])
        self.assertEqual(seen, set(scales))
        for invalid in ([0.0], [-1.0], [float("nan")], [.6, .6]):
            with self.assertRaises(ValueError):
                validate_spec(dict(spec, friction_scales=invalid))

    def test_native_wheel_friction_after_repeated_reset(self):
        with hydra.initialize(version_base=None, config_path="../configs"):
            c = hydra.compose(config_name="metadrive_tintersection_mass_friction_shallow",
                              overrides=["env.chassis.models=[default]", "env.traffic_density=0",
                                         "env.eval_num_scenarios=30"]).env
        env = MetaDrive(c, evaluation=True)
        try:
            seen = set()
            for seed in range(10000, 10030):
                obs = env.reset(seed=seed)
                p = env.chassis_parameters
                self.assertEqual(p["max_steering"], 40)
                seen.add(p["friction_scale"])
                self.assertTrue(env.observation_space.contains(obs))
                for _ in range(2):
                    self.assertAlmostEqual(float(obs["log_chassis_wheel_friction"]), p["wheel_friction"], places=6)
                    system = env._env.vehicle.system
                    self.assertEqual(system.getNumWheels(), 4)
                    self.assertTrue(all(np.isclose(system.getWheel(i).getFrictionSlip(), p["wheel_friction"], rtol=1e-6)
                                        for i in range(system.getNumWheels())))
                    self.assertAlmostEqual(env._env.vehicle.body.getMass(), p["mass"], places=3)
                    obs, _, _, _ = env.step(np.array([.05, .2], np.float32))
                    self.assertEqual(env.chassis_parameters, p)
            self.assertGreaterEqual(len(seen), 7)
            first = env.reset(seed=10000)
            p = env.chassis_parameters
            second = env.reset(seed=10000)
            self.assertEqual(p, env.chassis_parameters)
            np.testing.assert_allclose(first["state"], second["state"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
