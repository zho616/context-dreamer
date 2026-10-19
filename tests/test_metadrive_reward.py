import unittest
from types import SimpleNamespace

import numpy as np
from metadrive.component.lane.straight_lane import StraightLane

from envs.metadrive import RoundaboutEnv


class RewardTests(unittest.TestCase):
    def test_paper_reward_adds_events_without_lateral_or_heading_gate(self):
        for success, collision, off_road, terminal in (
            (False, False, False, 0), (True, False, False, 10),
            (False, True, False, -20), (False, False, True, -10),
        ):
            env = self.environment(success=success, collision=collision, off_road=off_road)
            env.config.update(reward_mode="paper", crash_vehicle_penalty=20,
                              out_of_road_penalty=10, lateral_penalty=0)
            vehicle = env.agents["agent"]
            vehicle.max_speed_km_h = 50
            vehicle.heading = np.array([-1, 0])
            vehicle.last_position = np.array([12, 1])
            vehicle.position = np.array([10, 1])
            reward, info = RoundaboutEnv.reward_function(env, "agent")
            self.assertAlmostEqual(info["reward_progress"], -2)
            self.assertAlmostEqual(info["reward_speed"], 0.08)
            self.assertEqual(info["reward_lateral"], 0)
            self.assertEqual(info["reward_terminal"], terminal)
            self.assertAlmostEqual(reward, -2 + 0.08 + terminal)

    def environment(self, success=False, collision=False, off_road=False):
        lane = StraightLane([0, 0], [100, 0], width=4)
        vehicle = SimpleNamespace(
            lane=lane, navigation=SimpleNamespace(current_ref_lanes=[lane], route_completion=0.25),
            last_position=np.array([10, 1]), position=np.array([12, 1]),
            heading=np.array([1, 0]), speed_km_h=40, max_speed_km_h=80, crash_vehicle=collision,
        )
        return SimpleNamespace(
            agents={"agent": vehicle},
            config={"reward_mode": "additive", "driving_reward": 1, "speed_reward": 0.1, "lateral_penalty": 0.5,
                    "success_reward": 10, "crash_vehicle_penalty": 5, "out_of_road_penalty": 5},
            _is_arrive_destination=lambda vehicle: success,
            _is_out_of_road=lambda vehicle: off_road,
        )

    def test_additive_terminal_rewards_and_component_sum(self):
        for success, collision, off_road, terminal in (
            (False, False, False, 0), (True, False, False, 10),
            (False, True, False, -5), (False, False, True, -5),
            (False, True, True, -5),
        ):
            env = self.environment(success, collision, off_road)
            reward, info = RoundaboutEnv.reward_function(env, "agent")
            self.assertAlmostEqual(reward, 2 + 0.05 - 0.125 + terminal)
            self.assertAlmostEqual(sum(info[f"reward_{key}"] for key in ("progress", "speed", "lateral", "terminal")),
                                   reward)

    def test_lateral_constraint_exists_while_stationary(self):
        env = self.environment()
        vehicle = env.agents["agent"]
        vehicle.last_position = vehicle.position.copy()
        vehicle.speed_km_h = 0
        reward, _ = RoundaboutEnv.reward_function(env, "agent")
        self.assertAlmostEqual(reward, -0.125)

    def test_reverse_heading_disables_progress_and_speed(self):
        env = self.environment()
        env.agents["agent"].heading = np.array([-1, 0])
        reward, _ = RoundaboutEnv.reward_function(env, "agent")
        self.assertAlmostEqual(reward, -0.125)

    def test_parallel_lane_uses_its_own_center(self):
        env = self.environment()
        vehicle = env.agents["agent"]
        lane = StraightLane([0, 4], [100, 4], width=4)
        vehicle.navigation.current_ref_lanes.append(lane)
        vehicle.lane = lane
        vehicle.last_position = np.array([10, 4])
        vehicle.position = np.array([12, 4])
        _, info = RoundaboutEnv.reward_function(env, "agent")
        self.assertEqual(info["reward_lateral"], 0)

    def test_gentle_lateral_weight_preserves_forward_reward(self):
        env = self.environment()
        env.config["lateral_penalty"] = 0.2
        vehicle = env.agents["agent"]
        vehicle.position = np.array([10 + 20 / 3.6 * 0.1, 2])
        vehicle.last_position = np.array([10, 2])
        vehicle.speed_km_h = 20
        reward, info = RoundaboutEnv.reward_function(env, "agent")
        self.assertAlmostEqual(info["reward_lateral"], -0.1)
        self.assertGreater(reward, 0.48)


if __name__ == "__main__":
    unittest.main()
