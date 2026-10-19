"""Episode-local endpoint history from raw MetaDrive physics, in SI units."""

from collections import deque

import numpy as np


class EgoHistory:
    def __init__(self, dt, include_kinematics=True):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive simulation control interval required")
        self.dt = float(dt)
        self.include_kinematics = bool(include_kinematics)
        self.rows = deque(maxlen=5)
        self.previous_velocity = None

    def observe(self, vehicle, action=None):
        first = action is None
        if first:
            self.rows.clear()
        velocity = np.asarray(vehicle.velocity, dtype=np.float64).copy()
        theta = float(vehicle.heading_theta)
        forward = np.array([np.cos(theta), np.sin(theta)])
        # World-z angular velocity, clockwise positive; planar driving contract.
        yaw_rate = -float(vehicle.body.getAngularVelocity()[2])
        acceleration = np.zeros(2) if first else (velocity - self.previous_velocity) / self.dt
        state = np.array([acceleration @ forward, yaw_rate], dtype=np.float32)
        if self.include_kinematics:
            # Read the actual front-wheel settings; do not feed max_steering or
            # any other randomized chassis labels into the history encoder.
            angle = -np.deg2rad(np.mean([vehicle.system.getSteeringValue(i) for i in (0, 1)]))
            state = np.concatenate((state, np.array([np.linalg.norm(velocity), angle], np.float32)))
        canonical = np.zeros(2, np.float32) if first else np.array([action[1], -action[0]], np.float32)
        if not np.isfinite(state).all() or not np.isfinite(canonical).all() or (np.abs(canonical) > 1).any():
            raise ValueError("Invalid physical state or executed action")
        self.previous_velocity = velocity
        if not first:
            self.rows.append(np.concatenate((state, canonical)))
        history = np.zeros((5, len(state) + 2), np.float32)
        mask = np.zeros(5, bool)
        if self.rows:
            history[-len(self.rows):] = np.stack(self.rows)
            mask[-len(self.rows):] = True
        collision = any(getattr(vehicle, key, False) for key in (
            "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"))
        return dict(ctx_ego_state=state, ctx_prev_action=canonical,
                    ctx_history=history, ctx_mask=mask, ctx_collision=bool(collision))
