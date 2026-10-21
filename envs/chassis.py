"""Episode-local chassis randomization for native MetaDrive vehicle presets."""

import numpy as np
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.vehicle_type import vehicle_type

MODELS = ("s", "default", "m", "l", "xl")
MODEL_IDS = {name: i for i, name in enumerate(MODELS)}
LOG_KEYS = ("model_id", "mass", "max_steering", "mass_scale", "steering_scale",
            "wheel_friction", "friction_scale")


def validate_spec(spec):
    models = spec["models"]
    if not models or len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ValueError(f"Chassis models must be a nonempty unique subset of {MODELS}")
    for key in ("mass_scales", "steering_scales", "friction_scales"):
        scales = np.asarray(spec.get(key, [1.0]), dtype=float)
        if scales.ndim != 1 or not scales.size or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError(f"{key} must contain finite positive scale choices")
        if len(set(scales)) != len(scales):
            raise ValueError(f"{key} choices must be unique")
    for model in models:
        cls = vehicle_type[model]
        nominal_steering = float(cls.PARAMETER_SPACE["max_steering"].low[0])
        if any(not 1 <= int(np.floor(nominal_steering * s + .5)) <= cls.MAX_STEERING
               for s in spec["steering_scales"]):
            raise ValueError("Scaled steering must round to 1..60 degrees")
        if any(int(np.floor(cls.MASS * s + .5)) < 1 for s in spec["mass_scales"]):
            raise ValueError("Scaled mass must round to a positive integer")


def sample_chassis(spec, seed):
    """Discrete independent scales; round positive physical values half upward."""
    rng = np.random.default_rng(int(seed))
    model = str(rng.choice(spec["models"]))
    cls = vehicle_type[model]
    nominal_mass = int(cls.MASS)
    nominal_steering = float(cls.PARAMETER_SPACE["max_steering"].low[0])
    sampled_mass_scale = float(rng.choice(spec["mass_scales"]))
    sampled_steering_scale = float(rng.choice(spec["steering_scales"]))
    sampled_friction_scale = float(rng.choice(spec.get("friction_scales", [1.0])))
    friction_space = cls.PARAMETER_SPACE["wheel_friction"]
    nominal_friction = float((float(friction_space.low[0]) + float(friction_space.high[0])) / 2)
    mass = int(np.floor(nominal_mass * sampled_mass_scale + .5))
    steering = int(np.floor(nominal_steering * sampled_steering_scale + .5))
    return dict(model=model, model_id=MODEL_IDS[model], nominal_mass=nominal_mass,
                nominal_max_steering=nominal_steering, mass=mass, max_steering=steering,
                sampled_mass_scale=sampled_mass_scale, sampled_steering_scale=sampled_steering_scale,
                mass_scale=mass / nominal_mass, steering_scale=steering / nominal_steering,
                nominal_wheel_friction=nominal_friction,
                wheel_friction=nominal_friction * sampled_friction_scale,
                sampled_friction_scale=sampled_friction_scale, friction_scale=sampled_friction_scale)


class ChassisManager(BaseManager):
    # After map/spawn selection (0/5), before native agent creation (10).
    PRIORITY = 6

    def reset(self):
        config = self.engine.global_config
        self.current = sample_chassis(config["chassis_spec"], config["chassis_random_seed"])
        model = self.current["model"]
        cls = vehicle_type[model]
        agent = config["agent_configs"][DEFAULT_AGENT]
        agent.update(dict(vehicle_model=model, mass=self.current["mass"],
                          max_steering=self.current["max_steering"],
                          wheel_friction=self.current["wheel_friction"]))
        # Native presets also randomly sample engine/brake forces. Pin them to
        # preset midpoints; model, mass, steering and friction follow the spec.
        for key in ("max_engine_force", "max_brake_force"):
            space = cls.PARAMETER_SPACE[key]
            agent[key] = float((float(space.low[0]) + float(space.high[0])) / 2)

    def after_reset(self):
        vehicle = self.engine.agent_manager.episode_created_agents[DEFAULT_AGENT]
        if type(vehicle) is not vehicle_type[self.current["model"]]:
            raise RuntimeError("Requested chassis class was not spawned")
        # Default/M/L MASS are class constants: updating config alone does not
        # update Bullet. Apply to the actual body, including pooled vehicles.
        vehicle.body.setMass(self.current["mass"])
        vehicle.config["mass"] = self.current["mass"]
        vehicle.max_steering = self.current["max_steering"]
        vehicle.config["max_steering"] = self.current["max_steering"]
        vehicle.system.setSteeringValue(0., 0)
        vehicle.system.setSteeringValue(0., 1)
        vehicle.config["wheel_friction"] = self.current["wheel_friction"]
        # Fetch fresh Bullet handles after all wheels have been created; handles
        # retained during createWheel() can be invalidated by native reallocation.
        for index in range(vehicle.system.getNumWheels()):
            wheel = vehicle.system.getWheel(index)
            wheel.setFrictionSlip(self.current["wheel_friction"])
            if not np.isclose(wheel.getFrictionSlip(), self.current["wheel_friction"], rtol=1e-6):
                raise RuntimeError("Wheel friction did not reach the physical engine")
        if not np.isclose(vehicle.body.getMass(), self.current["mass"], rtol=1e-6):
            raise RuntimeError("Chassis mass did not reach the physical engine")
        if vehicle.max_steering != self.current["max_steering"]:
            raise RuntimeError("Chassis steering did not reach the vehicle controller")
