"""Lane-change environment with custom observation."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation

FEATURES = ("presence", "x", "y", "vx", "vy", "heading")
FEATURE_SCALE = np.array([1.0, 100.0, 12.0, 30.0, 30.0, 1.0], dtype=np.float32)


class EgoCentricScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observation = env.unwrapped.config["observation"]
        if tuple(observation["features"]) != FEATURES or observation.get("absolute", False):
            raise ValueError("Expected ego-relative Kinematics features " + ", ".join(FEATURES))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=env.observation_space.shape,
                                                dtype=np.float32)

    def observation(self, observation):
        obs = np.array(observation, dtype=np.float32, copy=True)
        obs[0, 1] = 0.0
        return obs / FEATURE_SCALE


class OffroadPenalty(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        if not self.env.unwrapped.vehicle.on_road:
            reward = float(self.env.unwrapped.config["collision_reward"])
            info["offroad_penalised"] = True
        return obs, reward, terminated, truncated, info


def make_env(env_name, env_config, render_mode=None):
    import highway_env

    kwargs = {"config": deepcopy(env_config)}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_name, **kwargs)
    return FlattenObservation(EgoCentricScale(OffroadPenalty(env)))
