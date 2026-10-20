"""ego-centric-fixed-scale observation and reward protocol shared by training, validation, screening and benchmarks.

Observation and reward corrections (2026-10-02):
1. Ego-centric fixed-scale observation. highway-env Kinematics with absolute=False reports the ego row in
   absolute road coordinates (x grows to ~1.3 km) while other rows are ego-relative. The wrapper zeroes the ego x
   (absolute longitudinal progress carries no driving information) and divides each feature column by a
   fixed constant, following the earlier `lc` implementation (SCALE=[1,100,12,30,30,1,1] without cos/sin).
2. Off-road is an unsafe failure (R1.6): highway-env multiplies the step reward by on_road, giving 0 on
   off-road (better than a collision, -2) and also 0 for a collision that leaves the road. The wrapper assigns the
   collision reward whenever the ego is off the road. The original reward is kept in info['raw_reward'].
"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation

FEATURES = ('presence', 'x', 'y', 'vx', 'vy', 'heading')
FEATURE_SCALE = np.array([1., 100., 12., 30., 30., 1.], dtype=np.float32)
PROTOCOL = {'name': 'ego-centric-fixed-scale', 'ego_x_zeroed': True, 'feature_scale': FEATURE_SCALE.tolist(),
            'features': list(FEATURES), 'offroad_reward': 'collision_reward', 'learning_starts': 10000}


class EgoCentricScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        features = env.unwrapped.config['observation']['features']
        if tuple(features) != FEATURES:
            raise ValueError(f'ego-centric-fixed-scale expects features {FEATURES}, got {features}')
        if env.unwrapped.config['observation'].get('absolute', False):
            raise ValueError('ego-centric-fixed-scale expects ego-relative other vehicles (absolute=False)')
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def observation(self, observation):
        obs = np.array(observation, dtype=np.float32, copy=True)
        obs[0, 1] = 0.
        return obs / FEATURE_SCALE


class OffroadPenalty(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info['raw_reward'] = float(reward)
        if not self.env.unwrapped.vehicle.on_road:
            reward = float(self.env.unwrapped.config['collision_reward'])
            info['offroad_penalised'] = True
        return obs, reward, terminated, truncated, info


def make_env(name, config):
    import highway_env  # noqa: F401 registers environments
    env = gym.make(name, config=deepcopy(config))
    env = FlattenObservation(EgoCentricScale(OffroadPenalty(env)))
    if tuple(env.observation_space.shape) != (48,):
        env.close()
        raise ValueError('Expected flattened 8x6 observation')
    return env
