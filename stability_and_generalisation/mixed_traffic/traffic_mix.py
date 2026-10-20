"""Episode-level traffic interventions with independent, checkpointable RNGs."""
from copy import deepcopy
import hashlib

import gymnasium as gym
import numpy as np


BEHAVIOR_DISTRIBUTIONS = {
    'on': {
        'target_speed': [18.0, 30.0],
        'COMFORT_ACC_MAX': [2.0, 4.0],
        'COMFORT_ACC_MIN': [-6.0, -3.0],
        'TIME_WANTED': [1.0, 2.0],
        'POLITENESS': [0.0, 0.5],
        'LANE_CHANGE_MIN_ACC_GAIN': [0.1, 0.4],
        'LANE_CHANGE_DELAY': [0.5, 2.0],
    },
}
BEHAVIOR_DISTRIBUTIONS['aggressive'] = {
    **BEHAVIOR_DISTRIBUTIONS['on'],
    'TIME_WANTED': [0.5, 0.9],
    'LANE_CHANGE_MIN_ACC_GAIN': [0.0, 0.08],
    'LANE_CHANGE_DELAY': [0.25, 0.45],
}
# A fixed per-episode block prevents a different density (hence vehicle count)
# from shifting the behavior stream at every subsequent training episode.
TRAINING_BEHAVIOR_SLOTS = 60

PROTOCOL = {
    'version': 2,
    'rng': 'PCG64(int.from_bytes(SHA256(UTF8(f"{int(seed)}:{stream}")), "big"))',
    'density_stream': 'traffic-mix-density',
    'behavior_stream': 'traffic-mix-behavior',
    'evaluation_streams': 'same namespaces, derived anew from each evaluation seed',
    'density_uniform': [0.5, 2.0],
    'vehicles_count': 'int(round(30 * density))',
    'behavior_distributions': BEHAVIOR_DISTRIBUTIONS,
    'behavior_distribution_family': 'independent uniform per vehicle and parameter',
    'draw_order': {'vehicles': 'road.vehicles excluding controlled vehicles, in list order',
                   'parameters': list(BEHAVIOR_DISTRIBUTIONS['on'])},
    'training_behavior_slots': TRAINING_BEHAVIOR_SLOTS,
    'unused_training_slots': 'after actual vehicles, discard 7 uniform draws per unused slot',
    'evaluation_density': 'fixed by YAML; density RNG is derived but not consumed',
    'training_behavior_mode': 'on',
    'initial_speed': 'unchanged',
    'DELTA': 'unchanged environment randomization',
    'LANE_CHANGE_MAX_BRAKING_IMPOSED': 'unchanged',
    'timer': 'unchanged initial phase; subsequent check interval uses instance LANE_CHANGE_DELAY',
    'check_collisions': 'unchanged, False for HighwayEnvFast uncontrolled vehicles',
}


def dedicated_rng(seed, stream):
    """Do not use Python hash(), SeedSequence spawning, or any global RNG."""
    digest = hashlib.sha256(f'{int(seed)}:{stream}'.encode('utf8')).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))


def other_vehicles(env):
    base = env.unwrapped
    controlled = list(getattr(base, 'controlled_vehicles', ())) + [base.vehicle]
    return [v for v in base.road.vehicles if all(v is not ego for ego in controlled)]


def assign_behavior(env, rng, mode='on', *, slots=None):
    """Draw vehicle-major, parameter-minor; override instances, never class defaults.

    Training reserves 60 slots (maximum training population) so changing only the
    density stream cannot shift another episode's behavior draws. Unused draws
    are discarded after all real vehicles. Evaluation uses exactly its YAML count.
    """
    if mode not in BEHAVIOR_DISTRIBUTIONS:
        raise ValueError('Behavior mode must be on or aggressive')
    vehicles = other_vehicles(env)
    slots = len(vehicles) if slots is None else int(slots)
    if slots < len(vehicles):
        raise ValueError('Behavior slots cannot be less than actual other vehicles')
    distribution = BEHAVIOR_DISTRIBUTIONS[mode]
    values = {name: [] for name in distribution}
    for vehicle in vehicles:
        for name, (low, high) in distribution.items():
            value = float(rng.uniform(low, high))
            setattr(vehicle, name, value)
            values[name].append(value)
    rng.random((slots-len(vehicles))*len(distribution))
    stats = {'behavior_mode': mode, 'other_vehicles_count': len(vehicles)}
    for name, samples in values.items():
        stats[name+'_mean'] = float(np.mean(samples)) if samples else None
        stats[name+'_min'] = float(min(samples)) if samples else None
        stats[name+'_max'] = float(max(samples)) if samples else None
    return stats


class TrafficMixWrapper(gym.Wrapper):
    """Training only. reset(seed=...) seeds the environment, not the mix stream.

    Attach after the runner's initial seeded setup reset and before its first
    training reset. Separate dedicated streams draw density and behavior for
    each episode. Full environment pickles include this wrapper and both RNGs;
    state_dict also exposes the intervention state for checkpoint inspection.
    """
    def __init__(self, env, training_seed):
        super().__init__(env)
        self.training_seed = int(training_seed)
        self.default_env_config = deepcopy(env.unwrapped.config)
        self.density_rng = dedicated_rng(training_seed, PROTOCOL['density_stream'])
        self.behavior_rng = dedicated_rng(training_seed, PROTOCOL['behavior_stream'])
        self.last_episode = None

    def reset(self, **kwargs):
        density = float(self.density_rng.uniform(0.5, 2.0))
        count = int(round(30 * density))
        self.env.unwrapped.config.update(vehicles_density=density, vehicles_count=count)
        observation, info = self.env.reset(**kwargs)
        self.last_episode = {'density': density, 'vehicles_count': count,
                             **assign_behavior(self.env, self.behavior_rng,
                                               slots=TRAINING_BEHAVIOR_SLOTS)}
        return observation, info

    def state_dict(self):
        return deepcopy({'version': PROTOCOL['version'], 'training_seed': self.training_seed,
                         'density_rng': self.density_rng.bit_generator.state,
                         'behavior_rng': self.behavior_rng.bit_generator.state,
                         'last_episode': self.last_episode})

    def load_state_dict(self, state):
        if state['version'] != PROTOCOL['version'] or state['training_seed'] != self.training_seed:
            raise ValueError('Traffic-mix checkpoint protocol/seed mismatch')
        self.density_rng.bit_generator.state = deepcopy(state['density_rng'])
        self.behavior_rng.bit_generator.state = deepcopy(state['behavior_rng'])
        self.last_episode = deepcopy(state['last_episode'])


class HeteroTrafficWrapper(gym.Wrapper):
    """Evaluation only: per-seed behavior and unchanged YAML traffic density."""
    def __init__(self, env, mode='on', episode_records=None):
        super().__init__(env)
        if mode not in BEHAVIOR_DISTRIBUTIONS:
            raise ValueError('Behavior mode must be on or aggressive')
        self.mode = mode
        self.episode_records = episode_records

    def reset(self, *, seed=None, options=None):
        if seed is None:
            raise ValueError('Heterogeneous-traffic evaluation requires an episode seed')
        self.density_rng = dedicated_rng(seed, PROTOCOL['density_stream'])
        self.behavior_rng = dedicated_rng(seed, PROTOCOL['behavior_stream'])
        observation, info = self.env.reset(seed=seed, options=options)
        self.last_episode = {'episode_seed': int(seed),
                             'density': float(self.env.unwrapped.config['vehicles_density']),
                             'vehicles_count': int(self.env.unwrapped.config['vehicles_count']),
                             **assign_behavior(self.env, self.behavior_rng, self.mode)}
        if self.episode_records is not None:
            self.episode_records.append(deepcopy(self.last_episode))
        return observation, info
