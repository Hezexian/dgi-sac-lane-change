"""WP-I2 target clipping and replay-only potential shaping (no RNG draws)."""
import math

import numpy as np

STABILIZERS = ('none', 'qclip', 'pbrs', 'qclip_pbrs')
PBRS_C = .05


def validate_stabilizer(variant, stabilizer):
    if stabilizer not in STABILIZERS:
        raise ValueError(f'Unknown stabilizer: {stabilizer}')
    if stabilizer != 'none' and variant in {'ddpg', 'td3'}:
        raise ValueError(f'{variant} does not support stabilizer={stabilizer}')


def has_pbrs(stabilizer):
    return stabilizer in {'pbrs', 'qclip_pbrs'}


def has_qclip(stabilizer):
    return stabilizer in {'qclip', 'qclip_pbrs'}


def reward_upper_bound(config):
    """HighwayEnv._rewards components are in [0,1]; the off-road rule replaces r.

    Read from the resolved config or the frozen YAML. Fail closed if future
    reward terms/protocols cannot be justified by the audited HighwayEnv sum.
    """
    env = config['env']
    known = {'collision_reward', 'right_lane_reward', 'high_speed_reward',
             'lane_change_reward', 'on_road_reward'}
    unknown = {k for k in env if k.endswith('_reward')} - known - {'normalize_reward'}
    if unknown or env.get('normalize_reward', False):
        raise ValueError(f'Unaudited reward configuration: {sorted(unknown)}')
    if config['run']['env_name'] not in {'highway-fast-v0', 'highway-v0'}:
        raise ValueError('Q bound is audited only for the frozen highway reward')
    # _reward sums config weights against bounded nonnegative components.
    # lane_change is inactive in this env but a positive weight is conservative.
    upper = sum(max(0., float(env.get(k, 0.))) for k in known)
    return max(upper, float(env.get('collision_reward', -2.)))


def vehicle_max_speed():
    """Audited nominal limit; frozen [0,30] instance limits leave ample margin."""
    from importlib.metadata import version
    from highway_env.vehicle.kinematics import Vehicle
    if version('highway-env') != '1.10.2':
        raise ValueError('Shaped Q bound requires audited highway-env 1.10.2')
    maximum = float(Vehicle.MAX_SPEED)
    if not math.isfinite(maximum) or maximum <= 0 or float(Vehicle.MIN_SPEED) < -maximum:
        raise ValueError('Vehicle speed limits do not justify the shaped Q bound')
    return maximum


def q_upper_bound(config, stabilizer='qclip'):
    train = config['train']
    gamma, alpha = float(train['gamma']), float(train['alpha'])
    if not 0 <= gamma < 1 or alpha != .025 or train.get('automatic_entropy_tuning'):
        raise ValueError('Q clipping requires gamma in [0,1) and fixed alpha=.025')
    upper = (reward_upper_bound(config) + alpha * math.log(4.)) / (1. - gamma)
    if stabilizer == 'qclip_pbrs':
        maximum = vehicle_max_speed()
        speed_range = config['env'].get('action', {}).get('speed_range')
        if speed_range and max(abs(float(v)) for v in speed_range) > maximum:
            raise ValueError('Configured speed range exceeds audited Vehicle.MAX_SPEED')
        upper += PBRS_C * maximum
    return upper


def clip_target(y, q_max):
    """Only upper-clip; untouched elements keep their exact floating point bits."""
    fraction = float((y > q_max).float().mean())
    return y.clamp(max=q_max), fraction


def longitudinal_speed(env):
    vehicle = env.unwrapped.vehicle
    return float(vehicle.speed * math.cos(vehicle.heading))


def shaping_delta(speed, next_speed, terminated, gamma=.99, c=PBRS_C):
    # Truncation is deliberately not an argument: it must retain the bootstrap.
    return gamma * (0. if terminated else c * next_speed) - c * speed


def relabel_replay(memory, source_stabilizer, target_stabilizer, gamma=.99, c=PBRS_C):
    """Fork-only conversion, preserving sample order, priorities and all RNG.

    Earlier fixed-scale checkpoints lack physical speeds. Their ego vx is absolute
    longitudinal speed / 30, stored in float32. Record this approximation.
    Newly collected data use physical speeds and store the exact raw reward.
    """
    source, target = has_pbrs(source_stabilizer), has_pbrs(target_stabilizer)
    n = memory.size
    if source and getattr(memory, 'pbrs', None) is None:
        raise ValueError('PBRS source replay lacks raw reward/shaping metadata')
    if source == target:
        if target and (memory.pbrs['gamma'] != gamma or memory.pbrs['c'] != c):
            raise ValueError('PBRS source potential parameters mismatch')
        return {'action': 'preserved', 'transitions': n, 'approximate': False}
    if not target:
        memory.reward[:n] = memory.raw_reward[:n]
        memory.disable_pbrs()
        return {'action': 'restored_raw_rewards', 'transitions': n, 'approximate': False}
    memory.enable_pbrs(gamma, c)
    speed = memory.obs[:n].reshape(n, 8, 6)[:, 0, 3].astype(np.float64) * 30.
    next_speed = memory.next_obs[:n].reshape(n, 8, 6)[:, 0, 3].astype(np.float64) * 30.
    delta = gamma * c * next_speed * (1. - memory.terminal[:n]) - c * speed
    memory.raw_reward[:n] = memory.reward[:n]
    memory.shaping[:n] = delta
    memory.reward[:n] = memory.raw_reward[:n] + delta
    return {'action': 'relabelled_from_ego_vx', 'transitions': n,
            'approximate': True, 'speed_source': 'float32 observation[ego,vx] * 30',
            'priorities': 'preserved; recomputed when sampled',
            'new_transitions': 'physical vehicle.speed*cos(heading)'}
