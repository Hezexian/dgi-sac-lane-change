"""Frozen night-2 replay: exact branch distributions and restorable RNG state.

Only diagnostics scan the full buffer. Update, range normalization and sampling
use trees (inverse actor sampling uses exact rejection without an alias cache).
Priority values accepted by update_priorities are already transformed p, not TD.
"""
import copy
import math
import random

import numpy as np


VARIANTS = frozenset({"branch_uniform", "critic_per", "dual", "hre_uniform",
                      "hre_critic", "full_uniform", "full_critic", "dual_mirror"})
MIRROR_LEVELS = np.linspace(0., 1., 1001)
MIRROR_REFRESH = 1000
ARRAYS = ("obs", "next_obs", "action", "reward", "terminal", "steps",
          "priorities", "priority_updated")


class PriorityTree:
    def __init__(self, capacity):
        self.n = 1 << (int(capacity) - 1).bit_length()
        self.sum = np.zeros(2*self.n, np.float64)
        self.minimum = np.full(2*self.n, np.inf, np.float64)
        self.maximum = np.zeros(2*self.n, np.float64)

    def update(self, indices, raw, exponent):
        ids = np.asarray(indices, dtype=np.int64) + self.n
        raw = np.asarray(raw, dtype=np.float64)
        self.sum[ids] = raw**exponent
        self.minimum[ids] = raw
        self.maximum[ids] = raw
        while ids.size and ids[0] > 1:
            ids = np.unique(ids//2)
            self.sum[ids] = self.sum[ids*2] + self.sum[ids*2+1]
            self.minimum[ids] = np.minimum(self.minimum[ids*2], self.minimum[ids*2+1])
            self.maximum[ids] = np.maximum(self.maximum[ids*2], self.maximum[ids*2+1])

    def range_stats(self, start, end):
        left, right = self.n+int(start), self.n+int(end)
        mass, low, high = 0., np.inf, 0.
        while left < right:
            if left & 1:
                mass += self.sum[left]
                low = min(low, self.minimum[left])
                high = max(high, self.maximum[left])
                left += 1
            if right & 1:
                right -= 1
                mass += self.sum[right]
                low = min(low, self.minimum[right])
                high = max(high, self.maximum[right])
            left //= 2
            right //= 2
        return float(mass), float(low), float(high)

    def prefix(self, end):
        return self.range_stats(0, end)[0]

    def find(self, masses):
        masses = np.asarray(masses, dtype=np.float64).copy()
        nodes = np.ones(masses.shape, dtype=np.int64)
        while nodes.size and nodes[0] < self.n:
            left = 2*nodes
            go_right = masses >= self.sum[left]
            masses -= np.where(go_right, self.sum[left], 0.)
            nodes = left + go_right
        return nodes-self.n


class Replay:
    def __init__(self, capacity, shape, seed, variant="branch_uniform",
                 priority_mode="floor", alpha=.6):
        if int(capacity) < 1 or priority_mode not in {"floor", "epsilon", "standard"}:
            raise ValueError("Invalid capacity or priority mode")
        if variant not in VARIANTS or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid variant or exponent")
        self.capacity = int(capacity)
        self.shape = tuple(shape)
        self.variant, self.priority_mode, self.alpha = variant, priority_mode, float(alpha)
        self.size = self.ptr = 0
        self.rng = np.random.default_rng(seed)
        self.legacy_rng = random.Random(int(seed))
        self.obs = np.empty((self.capacity, *self.shape), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.action = np.empty((self.capacity, 2), np.float32)
        self.reward = np.empty(self.capacity, np.float32)
        self.terminal = np.empty(self.capacity, np.float32)
        self.steps = np.empty(self.capacity, np.int64)
        self.priorities = np.ones(self.capacity, np.float64)
        self.priority_updated = np.zeros(self.capacity, bool)
        self.tree = PriorityTree(self.capacity)
        self.mirror_table, self.mirror_calls = None, 0
        self.pbrs = None

    def enable_pbrs(self, gamma=.99, c=.05):
        if self.pbrs is None:
            self.raw_reward = np.empty(self.capacity, np.float64)
            self.shaping = np.zeros(self.capacity, np.float64)
            self.raw_reward[:self.size] = self.reward[:self.size]
        self.pbrs = {'gamma': float(gamma), 'c': float(c)}

    def disable_pbrs(self):
        self.pbrs = None
        for name in ('raw_reward', 'shaping'):
            if hasattr(self, name):
                delattr(self, name)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, terminal, step, *, raw_reward=None, shaping=0.):
        i = self.ptr
        if self.pbrs is not None:
            self.raw_reward[i] = reward - shaping if raw_reward is None else raw_reward
            self.shaping[i] = shaping
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.terminal[i], self.steps[i] = next_obs, terminal, step
        # No hidden floor in epsilon mode. Before the first item only, use 1.
        p = float(self.tree.maximum[1]) if self.size else 1.
        self.priorities[i] = p
        self.priority_updated[i] = False
        self.tree.update([i], [p], self.alpha)
        self.ptr = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def transformed_priorities(self, td):
        td = np.asarray(td, dtype=np.float64)
        if not np.all(np.isfinite(td)) or np.any(td < 0):
            raise ValueError("Expected finite nonnegative maximum absolute TD errors")
        if self.priority_mode == "standard":
            # Standard-priority update: standard proportional PER (Schaul et al. 2016), p = |delta| + eps.
            return td+1e-6
        return (np.maximum(td, 1.) if self.priority_mode == "floor" else td+1e-6)**.4

    def update_priorities(self, ids, values):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(ids) != len(values) or np.any(ids < 0) or np.any(ids >= self.size):
            raise ValueError("Invalid priority indices or value count")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Priorities must be positive and finite")
        if self.priority_mode == "floor" and np.any(values < 1.):
            raise ValueError("Floor mode expects already-transformed p >= 1")
        if not len(ids):
            return
        unique, inv = np.unique(ids, return_inverse=True)
        values = np.bincount(inv, weights=values)/np.bincount(inv)
        self.priorities[unique] = values
        self.priority_updated[unique] = True
        self.tree.update(unique, values, self.alpha)

    def _count(self, count):
        if not self.size:
            raise ValueError("Cannot sample an empty replay")
        return self.size if count is None else min(self.size, max(1, int(count)))

    def intervals(self, count):
        count = self._count(count)
        start, end = (self.ptr-count) % self.capacity, self.ptr
        if start < end:
            return [(start, end)]
        return [(start, self.capacity)] + ([(0, end)] if end else [])

    def recent_indices(self, count):
        return (self.ptr-1-np.arange(self._count(count))) % self.capacity

    def inside(self, ids, count):
        return ((self.ptr-1-np.asarray(ids)) % self.capacity) < self._count(count)

    def _range(self, count):
        stats = [self.tree.range_stats(a, b) for a, b in self.intervals(count)]
        return sum(s[0] for s in stats), min(s[1] for s in stats), max(s[2] for s in stats)

    def window_mass(self, count):
        return self._range(count)[0]

    def _outside_min(self, count):
        # Complement intervals in the ACTIVE physical domain, not unused leaves.
        parts, start = [], 0
        for a, b in sorted(self.intervals(count)):
            if start < a:
                parts.append((start, a))
            start = b
        if start < self.size:
            parts.append((start, self.size))
        return min((self.tree.range_stats(a, b)[1] for a, b in parts), default=np.inf)

    def sample_legacy(self, batch):
        if int(batch) > self.size or int(batch) < 0:
            raise ValueError("Invalid legacy batch")
        return np.asarray(self.legacy_rng.sample(range(self.size), int(batch)), np.int64)

    def uniform(self, n, count=None):
        count = self._count(count)
        return (self.ptr-1-self.rng.integers(0, count, size=n)) % self.capacity

    def prioritized(self, n, count=None):
        parts = self.intervals(self._count(count))
        masses = np.array([self.tree.range_stats(a, b)[0] for a, b in parts])
        chosen = self.rng.choice(len(parts), size=n, p=masses/masses.sum())
        u, targets = self.rng.random(n), np.empty(n)
        for j, (a, b) in enumerate(parts):
            ix = chosen == j
            offset = self.tree.prefix(a)
            # Avoid right-end rounding into a zero-mass/inactive leaf.
            targets[ix] = np.minimum(offset+u[ix]*masses[j],
                                     np.nextafter(offset+masses[j], offset))
        ids = self.tree.find(targets)
        if not np.all(self.inside(ids, count)):
            raise RuntimeError("Priority sampling escaped its window")
        return ids

    def inverse(self, n):
        hi, lo = float(self.tree.maximum[1]), float(self.tree.minimum[1])
        output, remaining = [], int(n)
        # This is an exact rejection sampler; no finite-attempt fallback to U.
        # A stalled pathological distribution raises instead of silently biasing.
        for _ in range(10000):
            if not remaining:
                return np.asarray(output, np.int64)
            ids = self.uniform(max(remaining*2, 16))
            accept = ((hi-self.priorities[ids]+lo)/hi)**self.alpha
            keep = ids[self.rng.random(len(ids)) < accept][:remaining]
            output.extend(keep.tolist())
            remaining -= len(keep)
        raise RuntimeError("Inverse-priority rejection sampler stalled")

    def mirror_weights(self, p):
        """Quantile-mirrored reverse priority: the sample at TD quantile q gets the
        positive weight of quantile 1-q, so Actor emphasis on low-TD samples mirrors the
        Critic emphasis on high-TD samples (same TV/ESS), immune to the long TD tail."""
        t = self.mirror_table
        F = np.interp(p, t, MIRROR_LEVELS)
        return np.interp(1.-F, MIRROR_LEVELS, t)**self.alpha

    def refresh_mirror(self):
        self.mirror_table = np.quantile(self.priorities[:self.size], MIRROR_LEVELS)

    def mirror_inverse(self, n):
        if self.mirror_table is None or self.mirror_calls % MIRROR_REFRESH == 0:
            self.refresh_mirror()
        self.mirror_calls += 1
        top = float(self.mirror_table[-1])**self.alpha
        output, remaining = [], int(n)
        for _ in range(100000):
            if not remaining:
                return np.asarray(output, np.int64)
            ids = self.uniform(max(remaining*4, 64))
            accept = self.mirror_weights(self.priorities[ids])/top
            keep = ids[self.rng.random(len(ids)) < accept][:remaining]
            output.extend(keep.tolist())
            remaining -= len(keep)
        raise RuntimeError("Mirror rejection sampler stalled")

    def critic_probabilities(self, ids, variant, window):
        """Return q(ids), min_i q(i), exact over the entire active domain."""
        if variant not in VARIANTS:
            raise ValueError(variant)
        count = self._count(window)
        ids = np.asarray(ids, np.int64)
        inside = self.inside(ids, count)
        if variant == "branch_uniform":
            return np.full(ids.shape, 1./self.size), 1./self.size
        if variant in {"hre_uniform", "hre_critic"}:
            q = .5/self.size + .5*inside/count
            return q, (.5/self.size if count < self.size else 1./self.size)
        mass = float(self.tree.sum[1])
        cond = self.priorities[ids]**self.alpha / mass
        if variant in {"critic_per", "dual", "dual_mirror"}:
            return .5/self.size+.5*cond, .5/self.size+.5*self.tree.minimum[1]**self.alpha/mass
        window_mass, window_min, _ = self._range(count)
        if variant == "full_critic":
            q = .5/self.size+.5*inside*self.priorities[ids]**self.alpha/window_mass
            low = .5/self.size if count < self.size else .5/self.size+.5*window_min**self.alpha/window_mass
            return q, low
        # full_uniform: global minimum p need not be outside the recent window.
        q = .5*inside/count+.5*cond
        low_in = .5/count+.5*window_min**self.alpha/mass
        low_out = .5*self._outside_min(count)**self.alpha/mass if count < self.size else np.inf
        return q, min(low_in, low_out)

    def draw(self, batch, variant=None, window=None, beta=.4):
        variant = self.variant if variant is None else variant
        if variant not in VARIANTS or batch < 2 or batch % 2:
            raise ValueError("Unknown variant or non-even batch")
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0,1]")
        count, half = self._count(window), int(batch)//2
        shared = self.uniform(half, count if variant in {"hre_uniform", "full_uniform"} else None)
        if variant in {"critic_per", "dual", "full_uniform", "dual_mirror"}:
            second_c = self.prioritized(half)
        elif variant == "full_critic":
            second_c = self.prioritized(half, count)
        else:
            second_c = self.uniform(half, count if variant == "hre_critic" else None)
        if variant == "dual_mirror":
            second_a = self.mirror_inverse(half)
        else:
            second_a = self.inverse(half) if variant in {"dual", "full_uniform", "full_critic"} else self.uniform(half)
        critic, actor = np.concatenate([shared, second_c]), np.concatenate([shared, second_a])
        q, low = self.critic_probabilities(critic, variant, count)
        weights = (low/q)**beta
        return critic, actor, weights.astype(np.float32)

    def diagnostics(self):
        if not self.size:
            return {"size": 0}
        p = self.priorities[:self.size]
        positive = p**self.alpha
        positive /= positive.sum()
        reverse = (p.max()-p+p.min())**self.alpha
        reverse /= reverse.sum()
        table = np.quantile(p, MIRROR_LEVELS)
        mirror = np.interp(1.-np.interp(p, table, MIRROR_LEVELS), MIRROR_LEVELS, table)**self.alpha
        mirror /= mirror.sum()
        return {"size": self.size, "priority_mode": self.priority_mode,
                "stored_floor_fraction": float(np.mean(p == 1.)) if self.priority_mode == "floor" else None,
                "not_updated_fraction": float(np.mean(~self.priority_updated[:self.size])),
                "priority_quantiles": np.quantile(p, [0, .01, .25, .5, .75, .99, 1]).tolist(),
                "positive_tv_uniform": float(.5*np.abs(positive-1/self.size).sum()),
                "positive_ess": float(1/(positive@positive)),
                "reverse_tv_uniform": float(.5*np.abs(reverse-1/self.size).sum()),
                "reverse_ess": float(1/(reverse@reverse)),
                "mirror_tv_uniform": float(.5*np.abs(mirror-1/self.size).sum()),
                "mirror_ess": float(1/(mirror@mirror))}

    def state_dict(self):
        state = {"format": "replay-buffer-v1", "capacity": self.capacity, "shape": self.shape,
                "size": self.size, "ptr": self.ptr, "alpha": self.alpha,
                "variant": self.variant, "priority_mode": self.priority_mode,
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "legacy_rng": self.legacy_rng.getstate(),
                "mirror_table": None if self.mirror_table is None else self.mirror_table.copy(),
                "mirror_calls": self.mirror_calls,
                **{k: getattr(self, k)[:self.size].copy() for k in ARRAYS}}
        if self.pbrs is not None:
            state['pbrs'] = {**self.pbrs, 'raw_reward': self.raw_reward[:self.size].copy(),
                             'shaping': self.shaping[:self.size].copy()}
        return state

    def load_state_dict(self, state):
        for name in ("capacity", "shape", "alpha", "variant", "priority_mode"):
            value = tuple(state[name]) if name == "shape" else state[name]
            if value != getattr(self, name):
                raise ValueError("Replay state mismatch: "+name)
        self.size, self.ptr = int(state["size"]), int(state["ptr"])
        if not 0 <= self.size <= self.capacity or not 0 <= self.ptr < self.capacity:
            raise ValueError("Invalid replay state size/pointer")
        if self.size < self.capacity and self.ptr != self.size:
            raise ValueError("Invalid partially-filled replay pointer")
        for name in ARRAYS:
            getattr(self, name)[:self.size] = state[name]
        if 'pbrs' in state:
            data = state['pbrs']
            self.enable_pbrs(data['gamma'], data['c'])
            self.raw_reward[:self.size] = data['raw_reward']
            self.shaping[:self.size] = data['shaping']
        else:
            self.disable_pbrs()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.legacy_rng.setstate(state["legacy_rng"])
        table = state.get("mirror_table")
        self.mirror_table = None if table is None else np.asarray(table, np.float64).copy()
        self.mirror_calls = int(state.get("mirror_calls", 0))
        self.tree = PriorityTree(self.capacity)
        self.tree.update(np.arange(self.size), self.priorities[:self.size], self.alpha)


def hre_window(size, step, episode_step, total_steps=1_000_000, horizon=200, min_window=5000, scale="legacy"):
    """legacy: graduate exponent horizon/k. ere: Wang & Ross (2019) Eq. 2 exponent k*1000/K, K=horizon."""
    if size < 0 or step < 0 or episode_step < 1 or total_steps <= 0 or horizon < 1 or min_window < 1:
        raise ValueError("Invalid HRE schedule input; episode_step starts at 1")
    if scale not in {"legacy", "ere"}:
        raise ValueError("Unknown HRE scale")
    eta = .996+.004*min(1., step/total_steps)
    exponent = horizon/episode_step if scale == "legacy" else min(episode_step, horizon)*1000./horizon
    return min(int(size), max(int(min_window), int(size*eta**exponent))), eta
