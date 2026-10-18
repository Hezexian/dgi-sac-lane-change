"""Uniform replay buffer shared by all five methods."""

import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim=48, action_dim=2):
        self.capacity = int(capacity)
        self.size = self.ptr = 0
        self.obs = np.empty((self.capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.action = np.empty((self.capacity, action_dim), np.float32)
        self.reward = np.empty(self.capacity, np.float32)
        self.terminal = np.empty(self.capacity, np.float32)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, terminal):
        i = self.ptr
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.terminal[i] = next_obs, terminal
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, indices=None):
        """Uniform sample without replacement: (obs, action, reward, next_obs, not_done).

        One minibatch is used for both the critic and the actor update.
        """
        ids = indices if indices is not None else random.sample(range(self.size), batch_size)
        return (
            self.obs[ids].reshape(batch_size, -1),
            self.action[ids],
            self.reward[ids],
            self.next_obs[ids].reshape(batch_size, -1),
            1.0 - self.terminal[ids],
        )
