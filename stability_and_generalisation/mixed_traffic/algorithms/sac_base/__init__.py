from .agent import SAC
from .replay import ReplayMemory


def create_agent(num_inputs, action_space, args):
    return SAC(num_inputs, action_space, args)


def create_replay_buffer(capacity, seed):
    return ReplayMemory(capacity, seed)


__all__ = [
    "SAC",
    "ReplayMemory",
    "create_agent",
    "create_replay_buffer",
]
