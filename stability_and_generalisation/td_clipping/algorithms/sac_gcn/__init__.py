from .agent import SAC
from .replay import ReplayMemory


# ============================================================
# SAC + GCN 算法工厂入口
# ============================================================
# 当前版本已经把 GCN 接入 policy：
# - YAML 中的 algorithm.name = sac_gcn 会加载本目录
# - algorithm.params 中的图参数会进入 GCN policy
# - critic / replay / train-test 主循环暂时仍保持 baseline SAC 逻辑
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
