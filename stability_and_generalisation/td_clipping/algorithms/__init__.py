from importlib import import_module


# ============================================================
# 算法加载入口
# ============================================================
# 顶层 train.py / test.py 不直接依赖某个具体算法实现，
# 而是根据 YAML 中的 algorithm.name 动态加载对应算法包。
#
# 约定：
# - 每个算法目录位于 algorithms/<algorithm_name>/
# - 算法包需要至少暴露：
#   - create_agent(...)
#   - create_replay_buffer(...)
def load_algorithm(name):
    if not name:
        raise ValueError("Algorithm name must be a non-empty string")

    try:
        return import_module(f"algorithms.{name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Unknown algorithm '{name}'. Expected package at algorithms/{name}/"
        ) from exc


__all__ = ["load_algorithm"]
