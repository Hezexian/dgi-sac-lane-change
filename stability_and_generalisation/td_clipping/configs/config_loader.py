import argparse
import ast
from copy import deepcopy
from pathlib import Path


# ============================================================
# YAML 配置加载工具
# ============================================================
# 目标：
# 1. 支持“一个实验一个 YAML”
# 2. 让 train.py 和 test.py 共用同一份配置
# 3. 即使当前环境没安装 PyYAML，也能解析本仓库使用的简单 YAML 子集


def _parse_scalar(value):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _preprocess_yaml_lines(text):
    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(raw_line.rstrip())
    return lines


def _parse_yaml_block(lines, start_index=0, indent_level=0):
    result = None
    index = start_index

    while index < len(lines):
        raw_line = lines[index]
        stripped_line = raw_line.lstrip(" ")
        current_indent = len(raw_line) - len(stripped_line)

        if current_indent < indent_level:
            break
        if current_indent > indent_level:
            raise ValueError(f"Unexpected indentation at line: {raw_line}")

        if stripped_line.startswith("- "):
            if result is None:
                result = []
            elif not isinstance(result, list):
                raise ValueError(f"Cannot mix list and mapping items at line: {raw_line}")

            item_value = stripped_line[2:].strip()
            if item_value == "":
                child, index = _parse_yaml_block(lines, index + 1, indent_level + 2)
                result.append(child)
                continue

            result.append(_parse_scalar(item_value))
            index += 1
            continue

        if ":" not in stripped_line:
            raise ValueError(f"Invalid mapping line: {raw_line}")

        if result is None:
            result = {}
        elif not isinstance(result, dict):
            raise ValueError(f"Cannot mix mapping and list items at line: {raw_line}")

        key, remainder = stripped_line.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()

        if remainder == "":
            child, index = _parse_yaml_block(lines, index + 1, indent_level + 2)
            result[key] = child
            continue

        result[key] = _parse_scalar(remainder)
        index += 1

    return result, index


def _load_yaml_text(text):
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        lines = _preprocess_yaml_lines(text)
        if not lines:
            return {}
        data, _ = _parse_yaml_block(lines, 0, 0)
        return data


def load_config(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = _load_yaml_text(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML config must be a mapping/dictionary")
    return path, config


def infer_config_path_from_checkpoint(checkpoint_path):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_dir = checkpoint.parent
    if checkpoint_dir.name != "checkpoints":
        return None

    candidate = checkpoint_dir.parent / "config.yaml"
    if candidate.exists():
        return candidate
    return None


def _section(config, name):
    section_value = deepcopy(config.get(name, {}))
    if not isinstance(section_value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping/dictionary")
    return section_value


def _extract_observation_metadata(config):
    env = _section(config, "env")
    observation = deepcopy(env.get("observation", {}))
    if not isinstance(observation, dict):
        raise ValueError("Config section 'env.observation' must be a mapping/dictionary")

    vehicles_count = observation.get("vehicles_count")
    feature_names = deepcopy(observation.get("features", []))

    if not isinstance(vehicles_count, int) or vehicles_count <= 0:
        raise ValueError("env.observation.vehicles_count must be a positive integer")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("env.observation.features must be a non-empty list")

    feature_dim = len(feature_names)
    presence_index = feature_names.index("presence") if "presence" in feature_names else None

    return {
        "obs_vehicles_count": vehicles_count,
        "obs_feature_names": feature_names,
        "obs_feature_dim": feature_dim,
        "obs_presence_index": presence_index,
        "obs_flat_dim": vehicles_count * feature_dim,
    }


def build_train_args(config, config_path, resume_checkpoint=None):
    algorithm = _section(config, "algorithm")
    run = _section(config, "run")
    train = _section(config, "train")
    observation_metadata = _extract_observation_metadata(config)

    return argparse.Namespace(
        config=str(config_path),
        algorithm_name=algorithm.get("name", "sac_base"),
        algorithm_variant=algorithm.get("variant"),
        algorithm_config=deepcopy(algorithm),
        algorithm_params=deepcopy(algorithm.get("params", {})),
        # 实验名直接统一使用配置文件名，避免再维护一层重复的 experiment.name。
        experiment_name=Path(config_path).stem,
        env_name=run.get("env_name", "highway-fast-v0"),
        seed=run.get("seed", 123456),
        cuda=run.get("cuda", False),
        policy=train.get("policy", "Gaussian"),
        eval=train.get("eval", True),
        gamma=train.get("gamma", 0.99),
        tau=train.get("tau", 0.005),
        lr=train.get("lr", 0.0003),
        critic_weight_decay=train.get("critic_weight_decay", 0.0),
        alpha=train.get("alpha", 1.0),
        automatic_entropy_tuning=train.get("automatic_entropy_tuning", True),
        batch_size=train.get("batch_size", 256),
        num_steps=train.get("num_steps", 1000001),
        hidden_size=train.get("hidden_size", 256),
        updates_per_step=train.get("updates_per_step", 1),
        start_steps=train.get("start_steps", 10000),
        target_update_interval=train.get("target_update_interval", 1),
        replay_size=train.get("replay_size", 1000000),
        save_every_episodes=train.get("save_every_episodes", 200),
        keep_last_checkpoints=train.get("keep_last_checkpoints"),
        resume_checkpoint=resume_checkpoint,
        **observation_metadata,
    )


def build_test_args(config, config_path, checkpoint, test_episodes=None, save_video=None, show_window=None, video_dir=None):
    algorithm = _section(config, "algorithm")
    run = _section(config, "run")
    train = _section(config, "train")
    test = _section(config, "test")
    observation_metadata = _extract_observation_metadata(config)

    return argparse.Namespace(
        config=str(config_path),
        checkpoint=str(checkpoint),
        algorithm_name=algorithm.get("name", "sac_base"),
        algorithm_variant=algorithm.get("variant"),
        algorithm_config=deepcopy(algorithm),
        algorithm_params=deepcopy(algorithm.get("params", {})),
        env_name=run.get("env_name", "highway-fast-v0"),
        seed=run.get("seed", 123456),
        cuda=run.get("cuda", False),
        policy=train.get("policy", "Gaussian"),
        gamma=train.get("gamma", 0.99),
        tau=train.get("tau", 0.005),
        lr=train.get("lr", 0.0003),
        critic_weight_decay=train.get("critic_weight_decay", 0.0),
        alpha=train.get("alpha", 1.0),
        automatic_entropy_tuning=train.get("automatic_entropy_tuning", True),
        hidden_size=train.get("hidden_size", 256),
        target_update_interval=train.get("target_update_interval", 1),
        test_episodes=(
            test_episodes
            if test_episodes is not None
            else test.get("test_episodes", 3)
        ),
        save_video=save_video if save_video is not None else test.get("save_video", False),
        show_window=show_window if show_window is not None else test.get("show_window", False),
        video_dir=video_dir if video_dir is not None else test.get("video_dir"),
        **observation_metadata,
    )


def get_env_config(config):
    env_config = deepcopy(config.get("env"))
    if not isinstance(env_config, dict):
        raise ValueError("Config section 'env' must be a mapping/dictionary")
    return env_config
