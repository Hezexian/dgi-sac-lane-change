"""Load and validate YAML configurations with command-line overrides."""

from pathlib import Path

import yaml

ALGORITHMS = ("dgi_sac", "sac", "mlp", "td3", "ddpg")
SAC_FAMILY = ("dgi_sac", "sac", "mlp")


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not key or not sep:
            raise ValueError(f"Override must look like key.subkey=value, got {item!r}")
        target = config
        *parents, leaf = key.split(".")
        for name in parents:
            target = target.setdefault(name, {})
        target[leaf] = yaml.safe_load(value)
    validate(config)
    return config


def validate(config):
    algorithm = config.get("algorithm")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    if algorithm not in SAC_FAMILY and config["train"].get("td_target_clip") is not None:
        raise ValueError("td_target_clip applies to the SAC family only")
    observation = config["env"]["observation"]
    if observation["features"] != ["presence", "x", "y", "vx", "vy", "heading"]:
        raise ValueError("The models expect Kinematics features presence, x, y, vx, vy, heading")


def obs_dim(config):
    observation = config["env"]["observation"]
    return observation["vehicles_count"] * len(observation["features"])
