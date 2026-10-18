"""Deterministic evaluation of a policy on a list of scenario seeds.

Each scenario seed fixes the initial traffic. The policy acts with its deterministic action and
every episode is summarised with the metrics of the paper (unsafe, success, mean speed, lane changes).
"""

import contextlib
import csv
import gzip
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .algorithms import make_agent
from .env import make_env
from .metrics import aggregate, behavior_events, episode_summary, geometry, lane_events
from .utils import isolated_rng

CHECKPOINT_FORMAT = "dgi-sac-lane-change-v1"
# Released models of the main comparison: models/<method>/seed_001.pt ... in the model archive
RELEASED_SEEDS = {"dgi_sac": 15, "sac": 15, "mlp": 15, "td3": 10, "ddpg": 10}


def load_checkpoint(path, device="cpu", seed=0):
    """Load a checkpoint written by train.py or a released model from the data archive.

    Returns (agent, config).
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") == CHECKPOINT_FORMAT:
        config, state = payload["config"], payload["agent"]
    else:  # released model, saved by the study code
        config, state = _from_study_checkpoint(payload)
    env = make_env(config["env_name"], config["env"])
    try:
        agent = make_agent(config, seed, env.action_space, device=device)
    finally:
        env.close()
    agent.load_state_dict(state)
    return agent, config


# Released models were saved by the study code; these two functions convert its configuration and
# weight layout to those of this repository. The weights themselves are unchanged.
_STUDY_NAMES = {"dgi": "dgi_sac", "sac": "sac", "capacity": "mlp", "td3": "td3", "ddpg": "ddpg"}


def _from_study_checkpoint(payload):
    state = payload["agent"]
    variant = payload.get("variant") or state["variant"]
    if variant not in _STUDY_NAMES:
        raise ValueError(f"Not one of the five paper methods: {variant!r}")
    algorithm = _STUDY_NAMES[variant]
    old = deepcopy(payload.get("config") or state["config"])
    params = old.get("algorithm", {}).get("params", {})
    train = old["train"]
    config = {
        "algorithm": algorithm,
        "train": {
            "gamma": train["gamma"],
            "tau": train["tau"],
            "lr": train["lr"],
            "hidden_size": train["hidden_size"],
            "batch_size": train["batch_size"],
            "replay_size": train["replay_size"],
            "total_steps": train["num_steps"],
        },
        "env_name": old["run"]["env_name"],
        "env": old["env"],
    }
    if algorithm in ("dgi_sac", "sac", "mlp"):
        config["train"].update(alpha=train["alpha"], td_target_clip=None)
    else:  # fixed in the study code, not stored in its configuration
        config["train"].update(exploration_noise=0.1)
        if algorithm == "td3":
            config["train"].update(target_noise=0.2, noise_clip=0.5, policy_delay=2)
    if algorithm == "dgi_sac":
        if (params.get("edge_weight_mode", "inverse_distance") != "inverse_distance"
                or params.get("distance_threshold") is not None
                or params.get("fusion_mode", "concat") != "concat"):
            raise ValueError("Checkpoint uses a graph variant that is not DGI-SAC")
        config["graph"] = {
            "hidden_size": params.get("graph_hidden_size", 256),
            "num_layers": params.get("graph_num_layers", 3),
            "position_scale": params.get("position_scale", [100.0, 12.0]),
            "ego_at_origin": params.get("ego_at_origin", True),
            "self_loop_weight": params.get("self_loop_weight", 1.0),
        }
    weights = {}
    for key, value in state.items():
        if key.endswith("_state_dict"):
            # TD3 and DDPG networks of the study divided their input by a constant 'obs_scale' of ones.
            scale = value.get("obs_scale")
            if scale is not None and not torch.equal(scale, torch.ones_like(scale)):
                raise ValueError("Unexpected input scaling in the released model")
            weights[key] = {k: v for k, v in value.items() if k != "obs_scale"}
    weights["updates"] = state.get("delay_counter", 0)  # critic updates (TD3 and DDPG only)
    if "exploration_rng" in state:
        weights["exploration_rng"] = state["exploration_rng"]
    return config, weights


def resolve_checkpoint(args):
    """Return (checkpoint path, display name) from --checkpoint or --method/--seed."""
    if args.checkpoint:
        path = Path(args.checkpoint)
        run = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        return path, f"{run.name}_{path.stem}"  # e.g. dgi_sac_seed26093001_final
    if not (args.method and args.seed):
        raise SystemExit("Give --checkpoint, or --method and --seed for a released model")
    if not 1 <= args.seed <= RELEASED_SEEDS[args.method]:
        raise SystemExit(f"No released model for {args.method} seed {args.seed}")
    path = Path(args.models_dir) / args.method / f"seed_{args.seed:03d}.pt"
    if not path.is_file():
        raise SystemExit(f"{path} not found; extract the model archive and set --models-dir")
    return path, f"{args.method}_seed{args.seed:03d}"


def set_ablation(agent, mode):
    """Evaluation-only interventions on the DGI-SAC actor (Section 4.3, Online Resource S1).

    cut_ego_edges       remove the edges between the ego vehicle and its neighbours
    zero_mlp_neighbors  hide the neighbours from the flattened branch
    """
    if mode == "none":
        return
    actor = agent.policy
    if not hasattr(actor, "graph_encoder"):
        raise ValueError("Ablations apply to the DGI-SAC (graph) actor only")
    if mode == "cut_ego_edges":
        actor.graph_encoder.ablation = mode
    elif mode == "zero_mlp_neighbors":
        actor.ablation = mode
    else:
        raise ValueError(f"Unknown ablation {mode!r}")


def snapshot(env, step, seed, previous_x=None, reward=0.0):
    """State of the ego vehicle and the geometry used by the lane-change metrics."""
    base = env.unwrapped
    v = base.vehicle
    vehicles = [{
        "ego": other is v,
        "x": float(other.position[0]),
        "y": float(other.position[1]),
        "length": float(other.LENGTH),
        "lane": int(other.lane_index[2]),
        "forward_speed": float(other.speed * math.cos(other.heading)),
    } for other in base.road.vehicles]
    lane, x = int(v.lane_index[2]), float(v.position[0])
    row = {
        "episode_seed": int(seed),
        "step": int(step),
        "time": step / 5.0,
        "x": x,
        "y": float(v.position[1]),
        "heading": float(v.heading),
        "lane": lane,
        "speed": float(v.speed * math.cos(v.heading)),
        "crashed": bool(v.crashed),
        "on_road": bool(v.on_road),
        "distance_delta": x - previous_x if previous_x is not None else 0.0,
        "reward": float(reward),
        "nearby_vehicles": sum(not c["ego"] and abs(c["x"] - x) <= 50 and abs(c["lane"] - lane) <= 1
                               for c in vehicles),
    }
    geom = {target: geometry({"vehicles": vehicles}, target) for target in range(4)}
    row.update(geom[lane])
    for target, values in geom.items():
        row.update({f"lane{target}_" + key: value for key, value in values.items()})
    row["vehicles"] = vehicles
    return row


def _attach_target_geometry(records, events):
    keys = ("front_gap_m", "rear_gap_m", "front_ttc_s", "rear_ttc_s")
    for row in records:
        row["target_lane"] = None
        for key in keys:
            row["target_" + key] = None
    by_tick = {r["step"]: r for r in records}
    for event in events:
        target = event["target_lane"]
        for tick in range(event["start_tick"], event["end_tick"] + 1):
            row = by_tick[tick]
            row["target_lane"] = target
            for key in keys:
                row["target_" + key] = row[f"lane{target}_" + key]
        for label, tick in [("start", event["start_tick"]), ("crossing", event["crossing_tick"]),
                            ("end", event["end_tick"])]:
            for key in keys:
                event[label + "_" + key] = by_tick[tick][f"lane{target}_" + key] if tick is not None else None
        for key in keys:
            vals = [by_tick[t][f"lane{target}_" + key] for t in range(event["start_tick"], event["end_tick"] + 1)]
            defined = [v for v in vals if v is not None]
            event["minimum_" + key] = min(defined) if defined else None
            event["defined_samples_" + key] = len(defined)


def run_episode(env, agent, seed):
    """One deterministic episode; returns (episode summary, lane-change events, behaviour events, records)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    config = env.unwrapped.config
    records = [snapshot(env, 0, seed)]
    initial_lane = records[0]["lane"]
    reward_sum, terminated, truncated = 0.0, False, False
    max_steps = math.ceil(float(config["duration"]) * float(config["policy_frequency"])) + 1
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            action = agent.select_action(np.asarray(state).reshape(-1), evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            row = snapshot(env, step, seed, records[-1]["x"], reward)
            row["crashed"] = row["crashed"] or bool(info.get("crashed", False))
            records.append(row)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Environment exceeded the configured episode length")
    changes = lane_events(records[1:], initial_lane)
    _attach_target_geometry(records, changes)
    behaviors = behavior_events(records, changes)
    episode = episode_summary(records, changes, behaviors, seed, reward_sum, terminated, truncated)
    return episode, changes, behaviors, records


def is_success(episode):
    """Paper success: safe, all 200 decisions completed, mean speed >= 25 m/s, <= 10 boundary oscillations."""
    oscillations = (episode["initial_10s_boundary_oscillation_count"]
                    + episode["after_10s_boundary_oscillation_count"])
    return (not episode["collision"] and not episode["offroad"] and bool(episode["truncated"])
            and episode["steps"] == 200 and episode["mean_speed"] >= 25 and oscillations <= 10)


def paper_metrics(episodes):
    """Unsafe rate, success rate and mean speed as reported in Table 2."""
    n = len(episodes)
    return {
        "episodes": n,
        "unsafe_rate": sum(bool(e["unsafe"]) for e in episodes) / n,
        "success_rate": sum(is_success(e) for e in episodes) / n,
        "mean_speed": sum(e["mean_speed"] for e in episodes) / n,
    }


def evaluate(agent, env_name, env_config, seeds, output_dir=None, save_trajectories=False):
    """Evaluate on the given scenario seeds; optionally write episodes.csv, events.jsonl and summary.json."""
    seeds = [int(s) for s in seeds]
    directory = Path(output_dir) if output_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "episodes.csv").exists():
            raise FileExistsError(f"{directory} already contains results; choose a new output directory")
    episodes = []
    with contextlib.ExitStack() as stack:
        env = make_env(env_name, env_config)
        stack.callback(env.close)
        episode_file = event_file = trajectory_file = episode_writer = trajectory_writer = None
        if directory is not None:
            episode_file = stack.enter_context((directory / "episodes.csv").open("w", encoding="utf-8", newline=""))
            event_file = stack.enter_context((directory / "events.jsonl").open("w", encoding="utf-8"))
            if save_trajectories:
                trajectory_file = stack.enter_context(
                    gzip.open(directory / "trajectories.csv.gz", "wt", encoding="utf-8", newline=""))
        with isolated_rng():
            for seed in seeds:
                episode, changes, behaviors, records = run_episode(env, agent, seed)
                episodes.append(episode)
                if directory is None:
                    continue
                if episode_writer is None:
                    episode_writer = csv.DictWriter(episode_file, fieldnames=list(episode))
                    episode_writer.writeheader()
                episode_writer.writerow(episode)
                for kind, items in [("lane_change", changes), ("behavior", behaviors)]:
                    for event in items:
                        event_file.write(json.dumps({"episode_seed": seed, "event_type": kind, **event},
                                                    allow_nan=False) + "\n")
                if trajectory_file is not None:
                    for row in records:
                        row = {**row, "vehicles": json.dumps(row["vehicles"], separators=(",", ":"))}
                        if trajectory_writer is None:
                            trajectory_writer = csv.DictWriter(trajectory_file, fieldnames=list(row))
                            trajectory_writer.writeheader()
                        trajectory_writer.writerow(row)
    summary = {**paper_metrics(episodes), "first_seed": seeds[0], "last_seed": seeds[-1],
               "details": aggregate(episodes)}
    if directory is not None:
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
