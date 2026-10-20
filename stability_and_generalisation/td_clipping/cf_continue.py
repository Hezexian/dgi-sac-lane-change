"""Fork a completed run's full checkpoint into a new counterfactual run.

Source files are opened read-only.  Only the initial source restore bypasses
plan equality; recovery of the new run uses Runner's ordinary strict checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path

import cloudpickle as pickle
import numpy as np
import torch

import run_training as train
from core import restore_rng
from stabilizers import PBRS_C, has_pbrs, has_qclip, q_upper_bound


STABILIZERS = ("none", "qclip", "pbrs", "qclip_pbrs")
VALIDATION_EVERY = 25000
VALIDATION_EPISODES = 50


def _read_json(path):
    with Path(path).open("r", encoding="utf8") as handle:
        return json.load(handle)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve_until_step(value, checkpoint_step):
    """Accept an absolute step, +N, checkpoint+N, or 检查点步数+N."""
    text = str(value).replace(" ", "")
    relative = re.fullmatch(r"(?:checkpoint|checkpoint_step|检查点步数)?\+(\d+)", text)
    if relative:
        result = int(checkpoint_step) + int(relative.group(1))
    elif re.fullmatch(r"\d+", text):
        result = int(text)
    else:
        raise ValueError("--until-step must be an integer or checkpoint step + N (for example +25000)")
    if result <= checkpoint_step:
        raise ValueError("--until-step must be greater than the source checkpoint step")
    return result


def derived_reseed_seed(checkpoint_sha256, offset):
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('--rng-reseed-offset must be a nonnegative integer')
    if offset == 0:
        return None
    return int(hashlib.sha256(f'{checkpoint_sha256}:{offset}'.encode()).hexdigest()[:8], 16)


def reseed_continuation(runner, seed):
    """Reseed only RNG state after full source restoration, never on recovery.

    In-place generator state changes retain HighwayEnv/Road aliases. Do not
    reset the environment, regenerate traffic, or touch replay/learned data.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    streams = ['python.random', 'numpy.global', 'torch.cpu']
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        streams.extend(f'torch.cuda:{index}' for index in range(torch.cuda.device_count()))
    seen = set()

    def reseed_rng(name, rng):
        # List every owner even when several owners share one physical stream.
        streams.append(name)
        if id(rng) in seen:
            return
        seen.add(id(rng))
        if isinstance(rng, np.random.Generator):
            rng.bit_generator.state = type(rng.bit_generator)(seed).state
        elif isinstance(rng, np.random.RandomState):
            rng.seed(seed)
        elif isinstance(rng, random.Random):
            rng.seed(seed)
        else:
            raise TypeError(f'Unaudited continuation RNG {name}: {type(rng).__name__}')

    reseed_rng('replay.rng', runner.memory.rng)
    reseed_rng('replay.legacy_rng', runner.memory.legacy_rng)
    if hasattr(runner.agent, 'exploration_rng'):
        reseed_rng('agent.exploration_rng', runner.agent.exploration_rng)
    base = runner.env.unwrapped
    reseed_rng('env.unwrapped.np_random', base.np_random)
    if hasattr(base, '_np_random_seed'):
        base._np_random_seed = seed
    road = getattr(base, 'road', None)
    if road is not None and hasattr(road, 'np_random'):
        reseed_rng('env.unwrapped.road.np_random', road.np_random)

    spaces_seen = set()

    def reseed_space(name, space):
        if space is None:
            return
        reseed_rng(name + '.np_random', space.np_random)
        if id(space) in spaces_seen:
            return
        spaces_seen.add(id(space))
        if hasattr(space, '_np_random_seed'):
            space._np_random_seed = seed
        children = getattr(space, 'spaces', None)
        if isinstance(children, dict):
            for key, child in children.items():
                reseed_space(f'{name}[{key}]', child)
        elif isinstance(children, (list, tuple)):
            for index, child in enumerate(children):
                reseed_space(f'{name}[{index}]', child)

    layer, level, layers_seen = runner.env, 0, set()
    while layer is not None and id(layer) not in layers_seen:
        layers_seen.add(id(layer))
        reseed_space(f'env.layer{level}.action_space', getattr(layer, 'action_space', None))
        reseed_space(f'env.layer{level}.observation_space', getattr(layer, 'observation_space', None))
        layer, level = getattr(layer, 'env', None), level + 1
    return streams


def plan_identity(checkpoint_sha256, stabilizer, until_step, code_sha256, rng_reseed_offset=0):
    return {"source_checkpoint_sha256": checkpoint_sha256,
            "stabilizer": stabilizer, "until_step": int(until_step),
            "code_sha256": code_sha256, "rng_reseed_offset": rng_reseed_offset,
            "rng_reseed_seed": derived_reseed_seed(checkpoint_sha256, rng_reseed_offset)}


def prepare(args):
    """Validate the completed source and return a read-only fork description.

    The full checkpoint is loaded on CPU for validation; parameter and optimizer
    restoration subsequently uses the already constructed target device agent.
    Nothing is created in the target until all source and path checks succeed.
    """
    offset = getattr(args, 'rng_reseed_offset', 0)
    derived_reseed_seed('', offset)  # Reject invalid offsets before any target write.
    source_root = Path(args.source_run_dir).resolve(strict=True)
    checkpoint = Path(args.source_checkpoint).resolve(strict=True)
    destination = Path(args.run_dir).resolve()
    if (destination == source_root or destination.is_relative_to(source_root)
            or source_root.is_relative_to(destination)):
        raise ValueError("Source and destination run directories must be disjoint")
    checkpoint_root = (source_root / "checkpoints").resolve(strict=True)
    if checkpoint_root.parent != source_root or checkpoint.parent != checkpoint_root:
        raise ValueError("Source checkpoint must be directly inside the source run's checkpoints directory")
    if not checkpoint.name.startswith("full-") or checkpoint.suffix != ".pt":
        raise ValueError("Counterfactual continuation requires a full-*.pt checkpoint")
    if not args.resume and destination.exists() and any(destination.iterdir()):
        raise ValueError("Initial counterfactual run requires a new or empty run directory")
    if args.resume and not (destination / "run-config.json").is_file():
        raise ValueError("--resume requires the counterfactual run's existing run-config.json")
    if destination.exists():
        # Recovery must not append through a pre-existing link back into the
        # source (or write/prune through an aliased checkpoints directory).
        for existing in destination.rglob("*"):
            if not existing.resolve().is_relative_to(destination):
                raise ValueError("Counterfactual output contains a path escaping its run directory")

    config_path = source_root / "run-config.json"
    status_path = source_root / "status.json"
    complete_path = source_root / "training-complete.json"
    manifest_path = checkpoint_root / "manifest.json"
    run_config = _read_json(config_path)
    status = _read_json(status_path)
    complete = _read_json(complete_path)
    manifest = _read_json(manifest_path)
    source_plan = run_config["plan_sha256"]
    if status.get("status") != "trained":
        raise ValueError("Source run is not completed (status must be trained)")
    if any(document.get("plan_sha256") != source_plan for document in (status, complete, manifest)):
        raise ValueError("Source run config/status/completion/manifest plan mismatch")
    if args.run_id == run_config.get("run_id"):
        raise ValueError("Counterfactual run must have a new run ID")
    if status.get("run_id") != run_config.get("run_id"):
        raise ValueError("Source status run ID mismatch")
    entry = next((x for x in manifest.get("entries", []) if x.get("path") == checkpoint.name), None)
    checkpoint_sha = train.sha256(checkpoint)
    if entry is None or entry.get("sha256") != checkpoint_sha:
        raise ValueError("Source checkpoint is absent from its manifest or has a mismatched hash")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"agent", "replay", "environment_pickle", "rng", "loop", "state",
                "done_episode", "device", "scientific", "plan_sha256", "run_id"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("Not a full resumable checkpoint; missing: " + ", ".join(missing))
    if train.sha256(checkpoint) != checkpoint_sha:
        raise ValueError("Source checkpoint changed while being read")
    if data["plan_sha256"] != source_plan or data["run_id"] != run_config.get("run_id"):
        raise ValueError("Source checkpoint plan/run ID mismatch")
    if data["scientific"] != run_config["scientific"]:
        raise ValueError("Source checkpoint scientific configuration differs from run-config")
    if data["device"] != args.device or data["scientific"]["device"] != args.device:
        raise ValueError("Counterfactual continuation must use the checkpoint's original device")
    step = int(data["loop"]["step"])
    if entry.get("step") != step:
        raise ValueError("Source manifest/checkpoint step mismatch")
    if int(complete["step"]) != int(status["step"]) or step > int(complete["step"]):
        raise ValueError("Source completion/checkpoint step mismatch")
    if int(complete["step"]) != int(status["budget"]):
        raise ValueError("Source run has not reached its declared budget")
    source_science = data["scientific"]
    source_environment = source_science.get("environment_fingerprint")
    current_environment = train.environment_fingerprint()
    if source_environment is None:
        raise ValueError("Source checkpoint has no environment fingerprint")
    # Source and candidate source-code hashes intentionally differ.  The
    # serialized simulator and installed runtime must nevertheless match.
    checked_environment_keys = ("versions", "highway_source_sha256", "torch_cuda", "python")
    mismatched_environment = [key for key in checked_environment_keys
                              if source_environment.get(key) != current_environment.get(key)]
    if mismatched_environment:
        raise ValueError("Source/runtime environment fingerprint mismatch: " + ", ".join(mismatched_environment))
    source_stabilizer = source_science.get("stabilizer", "none")
    variant = source_science["variant"]
    if variant in {"ddpg", "td3"} and args.stabilizer != "none":
        raise ValueError("Stabilizers are only implemented for SAC variants, not ddpg/td3")
    until_step = resolve_until_step(args.until_step, step)
    code_sha, source_files = train.code_digest(Path(__file__).parent)
    if args.expected_code_sha and args.expected_code_sha != code_sha:
        raise RuntimeError("Source hash does not match frozen command")
    identity = plan_identity(checkpoint_sha, args.stabilizer, until_step, code_sha, offset)
    plan_sha = _digest(identity)
    if args.plan_sha256 and args.plan_sha256 != plan_sha:
        raise RuntimeError("Requested plan_sha256 does not match counterfactual identity")

    # Preserve the source's sampling horizon.  Changing the stopping step must
    # not alter PER beta or the HRE/ERE window used in the none counterfactual.
    config = copy.deepcopy(source_science["config"])
    schedule_budget = int(source_science.get("schedule_budget", config["train"]["num_steps"]))
    scientific = copy.deepcopy(source_science)
    scientific.update(variant=variant, budget=until_step, config=config,
                      stabilizer=args.stabilizer, code_sha256=code_sha,
                      stabilizer_config={
                          "q_max": q_upper_bound(config, args.stabilizer) if has_qclip(args.stabilizer) else None,
                          "pbrs_c": PBRS_C if has_pbrs(args.stabilizer) else None,
                          "pbrs_speed": "signed vehicle.speed*cos(heading)",
                          "pbrs_terminal_phi": 0., "pbrs_truncated_bootstrap": True},
                      evaluation_every=VALIDATION_EVERY,
                      evaluation_episodes=VALIDATION_EPISODES,
                      schedule_budget=schedule_budget,
                      source_checkpoint_sha256=checkpoint_sha,
                      source_plan_sha256=source_plan,
                      until_step=until_step,
                      counterfactual_identity=identity,
                      rng_reseed_offset=offset,
                      rng_reseed_seed=identity['rng_reseed_seed'],
                      initial_restore_plan_equality_checked=False,
                      initial_restore_device_equality_checked=True,
                      recovery_plan_equality_checked=True)
    provenance = {
        "source_run_dir": str(source_root), "source_run_id": data["run_id"],
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_step": step, "source_plan_sha256": source_plan,
        "source_stabilizer": source_stabilizer, "stabilizer": args.stabilizer,
        "until_step": until_step, "schedule_budget": schedule_budget,
        "code_sha256": code_sha,
        "cf_continue_sha256": source_files.get("cf_continue.py", train.sha256(__file__)),
        "identity": identity,
        "rng_reseed_offset": offset,
        "rng_reseed_seed": identity['rng_reseed_seed'],
        "rng_reseed_applied_at_source_fork": offset > 0,
        "rng_reseed_streams": [],
        "initial_plan_equality_bypassed": True,
        "initial_device_equality_checked": True,
        "new_run_recovery_requires_plan_equality": True,
        "source_files_read_only": True,
        "environment_fingerprint_checked_keys": list(checked_environment_keys),
        "inherited_best": {"reward": data["loop"].get("best_reward"),
                           "step": data["loop"].get("best_step")},
        "best_selection_reset_for_new_run": True,
        "optimizer_state_keys_restored": sorted(k for k in data["agent"] if "optimizer" in k),
        "source_metadata_sha256": {p.name: train.sha256(p) for p in
                                   (config_path, status_path, complete_path, manifest_path)},
    }
    if args.resume:
        previous = _read_json(destination / "run-config.json")
        if previous.get("plan_sha256") != plan_sha or previous.get("run_id") != args.run_id:
            raise ValueError("Existing counterfactual run plan/run ID mismatch")
        conversion = previous.get("counterfactual", {}).get("replay_reward_conversion")
        if conversion is not None:
            provenance["replay_reward_conversion"] = conversion
        provenance['rng_reseed_streams'] = previous.get('counterfactual', {}).get('rng_reseed_streams', [])
    return {"checkpoint": data, "config": config, "scientific": scientific,
            "provenance": provenance, "plan_sha256": plan_sha,
            "identity": identity, "source_files": source_files}


def describe_plan(args):
    source = prepare(args)
    return {"plan_sha256": source["plan_sha256"], "scientific": source["scientific"],
            "identity": source["identity"], "counterfactual": source["provenance"],
            "source_files": source["source_files"]}


class CounterfactualRunner(train.Runner):
    def __init__(self, args):
        self.fork = prepare(args)
        science = self.fork["scientific"]
        source = self.fork["checkpoint"]
        runner_args = argparse.Namespace(
            variant=science["variant"], seed=int(science["seed"]),
            budget=int(science["until_step"]), device=args.device,
            stabilizer=args.stabilizer, run_dir=args.run_dir, run_id=args.run_id,
            threads=int(science["threads"]), priority_mode=science["priority_mode"],
            replay_capacity=int(source["replay"]["capacity"]),
            eval_every=VALIDATION_EVERY, eval_episodes=VALIDATION_EPISODES,
            full_every=int(science["full_every"]),
            protected_steps=list(science["protected_steps"]), kind=science["kind"],
            learning_starts=int(science.get("learning_starts", train.LEARNING_STARTS)),
            schedule_budget=int(science["schedule_budget"]),
            restore_diagnostic_window=True,
            resume=args.resume or "__source__", stop_after=args.stop_after,
            plan_sha256=self.fork["plan_sha256"], expected_code_sha=args.expected_code_sha,
            describe_plan=False)
        super().__init__(runner_args)
        target_config = self.root / "run-config.json"
        config = _read_json(target_config)
        config["counterfactual"] = self.fork["provenance"]
        config["plan_identity"] = self.fork["identity"]
        train.atomic_json(target_config, config)
        # A first-fork recovery checkpoint lets the standard scheduler resume
        # even when a worker fails before the next 100K periodic checkpoint.
        if not args.resume:
            self.full_checkpoint("counterfactual-start")
        self.fork.pop("checkpoint", None)

    def training_config(self):
        return copy.deepcopy(self.fork["config"])

    def build_scientific_spec(self):
        if self.code_sha != self.fork["identity"]["code_sha256"]:
            raise RuntimeError("Candidate code changed during counterfactual initialization")
        return copy.deepcopy(self.fork["scientific"])

    def plan_identity(self, scientific):
        return self.fork["identity"]

    def restore(self, reference):
        if reference != "__source__":
            return super().restore(reference)
        data = self.fork["checkpoint"]
        # The explicit intervention changes plan identity, but never device.
        if data["device"] != self.args.device:
            raise ValueError("Source checkpoint device mismatch")
        self.agent.load_state_dict(data["agent"])
        self.memory.load_state_dict(data["replay"])
        from stabilizers import relabel_replay
        self.fork["provenance"]["replay_reward_conversion"] = relabel_replay(
            self.memory, self.fork["provenance"]["source_stabilizer"],
            self.args.stabilizer, gamma=self.config["train"]["gamma"], c=PBRS_C)
        self.env.close()
        self.env = pickle.loads(data["environment_pickle"])
        self.state = data["state"]
        self.loop = copy.deepcopy(data["loop"])
        # Source best.json points into the source tree.  Keep its values in
        # provenance and select fresh weights at this run's validation nodes.
        self.loop["best_reward"], self.loop["best_step"] = None, None
        self.done_episode = data["done_episode"]
        self.actor_grad_norms = list(data.get("actor_grad_norms", []))
        restore_rng(data["rng"])
        seed = self.fork['identity']['rng_reseed_seed']
        if seed is not None:
            self.fork['provenance']['rng_reseed_streams'] = reseed_continuation(self, seed)
        self.started_wall = time.time()
        train.append_json(self.root / "resume-events.jsonl", {
            "step": self.loop["step"],
            "checkpoint": self.fork["provenance"]["source_checkpoint"],
            "source_checkpoint_sha256": self.fork["provenance"]["source_checkpoint_sha256"],
            "source_plan_sha256": data["plan_sha256"],
            "plan_sha256": self.plan_sha, "time": train.utcnow(),
            "mode": "counterfactual-fork", "plan_equality_bypassed": True,
            "rng_reseed_offset": self.fork['identity']['rng_reseed_offset'],
            "rng_reseed_seed": seed,
            "device_equality_checked": True})


CFRunner = CounterfactualRunner


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-checkpoint", required=True)
    result.add_argument("--source-run-dir", required=True)
    result.add_argument("--stabilizer", choices=("none", "qclip"), required=True)
    result.add_argument('--rng-reseed-offset', type=int, default=0,
                        help='0 restores RNG exactly; positive K reseeds only the initial source fork')
    result.add_argument("--until-step", required=True,
                        help="Absolute training step, or +N relative to the source checkpoint")
    result.add_argument("--run-dir", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--device", choices=["cpu", "cuda"], required=True)
    result.add_argument("--expected-code-sha", default=None)
    result.add_argument("--plan-sha256", default=None)
    result.add_argument("--describe-plan", action="store_true")
    result.add_argument("--resume", default=None,
                        help="Recover this counterfactual run (auto or a local checkpoint path)")
    result.add_argument("--stop-after", type=int, default=0,
                        help="Functional testing interruption; does not change scientific identity")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.describe_plan:
        print(json.dumps(describe_plan(args), indent=2))
    else:
        CounterfactualRunner(args).run()


if __name__ == "__main__":
    main()
