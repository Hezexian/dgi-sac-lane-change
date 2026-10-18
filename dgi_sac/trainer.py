"""Training loop shared by the five methods.

Actions are random during warm-up and policy-generated thereafter.
Each eligible step updates the agent before executing the action and
storing the transition. Validation and checkpointing run periodically
and at the final step when evaluation is enabled.

Losses are logged every 10,000 steps. TD3 retains the loss from its latest
actor update.
"""

import csv
import json
import time
from pathlib import Path

import torch
import yaml

from .algorithms import make_agent
from .env import make_env
from .evaluation import CHECKPOINT_FORMAT, evaluate
from .replay_buffer import ReplayBuffer
from .utils import isolated_rng, seed_everything

VALIDATION_SEED = 310000  # validation scenarios 310000, 310001, ...
LOG_EVERY = 10000


class CsvLog:
    def __init__(self, path):
        self.path, self.writer, self.file = Path(path), None, None

    def write(self, row):
        if self.writer is None:
            self.file = self.path.open("w", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def save_checkpoint(path, agent, config, seed, step):
    weights = {k: v for k, v in agent.state_dict().items() if "optimizer" not in k}
    torch.save({"format": CHECKPOINT_FORMAT, "method": config["algorithm"], "config": config,
                "seed": seed, "step": step, "agent": weights}, path)


def train(config, seed, output_dir, device="cpu", threads=1):
    train_cfg = config["train"]
    total_steps, random_steps = train_cfg["total_steps"], train_cfg["random_steps"]
    batch_size, eval_every = train_cfg["batch_size"], train_cfg["eval_every"]

    if train_cfg["replay_size"] <= batch_size:
        raise ValueError("train.replay_size must be greater than train.batch_size")
    if total_steps <= max(random_steps, batch_size + 1):
        raise ValueError(
            "train.total_steps must exceed both train.random_steps "
            "and train.batch_size + 1 to allow a gradient update"
        )

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new output directory")
    (out / "checkpoints").mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**config, "seed": seed}, sort_keys=False), encoding="utf-8")

    seed_everything(seed, threads)
    env = make_env(config["env_name"], config["env"])
    state, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    agent = make_agent(config, seed, env.action_space, device=device, threads=threads)

    exploration_rng = env.action_space.np_random
    state, _ = env.reset()
    env.action_space._np_random = exploration_rng
    memory = ReplayBuffer(train_cfg["replay_size"], state.shape[0], env.action_space.shape[0])

    episodes, losses_log = CsvLog(out / "episodes.csv"), CsvLog(out / "losses.csv")
    validation = CsvLog(out / "validation.csv")
    episode, episode_steps, episode_reward, last_losses = 1, 0, 0.0, {}
    started = time.time()
    print(f"Training {config['algorithm']} (seed {seed}) for {total_steps} steps -> {out}", flush=True)
    try:
        for step in range(total_steps):
            action = env.action_space.sample() if step < random_steps else agent.select_action(state)
            if memory.size > batch_size and step >= random_steps:
                last_losses = {**last_losses, **agent.update(memory, batch_size)}
            next_state, reward, terminated, truncated, _ = env.step(action)
            memory.add(state, action, reward, next_state, float(terminated))
            state = next_state
            episode_steps += 1
            episode_reward += float(reward)
            done_steps = step + 1

            if terminated or truncated:
                episodes.write({"step": done_steps, "episode": episode, "steps": episode_steps,
                                "reward": episode_reward, "terminated": bool(terminated)})
                exploration_rng = env.action_space.np_random
                state, _ = env.reset()
                env.action_space._np_random = exploration_rng
                episode, episode_steps, episode_reward = episode + 1, 0, 0.0

            if done_steps % LOG_EVERY == 0:
                losses_log.write({"step": done_steps, "critic_loss": last_losses.get("critic_loss"),
                                  "actor_loss": last_losses.get("actor_loss"),
                                  "clipped_target_fraction": last_losses.get("clipped_target_fraction")})
            if eval_every and (done_steps % eval_every == 0 or done_steps == total_steps):
                with isolated_rng():
                    result = evaluate(agent, config["env_name"], config["env"],
                                      range(VALIDATION_SEED, VALIDATION_SEED + train_cfg["eval_episodes"]))
                save_checkpoint(out / "checkpoints" / f"step_{done_steps:07d}.pt", agent, config, seed, done_steps)
                row = {"step": done_steps, **{k: result[k] for k in ("unsafe_rate", "success_rate", "mean_speed")},
                       "minutes": round((time.time() - started) / 60, 1)}
                validation.write(row)
                print(json.dumps(row), flush=True)
        save_checkpoint(out / "final.pt", agent, config, seed, total_steps)
    finally:
        for log in (episodes, losses_log, validation):
            log.close()
        env.close()
    return out / "final.pt"
