"""Environment wrapper, graph encoder, the five agents and checkpoints."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from dgi_sac.algorithms import make_agent
from dgi_sac.algorithms.dgi_sac import GraphEncoder
from dgi_sac.algorithms.sac import td_target_bound
from dgi_sac.config import load_config
from dgi_sac.env import FEATURE_SCALE, make_env
from dgi_sac.evaluation import evaluate, load_checkpoint, resolve_checkpoint
from dgi_sac.replay_buffer import ReplayBuffer
from dgi_sac.trainer import train

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# Actor and critic parameter counts reported in Table 4.
TABLE4 = {"dgi_sac": (476932, 158210), "sac": (79364, 158210), "mlp": (476756, 158210),
          "td3": (78850, 158210), "ddpg": (78850, 79105)}


def agent_for(name, **overrides):
    config = load_config(CONFIGS / f"{name}.yaml", [f"{k}={v}" for k, v in overrides.items()])
    env = make_env(config["env_name"], config["env"])
    return config, env, make_agent(config, 1, env.action_space)


@pytest.mark.parametrize("name", list(TABLE4))
def test_config_selects_method_with_table4_sizes(name):
    config, env, agent = agent_for(name)
    env.close()
    assert config["algorithm"] == name
    counts = agent.parameter_counts()
    assert (counts["actor"], counts["critic"]) == TABLE4[name]


def test_observation_is_ego_centric_and_scaled():
    config, env, _ = agent_for("sac")
    try:
        obs, _ = env.reset(seed=310000)
        raw = env.unwrapped.observation_type.observe().astype(np.float32)
        raw[0, 1] = 0.0
        assert obs.shape == (48,) and np.array_equal(obs.reshape(8, 6), raw / FEATURE_SCALE)
    finally:
        env.close()


def test_leaving_the_road_receives_collision_penalty():
    _, env, _ = agent_for("sac")
    try:
        seen = 0
        for seed in range(310001, 310011):
            env.reset(seed=seed)
            for _ in range(200):
                _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.3]))
                if not env.unwrapped.vehicle.on_road:
                    assert reward == -2.0 and terminated
                    seen += 1
                if terminated or truncated:
                    break
        assert seen > 0
    finally:
        env.close()


def test_edge_weights_are_inverse_distances_in_metres():
    state = torch.zeros(1, 8, 6)
    state[0, :3, 0] = 1
    state[0, 1, 1] = 10.0 / 100   # 10 m ahead
    state[0, 2, 2] = 4.0 / 12     # 4 m to the side
    encoder = GraphEncoder(hidden_size=8, num_layers=1)
    positions = state.index_select(2, encoder.position_index_tensor) * torch.tensor([100.0, 12.0])
    adjacency = encoder.adjacency(positions, state[:, :, 0] > 0.5)[0]
    assert torch.isclose(adjacency[0, 1], torch.tensor(0.1)) and torch.isclose(adjacency[0, 2], torch.tensor(0.25))
    assert adjacency[0, 0] == 1.0 and adjacency[0, 3:].abs().sum() == 0  # self-loop; absent vehicles


def test_cut_ego_edges_hides_neighbours_from_the_graph_branch():
    torch.manual_seed(0)
    state = torch.randn(4, 8, 6)
    state[:, :, 0] = 1
    encoder = GraphEncoder(hidden_size=8, num_layers=3)
    encoder.ablation = "cut_ego_edges"
    changed = state.clone()
    changed[:, 1:, 3:] += 5.0  # neighbours' velocities and headings
    assert torch.allclose(encoder(state), encoder(changed), atol=1e-6)


@pytest.mark.parametrize("name", list(TABLE4))
def test_every_method_trains_on_a_replay_minibatch(name):
    _, env, agent = agent_for(name)
    env.close()
    rng = np.random.default_rng(0)
    memory = ReplayBuffer(512)
    for _ in range(512):
        memory.add(rng.normal(size=48), rng.uniform(-1, 1, 2), rng.uniform(-1, 0.5), rng.normal(size=48), 0.0)
    before = [p.clone() for p in agent.policy.parameters()]
    losses = [agent.update(memory, 256) for _ in range(2)]
    assert all(np.isfinite(loss["critic_loss"]) for loss in losses)
    moved = any(not torch.equal(a, b) for a, b in zip(before, agent.policy.parameters(), strict=True))
    if name == "td3":
        assert "actor_loss" not in losses[0] and "actor_loss" in losses[1]  # delayed actor update
    assert moved


def test_td_target_clip_bound_and_effect():
    config, env, agent = agent_for("dgi_sac", **{"train.td_target_clip": "auto"})
    env.close()
    assert abs(td_target_bound(config) - (0.5 + 0.025 * np.log(4.0)) / 0.01) < 1e-12
    assert abs(agent.td_target_clip - 53.4657) < 1e-3
    memory = ReplayBuffer(512)
    rng = np.random.default_rng(0)
    for _ in range(512):  # rewards far above the bound make clipping active
        memory.add(rng.normal(size=48), rng.uniform(-1, 1, 2), 100.0, rng.normal(size=48), 0.0)
    assert agent.update(memory, 256)["clipped_target_fraction"] == 1.0


def test_short_training_run_saves_a_loadable_checkpoint(tmp_path):
    config = load_config(CONFIGS / "dgi_sac.yaml", ["train.total_steps=400", "train.random_steps=300",
                                                    "train.eval_every=400", "train.eval_episodes=1"])
    final = train(config, 1, tmp_path / "run")
    assert (tmp_path / "run" / "validation.csv").is_file() and (tmp_path / "run" / "checkpoints").is_dir()
    agent, loaded = load_checkpoint(final)
    assert loaded == config and agent.updates == 100  # one update per step after the 300 random steps
    summary = evaluate(agent, loaded["env_name"], loaded["env"], [380000], tmp_path / "eval")
    assert summary["episodes"] == 1 and (tmp_path / "eval" / "episodes.csv").is_file()


@pytest.mark.parametrize("name", ["td3", "ddpg"])
def test_study_checkpoint_layout_is_converted(name, tmp_path):
    """Released TD3/DDPG models store a constant input scale of ones and the update count as delay_counter."""
    config, env, agent = agent_for(name)
    env.close()
    state = agent.state_dict()
    study_state = {"variant": name, "config": None, "delay_counter": 490000,
                   "exploration_rng": state["exploration_rng"]}
    for key, weights in state.items():
        if key.endswith("_state_dict") and "optimizer" not in key:
            study_state[key] = {**weights, "obs_scale": torch.ones(50 if "critic" in key else 48)}
    study_config = {"train": {"gamma": 0.99, "tau": 0.005, "lr": 3e-4, "hidden_size": 256, "batch_size": 256,
                              "replay_size": 1000000, "num_steps": 500000, "alpha": 0.025},
                    "run": {"env_name": config["env_name"]}, "env": config["env"]}
    torch.save({"agent": study_state, "variant": name, "config": study_config}, tmp_path / "model.pt")
    loaded, loaded_config = load_checkpoint(tmp_path / "model.pt")
    assert loaded_config["algorithm"] == name and loaded.updates == 490000
    for a, b in zip(agent.policy.parameters(), loaded.policy.parameters(), strict=True):
        assert torch.equal(a, b)


def test_released_model_path_is_resolved_inside_the_models_folder(tmp_path):
    target = tmp_path / "models" / "dgi_sac" / "seed_002.pt"
    target.parent.mkdir(parents=True)
    target.touch()
    args = Namespace(checkpoint=None, method="dgi_sac", seed=2, models_dir=str(tmp_path / "models"))
    assert resolve_checkpoint(args) == (target, "dgi_sac_seed002")


def test_own_checkpoints_get_distinct_result_names():
    def name(path):
        return resolve_checkpoint(Namespace(checkpoint=path, method=None, seed=None, models_dir="models"))[1]

    assert name("runs/dgi_sac_seed1/final.pt") == "dgi_sac_seed1_final"
    assert name("runs/dgi_sac_seed1/checkpoints/step_0025000.pt") == "dgi_sac_seed1_step_0025000"
