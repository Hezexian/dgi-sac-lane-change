"""Evaluate a trained policy on test or screening scenarios using deterministic actions."""

import argparse
import json

from dgi_sac.evaluation import evaluate, load_checkpoint, resolve_checkpoint, set_ablation

SCENARIOS = {"test": 380000, "screening": 350000}  # first scenario seed of each set


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="checkpoint written by train.py or a released .pt file")
    parser.add_argument("--method", choices=["dgi_sac", "sac", "mlp", "td3", "ddpg"])
    parser.add_argument("--seed", type=int, help="paper seed 1-15 (TD3 and DDPG 1-10) of a released model")
    parser.add_argument("--models-dir", default="models", help="the extracted 'models' folder of the model archive")
    parser.add_argument("--scenarios", choices=SCENARIOS, default="test")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--ablation", default="none", choices=["none", "cut_ego_edges", "zero_mlp_neighbors"])
    parser.add_argument("--output", help="new directory for episodes.csv, events.jsonl and summary.json")
    parser.add_argument("--trajectories", action="store_true", help="also write per-step trajectories.csv.gz")
    args = parser.parse_args()

    if not 1 <= args.episodes <= 1000:
        parser.error("--episodes must be between 1 and 1000")
    if args.checkpoint and (args.method is not None or args.seed is not None):
        parser.error("Use either --checkpoint or --method with --seed")

    checkpoint, name = resolve_checkpoint(args)
    agent, config = load_checkpoint(checkpoint)
    set_ablation(agent, args.ablation)
    first = SCENARIOS[args.scenarios]
    suffix = "" if args.ablation == "none" else f"_{args.ablation}"
    output = args.output or f"results/{name}_{args.scenarios}{suffix}"
    summary = evaluate(agent, config["env_name"], config["env"], range(first, first + args.episodes), output,
                       save_trajectories=args.trajectories)
    print(json.dumps({k: summary[k] for k in ("episodes", "unsafe_rate", "success_rate", "mean_speed")}, indent=2))
    print(f"per-episode results: {output}/episodes.csv")


if __name__ == "__main__":
    main()
