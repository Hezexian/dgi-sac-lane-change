"""Train a method using a YAML configuration."""

import argparse

from dgi_sac.config import load_config
from dgi_sac.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="configs/dgi_sac.yaml, sac.yaml, mlp.yaml, td3.yaml or ddpg.yaml")
    parser.add_argument("--seed", type=int, required=True, help="training seed (the paper used 26093001-26093015)")
    parser.add_argument("--output", help="output directory (default: runs/<method>_seed<seed>)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. train.total_steps=20000 (repeatable)")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    output = args.output or f"runs/{config['algorithm']}_seed{args.seed}"
    train(config, args.seed, output, device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()
