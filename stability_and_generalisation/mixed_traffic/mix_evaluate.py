"""Traffic behavior interventions around the unchanged evaluator and loader."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import evaluate as evaluator
from traffic_mix import HeteroTrafficWrapper, PROTOCOL, BEHAVIOR_DISTRIBUTIONS


@contextmanager
def traffic_environment(hetero_traffic, episode_records=None):
    if hetero_traffic not in ('off', 'on', 'aggressive'):
        raise ValueError('hetero_traffic must be off, on or aggressive')
    if hetero_traffic == 'off':
        yield
        return
    original = evaluator.make_env
    try:
        evaluator.make_env = lambda config: HeteroTrafficWrapper(
            original(config), hetero_traffic, episode_records)
        yield
    finally:
        evaluator.make_env = original


def write_traffic_outputs(output, mode, records, **provenance):
    directory = Path(output)
    evaluator.atomic_json(directory/'traffic-evaluation.json', {
        'hetero_traffic': mode, 'protocol': PROTOCOL,
        'active_behavior_distributions': BEHAVIOR_DISTRIBUTIONS.get(mode, {}),
        'density_source': 'YAML unchanged',
        'density_rng': ('not constructed in off mode' if mode == 'off'
                        else 'derived from evaluation seed but not consumed'),
        **provenance})
    with (directory/'traffic-episodes.jsonl').open('w', encoding='utf8') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+'\n')


def evaluate(agent, env_config, seeds, output_dir=None, tag='validation',
             save_trajectories=False, *, hetero_traffic='off'):
    # Scoped process-local replacement. Intended for serial evaluation, not threads.
    records = []
    with traffic_environment(hetero_traffic, records):
        result = evaluator.evaluate(agent, env_config, seeds, output_dir, tag, save_trajectories)
    if output_dir is not None:
        write_traffic_outputs(output_dir, hetero_traffic, records)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hetero-traffic', '--hetero-speed', dest='hetero_traffic',
                        choices=['off', 'on', 'aggressive'], default='off')
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if '--help' in remaining or '-h' in remaining:
        print('Additional option: --hetero-traffic {off,on,aggressive} (default: off; alias: --hetero-speed)')
    previous = sys.argv
    records = []
    try:
        sys.argv = [str(Path(__file__)), *remaining]
        with traffic_environment(args.hetero_traffic, records):
            # Keep the frozen CLI's checkpoint loading, config and metrics behavior.
            evaluator.main()
    finally:
        sys.argv = previous
    # Preserve summary.json and episodes.csv exactly; record intervention provenance
    # separately, because the frozen summary's evaluator hash does not cover wrappers.
    metadata = argparse.ArgumentParser(add_help=False)
    for flag in ('output', 'checkpoint', 'config', 'variant'):
        metadata.add_argument('--' + flag)
    paths, _ = metadata.parse_known_args(remaining)
    from run_training import code_digest, sha256
    code_sha, sources = code_digest(Path(__file__).parent)
    write_traffic_outputs(paths.output, args.hetero_traffic, records,
        checkpoint=paths.checkpoint, checkpoint_sha256=sha256(paths.checkpoint),
        config=paths.config, config_sha256=sha256(paths.config),
        variant_argument=paths.variant, code_sha256=code_sha, source_files=sources)


if __name__ == '__main__':
    main()
