"""Persistent experiment training runner; no scientific queue mutations here."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import argparse
import csv
import hashlib
import importlib.metadata
import json
import cloudpickle as pickle
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 registers environment
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from core import Agent, LEGACY, historical_config, isolated_rng, rng_state, restore_rng, seed_everything
from replay import Replay, VARIANTS
from deterministic_agents import TDAgent
from observation import PROTOCOL as OBS_PROTOCOL, make_env
from traffic_mix import TrafficMixWrapper, PROTOCOL as TRAFFIC_PROTOCOL

LEARNING_STARTS = 10000


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    for attempt in range(6):
        try:
            with open(tmp, 'w', encoding='utf8') as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with open(tmp, 'wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256(path)


def append_json(path, data):
    with open(path, 'a', encoding='utf8') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')
        f.flush()


def code_digest(root):
    files = {}
    for pattern in ('algorithms/**/*.py', 'configs/**/*.py', 'configs/*.yaml', 'utils.py',
                    'core.py', 'run_training.py', 'replay.py',
                    'evaluate.py', 'metrics.py', 'deterministic_agents.py', 'observation.py',
                    'traffic_mix.py', 'mix_evaluate.py'):
        for path in sorted(Path(root).glob(pattern)):
            files[path.relative_to(root).as_posix()] = sha256(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


def environment_fingerprint():
    versions = {name: importlib.metadata.version(name) for name in
                ['torch','numpy','gymnasium','highway-env','cloudpickle']}
    envroot = Path(highway_env.__file__).parent
    environment_sources = {p.relative_to(envroot).as_posix(): sha256(p) for p in sorted(envroot.rglob('*.py'))}
    lock = Path(__file__).parent.parent/'uv.lock'
    return {'versions': versions, 'highway_source_sha256': hashlib.sha256(json.dumps(environment_sources,sort_keys=True).encode()).hexdigest(),
            'uv_lock_sha256': sha256(lock) if lock.exists() else None,
            'torch_cuda': torch.version.cuda, 'python': __import__('sys').version}


def scientific_spec(args, config, code_sha, env):
    return {'variant': args.variant, 'seed': args.seed, 'budget': args.budget,
            'traffic_mix': getattr(args, 'traffic_mix', 'none'),
            'traffic_mix_protocol': TRAFFIC_PROTOCOL if getattr(args, 'traffic_mix', 'none') == 'mix' else None,
            'device': args.device, 'priority_mode': args.priority_mode,
            'config': config, 'code_sha256': code_sha,
            'evaluation_every': args.eval_every, 'evaluation_episodes': args.eval_episodes,
            'full_every': args.full_every, 'threads': args.threads,
            'protected_steps': args.protected_steps, 'kind': args.kind,
            'learning_starts': args.learning_starts,
            'environment_fingerprint': environment_fingerprint(),
            'resolved_env_config': (env.default_env_config if isinstance(env, TrafficMixWrapper)
                                    else env.unwrapped.config),
            'observation_reward_protocol': OBS_PROTOCOL,
            'resolved_see_behind': getattr(env.unwrapped.observation_type,'see_behind',None)}


def describe_plan(args):
    config = historical_config('sac' if args.variant in {'ddpg','td3'} else args.variant)
    config['run'].update(seed=args.seed,cuda=args.device=='cuda')
    config['train'].update(num_steps=args.budget,replay_size=args.replay_capacity)
    env = make_env(config['run']['env_name'],config['env'])
    try:
        env.reset(seed=args.seed)
        code_sha,files = code_digest(Path(__file__).parent)
        spec = scientific_spec(args,config,code_sha,env)
        return {'plan_sha256': hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                'scientific':spec,'source_files':files}
    finally:
        env.close()


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'checkpoints').mkdir(exist_ok=True)
        (self.root/'weights').mkdir(exist_ok=True)
        self.started_wall = time.time()
        self.attempt = len((self.root/'resume-events.jsonl').read_text().splitlines())+1 if (self.root/'resume-events.jsonl').exists() else 0
        if args.resume:
            self.attempt = max(1, self.attempt)
        self.code_sha, self.source_files = code_digest(Path(__file__).parent)
        if args.expected_code_sha and self.code_sha != args.expected_code_sha:
            raise RuntimeError('Source hash does not match frozen command')
        self.config = historical_config('sac' if args.variant in {'ddpg','td3'} else args.variant)
        self.config['run'].update(seed=args.seed, cuda=args.device=='cuda')
        self.config['train']['num_steps'] = args.budget
        self.config['train']['replay_size'] = args.replay_capacity
        scientific = {'variant': args.variant, 'seed': args.seed, 'budget': args.budget,
                      'device': args.device, 'priority_mode': args.priority_mode,
                      'config': self.config, 'code_sha256': self.code_sha,
                      'evaluation_every': args.eval_every, 'evaluation_episodes': args.eval_episodes,
                      'full_every': args.full_every, 'threads': args.threads,
                      'protected_steps': args.protected_steps, 'kind': args.kind,
                      'learning_starts': args.learning_starts}
        seed_everything(args.seed, args.threads)
        self.env = make_env(self.config['run']['env_name'], self.config['env'])
        self.state, _ = self.env.reset(seed=args.seed)
        self.env.action_space.seed(args.seed)
        if args.variant in {'ddpg','td3'}:
            self.agent = TDAgent(args.variant, args.seed, args.device, self.config,
                                 self.env.action_space, args.threads)
        else:
            self.agent = Agent(args.variant, args.seed, args.device, self.config,
                               self.env.action_space, args.threads, args.priority_mode)
        # Preserve original train.py's second, unseeded reset before first training episode.
        # highway-env recreates action spaces at reset: carry the seeded generator
        # forward rather than losing it or restarting the same action stream each episode.
        exploration_rng = self.env.action_space.np_random
        if getattr(args, 'traffic_mix', 'none') == 'mix':
            self.env = TrafficMixWrapper(self.env, args.seed)
        self.state, _ = self.env.reset()
        self.env.action_space._np_random = exploration_rng
        replay_variant = 'branch_uniform' if args.variant in {'sac','dgi','capacity','ddpg','td3'} else args.variant
        self.memory = Replay(args.replay_capacity, (48,), args.seed, variant=replay_variant,
                             priority_mode=args.priority_mode)
        scientific = scientific_spec(args,self.config,self.code_sha,self.env)
        computed = hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
        if args.plan_sha256 and args.plan_sha256 != computed:
            raise RuntimeError('Requested plan_sha256 does not match computed science+environment identity')
        self.plan_sha = computed
        self.scientific = scientific
        self.loop = {'step': 0, 'episode': 1, 'episode_step': 0, 'episode_reward': 0.,
                     'best_reward': None, 'best_step': None, 'elapsed_seconds': 0.,
                     'validation_seconds': 0., 'checkpoint_seconds': 0., 'train_seconds': 0.,
                     'last_losses': None}
        self.recovery_manifest = {'plan_sha256': self.plan_sha, 'latest': None, 'entries': [], 'pruned': []}
        self.done_episode = False
        self.actor_grad_norms = []
        if args.resume:
            self.restore(args.resume)
        elif (self.root/'status.json').exists():
            raise RuntimeError('Existing run directory requires explicit --resume; refusing duplicate')
        (self.root/'online-diagnostics.jsonl').touch(exist_ok=True)
        atomic_json(self.root/'run-config.json', {'run_id': args.run_id, 'plan_sha256': self.plan_sha,
                    'computed_scientific_sha256': computed, 'scientific': scientific,
                    'source_files': self.source_files, 'params': self.agent.parameter_counts(),
                    'capacity_match': self.agent.capacity_match,
                    'resolved_env_config': self.env.unwrapped.config,
                    'resolved_see_behind': getattr(self.env.unwrapped.observation_type, 'see_behind', None)})
        self.status('running')
        if not args.resume:
            self.log_traffic_episode()

    def log_traffic_episode(self):
        if isinstance(self.env, TrafficMixWrapper):
            append_json(self.root/'traffic-episodes.jsonl', {
                'episode': self.loop['episode'], 'start_step': self.loop['step'],
                'attempt': self.attempt, **self.env.last_episode})

    def status(self, status, **extra):
        elapsed = self.loop['elapsed_seconds'] + time.time()-self.started_wall
        atomic_json(self.root/'status.json', {'run_id': self.args.run_id, 'status': status,
                    'plan_sha256': self.plan_sha, 'step': self.loop['step'], 'budget': self.args.budget,
                    'episode': self.loop['episode'], 'updated_at': utcnow(), 'pid': os.getpid(),
                    'elapsed_seconds': elapsed, 'counts': self.agent.counts, 'device': self.args.device,
                    'best_step': self.loop['best_step'], **extra})

    def weights(self, step):
        agent = {k:v for k,v in self.agent.state_dict().items() if 'optimizer' not in k}
        path = self.root/'weights'/f'step-{step:07d}.pt'
        if path.exists():
            path = self.root/'weights'/f'step-{step:07d}-attempt-{self.attempt}-{time.time_ns()}.pt'
        digest = atomic_torch(path, {'agent': agent, 'variant': self.args.variant, 'config': self.config,
                    'run_id': self.args.run_id, 'step': step, 'plan_sha256': self.plan_sha})
        atomic_json(path.with_suffix('.json'), {'sha256': digest, 'step': step, 'plan_sha256': self.plan_sha})
        return path, digest

    def full_checkpoint(self, reason):
        start = time.time()
        step = self.loop['step']
        name = f'full-{step:07d}-{reason}-attempt-{self.attempt}.pt'
        path = self.root/'checkpoints'/name
        loop = self.loop.copy()
        loop['elapsed_seconds'] += time.time()-self.started_wall
        payload = {'schema': 1, 'run_id': self.args.run_id, 'plan_sha256': self.plan_sha,
                   'scientific': self.scientific, 'code_sha256': self.code_sha,
                   'agent': self.agent.state_dict(), 'replay': self.memory.state_dict(),
                   'environment_pickle': pickle.dumps(self.env, protocol=5),
                   'rng': rng_state(), 'loop': loop, 'state': np.asarray(self.state).copy(),
                   'done_episode': self.done_episode, 'device': self.args.device}
        if isinstance(self.env, TrafficMixWrapper):
            payload['traffic_mix'] = self.env.state_dict()
            payload['actor_grad_norms'] = self.actor_grad_norms.copy()
        digest = atomic_torch(path, payload)
        del payload
        check = torch.load(path, map_location='cpu', weights_only=False)
        if check['plan_sha256'] != self.plan_sha or check['loop']['step'] != step or check['replay']['size'] != self.memory.size:
            raise RuntimeError('Written recovery checkpoint failed readback')
        del check
        protected = step in self.args.protected_steps or step == self.args.budget or reason == 'failure'
        old = self.recovery_manifest
        entries = [x for x in old['entries'] if x['path'] != name]
        entries.append({'path': name, 'step': step, 'sha256': digest, 'protected': protected,
                        'reason': reason, 'saved_at': utcnow()})
        rolling = sorted([x for x in entries if not x['protected']], key=lambda x: (x['step'], x['saved_at']))
        prune = rolling[:-2]
        keep = [x for x in entries if x not in prune]
        new = {'plan_sha256': self.plan_sha, 'latest': name, 'entries': keep,
               'pruned': old['pruned'] + [{**x, 'pruned_at': utcnow()} for x in prune]}
        atomic_json(self.root/'checkpoints'/'manifest.json', new)
        self.recovery_manifest = new
        checkpoint_root = (self.root/'checkpoints').resolve()
        for entry in prune:
            target = (checkpoint_root/entry['path']).resolve()
            if target.parent != checkpoint_root or not target.name.startswith('full-') or target == path.resolve():
                raise RuntimeError('Checkpoint prune boundary violation')
            if target.exists():
                target.unlink()
        duration = time.time()-start
        self.loop['checkpoint_seconds'] += duration
        append_json(self.root/'checkpoint-events.jsonl', {'step': step, 'path': name,
                    'sha256': digest, 'seconds': duration, 'protected': protected, 'pruned': len(prune)})
        return path

    def restore(self, ref):
        manifest = json.loads((self.root/'checkpoints'/'manifest.json').read_text())
        if manifest['plan_sha256'] != self.plan_sha:
            raise RuntimeError('Recovery manifest scientific identity mismatch')
        path = self.root/'checkpoints'/manifest['latest'] if ref == 'auto' else Path(ref)
        entry = next((x for x in manifest['entries'] if x['path'] == path.name), None)
        if entry is None or sha256(path) != entry['sha256']:
            raise RuntimeError('Recovery checkpoint hash absent/mismatched')
        data = torch.load(path, map_location=self.args.device, weights_only=False)
        if data['plan_sha256'] != self.plan_sha or data['scientific'] != self.scientific or data['device'] != self.args.device:
            raise RuntimeError('Recovery plan/config/device mismatch')
        self.agent.load_state_dict(data['agent'])
        self.memory.load_state_dict(data['replay'])
        self.env.close()
        self.env = pickle.loads(data['environment_pickle'])
        self.state = data['state']
        self.loop = data['loop']
        self.done_episode = data['done_episode']
        self.actor_grad_norms = []
        if isinstance(self.env, TrafficMixWrapper):
            self.env.load_state_dict(data['traffic_mix'])
            self.actor_grad_norms = data['actor_grad_norms']
        self.recovery_manifest = manifest
        restore_rng(data['rng'])
        self.started_wall = time.time()
        append_json(self.root/'resume-events.jsonl', {'step': self.loop['step'], 'checkpoint': str(path), 'time': utcnow()})

    def validate(self):
        from evaluate import evaluate
        start = time.time()
        step = self.loop['step']
        weight_path, digest = self.weights(step)
        with isolated_rng():
            summary = evaluate(self.agent, {'env_name': self.config['run']['env_name'], 'config': self.config['env']},
                               list(range(310000, 310000+self.args.eval_episodes)),
                               output_dir=self.root/'validation'/f'step-{step:07d}'/f'attempt-{self.attempt}', tag='validation',
                               save_trajectories=False)
        reward = summary.get('mean_reward', summary.get('avg_reward'))
        if reward is None or not np.isfinite(reward):
            raise RuntimeError('Invalid validation reward')
        if self.loop['best_reward'] is None or reward > self.loop['best_reward']:
            self.loop['best_reward'], self.loop['best_step'] = float(reward), step
            atomic_json(self.root/'best.json', {'step': step, 'mean_reward': reward,
                        'weights': str(weight_path.relative_to(self.root)), 'sha256': digest})
        duration = time.time()-start
        self.loop['validation_seconds'] += duration
        append_json(self.root/'validation-nodes.jsonl', {'step': step, 'attempt': self.attempt, 'seconds': duration, 'summary': summary})
        if self.args.variant not in {'sac','dgi','capacity'}:
            diagnostic = self.agent.diagnostics(self.memory, step)
            if hasattr(self.memory, 'diagnostics'):
                diagnostic['stored'] = self.memory.diagnostics()
            append_json(self.root/'sampling-diagnostics.jsonl', {'step': step, **diagnostic})

    def run(self):
        a = self.args
        next_status = time.monotonic()
        try:
            if self.loop['step'] == 0 and self.recovery_manifest['latest'] is None:
                self.full_checkpoint('initial')
            while self.loop['step'] < a.budget:
                if self.done_episode:
                    exploration_rng = self.env.action_space.np_random
                    self.state, _ = self.env.reset()
                    self.env.action_space._np_random = exploration_rng
                    self.loop['episode'] += 1
                    self.loop['episode_step'], self.loop['episode_reward'] = 0, 0.
                    self.done_episode = False
                    self.log_traffic_episode()
                step = self.loop['step']
                start = time.perf_counter()
                # Preserve historical action/update/interaction order exactly.
                action = self.env.action_space.sample() if step < 10000 else self.agent.select_action(self.state)
                # Learning-start protocol: no gradient updates during the 10K uniform-random phase. With ego-centric
                # normalised observations the critic otherwise extrapolates to the actor's own (unseen) actions
                # and diverges (Q median 75 at 10K vs bound 50); following the TD3 reference implementation
                # (Fujimoto et al. 2018), training starts after start_timesteps. Applies to every method.
                if self.memory.size > 256 and step >= self.args.learning_starts:
                    actor_updates = self.agent.counts['actor_updates']
                    self.loop['last_losses'] = self.agent.update(self.memory, 256, step,
                            self.loop['episode_step']+1, a.budget)
                    grad_norm = self.loop['last_losses'].get('actor_grad_norm')
                    if grad_norm is not None and self.agent.counts['actor_updates'] > actor_updates:
                        self.actor_grad_norms.append(float(grad_norm))
                        del self.actor_grad_norms[:-1000]
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.memory.add(self.state, action, reward, next_state, float(terminated), step)
                self.state = next_state
                self.loop['step'] += 1
                self.loop['episode_step'] += 1
                self.loop['episode_reward'] += float(reward)
                self.done_episode = bool(terminated or truncated)
                self.loop['train_seconds'] += time.perf_counter()-start
                if self.done_episode:
                    append_json(self.root/'train-episodes.jsonl', {'step': self.loop['step'],
                            'attempt': self.attempt,
                            'episode': self.loop['episode'], 'episode_steps': self.loop['episode_step'],
                            'reward': self.loop['episode_reward'], 'terminated': bool(terminated), 'truncated': bool(truncated)})
                step = self.loop['step']
                if step % 10000 == 0:
                    losses = self.loop['last_losses'] or {}
                    append_json(self.root/'online-diagnostics.jsonl', {'step': step,
                            'last_losses': self.loop['last_losses'], **losses,
                            'actor_grad_norm_window_n': len(self.actor_grad_norms),
                            'actor_grad_norm_mean': float(np.mean(self.actor_grad_norms)) if self.actor_grad_norms else None,
                            'actor_grad_norm_variance': float(np.var(self.actor_grad_norms)) if self.actor_grad_norms else None})
                if a.eval_every and (step % a.eval_every == 0 or step == a.budget):
                    self.status('validating')
                    self.validate()
                if step == a.budget or step in a.protected_steps or (a.full_every and step % a.full_every == 0):
                    self.full_checkpoint('final' if step == a.budget else ('pilot' if step in a.protected_steps else 'rolling'))
                if time.monotonic() >= next_status:
                    self.status('running', last_losses=self.loop['last_losses'])
                    next_status = time.monotonic()+15
                if a.stop_after and step >= a.stop_after:
                    self.full_checkpoint('test-interruption')
                    self.status('interrupted_for_test')
                    return
            if not a.eval_every:
                self.weights(self.loop['step'])
            self.status('trained', throughput_train=self.loop['step']/max(self.loop['train_seconds'],1e-9),
                        timing=self.loop, parameters=self.agent.parameter_counts())
            atomic_json(self.root/'training-complete.json', {'step': self.loop['step'], 'plan_sha256': self.plan_sha,
                         'counts': self.agent.counts, 'time': utcnow()})
        except BaseException as error:
            # A failure may occur midway through an optimizer/env transition.
            # Preserve forensic state but NEVER promote it to an auto-resume checkpoint.
            try:
                forensic = self.root/'weights'/f'failure-step-{self.loop["step"]:07d}-attempt-{self.attempt}.pt'
                atomic_torch(forensic, {'forensic_only_not_resumable': True,
                    'error': repr(error), 'agent': self.agent.state_dict(), 'loop': self.loop,
                    'state': self.state, 'rng': rng_state(), 'plan_sha256': self.plan_sha,
                    'replay_size': self.memory.size, 'replay_pointer': self.memory.ptr})
            except BaseException as save_error:
                append_json(self.root/'failure-events.jsonl', {'time': utcnow(), 'checkpoint_error': repr(save_error)})
            self.status('failed', error=repr(error))
            raise
        finally:
            self.env.close()


class TrafficArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        parsed, remaining = super().parse_known_args(args, namespace)
        if parsed.eval_every is None:
            parsed.eval_every = 25000 if parsed.traffic_mix == 'mix' else 50000
        return parsed, remaining


def parser():
    p = TrafficArgumentParser()
    p.add_argument('--variant', choices=['sac', 'dgi', 'capacity'], required=True)
    p.add_argument('--traffic-mix', choices=['none', 'mix'], default='none')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--learning-starts', type=int, default=LEARNING_STARTS,
                   help='First environment step with gradient updates (default 10000; tests may use 0)')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--priority-mode', choices=['floor','epsilon','standard'], default='floor', help=argparse.SUPPRESS)
    p.add_argument('--replay-capacity', type=int, default=1000000)
    p.add_argument('--eval-every', type=int, default=None,
                   help='validation interval: mix defaults to 25000; none preserves legacy 50000')
    p.add_argument('--eval-episodes', type=int, default=50)
    p.add_argument('--full-every', type=int, default=100000)
    p.add_argument('--protected-steps', type=int, nargs='*', default=[])
    p.add_argument('--resume', default=None)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--plan-sha256', default=None)
    p.add_argument('--expected-code-sha', default=None)
    p.add_argument('--kind', choices=['functional','benchmark','science','screening'], default='science')
    p.add_argument('--describe-plan', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.kind == 'science' and args.budget < 500000:
        raise SystemExit('Formal budget below500K forbidden')
    if args.describe_plan:
        print(json.dumps(describe_plan(args),indent=2))
    else:
        Runner(args).run()
