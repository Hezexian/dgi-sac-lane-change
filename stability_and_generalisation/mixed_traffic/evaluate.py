"""Isolated deterministic evaluation and trusted-local-checkpoint CLI.

API: evaluate(agent, env_config, seeds, output_dir=None, tag='validation',
              save_trajectories=False) -> JSON-safe summary.
The action API consumes flattened 48D observations and evaluate=True.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from metrics import (DEFINITION, aggregate, behavior_events, episode_summary,
                            geometry, lane_events)


@contextmanager
def preserve_rng():
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try: yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)


def make_env(env_config):
    import gymnasium as gym
    import highway_env  # registers environments; resolves caller's frozen environment
    from gymnasium.wrappers import FlattenObservation
    config=deepcopy(env_config)
    name=config.pop('env_name','highway-fast-v0')
    if 'config' in config: config=config['config']
    if float(config.get('policy_frequency',5)) != 5:
        raise ValueError('Frozen evaluation event metrics require policy_frequency=5')
    if int(config.get('lanes_count',4)) != 4:
        raise ValueError('Frozen evaluation lane metrics require four lanes')
    from observation import make_env as make_observation_env
    env=make_observation_env(name,config)
    if tuple(env.observation_space.shape)!=(48,):
        env.close();raise ValueError('Expected flattened 8x6 observation')
    return env


def snapshot(env, step, seed, previous_x=None, reward=0.):
    base=env.unwrapped;v=base.vehicle
    vehicles=[]
    for other in base.road.vehicles:
        vehicles.append({'ego':other is v,'x':float(other.position[0]),
                         'y':float(other.position[1]),'length':float(other.LENGTH),
                         'lane':int(other.lane_index[2]),
                         'forward_speed':float(other.speed*math.cos(other.heading))})
    lane=int(v.lane_index[2]);x=float(v.position[0])
    row={'episode_seed':int(seed),'step':int(step),'time':step/5.,
         'x':x,'y':float(v.position[1]),'heading':float(v.heading),'lane':lane,
         'speed':float(v.speed*math.cos(v.heading)),'crashed':bool(v.crashed),
         'on_road':bool(v.on_road),'distance_delta':x-previous_x if previous_x is not None else 0.,
         'reward':float(reward),'nearby_vehicles':sum(not c['ego'] and abs(c['x']-x)<=50
                   and abs(c['lane']-lane)<=1 for c in vehicles)}
    geom={target:geometry({'vehicles':vehicles},target) for target in range(4)}
    row.update(geom[lane])
    for target,values in geom.items():
        row.update({f'lane{target}_'+key:value for key,value in values.items()})
    # Raw geometry witnesses retained only when caller saves trajectories.
    row['vehicles']=vehicles
    return row


def attach_target_geometry(records, events):
    for row in records:
        row['target_lane']=None
        for key in ('front_gap_m','rear_gap_m','front_ttc_s','rear_ttc_s'):
            row['target_'+key]=None
    by_tick={r['step']:r for r in records}
    for event in events:
        target=event['target_lane']
        for tick in range(event['start_tick'],event['end_tick']+1):
            row=by_tick[tick];row['target_lane']=target
            for key in ('front_gap_m','rear_gap_m','front_ttc_s','rear_ttc_s'):
                row['target_'+key]=row[f'lane{target}_'+key]
        for label,tick in [('start',event['start_tick']),('crossing',event['crossing_tick']),('end',event['end_tick'])]:
            for key in ('front_gap_m','rear_gap_m','front_ttc_s','rear_ttc_s'):
                event[label+'_'+key]=by_tick[tick][f'lane{target}_'+key] if tick is not None else None
        for key in ('front_gap_m','rear_gap_m','front_ttc_s','rear_ttc_s'):
            vals=[by_tick[t][f'lane{target}_'+key] for t in range(event['start_tick'],event['end_tick']+1)]
            defined=[v for v in vals if v is not None]
            event['minimum_'+key]=min(defined) if defined else None
            event['defined_samples_'+key]=len(defined)


def atomic_json(path, value):
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def evaluate(agent, env_config, seeds, output_dir=None, tag='validation', save_trajectories=False):
    seeds=[int(s) for s in seeds]
    if not seeds or len(seeds)!=len(set(seeds)): raise ValueError('Nonempty unique seeds required')
    directory=Path(output_dir) if output_dir is not None else None
    if directory:
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'episodes.csv').exists():
            raise FileExistsError('Evaluation output already exists; use a fresh directory')
    started=time.monotonic();episodes=[];event_count=0
    env=None;epfile=None;eventfile=None;trajectoryfile=None;epwriter=None;trajectorywriter=None
    modes=[]
    # Preserve every module's actual training flag, including mixed eval/train trees.
    seen=set()
    for owner in (agent,getattr(agent,'legacy',None)):
        if owner is None:continue
        for value in vars(owner).values():
            if isinstance(value,torch.nn.Module):
                for module in value.modules():
                    if id(module) not in seen:
                        modes.append((module,module.training));seen.add(id(module))
    try:
        with preserve_rng():
            for module,_ in modes: module.training=False
            env=make_env(env_config)
            effective_config=deepcopy(env.unwrapped.config)
            if directory:
                epfile=(directory/'episodes.csv').open('w',encoding='utf-8',newline='')
                eventfile=(directory/'events.jsonl').open('w',encoding='utf-8')
                if save_trajectories:
                    trajectoryfile=gzip.open(directory/'trajectories.csv.gz','wt',encoding='utf-8',newline='')
            for seed in seeds:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                state,_=env.reset(seed=seed)
                env.action_space.seed(seed)
                records=[snapshot(env,0,seed)]
                initial_lane=records[0]['lane'];reward_sum=0.;terminated=False;truncated=False
                max_steps=math.ceil(float(effective_config['duration'])*float(effective_config['policy_frequency']))+1
                with torch.no_grad():
                    for step in range(1,max_steps+1):
                        action=agent.select_action(np.asarray(state).reshape(-1),evaluate=True)
                        state,reward,terminated,truncated,info=env.step(action)
                        reward_sum+=float(reward)
                        row=snapshot(env,step,seed,records[-1]['x'],reward)
                        row['crashed']=row['crashed'] or bool(info.get('crashed',False))
                        records.append(row)
                        if terminated or truncated:break
                    else: raise RuntimeError('Environment exceeded configured episode horizon')
                # v1.1 consumes post-step records; initial point serves geometric witness.
                changes=lane_events(records[1:],initial_lane)
                attach_target_geometry(records,changes)
                behaviors=behavior_events(records,changes)
                episode=episode_summary(records,changes,behaviors,seed,reward_sum,terminated,truncated)
                episodes.append(episode)
                if directory:
                    if epwriter is None:
                        epwriter=csv.DictWriter(epfile,fieldnames=list(episode));epwriter.writeheader()
                    epwriter.writerow(episode);epfile.flush()
                    for kind,items in [('lane_change',changes),('behavior',behaviors)]:
                        for event in items:
                            eventfile.write(json.dumps({'episode_seed':seed,'event_type':kind,**event},allow_nan=False)+'\n')
                            event_count+=1
                    eventfile.flush()
                    if trajectoryfile:
                        for row in records:
                            serialized={**row,'vehicles':json.dumps(row['vehicles'],separators=(',',':'))}
                            if trajectorywriter is None:
                                trajectorywriter=csv.DictWriter(trajectoryfile,fieldnames=list(serialized));trajectorywriter.writeheader()
                            trajectorywriter.writerow(serialized)
                        trajectoryfile.flush()
                if directory:
                    atomic_json(directory/'progress.json',{'tag':tag,'completed_episodes':len(episodes),
                                'requested_episodes':len(seeds),'elapsed_seconds':time.monotonic()-started})
            result=aggregate(episodes)
            result.update(tag=tag,seeds=seeds,elapsed_seconds=time.monotonic()-started,
                          metrics_definition=DEFINITION,trajectories_saved=bool(save_trajectories),
                          effective_see_behind=effective_config.get('observation',{}).get('see_behind','unspecified'),
                          observation_see_behind=getattr(env.unwrapped.observation_type,'see_behind',None),
                          evaluation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          metrics_code_sha256=hashlib.sha256(Path(__file__).with_name('metrics.py').read_bytes()).hexdigest(),
                          episode_metrics=episodes)
            # Validate before writing. No NaN/Infinity strings allowed.
            json.dumps(result,allow_nan=False)
            # Finish gzip trailer and CSV writes before publishing complete summary.
            for stream in (epfile,eventfile,trajectoryfile):
                if stream is not None:stream.close()
            epfile=eventfile=trajectoryfile=None
            if directory:atomic_json(directory/'summary.json',result)
            return result
    except BaseException as exc:
        if directory:
            atomic_json(directory/'failure.json',{'tag':tag,'completed_episodes':len(episodes),
                        'requested_episodes':len(seeds),'exception':type(exc).__name__,
                        'message':str(exc),'incomplete':True})
        raise
    finally:
        for module,mode in modes:module.training=mode
        for stream in (epfile,eventfile,trajectoryfile):
            if stream is not None:stream.close()
        if env is not None:env.close()


def apply_ablation(agent, mode):
    """Evaluation-only graph ablations (graph-use diagnostics). Returns the number of modules switched."""
    if mode == 'none':
        return 0
    count, seen = 0, set()
    for owner in (agent, getattr(agent, 'legacy', None)):
        if owner is None:
            continue
        for value in vars(owner).values():
            if not isinstance(value, torch.nn.Module):
                continue
            for module in value.modules():
                if id(module) in seen or not hasattr(module, 'ablation'):
                    continue
                seen.add(id(module))
                if (mode == 'zero_mlp_neighbors' and hasattr(module, 'graph_encoder')) or                         (mode in ('cut_ego_edges', 'ego_origin') and hasattr(module, 'convs')):
                    module.ablation = mode
                    count += 1
    if count == 0:
        raise ValueError(f'No module accepts ablation {mode!r}; graph agent required')
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--seeds-start',type=int,required=True);parser.add_argument('--episodes',type=int,default=1000)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--output',required=True)
    parser.add_argument('--tag',default='historical-best');parser.add_argument('--save-trajectories',action='store_true')
    parser.add_argument('--variant',choices=['sac', 'dgi', 'capacity'],default=None)
    parser.add_argument('--ablation',choices=['none','cut_ego_edges','ego_origin','zero_mlp_neighbors'],default='none')
    args=parser.parse_args()
    from configs import load_config,get_env_config,build_test_args
    from algorithms import load_algorithm
    path,config=load_config(args.config)
    testargs=build_test_args(config,path,Path(args.checkpoint))
    testargs.cuda=args.device=='cuda';testargs.device=args.device
    env_config=get_env_config(config)
    # Checkpoints are explicitly supplied trusted local artifacts. No monkeypatch.
    checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    env=make_env({'env_name':testargs.env_name,'config':env_config})
    try:
        if 'agent' in checkpoint:
            from core import Agent
            stored=checkpoint['agent']
            variant=args.variant or checkpoint.get('variant') or stored['variant']
            if variant in {'ddpg','td3'}:
                from deterministic_agents import TDAgent
                agent=TDAgent(variant,args.seeds_start,device=args.device,config=checkpoint.get('config',config),
                              action_space=env.action_space)
            else:
                agent=Agent(variant,args.seeds_start,device=args.device,config=checkpoint.get('config',config),
                            action_space=env.action_space,priority_mode=stored.get('priority_mode','floor'))
            agent.load_state_dict(stored)
        else:
            agent=load_algorithm(testargs.algorithm_name).create_agent(48,env.action_space,testargs)
            for field,key in [('policy','policy_state_dict'),('critic','critic_state_dict'),('critic_target','critic_target_state_dict')]:
                if hasattr(agent,field) and key in checkpoint:getattr(agent,field).load_state_dict(checkpoint[key])
            if 'policy_state_dict' not in checkpoint:raise ValueError('Unsupported checkpoint: policy_state_dict missing')
    finally:env.close()
    switched=apply_ablation(agent,args.ablation)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    atomic_json(Path(args.output)/'ablation.json',{'ablation':args.ablation,'modules_switched':switched,'checkpoint':str(args.checkpoint)})
    result=evaluate(agent,{'env_name':testargs.env_name,'config':env_config},
                    range(args.seeds_start,args.seeds_start+args.episodes),args.output,args.tag,args.save_trajectories)
    print(json.dumps({k:v for k,v in result.items() if k not in ('episode_metrics','seeds','metrics_definition')},allow_nan=False))


if __name__=='__main__':main()
