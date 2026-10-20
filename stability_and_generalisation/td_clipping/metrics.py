"""Frozen evaluation metrics; all durations are integer decision-tick intervals.

lane_events and finish preserve evaluation-v1.1 semantics, including its abort
minimum elapsed span (not an extra consecutive-return confirmation rule).
Target geometry preserves target-lane-proxy-v1 nearest-vehicle semantics.
"""
import math
from statistics import mean

DT = 0.2
DEFINITION = {
    'version': 'lane-change-evaluation-v1', 'lane_event_source': 'evaluation-v1.1',
    'geometry_source': 'target-lane-proxy-v1', 'decision_dt_seconds': DT,
    'lane_width_m': 4, 'attempt_offset_m': .8, 'attempt_confirmation_points': 2,
    'completion_offset_m': .8, 'completion_heading_rad': .1,
    'completion_stable_intervals': 5, 'abort_offset_m': .6,
    'abort_min_elapsed_intervals': 2, 'attempt_timeout_intervals': 40,
    'low_speed_mps': 20, 'low_speed_intervals': 10,
    'braking_drop_mps': 5, 'braking_window_intervals': 25,
    'recovery_gain_mps': 5, 'recovery_window_intervals': 50,
    'round_trip_completion_window_intervals': 25,
    'boundary_oscillation_window_intervals': 20, 'boundary_hysteresis_m': .2,
    'nearby_definition': 'within 50m longitudinal and current or adjacent lane',
    'event_counting': 'nonoverlapping per behavior; phase assigned by start tick <50',
    'low_speed_duration': 'consecutive below-threshold samples span (last-first)*dt',
    'censoring': 'episode-ended active windows retained with right_censored=true',
    'speed_definition': 'post-step forward speed, vx=speed*cos(heading)',
    'distance_definition': 'sum signed post-step x minus pre-step x',
}


def lane_events(records, initial_lane):
    lane=int(initial_lane);pending=0;direction=0;attempt=None;events=[];previous=None
    for row in records:
        tick=int(row['step']);y=float(row['y']);heading=float(row['heading'])
        if previous is not None and tick!=previous['step']+1:raise ValueError('Nonconsecutive trajectory')
        offset=y-lane*4
        if attempt is None:
            sign=1 if offset>0 else -1
            if abs(offset)>=.8 and 0<=lane+sign<4:
                pending=pending+1 if sign==direction else 1;direction=sign
                if pending>=2:
                    target=lane+sign
                    first_cross=tick-1 if previous is not None and abs(previous['y']-target*4)<2 else None
                    first_stable=tick-1 if previous is not None and abs(previous['y']-target*4)<=.8 and abs(previous['heading'])<=.1 else None
                    attempt={'start_tick':tick-1,'confirmation_tick':tick,'from_lane':lane,'target_lane':target,'stable_start_tick':first_stable,'crossing_tick':first_cross}
                    pending=0;direction=0
            else:pending=0;direction=0
        if attempt is not None:
            target=attempt['target_lane'];origin=attempt['from_lane'];elapsed=tick-attempt['start_tick']
            if attempt['crossing_tick'] is None and abs(y-target*4)<2:attempt['crossing_tick']=tick
            stable=abs(y-target*4)<=.8 and abs(heading)<=.1
            if not stable:attempt['stable_start_tick']=None
            elif attempt['stable_start_tick'] is None:attempt['stable_start_tick']=tick
            outcome=None
            if row['crashed'] or not row['on_road']:outcome='unsafe_failure'
            elif attempt['stable_start_tick'] is not None and tick-attempt['stable_start_tick']>=5:outcome='completed'
            elif elapsed>=40:outcome='duration_limit'
            elif elapsed>=2 and abs(y-origin*4)<=.6:outcome='aborted'
            if outcome:
                events.append(finish(attempt,tick,outcome))
                lane=target if outcome=='completed' else max(0,min(3,round(y/4)))
                attempt=None
        previous=row
    if attempt is not None:events.append(finish(attempt,int(records[-1]['step']),'episode_ended'))
    return events


def finish(a,tick,outcome):
    return {'start_tick':a['start_tick'],'confirmation_tick':a['confirmation_tick'],'end_tick':tick,'from_lane':a['from_lane'],'target_lane':a['target_lane'],'start':a['start_tick']/5,'confirmation_time':a['confirmation_tick']/5,'end':tick/5,'duration':(tick-a['start_tick'])/5,'crossed':a['crossing_tick'] is not None,'crossing_tick':a['crossing_tick'],'crossing_time':a['crossing_tick']/5 if a['crossing_tick'] is not None else None,'stable_observed_span_seconds':(tick-a['stable_start_tick'])/5 if a['stable_start_tick'] is not None else 0.,'outcome':outcome}


def geometry(snapshot, target):
    ego = next(v for v in snapshot['vehicles'] if v['ego'])
    values = {'front_gap_m': None, 'rear_gap_m': None, 'front_ttc_s': None, 'rear_ttc_s': None}
    candidates = {'front': [], 'rear': []}
    for other in snapshot['vehicles']:
        if other['ego'] or other['lane'] != target: continue
        dx = other['x'] - ego['x']; side = 'front' if dx >= 0 else 'rear'
        gap = abs(dx) - (other['length'] + ego['length']) / 2
        closing = (ego['forward_speed'] - other['forward_speed']) * (1 if side == 'front' else -1)
        candidates[side].append((abs(dx), gap, max(gap, 0) / closing if closing > 0 else None))
    for side, cars in candidates.items():
        if cars:
            _, gap, ttc = min(cars, key=lambda v: v[0])
            values[side+'_gap_m'] = gap; values[side+'_ttc_s'] = ttc
    return values


def _event(kind, start, end, qualified=True, censored=False, **extra):
    return dict(kind=kind, start_tick=int(start), end_tick=int(end),
                duration_seconds=(end-start)*DT, qualified=bool(qualified),
                right_censored=bool(censored),
                phase='initial_10s' if start < 50 else 'after_10s', **extra)


def behavior_events(records, changes):
    """Records include reset point tick 0; incomplete windows are not successes."""
    if not records: return []
    for left, right in zip(records, records[1:]):
        if int(right['step']) != int(left['step'])+1:
            raise ValueError('Nonconsecutive trajectory')
    out=[]; start=None
    for i,row in enumerate(records):
        if row['speed'] < 20:
            if start is None: start=i
        elif start is not None:
            a,b=records[start]['step'],records[i-1]['step']
            if b-a>=10: out.append(_event('low_speed',a,b))
            start=None
    if start is not None:
        a,b=records[start]['step'],records[-1]['step']
        out.append(_event('low_speed',a,b,b-a>=10,True))

    # Earliest qualifying drop; earliest maximum over preceding 25 intervals.
    # Recovery is >=5 m/s above speed at drop confirmation within 50 intervals.
    # Consume through recovery/deadline so sliding windows cannot double count.
    available=0; i=1
    while i<len(records):
        left=max(available,i-25)
        if left>=i: i+=1; continue
        peak=max(range(left,i), key=lambda j: records[j]['speed'])
        if records[peak]['speed']-records[i]['speed'] < 5:
            i+=1; continue
        last=min(len(records)-1,i+50)
        recovery=next((j for j in range(i+1,last+1)
                       if records[j]['speed']-records[i]['speed']>=5),None)
        end=recovery if recovery is not None else last
        out.append(_event('decelerate_reaccelerate',records[peak]['step'],records[end]['step'],
                          recovery is not None,recovery is None and i+50>len(records)-1,
                          drop_tick=records[i]['step'],drop_speed=records[i]['speed'],
                          peak_speed=records[peak]['speed'],
                          drop_front_gap_m=records[i].get('front_gap_m'),
                          drop_nearby_vehicles=records[i].get('nearby_vehicles')))
        available=end+1; i=end+1

    completed=[e for e in changes if e['outcome']=='completed']; i=0
    while i<len(completed):
        first=completed[i]
        if i+1<len(completed):
            second=completed[i+1]
            if (second['end_tick']-first['end_tick']<=25 and
                    second['from_lane']==first['target_lane'] and
                    second['target_lane']==first['from_lane']):
                out.append(_event('round_trip',first['end_tick'],second['end_tick']))
                i+=2; continue
        # Unfinished return horizon at episode end retained separately.
        if records[-1]['step']-first['end_tick']<25:
            out.append(_event('round_trip',first['end_tick'],records[-1]['step'],False,True))
        i+=1

    for boundary in (2.,6.,10.):
        side=0; crossing=None
        for row in records:
            current=1 if row['y']>=boundary+.2 else (-1 if row['y']<=boundary-.2 else 0)
            if not current: continue
            if side and current!=side:
                tick=row['step']
                if crossing is not None and tick-crossing<=20:
                    stable=any(crossing<=e['end_tick']<=tick for e in completed)
                    if not stable:
                        out.append(_event('boundary_oscillation',crossing,tick,boundary_m=boundary))
                    crossing=None
                else: crossing=tick
            side=current
        if crossing is not None and records[-1]['step']-crossing<20:
            out.append(_event('boundary_oscillation',crossing,records[-1]['step'],False,True,boundary_m=boundary))
    return out


def episode_summary(records, changes, behaviors, seed, reward, terminated, truncated):
    rows=[r for r in records if r['step']>0]
    if not rows: raise ValueError('Empty episode')
    collision=any(r['crashed'] for r in rows); offroad=any(not r['on_road'] for r in rows)
    completed=[e for e in changes if e['outcome']=='completed']
    result={'episode_seed':int(seed),'reward':float(reward),'steps':len(rows),
            'duration_seconds':len(rows)*DT,'collision':int(collision),'offroad':int(offroad),
            'unsafe':int(collision or offroad),'terminated':bool(terminated),'truncated':bool(truncated),
            'speed_sum':sum(r['speed'] for r in rows),'mean_speed':mean(r['speed'] for r in rows),
            'distance_m':sum(r['distance_delta'] for r in rows),'attempts':len(changes),
            'completed':len(completed),'completion_duration_sum':sum(e['duration'] for e in completed),
            'aborted':sum(e['outcome']=='aborted' for e in changes),
            'failed_attempts':len(changes)-len(completed),
            'censored_attempts':sum(e['outcome']=='episode_ended' for e in changes)}
    for key in ('front_gap_m','rear_gap_m','front_ttc_s','rear_ttc_s'):
        vals=[r.get(key) for r in rows if r.get(key) is not None]
        result[key+'_min']=min(vals) if vals else None
    for phase in ('initial_10s','after_10s'):
        phase_rows=[r for r in rows if (r['step']<=50)==(phase=='initial_10s')]
        result[phase+'_duration_seconds']=len(phase_rows)*DT
        result[phase+'_distance_m']=sum(r['distance_delta'] for r in phase_rows)
        for kind in ('low_speed','decelerate_reaccelerate','round_trip','boundary_oscillation'):
            events=[e for e in behaviors if e['kind']==kind and e['phase']==phase]
            result[phase+'_'+kind+'_count']=sum(e['qualified'] for e in events)
            result[phase+'_'+kind+'_censored']=sum(e['right_censored'] for e in events)
    return result


def aggregate(rows):
    if not rows: raise ValueError('No episodes')
    n=len(rows); steps=sum(r['steps'] for r in rows)
    result={'episodes':n,'steps_total':steps,'mean_reward':mean(r['reward'] for r in rows),
            'mean_speed_episode':mean(r['mean_speed'] for r in rows),
            'mean_speed_step':sum(r['speed_sum'] for r in rows)/steps,
            'mean_episode_steps':steps/n,'mean_distance_m':mean(r['distance_m'] for r in rows),
            'duration_seconds_total':sum(r['duration_seconds'] for r in rows),
            'distance_m_total':sum(r['distance_m'] for r in rows)}
    for key in ('collision','offroad','unsafe','truncated'):
        result[key+'_rate']=sum(r[key] for r in rows)/n
    for key in ('attempts','completed','aborted','failed_attempts','censored_attempts'):
        result[key]=sum(r[key] for r in rows)
    result['lane_change_success_rate']=result['completed']/result['attempts'] if result['attempts'] else None
    result['mean_lane_change_duration']=sum(r['completion_duration_sum'] for r in rows)/result['completed'] if result['completed'] else None
    result['no_attempt_episode_rate']=sum(r['attempts']==0 for r in rows)/n
    for phase in ('initial_10s','after_10s'):
        duration=sum(r[phase+'_duration_seconds'] for r in rows)
        distance=sum(r[phase+'_distance_m'] for r in rows)
        result[phase+'_duration_seconds']=duration;result[phase+'_distance_m']=distance
        for kind in ('low_speed','decelerate_reaccelerate','round_trip','boundary_oscillation'):
            stem=phase+'_'+kind
            count=sum(r[stem+'_count'] for r in rows)
            result[stem+'_count']=count
            result[stem+'_censored']=sum(r[stem+'_censored'] for r in rows)
            result[stem+'_per_episode']=count/n
            result[stem+'_per_km']=count/(distance/1000) if distance>0 else None
            result[stem+'_per_hour']=count/(duration/3600) if duration>0 else None
    result['avg_reward']=result['mean_reward']; result['crash_rate']=result['collision_rate']
    result['avg_speed']=result['mean_speed_episode'];result['mean_speed']=result['mean_speed_step']
    return result
