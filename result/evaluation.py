"""Greedy fixed-case evaluation, raw traces and shared-state serial timing."""
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import torch
from agent.observation import observe
from agent.rules import RulePolicy
from agent.model import Policy
from configs.experiment import environment_config, identity, ALL_RULES
from data.benchmark import fixtures, prepare
from environment.public import SchedulingEnv
from result.recording import Recorder, PHASES, phase_at, split_time, verify
from result.storage import atomic_json, digest, object_hash, write_csv, read_csv, atomic_torch_save
from result.provenance import source_hash, evaluation_hash


@torch.no_grad()
def infer(policy, obs, details=False):
    if not isinstance(policy, Policy): return policy.act(obs)[0], None
    out=policy(policy.batch([obs]),include_impact=False,details=details)
    choice=int((out.logits if policy.method in ('dqn','ddqn') else out.logp).argmax())
    return choice,out


@torch.no_grad()
def evaluate_instances(policy, instances, c, recorder=None, kind='evaluation', panel=None):
    rows=[]
    for inst in instances:
        begun=time.perf_counter();io_before=recorder.io_seconds if recorder else 0.
        env=SchedulingEnv(inst,environment_config(c))
        if recorder: env.attach_recorder(recorder,kind=kind)
        phases={p:dict(steps=0,waits=0,held_machine_time=0.,capacity_time=0.) for p in PHASES}
        timings=[];steps=waits=0;seen=set();build_total=infer_total=environment_total=0.
        last=float(inst.arrival_times[-1]);phase_snapshots={}
        while not env.done:
            phase=phase_at(env.now,last);start=time.perf_counter();obs=observe(env);build=time.perf_counter()-start
            start=time.perf_counter();a,out=infer(policy,obs,details=recorder is not None);inference=time.perf_counter()-start
            first=phase not in seen;seen.add(phase)
            if panel is not None and first: panel.append(dict(instance_id=inst.instance_id,phase=phase,observation=obs))
            scores=None
            if recorder:
                candidates=[asdict(x) if asdict(x) else {'wait':True} for x in obs.actions.actions]
                if out is not None:
                    scores=dict(scores=out.logits.tolist(),base_scores=out.base_logits.tolist(),correction=out.correction.tolist(),
                                gates=out.gates.tolist() if out.gates is not None else None,
                                probabilities=None if policy.method in ('dqn','ddqn') else out.logp.exp().tolist(),
                                value=None if policy.method in ('dqn','ddqn') else float(out.value[0]))
                recorder.annotate(env,action_index=a,candidates=candidates,model=scores,orders_visible=len(obs.order),
                                  operations_visible=len(obs.operation),machines=len(obs.machine),edges=len(obs.eligibility),
                                  build_seconds=build,inference_seconds=inference)
                if first:
                    key=recorder.artifact(dict(public_state=env.public_state(),observation=obs,action=a,model=scores,
                         representations=out.representations.detach() if out is not None else None),'mechanisms')
                    recorder.emit('mechanisms',dict(episode_id=env._record_id,instance_id=inst.instance_id,phase=phase,artifact=key))
                    phase_snapshots[phase]=key
            now=env.now;held=env.stats.held_time;start=time.perf_counter()
            env.step(obs.actions.actions[a]);elapsed=time.perf_counter()-start
            env_seconds=env.last_step_seconds
            wait=obs.candidate[a,0]<0;steps+=1;waits+=int(wait)
            phases[phase]['steps']+=1;phases[phase]['waits']+=int(wait)
            for ph,dt in split_time(now,env.now,last): phases[ph]['capacity_time']+=dt*inst.machine_count
            for event in env.recorded_events():
                if event['kind']=='wait':
                    for ph,dt in split_time(event['time'],event['end'],last): phases[ph]['held_machine_time']+=dt*len(event['machines'])
            timings.append(inference+build);build_total+=build;infer_total+=inference;environment_total+=env_seconds
        if env.truncated: raise RuntimeError(f'Evaluation truncated: {inst.instance_id}')
        if np.any(env.order_outcome<0): raise AssertionError('Unresolved evaluation orders')
        if recorder: recorder.flush()
        wall=time.perf_counter()-begun
        row=dict(instance_id=inst.instance_id,eta=env.eta,iota=inst.meta['iota'],rho=inst.meta['rho_sys'],
                 orders=inst.order_count,products=inst.product_count,stages=inst.stage_count,machines=inst.machine_count,
                 due_factor=inst.meta['due_factor'],parameter_hash=inst.meta.get('parameter_hash'),
                 steps=steps,wait_fraction=waits/max(steps,1),held_share=env.held_share,
                 completed=env.n_completed,failed=env.n_discarded,decision_ms=1000*infer_total/max(steps,1),
                 observation_ms=1000*build_total/max(steps,1),decision_p50_ms=1000*float(np.median(timings)) if timings else 0.,
                 decision_p95_ms=1000*float(np.quantile(timings,.95)) if timings else 0.,
                 build_seconds=build_total,inference_seconds=infer_total,environment_seconds=environment_total,
                 recording_seconds=recorder.io_seconds-io_before if recorder else 0.,wall_seconds=wall,
                 simulation_end=env.now,episode_id=getattr(env,'_record_id',None),
                 **{f'{p}_{k}':v for p,values in phases.items() for k,v in values.items()})
        rows.append(row)
    return rows


def load_policy(path, spec):
    state=torch.load(path,weights_only=False,map_location='cpu')
    if state['seed']!=spec.seed or state['identity']!=identity(spec.config) or state['source_hash']!=source_hash():
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    p=Policy(state['cfg']).eval();p.load_state_dict(state['model']);return p,state


def policy_identity(state):
    """The deployed function, independent of checkpoint filename or validation metadata."""
    c=state['cfg'];h=hashlib.sha256(json.dumps(dict(method=c['method'],network=c['network'],
         classic_width=c['classic']['width']),sort_keys=True).encode())
    for name,value in sorted(state['model'].items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def cached_case(root, inst, policy, model_id, c, instance_sha, panel=False):
    identity_value=dict(model=model_id,instance=instance_sha,evaluation=evaluation_hash())
    # A short directory key avoids MAX_PATH on Windows; the full identity is still checked.
    key=object_hash(identity_value)[:24]; cache=Path(root)/'evaluations'/key;complete=cache/'complete.json'
    if complete.exists():
        record=json.loads(complete.read_text(encoding='utf-8'))
        if record['identity']!=identity_value: raise ValueError('Evaluation cache mismatch')
        if digest(cache/record['raw_manifest'])!=record['raw_sha256']: raise ValueError('Raw manifest changed')
        verify(cache/record['raw_directory'])
        if record['panel'] and digest(cache/record['panel'])!=record['panel_sha256']: raise ValueError('Timing panel changed')
        if panel and not record['panel']: raise ValueError('Evaluation lacks requested timing panel')
        return record
    attempt=cache/f'attempt_{time.time_ns()}'; recorder=Recorder(attempt/'raw',c);states=[]
    try: row=evaluate_instances(policy,[inst],c,recorder,panel=states if panel else None)[0]
    finally: recorder.commit(recorder.snapshot())
    if states: atomic_torch_save(states,attempt/'panel.pt')
    record=dict(identity=identity_value,row=row,raw_directory=(attempt/'raw').relative_to(cache).as_posix(),
                raw_manifest=(attempt/'raw/manifest.json').relative_to(cache).as_posix(),raw_sha256=digest(attempt/'raw/manifest.json'),
                panel=(attempt/'panel.pt').relative_to(cache).as_posix() if states else None,
                panel_sha256=digest(attempt/'panel.pt') if states else None)
    atomic_json(complete,record)
    return record


def evaluate_matrix(root,data,c,specs):
    root=Path(root); manifest=prepare(c,data);case_index={r['instance_id']:r for r in manifest['cases']}
    core=[s for s in specs if s.group!='sensitivity'];outputs={};costs=[]
    def evaluate_set(name,split,selected,checkpoint,include_rules=True):
        rows=[];instances=fixtures(c,data,split)
        for spec in selected:
            path=root/'runs'/spec.name/(checkpoint(spec) if callable(checkpoint) else checkpoint)
            policy,state=load_policy(path,spec);cp_hash=digest(path);model_id=policy_identity(state)
            for inst in instances:
                start=time.perf_counter();item=cached_case(root,inst,policy,model_id,c,case_index[inst.instance_id]['sha256'])
                rows.append(dict(variant=spec.name.rsplit('_s',1)[0] if spec.group=='sensitivity' else spec.method,
                    run=spec.name,seed=spec.seed,budget=state['steps'],checkpoint_sha256=cp_hash,policy_sha256=model_id,
                    evaluation_key=object_hash(item['identity'])[:24],**item['row']))
                costs.append(dict(run=spec.name,split=name,lookup_wall_seconds=time.perf_counter()-start))
        if include_rules:
            for rule in ALL_RULES:
                for inst in instances:
                    item=cached_case(root,inst,RulePolicy(rule),'rule:'+rule,c,case_index[inst.instance_id]['sha256'],panel=rule=='SPT')
                    rows.append(dict(variant=rule,run=rule,seed=0,budget=0,checkpoint_sha256='',evaluation_key=object_hash(item['identity'])[:24],**item['row']))
        write_csv(root/f'{name}.csv',rows);outputs[name]=dict(file=f'{name}.csv',sha256=digest(root/f'{name}.csv'),rows=len(rows))
    for split in ('main','small','long_stream','large_shop'):
        evaluate_set(split,split,core,lambda s:f'checkpoint_{s.budget}.pt')
    points=sorted(set(x for s in core for x in s.config['training']['milestones'] if x<s.budget and x>= (64 if c['purpose']=='engineering-smoke' else 100000)))
    for point in points: evaluate_set(f'main_{point}','main',core,f'checkpoint_{point}.pt',False)
    evaluate_set('main_best','main',core,'checkpoint_best.pt',False)
    from configs.experiment import uses_demo
    evaluate_set('initialization','validation',[s for s in core if uses_demo(s.method)],'checkpoint_initialization.pt',False)
    teacher=[]
    for inst in fixtures(c,data,'validation'):
        item=cached_case(root,inst,RulePolicy(c['teacher']),'rule:'+c['teacher'],c,case_index[inst.instance_id]['sha256'])
        teacher.append(dict(variant=c['teacher'],run=c['teacher'],seed=0,budget=0,checkpoint_sha256='',
                            evaluation_key=object_hash(item['identity'])[:24],**item['row']))
    initial=read_csv(root/'initialization.csv')+teacher;write_csv(root/'initialization.csv',initial)
    outputs['initialization']=dict(file='initialization.csv',sha256=digest(root/'initialization.csv'),rows=len(initial))
    sens=[s for s in specs if s.group=='sensitivity']
    evaluate_set('sensitivity_changes','sensitivity',sens,lambda s:f'checkpoint_{s.budget}.pt',False)
    defaults=[s for s in core if s.method=='full' and s.seed in c['experiment']['sensitivity_seeds']]
    evaluate_set('sensitivity_default','sensitivity',defaults,f"checkpoint_{c['experiment']['sensitivity_budget']}.pt",False)
    sensitive=read_csv(root/'sensitivity_changes.csv')+read_csv(root/'sensitivity_default.csv');write_csv(root/'sensitivity.csv',sensitive)
    outputs['sensitivity']=dict(file='sensitivity.csv',sha256=digest(root/'sensitivity.csv'),rows=len(sensitive))
    write_csv(root/'evaluation_lookups.csv',costs)
    atomic_json(root/'evaluation_manifest.json',dict(outputs=outputs,data=digest(Path(data)/'manifest.json'),evaluation=evaluation_hash()))
    return outputs


def latency(root,c,specs):
    root=Path(root);target=root/'latency.csv';panels=[]
    for p in sorted((root/'evaluations').glob('*/complete.json')):
        r=json.loads(p.read_text())
        if r['identity']['evaluation']==evaluation_hash() and r['identity']['model']=='rule:SPT' and r['panel']:
            if digest(p.parent/r['panel'])!=r['panel_sha256']:raise ValueError('Timing panel changed')
            panels.append((p.parent/r['panel'],r['panel_sha256']))
    identity_value=dict(evaluation=evaluation_hash(),panels=[sha for _,sha in panels],
        checkpoints={s.name:digest(root/'runs'/s.name/f'checkpoint_{s.budget}.pt') for s in specs if s.group!='sensitivity'},
        warmup=c['experiment']['latency_warmup'],repeats=c['experiment']['latency_repeats'])
    marker=root/'latency_manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text());
        if old['identity']!=identity_value or old['sha256']!=digest(target): raise ValueError('Latency identity mismatch')
        return
    panel=[]
    for path,_ in panels:panel.extend(torch.load(path,weights_only=False))
    if not panel: raise ValueError('Missing SPT timing panel')
    models=[]
    for s in specs:
        if s.group!='sensitivity':models.append((s.name,s.seed,load_policy(root/'runs'/s.name/f'checkpoint_{s.budget}.pt',s)[0]))
    models.extend((r,0,RulePolicy(r)) for r in ALL_RULES);rows=[];torch.set_num_threads(1)
    for name,seed,policy in models:
        for state in panel:
            o=state['observation']
            for _ in range(c['experiment']['latency_warmup']): infer(policy,o)
            for repeat in range(c['experiment']['latency_repeats']):
                start=time.perf_counter();infer(policy,o);seconds=time.perf_counter()-start
                rows.append(dict(run=name,seed=seed,instance_id=state['instance_id'],phase=state['phase'],repeat=repeat,
                                 inference_seconds=seconds,candidates=len(o.candidate),orders=len(o.order),operations=len(o.operation),machines=len(o.machine)))
    write_csv(target,rows);atomic_json(marker,dict(identity=identity_value,sha256=digest(target),rows=len(rows),threads=1))
