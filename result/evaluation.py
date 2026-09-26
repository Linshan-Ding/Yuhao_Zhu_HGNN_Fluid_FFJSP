"""Greedy fixed-case evaluation, raw traces and shared-state serial timing."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from agent.observation import observe
from agent.rules import RulePolicy
from agent.model import Policy
from configs.experiment import environment_config, identity, uses_demo, ALL_RULES
from data.benchmark import fixtures, prepare
from environment.public import SchedulingEnv
from result.recording import Recorder, PHASES, phase_at, split_time, verify
from result.storage import atomic_json, digest, object_hash, write_csv, atomic_torch_save, state_hash
from result.provenance import source_hash, evaluation_hash

PANEL_RULE='SPT'  # rule whose test-split trajectories supply the shared-state timing panel


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
        last=float(inst.arrival_times[-1]);startup=c['recording']['phase_startup_fraction']
        while not env.done:
            phase=phase_at(env.now,last,startup);start=time.perf_counter();obs=observe(env);build=time.perf_counter()-start
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
            now=env.now;env.step(obs.actions.actions[a]);env_seconds=env.last_step_seconds
            wait=obs.candidate[a,0]<0;steps+=1;waits+=int(wait)
            phases[phase]['steps']+=1;phases[phase]['waits']+=int(wait)
            for ph,dt in split_time(now,env.now,last,startup): phases[ph]['capacity_time']+=dt*inst.machine_count
            for event in env.recorded_events():
                if event['kind']=='wait':
                    for ph,dt in split_time(event['time'],event['end'],last,startup): phases[ph]['held_machine_time']+=dt*len(event['machines'])
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
    c=state['cfg']
    return state_hash(state['model'],json.dumps(dict(method=c['method'],network=c['network'],classic_width=c['classic']['width']),sort_keys=True).encode())


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


def milestone_points(c,core):
    """Intermediate core checkpoints that are evaluated on the main split (fixed-budget learning curves)."""
    start=c['experiment']['evaluate_milestones_from']
    return sorted({x for s in core for x in s.config['training']['milestones'] if start<=x<s.budget})


def variant_name(spec):
    return spec.name.rsplit('_s',1)[0] if spec.group=='sensitivity' else spec.method


def evaluation_plan(c,specs):
    """Every planned evaluation table: name, fixed split, (job, checkpoint file) pairs and rule policies.
    evaluate_matrix runs this plan and statistics.validate checks the persisted tables against the same plan."""
    e=c['experiment'];core=[s for s in specs if s.group!='sensitivity'];final=lambda s:f'checkpoint_{s.budget}.pt'
    plan=[dict(name=split,split=split,models=[(s,final(s)) for s in core],rules=ALL_RULES,panel=PANEL_RULE)
          for split in ('main','small','long_stream','large_shop')]
    plan+=[dict(name=f'main_{p}',split='main',models=[(s,f'checkpoint_{p}.pt') for s in core],rules=(),panel=None) for p in milestone_points(c,core)]
    plan.append(dict(name='main_best',split='main',models=[(s,'checkpoint_best.pt') for s in core],rules=(),panel=None))
    plan.append(dict(name='initialization',split='validation',models=[(s,'checkpoint_initialization.pt') for s in core if uses_demo(s.method)],
                     rules=(c['teacher'],),panel=None))
    changed=[(s,final(s)) for s in specs if s.group=='sensitivity']
    defaults=[(s,f"checkpoint_{e['sensitivity_budget']}.pt") for s in core if s.method=='full' and s.seed in e['sensitivity_seeds']]
    plan.append(dict(name='sensitivity',split='sensitivity',models=changed+defaults,rules=(),panel=None))
    return plan


def evaluate_matrix(root,data,c,specs):
    root=Path(root);manifest=prepare(c,data);case_index={r['instance_id']:r for r in manifest['cases']};outputs={};costs=[]
    for item in evaluation_plan(c,specs):
        rows=[];instances=fixtures(c,data,item['split'])
        for spec,filename in item['models']:
            path=root/'runs'/spec.name/filename
            policy,state=load_policy(path,spec);cp_hash=digest(path);model_id=policy_identity(state)
            for inst in instances:
                start=time.perf_counter();entry=cached_case(root,inst,policy,model_id,c,case_index[inst.instance_id]['sha256'])
                rows.append(dict(variant=variant_name(spec),run=spec.name,seed=spec.seed,budget=state['steps'],checkpoint_sha256=cp_hash,
                    policy_sha256=model_id,evaluation_key=object_hash(entry['identity'])[:24],**entry['row']))
                costs.append(dict(run=spec.name,split=item['name'],lookup_wall_seconds=time.perf_counter()-start))
        for rule in item['rules']:
            for inst in instances:
                entry=cached_case(root,inst,RulePolicy(rule),'rule:'+rule,c,case_index[inst.instance_id]['sha256'],panel=rule==item['panel'])
                rows.append(dict(variant=rule,run=rule,seed=0,budget=0,checkpoint_sha256='',evaluation_key=object_hash(entry['identity'])[:24],**entry['row']))
        write_csv(root/f"{item['name']}.csv",rows)
        outputs[item['name']]=dict(file=f"{item['name']}.csv",sha256=digest(root/f"{item['name']}.csv"),rows=len(rows))
    write_csv(root/'evaluation_lookups.csv',costs)
    atomic_json(root/'evaluation_manifest.json',dict(outputs=outputs,data=digest(Path(data)/'manifest.json'),evaluation=evaluation_hash()))
    return outputs


def latency(root,c,specs):
    root=Path(root);target=root/'latency.csv';panels=[]
    for p in sorted((root/'evaluations').glob('*/complete.json')):
        r=json.loads(p.read_text())
        if r['identity']['evaluation']==evaluation_hash() and r['identity']['model']=='rule:'+PANEL_RULE and r['panel']:
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
    if not panel: raise ValueError(f'Missing {PANEL_RULE} timing panel')
    models=[]
    for s in specs:
        if s.group!='sensitivity':models.append((s.name,s.seed,load_policy(root/'runs'/s.name/f'checkpoint_{s.budget}.pt',s)[0]))
    models.extend((r,0,RulePolicy(r)) for r in ALL_RULES);rows=[];threads=c['runtime']['threads'];torch.set_num_threads(threads)
    for name,seed,policy in models:
        for state in panel:
            o=state['observation']
            for _ in range(c['experiment']['latency_warmup']): infer(policy,o)
            for repeat in range(c['experiment']['latency_repeats']):
                start=time.perf_counter();infer(policy,o);seconds=time.perf_counter()-start
                rows.append(dict(run=name,seed=seed,instance_id=state['instance_id'],phase=state['phase'],repeat=repeat,
                                 inference_seconds=seconds,candidates=len(o.candidate),orders=len(o.order),operations=len(o.operation),machines=len(o.machine)))
    write_csv(target,rows);atomic_json(marker,dict(identity=identity_value,sha256=digest(target),rows=len(rows),threads=threads))
