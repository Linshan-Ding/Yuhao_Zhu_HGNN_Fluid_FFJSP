"""Resumable training with conservative crash accounting and immutable source archives."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import time
import zipfile
import numpy as np
import torch
from configs.experiment import ROOT, identity, environment_config, uses_demo, uses_scenario
from data.online import sample
from data.benchmark import fixtures
from environment.public import SchedulingEnv
from agent.observation import observe
from agent.rules import RulePolicy
from agent.model import Policy
from agent.preference import ScenarioEvaluator,PreferenceReplay
from agent.learning import on_policy,preference_update,q_update
from result.storage import atomic_torch_save
from result.provenance import atomic_json,hash_files

from result.provenance import training_files as sources, source_hash, snapshot
from result.recording import Recorder

@dataclass
class DemonstrationDataset:
    observations: list
    actions: list
    steps: int=0
    teacher: str='SPT'

def evaluate(policy, instances, c, recorder=None, kind='evaluation'):
    from result.evaluation import evaluate_instances
    return evaluate_instances(policy, instances, c, recorder=recorder, kind=kind)


def _train(c,run,seed,budget,data,holders):
    run=Path(run);run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    policy=Policy(c);optimizer=torch.optim.Adam(policy.parameters(),lr=c['training']['learning_rate'])
    demo_opt=torch.optim.Adam(policy.parameters(),lr=c['demonstration']['learning_rate'])
    target=deepcopy(policy).eval();teacher=deepcopy(policy).eval()
    irng=np.random.default_rng(np.random.SeedSequence([seed,11]));arng=np.random.default_rng(np.random.SeedSequence([seed,22]));ur=np.random.default_rng(np.random.SeedSequence([seed,33]))
    scenario=ScenarioEvaluator(c,np.random.SeedSequence([seed,44]));pref=PreferenceReplay(c['scenario']['replay_size'],c['scenario']['versions'])
    replay=deque(maxlen=c['classic']['replay_size']);demo=DemonstrationDataset([],[],0,c['teacher'])
    demo_env=None;envs=[None]*c['training']['environments'];next_obs=[None]*len(envs)
    real=branch=lost=epoch=demo_epoch=q_steps=eval_steps=0;elapsed_offset=0.;next_eval=0;best=-1.
    labels=failed=0;next_scenario=0;teacher_version=0;next_teacher=c['scenario']['teacher_interval'];demo_finished=not uses_demo(c['method'])
    cp=run/'checkpoint_last.pt';code=source_hash();fp=identity(c);ledger=run/'budget.json';log=run/'log.jsonl'
    history=[];monitor=fixtures(c,data,'monitor');began=time.perf_counter();scenario_seconds=0.
    recorder_state=None
    if cp.exists():
        s=torch.load(cp,weights_only=False,map_location='cpu')
        if (s['source_hash'],s['identity'],s['seed'])!=(code,fp,seed):raise ValueError('incompatible training resume')
        policy.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);demo_opt.load_state_dict(s['demo_optimizer'])
        target.load_state_dict(s['target']);teacher.load_state_dict(s['teacher_model'])
        real,branch,lost,epoch,demo_epoch,q_steps,eval_steps=[s[k] for k in ('real_steps','scenario_steps','lost_upper_bound','epoch','demo_epoch','q_steps','evaluation_steps')]
        irng.bit_generator.state=s['instance_rng'];arng.bit_generator.state=s['action_rng'];ur.bit_generator.state=s['update_rng'];torch.set_rng_state(s['torch_rng'])
        demo=s['demonstrations'];demo_finished=s['demo_finished'];demo_env=SchedulingEnv.from_state_dict(s['demo_environment']) if s['demo_environment'] else None
        envs=[SchedulingEnv.from_state_dict(e) if e else None for e in s['environments']];next_obs=s['next_observations']
        scenario.load_state_dict(s['scenario_state']);pref=s['preferences'];replay=s['replay'];elapsed_offset=s['elapsed_seconds']
        best=s['best'];next_eval=s['next_eval'];labels=s['labels'];failed=s['failed_labels'];next_scenario=s['next_scenario']
        teacher_version=s['teacher_version'];next_teacher=s['next_teacher'];history=s['history'];scenario_seconds=s['scenario_seconds']
        recorder_state=s['recorder']
        reserved=json.loads(ledger.read_text())['charged_upper_bound'] if ledger.exists() else s['steps']
        lost+=max(0,reserved-s['steps'])
        if log.exists():
            lines=log.read_text(encoding='utf-8').splitlines();keep=[x for x in lines if json.loads(x)['epoch']<=epoch]
            if len(keep)!=len(lines):
                (run/f'log_recovery_{time.time_ns()}.jsonl').write_text('\n'.join(lines)+'\n',encoding='utf-8');log.write_text('\n'.join(keep)+'\n',encoding='utf-8')
    else:
        import yaml
        (run/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        archive=snapshot(run,c)
        atomic_json(run/'manifest.json',dict(format='run-1',source_hash=code,identity=fp,seed=seed,config=c,
            archive_sha256=archive,torch=torch.__version__,numpy=np.__version__))
    recorder=Recorder(run/'raw',c,recorder_state);holders.append(recorder)
    scenario.recorder=recorder
    if demo_env is not None: demo_env.attach_recorder(recorder,demo_env._record_id,'demonstration')
    for e in envs:
        if e is not None: e.attach_recorder(recorder,e._record_id,'real')
    def make_env(kind):
        return SchedulingEnv(sample(c,irng),environment_config(c)).attach_recorder(recorder,kind=kind)

    def total():return real+branch+demo.steps+lost
    def reserve(limit):atomic_json(ledger,dict(charged_upper_bound=limit,committed=total(),limit=budget))
    def state():
        return dict(schema_version='schedule-data-1',cfg=c,model=policy.state_dict(),source_hash=code,identity=fp,seed=seed,steps=total(),
            optimizer=optimizer.state_dict(),demo_optimizer=demo_opt.state_dict(),target=target.state_dict(),teacher_model=teacher.state_dict(),
            real_steps=real,scenario_steps=branch,lost_upper_bound=lost,epoch=epoch,demo_epoch=demo_epoch,q_steps=q_steps,evaluation_steps=eval_steps,
            instance_rng=irng.bit_generator.state,action_rng=arng.bit_generator.state,update_rng=ur.bit_generator.state,torch_rng=torch.get_rng_state(),
            demonstrations=demo,demo_finished=demo_finished,demo_environment=demo_env.state_dict() if demo_env else None,
            environments=[e.state_dict() if e else None for e in envs],next_observations=next_obs,scenario_state=scenario.state_dict(),
            preferences=pref,replay=replay,elapsed_seconds=elapsed_offset+time.perf_counter()-began,best=best,next_eval=next_eval,labels=labels,
            failed_labels=failed,next_scenario=next_scenario,teacher_version=teacher_version,next_teacher=next_teacher,history=history,scenario_seconds=scenario_seconds,recorder=recorder.snapshot())
    def save():
        import shutil
        current=state()
        if cp.exists(): shutil.copy2(cp,run/'checkpoint_previous.pt')
        atomic_torch_save(current,cp);recorder.commit(current['recorder']);reserve(total())
    def lightweight(path,**extra):
        atomic_torch_save(dict(schema_version='schedule-data-1',cfg=c,model=policy.state_dict(),seed=seed,steps=total(),source_hash=code,identity=fp,**extra),path)
    if total()>=budget and demo_finished:
        if not (run/f'checkpoint_{budget}.pt').exists():lightweight(run/f'checkpoint_{budget}.pt')
        return dict(status='complete',steps=total(),real=real,scenario=branch,demonstration=demo.steps,lost=lost)
    if not cp.exists():save()
    if not demo_finished:
        desired=c['demonstration']['steps'];policy.residual_enabled=False
        while demo.steps<desired and total()<budget:
            end=min(desired,demo.steps+256);reserve(min(budget,total()+end-demo.steps))
            while demo.steps<end and total()<budget:
                while demo_env is None or demo_env.done:demo_env=make_env('demonstration')
                o=observe(demo_env);a=RulePolicy(c['teacher']).act(o)[0]
                demo.observations.append(o);demo.actions.append(a);demo_env.step(o.actions.actions[a]);demo.steps+=1
            save()
        if demo.steps<desired:return dict(status='partial',steps=total())
        while demo_epoch<c['demonstration']['epochs']:
            losses=[]
            for ids in np.array_split(ur.permutation(len(demo.actions)),max(1,int(np.ceil(len(demo.actions)/c['demonstration']['minibatch'])))):
                b=policy.batch([demo.observations[int(i)] for i in ids]);out=policy(b)
                idx=b.offsets[:-1]+torch.tensor([demo.actions[int(i)] for i in ids]);loss=-out.logp[idx].mean()
                demo_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);demo_opt.step();losses.append(float(loss.detach()))
            demo_epoch+=1;save();print(f'demo epoch={demo_epoch} loss={np.mean(losses):.4f}',flush=True)
        policy.residual_enabled=True;demo_finished=True;teacher.load_state_dict(policy.state_dict());target.load_state_dict(policy.state_dict())
        values=evaluate(policy,monitor,c,recorder,'initialization');eval_steps+=sum(v['steps'] for v in values)
        atomic_json(run/'initialization_evaluation.json',dict(steps=demo.steps,rows=values));lightweight(run/'checkpoint_initialization.pt')
        # Demonstrations remain a separately archived artifact, not continually replayed by PPO.
        atomic_torch_save(demo,run/'demonstrations.pt');demo.observations=[];demo.actions=[];demo_env=None;save()
    policy.residual_enabled=True
    while total()<budget:
        milestone=min([x for x in c['training']['milestones'] if x>total()]+[budget]);limit=min(budget,milestone)
        rollout=c['classic']['a2c_rollout'] if c['method']=='a2c' else c['training']['rollout_steps']
        end=min(limit,total()+rollout+(max(4096,2*scenario.last_cost) if uses_scenario(c['method']) else 0))
        reserve(end);records=[];metrics={};used=total();is_q=c['method'] in ('dqn','ddqn')
        # Reservation covers every simulator call, including unfinished scenario branches.
        while len(records)<rollout and total()<end:
            count=min(len(envs),end-total(),rollout-len(records));obs=[]
            for i in range(count):
                while envs[i] is None or envs[i].done:
                    envs[i]=make_env('real');next_obs[i]=None
                obs.append(next_obs[i] if next_obs[i] is not None else observe(envs[i]));next_obs[i]=None
            epsilon=max(c['classic']['epsilon_final'],1-(1-c['classic']['epsilon_final'])*real/c['classic']['epsilon_steps'])
            choices=policy.act_many(obs,arng,epsilon)
            for i,(o,(a,lp,val)) in enumerate(zip(obs,choices)):
                if total()>=end:break
                env=envs[i]
                if uses_scenario(c['method']) and real>=next_teacher:
                    teacher.load_state_dict(policy.state_dict());teacher_version+=1;next_teacher=real+c['scenario']['teacher_interval'];pref.expire(teacher_version)
                fraction=c['scenario']['budget_fraction'];allowance=int((real+demo.steps)*fraction/(1-fraction))-branch
                minimum=max(64,int(scenario.last_cost))
                if uses_scenario(c['method']) and real>=c['scenario']['warmup_real_steps'] and real>=next_scenario and allowance>=minimum and end-total()>minimum:
                    label=scenario.evaluate(env.public_state(),o,teacher,min(allowance,end-total()-1),c['teacher'],teacher_version,candidate_policy=policy)
                    next_scenario=real+c['scenario']['interval']
                    if label is not None:
                        branch+=label.steps;scenario_seconds+=label.seconds;pref.add(o,label);labels+=int(label.complete);failed+=int(not label.complete)
                        with (run/'scenario_labels.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch+1,real_steps=real,**label.record()))+'\n')
                recorder.annotate(env,action_index=a,log_probability=lp,value=val,candidates=len(o.candidate),
                    orders_visible=len(o.order),operations_visible=len(o.operation),machines=len(o.machine))
                reward,done,info=env.step(o.actions.actions[a]);real+=1
                rec=dict(obs=o,action=a,logp=lp,value=val,reward=reward,stream=i,terminated=info['terminated'],truncated=info['truncated'],boundary=done,scenario=None)
                if env.truncated:
                    query=SchedulingEnv.from_state_dict(env.state_dict());query.done=False;query.truncated=False
                    no=observe(query);rec['bootstrap']=policy.act(no)[2]
                else:no=observe(env) if is_q and not done else None
                records.append(rec)
                if is_q:
                    replay.append((o,a,reward,no));next_obs[i]=no
                    if real>=c['classic']['start'] and len(replay)>=c['classic']['batch'] and real%c['classic']['train_every']==0:
                        update_start=time.perf_counter();metrics=q_update(policy,target,optimizer,replay,c,ur);q_steps+=1
                        recorder.emit('updates',dict(kind='q',real_steps=real,number=q_steps,seconds=time.perf_counter()-update_start,**metrics))
                        if q_steps%c['classic']['target_every']==0:target.load_state_dict(policy.state_dict())
        if not records:raise RuntimeError('training made no progress')
        if not is_q:
            boot={i:0. for i in range(len(envs))};active=[i for i,e in enumerate(envs) if e and not e.done]
            for i,x in zip(active,policy.act_many([observe(envs[i]) for i in active])):boot[i]=x[2]
            update_start=time.perf_counter();metrics=on_policy(policy,optimizer,records,boot,c,ur)
            metrics['on_policy_seconds']=time.perf_counter()-update_start
            if 'wait_probability' in metrics: metrics['wait_probability_before_preference']=metrics.pop('wait_probability')
            if uses_scenario(c['method']):
                update_start=time.perf_counter();metrics.update(preference_update(policy,optimizer,pref,c,ur,teacher_version));metrics['preference_seconds']=time.perf_counter()-update_start
            recorder.emit('updates',dict(kind='on_policy',real_steps=real,epoch=epoch+1,**metrics))
        epoch+=1;eta=None
        if total()>=next_eval or total() in c['training']['milestones'] or total()>=budget:
            values=evaluate(policy,monitor,c,recorder,'monitor');eval_steps+=sum(v['steps'] for v in values);eta=float(np.mean([v['eta'] for v in values]));next_eval=total()+c['training']['evaluate_every']
            atomic_json(run/f'monitor_{total()}.json',dict(rows=values))
            if eta>best:best=eta;lightweight(run/'checkpoint_best.pt',eta_validation=eta)
        row=dict(epoch=epoch,total_steps=total(),real_steps=real,scenario_steps=branch,demonstration_steps=demo.steps,lost_upper_bound=lost,
            eta_validation=eta,labels=labels,failed_labels=failed,teacher_version=teacher_version,evaluation_steps=eval_steps,
            elapsed_seconds=elapsed_offset+time.perf_counter()-began,scenario_seconds=scenario_seconds,recording_seconds=recorder.io_seconds,**metrics)
        history.append(row)
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if total() in c['training']['milestones'] or total()>=budget:lightweight(run/f'checkpoint_{total()}.pt')
        if not is_q or epoch%8==0 or eta is not None or total()>=budget:save()
        if eta is not None:print(f"[{c['method']} s{seed}] {total()}/{budget} eta={eta:.4f} labels={labels} real={real} branch={branch}",flush=True)
    for e in envs:
        if e is not None and not e.done: recorder.outcomes(e,'training_budget_end')
    save()
    return dict(status='complete',steps=total(),real=real,scenario=branch,demonstration=demo.steps,lost=lost)


def train(c,run,seed,budget,data):
    holders=[]
    try:
        return _train(c,run,seed,budget,data,holders)
    except BaseException:
        for recorder in holders:
            try:atomic_json(Path(run)/f'failed_raw_{time.time_ns()}.json',recorder.snapshot())
            except OSError:print('Raw flush failed; durable checkpoint and charged reservation remain authoritative.',flush=True)
        raise
