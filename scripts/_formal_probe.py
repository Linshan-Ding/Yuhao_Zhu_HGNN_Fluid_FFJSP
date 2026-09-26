"""Internal subprocess: actual formal tensor sizes and full-capacity serialized buffers."""
import _bootstrap
from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import gc
import json
import pickle
import sys
import time
import numpy as np
import torch

from configs.experiment import ROOT,config,environment_config,uses_demo,uses_scenario
from agent.model import Policy
from agent.observation import observe
from agent.rules import RulePolicy
from agent.training import DemonstrationDataset
from agent.learning import on_policy,q_update,preference_update
from agent.preference import PreferenceReplay,ScenarioPreference,ScenarioEvaluator
from data.benchmark import fixtures
from data.online import sample
from environment.public import SchedulingEnv
from result.engineering import counted
from result.recording import Recorder,verify
from result.resources import memory_info
from result.storage import atomic_json,atomic_torch_save,digest,state_hash


def probe(method,root):
    root=Path(root);c=config();c['method']=method
    if method in c['classic']['learning_rates']:c['training']['learning_rate']=c['classic']['learning_rates'][method]
    torch.set_num_threads(c['runtime']['threads']);torch.manual_seed(1701);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();policy=Policy(c);target=deepcopy(policy)
    optimizer=torch.optim.Adam(policy.parameters(),lr=c['training']['learning_rate'])
    sampler=deepcopy(c);sampler['data']['orders']=[96,96]
    rng=np.random.default_rng(1701);observations=[];env=None
    recorder=Recorder(root/'raw',c)
    # Collect distinct public observations from real formal-size episodes.
    for _ in range(c['training']['rollout_steps']):
        if env is None or env.done:env=SchedulingEnv(sample(sampler,rng),environment_config(c)).attach_recorder(recorder,kind='formal_shape')
        o=observe(env);a=RulePolicy('SPT').act(o)[0];observations.append(o)
        recorder.annotate(env,action_index=a,candidates=len(o.candidate),orders_visible=len(o.order),operations_visible=len(o.operation),machines=len(o.machine))
        env.step(o.actions.actions[a])
    if uses_scenario(method):
        if env.done:env=SchedulingEnv(sample(sampler,rng),environment_config(c)).attach_recorder(recorder,kind='formal_shape')
        o=observe(env);scenario=ScenarioEvaluator(c,17);scenario.recorder=recorder
        scenario.evaluate(env.public_state(),o,policy,c['scenario']['reservation_floor'],'SPT',0,
                          indices=(0,len(o.candidate)-1),candidate_policy=policy)
    recorder.commit(recorder.snapshot());verify(root/'raw')
    records=[]
    for j,o in enumerate(observations):
        a,lp,v=policy.act(o)
        records.append(dict(obs=o,action=a,logp=lp,value=v,reward=(j%3)*.01,
                            stream=0,terminated=False,truncated=False,boundary=False))
    is_q=method in ('dqn','ddqn')
    if is_q:
        rows=[(o,r['action'],r['reward'],observations[(j+1)%len(observations)]) for j,(o,r) in enumerate(zip(observations,records))]
        encoded=[pickle.dumps(row,protocol=5) for row in rows]
        replay=deque((pickle.loads(encoded[j%len(encoded)]) for j in range(c['classic']['replay_size'])),maxlen=c['classic']['replay_size'])
        metrics=q_update(policy,target,optimizer,replay,c,np.random.default_rng(11))
    else:
        replay=None
        rows=records[:c['classic']['a2c_rollout']] if method=='a2c' else records
        metrics=on_policy(policy,optimizer,rows,{0:0.},c,np.random.default_rng(11))
    for key,value in metrics.items():
        if value is not None and not np.isfinite(value):raise AssertionError(f'Nonfinite {key}')
    demo=None;demonstration_bytes=0
    if uses_demo(method):
        encoded=[pickle.dumps(o,protocol=5) for o in observations]
        demo=DemonstrationDataset([pickle.loads(encoded[j%len(encoded)]) for j in range(c['demonstration']['steps'])],
                                  [records[j%len(records)]['action'] for j in range(c['demonstration']['steps'])],c['demonstration']['steps'])
        policy.residual_enabled=False
        batch=policy.batch(demo.observations[:c['demonstration']['minibatch']]);out=policy(batch)
        ix=batch.offsets[:-1]+torch.tensor(demo.actions[:c['demonstration']['minibatch']]);loss=-out.logp[ix].mean()
        demo_opt=torch.optim.Adam(policy.parameters(),lr=c['demonstration']['learning_rate'])
        demo_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),c['demonstration']['max_grad_norm']);demo_opt.step()
        policy.residual_enabled=True
        atomic_torch_save(demo,root/'demonstrations.pt');demonstration_bytes=(root/'demonstrations.pt').stat().st_size
        del batch,out,loss,demo;demo=None;gc.collect()
    preferences=None
    if uses_scenario(method):
        preferences=PreferenceReplay(c['scenario']['replay_size'],c['scenario']['versions'])
        valid=[o for o in observations if len(o.candidate)>1]
        label=ScenarioPreference((0,1),np.array([[0.,1.],[0.,1.]]),1.,0.,.9,.9,0,8,0.,True,'resource_fixture',[],[],'',[],[])
        encoded=[pickle.dumps((o,label),protocol=5) for o in valid]
        for j in range(c['scenario']['replay_size']):preferences.items.append(pickle.loads(encoded[j%len(encoded)]))
        metrics.update(preference_update(policy,optimizer,preferences,c,np.random.default_rng(13),0))
    if not all(torch.isfinite(p).all() for p in policy.parameters()):raise AssertionError('Nonfinite parameters')
    state=dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),target=target.state_dict(),
               frozen=deepcopy(policy.state_dict()),replay=replay,preferences=preferences,
               observations=observations,cfg=c,engineering_only=True)
    atomic_torch_save(state,root/'recovery.pt')
    restored=torch.load(root/'recovery.pt',weights_only=False,map_location='cpu')
    if state_hash(restored['model'])!=state_hash(policy.state_dict()):raise AssertionError('Formal checkpoint roundtrip differs')
    if is_q and len(restored['replay'])!=c['classic']['replay_size']:raise AssertionError('Formal replay capacity missing')
    atomic_torch_save(dict(model=policy.state_dict(),cfg=c),root/'inference.pt')
    del restored,state,replay,preferences,records,observations;gc.collect()
    # Full largest-shop input, with all orders visible, exercises worst candidate/graph shape.
    inst=next(i for i in fixtures(c,ROOT/'data/instances/fixed','large_shop') if i.stage_count==7)
    inst=replace(inst,arrival_times=np.zeros(inst.order_count),due_dates=np.full(inst.order_count,100000.))
    obs=observe(SchedulingEnv(inst,environment_config(c)))
    with torch.no_grad():
        out=policy(policy.batch([obs]))
        if not torch.isfinite(out.logits).all():raise AssertionError('Largest graph nonfinite')
    raw_bytes=sum(p.stat().st_size for p in (root/'raw').rglob('*') if p.is_file())
    files={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file() and p.name not in ('stdout.log','complete.json')}
    result=dict(method=method,seconds=time.perf_counter()-start,peak_memory_bytes=memory_info()['peak_bytes'],
                formal_config=c,orders=96,rollout=c['training']['rollout_steps'],metrics=metrics,
                largest_orders=inst.order_count,largest_stages=inst.stage_count,largest_candidates=len(obs.candidate),
                recovery_bytes=(root/'recovery.pt').stat().st_size,inference_bytes=(root/'inference.pt').stat().st_size,
                demonstration_bytes=demonstration_bytes,raw_bytes_per_interaction=raw_bytes/recorder.steps,
                files=files,synthetic_rewards=True,
                interpretation='Engineering size/update fixture, not training or performance evidence.')
    atomic_json(root/'complete.json',result)
    print(json.dumps({k:result[k] for k in ('method','seconds','peak_memory_bytes','recovery_bytes')}),flush=True)


if __name__=='__main__':
    with counted('formal-shape:'+sys.argv[1]):probe(sys.argv[1],sys.argv[2])
