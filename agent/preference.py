"""Paired futures, mixed frozen continuations and bounded preference replay."""
from dataclasses import dataclass,asdict
from collections import deque
import hashlib
import time
import numpy as np
from scipy.special import expit
from agent.future import FutureGenerator
from agent.observation import observe
from agent.rules import RulePolicy
from configs.experiment import RULES

@dataclass
class ScenarioPreference:
    actions: tuple
    returns: np.ndarray
    mean: float
    standard_error: float
    reliability: float
    probability: float
    version: int
    steps: int
    seconds: float
    complete: bool
    source: str
    scene_hashes: list
    branch_hashes: list
    teacher_hash: str
    continuations: list
    cohort_sizes: list
    candidate_policy_hash: str = ""
    def record(self):
        d=asdict(self);d['returns']=self.returns.tolist();return d

class ScenarioEvaluator(FutureGenerator):
    def __init__(self,cfg,seed):
        super().__init__(cfg,seed);self.last_cost=64
    def choose(self,obs,policy,teacher):
        anchor=RulePolicy(teacher).act(obs)[0];best=policy.act(obs)[0]
        proposed=list(dict.fromkeys([best]+[RulePolicy(r).act(obs)[0] for r in RULES]))
        alternatives=[a for a in proposed if a!=anchor and obs.candidate[a,0]>=0]
        if alternatives:return (anchor,alternatives[self.calls%len(alternatives)]),'dispatch_disagreement'
        # Query timing sparsely; all online actions are always retained.
        waits=np.flatnonzero(obs.candidate[:,0]<0)
        if len(waits) and self.calls%4==0:return (anchor,int(waits[0])),'dispatch_wait'
        return None,'no_disagreement'

    def evaluate(self,snapshot,obs,policy,step_budget,teacher,version,indices=None,candidate_policy=None):
        candidate_policy=policy if candidate_policy is None else candidate_policy
        chosen,source=self.choose(obs,candidate_policy,teacher) if indices is None else (tuple(indices),'explicit')
        self.calls+=1
        if chosen is None:return None
        start=time.perf_counter();branches=[];hashes=[];actual=[];cohorts=[];continuations=[]
        recorder=getattr(self,'recorder',None)
        query_id=f'query_{self.calls:08d}'
        if recorder is not None:
            public_key=recorder.artifact(snapshot,'public_queries');teacher_key=recorder.artifact(policy.state_dict(),'teachers')
            recorder.emit('queries',dict(query_id=query_id,public_state=public_key,teacher=teacher_key,actions=chosen,source=source,version=version))
        teacher_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in policy.state_dict().values())).hexdigest()
        count=self.cfg['scenario']['count']
        for j in range(count):
            future=self.future(snapshot)
            if recorder is not None: recorder.emit('futures',dict(query_id=query_id,scenario=j,products=future[0],arrivals=future[1],deadlines=future[2]))
            hashes.append(hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in future)).hexdigest())
            cohort=int(np.sum(snapshot.outcomes<0))+len(future[0]);cohorts.append(cohort)
            continuations.append(teacher if (j+self.calls)%2==0 else 'frozen_policy')
            local=[]
            for idx in chosen:
                env=snapshot.fork(*future)
                if recorder is not None:
                    env.attach_recorder(recorder,kind='scenario')
                    recorder.emit('branches',dict(query_id=query_id,scenario=j,action_index=idx,episode_id=env._record_id))
                branches.append(env)
                n=len(snapshot.status)
                local.append(hashlib.sha256(b''.join(np.ascontiguousarray(v[n:]).tobytes() for v in (env.inst.order_product,env.inst.arrival_times,env.inst.due_dates))).hexdigest())
            actual.append(local)
        steps=0;counts=np.zeros(len(branches),int);complete=True
        for k,e in enumerate(branches):
            if steps>=step_budget:complete=False;break
            e.step(snapshot.branch_action(obs.actions.actions[chosen[k%2]]));steps+=1;counts[k]+=1
        while complete:
            active=[k for k,e in enumerate(branches) if not e.done]
            if not active:break
            if steps>=step_budget or any(counts[k]>=self.cfg['scenario']['max_branch_steps'] for k in active):complete=False;break
            active=active[:step_budget-steps];observations=[observe(branches[k]) for k in active]
            pi=[j for j,k in enumerate(active) if continuations[k//2]=='frozen_policy']
            choices={j:p[0] for j,p in zip(pi,policy.act_many([observations[j] for j in pi]))} if pi else {}
            for j,k in enumerate(active):
                choice=choices[j] if j in choices else RulePolicy(teacher).act(observations[j])[0]
                branches[k].step(observations[j].actions.actions[choice]);steps+=1;counts[k]+=1
        complete=complete and all(e.done and not e.truncated for e in branches)
        values=np.asarray([e.n_completed-snapshot.completed for e in branches],np.float32).reshape(count,2)
        values/=np.maximum(np.asarray(cohorts)[:,None],1)
        delta=values[:,1]-values[:,0];mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(count)) if count>1 else 1.
        floor=self.cfg['scenario']['reliability_floor'];scale=se+floor
        reliability=abs(mean)/(abs(mean)+scale) if complete else 0.
        preference=float(expit(np.clip(mean/scale,-self.cfg['scenario']['max_preference'],self.cfg['scenario']['max_preference'])))
        self.last_cost=max(steps,1) if complete else max(self.last_cost,2*steps)
        label=ScenarioPreference(chosen,values,mean,se,reliability,preference,version,steps,time.perf_counter()-start,
                                  complete,source,hashes,actual,teacher_hash,continuations,cohorts,
                                  hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in candidate_policy.state_dict().values())).hexdigest() if hasattr(candidate_policy,"state_dict") else type(candidate_policy).__name__)
        if recorder is not None:
            recorder.emit('preferences',dict(query_id=query_id,branch_steps=counts,**label.record()))
            for e in branches:
                if not e.done: recorder.outcomes(e,'scenario_budget_or_branch_limit')
        return label

class PreferenceReplay:
    def __init__(self,capacity=4096,versions=2):self.items=deque(maxlen=capacity);self.versions=versions;self.reuses=0
    def add(self,obs,label):
        if label.complete:self.items.append((obs,label))
    def expire(self,version):
        self.items=deque(((o,l) for o,l in self.items if version-self.versions<l.version<=version),maxlen=self.items.maxlen)
    def sample(self,rng,size):
        if not self.items:return []
        idx=rng.choice(len(self.items),min(size,len(self.items)),replace=False);self.reuses+=len(idx)
        return [self.items[int(i)] for i in idx]

