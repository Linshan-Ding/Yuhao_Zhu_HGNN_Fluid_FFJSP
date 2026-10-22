"""Contested-state paired queries, frozen-policy continuations, adaptive shared futures and bounded preference replay."""
from dataclasses import dataclass,asdict
from collections import deque
import hashlib
import time
import numpy as np
from scipy.special import expit
from agent.future import FutureGenerator
from agent.observation import observe
from agent.rules import RulePolicy
from result.storage import state_hash

@dataclass
class ScenarioPreference:
    actions: tuple            # (anchor, challenger) candidate indices of the queried observation
    returns: np.ndarray       # [k_used, 2] on-time completions beyond the snapshot, per shared future and branch
    mean: float               # mean of challenger minus anchor, in orders
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
    frozen_hash: str
    cohort_sizes: list
    k_used: int
    winner: str               # 'challenger' | 'anchor' | 'tie'
    late_sets: list           # orders predicted late (known commitments) under the anchor and under the challenger
    candidate_policy_hash: str = ""
    def record(self):
        d=asdict(self);d['returns']=self.returns.tolist();return d

def winner_of(mean):
    return 'tie' if mean==0 else ('challenger' if mean>0 else 'anchor')

class ScenarioEvaluator(FutureGenerator):
    """Paired first-action comparison at contested states.

    The anchor is the live policy's greedy action; the challenger is the most probable alternative (the policy's
    runner-ups, then the teacher rule's action) whose known commitment changes which orders can still meet their
    deadlines (`Observation.late`). Both branches continue with the frozen graph policy under the same sampled
    futures; futures are added until the sign of the difference is resolved or `scenario.max_count` is reached."""
    def state_dict(self):
        return {**super().state_dict(),'checks':getattr(self,'checks',0)}

    def load_state_dict(self,state):
        super().load_state_dict(state);self.checks=state.get('checks',0)

    def choose(self,obs,probabilities,rule):
        self.checks=getattr(self,'checks',0)+1
        order=np.argsort(-np.asarray(probabilities,dtype=np.float64),kind='stable');anchor=int(order[0])
        runner_ups=[int(i) for i in order[1:3]];teacher=int(RulePolicy(rule).act(obs)[0])
        for i in dict.fromkeys(runner_ups+[teacher]):
            if i!=anchor and np.any(obs.late[i]!=obs.late[anchor]):
                return (anchor,i),('contested_policy' if i in runner_ups else 'contested_rule')
        return None,'uncontested'

    def evaluate(self,snapshot,obs,frozen,step_budget,rule,version,probabilities=None,indices=None,candidate_policy=None):
        if indices is not None:chosen,source=tuple(int(i) for i in indices),'explicit'
        else:
            if probabilities is None:probabilities=candidate_policy.act_many([obs],return_probabilities=True)[1][0]
            chosen,source=self.choose(obs,probabilities,rule)
        if chosen is None:return None
        self.calls+=1;sc=self.cfg['scenario'];count=int(sc['count']);max_count=max(int(sc['max_count']),count)
        start=time.perf_counter();hashes=[];actual=[];cohorts=[];values=[];steps=0;complete=True
        recorder=getattr(self,'recorder',None);query_id=f'query_{self.calls:08d}'
        if recorder is not None:
            public_key=recorder.artifact(snapshot,'public_queries');frozen_key=recorder.artifact(frozen.state_dict(),'frozen_policies')
            recorder.emit('queries',dict(query_id=query_id,public_state=public_key,frozen_policy=frozen_key,actions=chosen,source=source,version=version))
        frozen_hash=state_hash(frozen.state_dict());n=len(snapshot.status)
        for j in range(max_count):
            future=self.future(snapshot)
            if recorder is not None: recorder.emit('futures',dict(query_id=query_id,scenario=j,products=future[0],arrivals=future[1],deadlines=future[2]))
            hashes.append(hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in future)).hexdigest())
            cohorts.append(int(np.sum(snapshot.outcomes<0))+len(future[0]))
            branches=[];local=[]
            for idx in chosen:
                env=snapshot.fork(*future)
                if recorder is not None:
                    env.attach_recorder(recorder,kind='scenario')
                    recorder.emit('branches',dict(query_id=query_id,scenario=j,action_index=idx,episode_id=env._record_id))
                branches.append(env)
                local.append(hashlib.sha256(b''.join(np.ascontiguousarray(v[n:]).tobytes() for v in (env.inst.order_product,env.inst.arrival_times,env.inst.due_dates))).hexdigest())
            actual.append(local);counts=np.zeros(2,int)
            for k,e in enumerate(branches):
                if steps>=step_budget:complete=False;break
                e.step(snapshot.branch_action(obs.actions.actions[chosen[k]]));steps+=1;counts[k]+=1
            while complete:
                active=[k for k,e in enumerate(branches) if not e.done]
                if not active:break
                if steps>=step_budget or any(counts[k]>=sc['max_branch_steps'] for k in active):complete=False;break
                active=active[:step_budget-steps];observations=[observe(branches[k]) for k in active]
                for k,o,(choice,_,_) in zip(active,observations,frozen.act_many(observations)):
                    branches[k].step(o.actions.actions[choice]);steps+=1;counts[k]+=1
            complete=complete and all(e.done and not e.truncated for e in branches)
            if recorder is not None:
                for e in branches:
                    if not e.done: recorder.outcomes(e,'scenario_budget_or_branch_limit')
            if not complete:break
            values.append([e.n_completed-snapshot.completed for e in branches])
            delta=np.diff(np.asarray(values,np.float64),axis=1)[:,0];k=len(delta)
            if k>=count:
                mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(k))
                if abs(mean)>2*se or not np.any(delta) or k>=max_count:break   # sign resolved, no effect, or exhausted
        values=np.asarray(values,np.float32).reshape(-1,2)
        if complete:
            delta=values[:,1]-values[:,0];mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        else:mean=se=0.
        floor=sc['reliability_floor'];scale=se+floor
        reliability=abs(mean)/(abs(mean)+scale) if complete else 0.
        preference=float(expit(np.clip(mean/scale,-sc['max_preference'],sc['max_preference'])))
        self.last_cost=max(steps,1) if complete else max(self.last_cost,2*steps)
        late=[np.flatnonzero(obs.late[i]).tolist() for i in chosen]
        label=ScenarioPreference(chosen,values,mean,se,reliability,preference,version,steps,time.perf_counter()-start,
                                  complete,source,hashes,actual,frozen_hash,cohorts,len(values),winner_of(mean) if complete else 'tie',late,
                                  state_hash(candidate_policy.state_dict()) if hasattr(candidate_policy,'state_dict') else type(candidate_policy).__name__)
        if recorder is not None:recorder.emit('preferences',dict(query_id=query_id,**label.record()))
        return label

class PreferenceReplay:
    def __init__(self,capacity=4096,versions=2):self.items=deque(maxlen=capacity);self.versions=versions;self.reuses=0
    def add(self,obs,label):
        if label.complete:self.items.append((obs,label))
    def expire(self,version):
        self.items=deque(((o,l) for o,l in self.items if version-self.versions<l.version<=version),maxlen=self.items.maxlen)
    def usable(self,threshold):
        """Labels that name an improvement: the challenger won with reliability at or above `threshold`."""
        return [(o,l) for o,l in self.items if l.winner=='challenger' and l.reliability>=threshold]
    def sample(self,rng,size,pool=None):
        pool=list(self.items) if pool is None else pool
        if not pool:return []
        idx=rng.choice(len(pool),min(size,len(pool)),replace=False);self.reuses+=len(idx)
        return [pool[int(i)] for i in idx]
