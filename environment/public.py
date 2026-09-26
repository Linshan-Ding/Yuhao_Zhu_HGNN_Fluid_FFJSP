"""Causal public state and versioned simulator restoration, owned by the environment."""
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np
from configs.config import Config
from data.generator import Instance
from environment.env import SchedulingEnv as BaseEnv, NOT_ARRIVED, StepStats
from environment.problem import Problem
from environment.interfaces import Dispatch, Wait

def readonly(x):
    a=np.array(x,copy=True); a.flags.writeable=False
    return a

@dataclass(frozen=True)
class ActionSet:
    actions: tuple
    def __len__(self): return len(self.actions)

@dataclass(frozen=True)
class DecisionObservation:
    order: np.ndarray
    operation: np.ndarray
    machine: np.ndarray
    owner: np.ndarray
    precedence: np.ndarray
    eligibility: np.ndarray
    edge_features: np.ndarray
    candidate: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    actions: ActionSet
    time_ratio: float = 1.0
    schema_version: str = 'schedule-data-1'
    def __post_init__(self):
        for name in self.__dataclass_fields__:
            x=getattr(self,name)
            if isinstance(x,np.ndarray): object.__setattr__(self,name,readonly(x))

@dataclass(frozen=True)
class PublicSchedulingState:
    instance: Instance
    cfg: dict
    now: float
    status: np.ndarray
    stage: np.ndarray
    machine_free: np.ndarray
    machine_busy: np.ndarray
    machine_busy_time: np.ndarray
    outcomes: np.ndarray
    original_ids: np.ndarray
    completed: int
    discarded: int
    t_ref: float
    p_ref: float
    wait_interval: float
    raw_actions: tuple

    @classmethod
    def capture(cls,env):
        known=np.flatnonzero(env.status!=NOT_ARRIVED)
        mapping={int(o):k for k,o in enumerate(known)}; i=env.inst
        safe=Instance('public','public',i.product_count,i.stage_count,i.machines_per_stage,readonly(i.proc_times),
            readonly(i.order_product[known]),readonly(i.arrival_times[known]),readonly(i.due_dates[known]),
            {'DDT':env.t_ref,'schema_version':'schedule-data-1'})
        raw=tuple((t,m,mapping[o]) if t>=0 else (t,m,o) for t,m,o in env.candidate_actions())
        return cls(safe,deepcopy(env.cfg.raw),env.now,readonly(env.status[known]),readonly(env.stage[known]),
            readonly(env.machine_free_at),readonly([mapping[int(o)] if o>=0 else -1 for o in env.machine_busy_with]),
            readonly(env.machine_busy_time),readonly(env.order_outcome[known]),readonly(known),
            env.n_completed,env.n_discarded,env.t_ref,env.p_ref,env.wait_interval,raw)

    @property
    def order_map(self): return {int(o):k for k,o in enumerate(self.original_ids)}

    def branch_action(self,action):
        return Dispatch(self.order_map[action.order],action.machine) if isinstance(action,Dispatch) else Wait()

    def fork(self,products,arrivals,deadlines):
        i=self.instance
        inst=Instance('scenario','scenario',i.product_count,i.stage_count,i.machines_per_stage,i.proc_times.copy(),
            np.r_[i.order_product,products].astype(np.int64),np.r_[i.arrival_times,arrivals],
            np.r_[i.due_dates,deadlines],dict(i.meta))
        env=SchedulingEnv(inst,Config(deepcopy(self.cfg))); n=len(self.status)
        env.now=self.now; env.status[:n]=self.status; env.status[n:]=NOT_ARRIVED
        env.stage[:n]=self.stage; env.stage[n:]=0
        env.machine_free_at=self.machine_free.copy();env.machine_busy_with=self.machine_busy.copy()
        env.machine_busy_time=self.machine_busy_time.copy();env.order_outcome[:n]=self.outcomes
        env.order_outcome[n:]=-1;env.n_completed=self.completed;env.n_discarded=self.discarded
        env.done=False;env.truncated=False;env.step_count=0;env.stats=StepStats()
        env._cand_stamp=-1;env._events=[]
        return env

    def graph_view(self):
        """Graph view containing ONLY sanitized state; no simulator handle."""
        from types import SimpleNamespace
        return SimpleNamespace(inst=self.instance,problem=Problem(self.instance),now=self.now,
            status=self.status,stage=self.stage,machine_free_at=self.machine_free,machine_busy_with=self.machine_busy,
            t_ref=self.t_ref,p_ref=self.p_ref,wait_interval=self.wait_interval,n_completed=self.completed,
            n_discarded=self.discarded,candidate_actions=lambda:self.raw_actions,
            _idle_mask=lambda:(self.machine_busy<0))

class SchedulingEnv(BaseEnv):
    def public_state(self): return PublicSchedulingState.capture(self)

    def attach_recorder(self, recorder, episode_id=None, kind='real'):
        self._recorder = recorder
        self._record_id = recorder.attach(self, episode_id, kind)
        return self

    def step(self, action):
        from environment.accounting import charge
        import time
        charge()
        recorder = getattr(self, '_recorder', None)
        if recorder is not None: recorder.before(self, action)
        start = time.perf_counter()
        result = super().step(action)
        seconds = time.perf_counter() - start
        self.last_step_seconds = seconds
        if recorder is not None: recorder.after(self, action, result, seconds)
        else: self.take_events()
        return result

    def state_dict(self):
        names=('now','status','stage','machine_free_at','machine_busy_with','machine_busy_time','n_completed',
               'n_discarded','step_count','done','truncated','order_outcome')
        return {'schema':'schedule-data-1','instance':asdict(self.inst),'config':deepcopy(self.cfg.raw),
                'dynamic':{k:deepcopy(getattr(self,k)) for k in names},'stats':asdict(self.stats),
                'record_id':getattr(self,'_record_id',None)}

    @classmethod
    def from_state_dict(cls,state):
        if state['schema']!='schedule-data-1': raise ValueError('simulator schema mismatch')
        env=cls(Instance(**deepcopy(state['instance'])),Config(deepcopy(state['config'])))
        for k,v in state['dynamic'].items(): setattr(env,k,deepcopy(v))
        env.stats=StepStats(**deepcopy(state['stats']));env._cand_stamp=-1
        env._events=[];env._record_id=state.get('record_id')
        return env
