"""Known commitment bounds, not a simulated future or a completed operation."""
from dataclasses import dataclass
import numpy as np
from agent.graph import observe as base_observe
from environment.public import PublicSchedulingState, readonly
from environment.interfaces import Wait

EXTRA_DIM=11
CONDITION_DIM=8

@dataclass(frozen=True)
class Observation:
    base: object
    extra: np.ndarray
    keys: np.ndarray
    local_action: np.ndarray
    local_op: np.ndarray
    condition: np.ndarray
    def __getattr__(self, name):
        if name=='base': raise AttributeError(name)
        return getattr(self.base,name)

def bounds(o, selected=None):
    """Earliest starts/completions under known releases; allow optimistic reuse downstream."""
    n=len(o.operation); a=1 if selected is None else len(selected)
    release=np.tile(o.machine[:,1]*o.time_ratio,(a,1))
    chosen=np.full(a,-1,dtype=int); durations=np.zeros(a)
    if selected is not None:
        for k,idx in enumerate(selected):
            op,m,_=o.candidate[idx]
            if m>=0:
                chosen[k]=op;durations[k]=o.candidate_features[idx,1]*o.time_ratio
                release[k,m]=durations[k]
    predecessor=np.full(n,-1,dtype=int)
    if len(o.precedence): predecessor[o.precedence[:,1]]=o.precedence[:,0]
    starts=np.zeros((a,n)); finishes=np.zeros((a,n))
    for stage in np.unique(o.operation[:,0]):
        nodes=np.flatnonzero(o.operation[:,0]==stage)
        edge_ids=np.flatnonzero(np.isin(o.eligibility[:,0],nodes))
        op,m=o.eligibility[edge_ids].T
        ready=np.tile(o.operation[:,6],(a,1))
        has=predecessor>=0
        ready[:,has]=finishes[:,predecessor[has]]
        st=np.maximum(ready[:,op],release[:,m])
        ft=st+o.edge_features[edge_ids,0]*o.time_ratio
        starts[:,nodes]=np.inf;finishes[:,nodes]=np.inf
        for k in range(a):
            np.minimum.at(starts[k],op,st[k]);np.minimum.at(finishes[k],op,ft[k])
        running=nodes[o.operation[nodes,5]>0]
        for j in running:
            busy=np.flatnonzero((o.eligibility[:,0]==j)&(o.edge_features[:,5]>0))
            starts[:,j]=0.;finishes[:,j]=o.edge_features[busy[0],3]
        for k in range(a):
            if chosen[k] in nodes:
                starts[k,chosen[k]]=0.;finishes[k,chosen[k]]=durations[k]
    return starts,finishes

def observe(state):
    state=state if isinstance(state,PublicSchedulingState) else state.public_state()
    o=base_observe(state);n=len(o.candidate)
    extra=np.zeros((n,EXTRA_DIM),np.float32);keys=np.full((n,9),np.inf)
    old_start,old_finish=bounds(o); la=[];lo=[];conditions=[]
    # Small bounded temporary arrays; no full environment is retained in the observation.
    for begin in range(0,n,128):
        indices=np.arange(begin,min(begin+128,n));new_start,new_finish=bounds(o,indices)
        for k,idx in enumerate(indices):
            op,m,owner=o.candidate[idx]
            if m<0: continue
            original=o.actions.actions[idx].order;oid=state.order_map[original]
            arrival=state.instance.arrival_times[oid];due=state.instance.due_dates[oid]
            need=o.order[owner,1];slack=o.order[owner,0];proc=o.candidate_features[idx,1]
            keys[idx]=[proc,arrival,due,slack-need,slack/max(need,1e-8),need,original,m,owner]
            ds=np.maximum(new_start[k]-old_start[0],0);df=np.maximum(new_finish[k]-old_finish[0],0)
            other=o.owner!=owner
            shared=np.zeros(len(o.operation),bool)
            shared[o.eligibility[o.eligibility[:,1]==m,0]]=True
            relevant=((shared&(old_start[0]<proc*o.time_ratio))|(df>1e-8)|(o.owner==owner))
            nodes=np.flatnonzero(relevant)
            cond=np.column_stack((nodes==op,np.full(len(nodes),proc*o.time_ratio),ds[nodes],df[nodes],
                old_start[0,nodes],new_start[k,nodes],new_finish[k,nodes],o.operation[nodes,3]-new_finish[k,nodes]))
            la.extend([idx]*len(nodes));lo.extend(nodes);conditions.extend(cond)
            ratio=proc/max(o.operation[op,1],1e-8)
            extra[idx]=[ratio,ratio-1,slack/max(need,1e-8),o.order[owner,2],o.machine[m,2],
                o.operation[op,7],np.count_nonzero((df>1e-8)&other)/10,
                ds[other].sum(),df[other].max(initial=0),o.machine[m,3],df[other].sum()]
    return Observation(o,readonly(extra),readonly(keys),readonly(np.array(la,np.int64)),
                       readonly(np.array(lo,np.int64)),readonly(np.asarray(conditions,np.float32).reshape(-1,CONDITION_DIM)))
