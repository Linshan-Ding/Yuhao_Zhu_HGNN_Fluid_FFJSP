"""Known commitment bounds, not a simulated future or a completed operation."""
from dataclasses import dataclass
import numpy as np
from agent.graph import observe as base_observe
from environment.public import PublicSchedulingState, readonly

EXTRA_DIM=12
CONTENDER_DIM=8

@dataclass(frozen=True)
class Observation:
    """`base` is the public graph; the rest are candidate-conditioned, scale-free summaries of known commitments.

    contender_action/contender_order/contender_features: one row per (candidate, contending order) pair, where a
    contending order either competes for the candidate's machine during the committed occupancy or has an operation
    whose earliest known completion the commitment delays. `late[idx, q]` says whether order q's last operation can no
    longer meet its deadline (under known releases) once candidate idx is committed; the Wait candidate carries the
    uncommitted baseline."""
    base: object
    extra: np.ndarray
    keys: np.ndarray
    contender_action: np.ndarray
    contender_order: np.ndarray
    contender_features: np.ndarray
    late: np.ndarray
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

def last_operations(o):
    """Index of each order's last listed operation (operations are listed stage by stage per order)."""
    last=np.full(len(o.order),-1,dtype=int);last[o.owner]=np.arange(len(o.operation))
    return last

def observe(state):
    state=state if isinstance(state,PublicSchedulingState) else state.public_state()
    o=base_observe(state);n=len(o.candidate);no=len(o.order)
    extra=np.zeros((n,EXTRA_DIM),np.float32);keys=np.full((n,9),np.inf)
    old_start,old_finish=bounds(o);last=last_operations(o);slack=o.order[:,0];need=np.maximum(o.order[:,1],1e-8)
    late=np.zeros((n,no),bool);baseline=old_finish[0,last]>slack+1e-9
    idle=o.machine[:,0]>0
    # Current operation of every order and its cheapest idle alternative (both from public eligibility only).
    current=np.flatnonzero(o.operation[:,4]>0);current_of=np.full(no,-1,dtype=int);current_of[o.owner[current]]=current
    alternative=np.zeros(no);on_machine=np.full((no,len(o.machine)),0.)
    for j in current:
        edges=np.flatnonzero(o.eligibility[:,0]==j);machines=o.eligibility[edges,1];durations=o.edge_features[edges,0]
        q=o.owner[j];on_machine[q,machines]=durations
        if idle[machines].any():alternative[q]=durations[idle[machines]].min()
    ca=[];co=[];cf=[]
    # Small bounded temporary arrays; no full environment is retained in the observation.
    for begin in range(0,n,128):
        indices=np.arange(begin,min(begin+128,n));new_start,new_finish=bounds(o,indices)
        late[indices]=new_finish[:,last]>slack[None,:]+1e-9
        for k,idx in enumerate(indices):
            op,m,owner=o.candidate[idx]
            if m<0: late[idx]=baseline;continue
            original=o.actions.actions[idx].order;oid=state.order_map[original]
            arrival=state.instance.arrival_times[oid];due=state.instance.due_dates[oid]
            proc=o.candidate_features[idx,1]
            keys[idx]=[proc,arrival,due,slack[owner]-need[owner],slack[owner]/need[owner],need[owner],original,m,owner]
            ds=np.maximum(new_start[k]-old_start[0],0);df=np.maximum(new_finish[k]-old_finish[0],0)
            other=o.owner!=owner
            shared=np.zeros(len(o.operation),bool)
            shared[o.eligibility[o.eligibility[:,1]==m,0]]=True
            relevant=((shared&(old_start[0]<proc*o.time_ratio))|(df>1e-8))&other
            contenders=np.unique(o.owner[relevant])
            for q in contenders:
                ops=np.flatnonzero(o.owner==q)
                cf.append([slack[q]-new_finish[k,last[q]],slack[q]-old_finish[0,last[q]],df[ops].max(),ds[ops].max(),
                           o.order[q,1],on_machine[q,m],alternative[q],np.clip(slack[q]/need[q],-5,5)])
            ca.extend([idx]*len(contenders));co.extend(contenders)
            ratio=proc/max(o.operation[op,1],1e-8);delayed=(df>1e-8)&other
            extra[idx]=[ratio,ratio-1,slack[owner]/need[owner],o.order[owner,2],o.machine[m,2],
                o.operation[op,7],np.log1p(np.count_nonzero(delayed)),ds[other].mean() if other.any() else 0.,
                df[other].max(initial=0),o.machine[m,3],df[other].mean() if other.any() else 0.,np.log1p(len(contenders))]
    return Observation(o,readonly(extra),readonly(keys),readonly(np.array(ca,np.int64)),readonly(np.array(co,np.int64)),
                       readonly(np.asarray(cf,np.float32).reshape(-1,CONTENDER_DIM)),readonly(late))
