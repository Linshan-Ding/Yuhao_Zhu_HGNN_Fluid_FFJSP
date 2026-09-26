"""Build the public graph exclusively from arrived orders and the public catalog."""
import numpy as np
from environment.env import WAITING, IN_PROCESS, NOT_ARRIVED
from environment.interfaces import Dispatch, Wait
from environment.public import ActionSet, DecisionObservation, PublicSchedulingState

ORDER_DIM, OP_DIM, MACHINE_DIM, EDGE_DIM, ACTION_DIM, GLOBAL_DIM = 8, 8, 6, 6, 7, 8

def observe(state):
    if not isinstance(state, PublicSchedulingState):
        state = PublicSchedulingState.capture(state)
    env = state.graph_view()
    p, inst, now = env.problem, env.inst, env.now
    raw = tuple(env.candidate_actions())
    if not raw:
        raise ValueError("observation requires a decision state")
    active = np.flatnonzero((env.status == WAITING) | (env.status == IN_PROCESS))
    arrived = np.flatnonzero(env.status != NOT_ARRIVED)
    tref, pref = env.t_ref, env.p_ref
    nseen = max(len(arrived), 1)
    machine = np.zeros((p.n_machine, MACHINE_DIM), np.float32)
    idle = env._idle_mask()
    machine[:, 0] = idle
    machine[:, 1] = np.maximum(env.machine_free_at - now, 0) / pref
    mstage = np.repeat(np.arange(p.n_stage), inst.machines_per_stage)
    machine[:, 5] = mstage / max(p.n_stage - 1, 1)
    order, op, owner, prec, elig, edge = [], [], [], [], [], []
    lookup = {}
    local_order = {int(o): k for k, o in enumerate(active)}
    for k, o in enumerate(active):
        r, stage = int(inst.order_product[o]), int(env.stage[o])
        running = env.status[o] == IN_PROCESS
        busy = np.flatnonzero(env.machine_busy_with == o)
        rest = max(float(env.machine_free_at[busy[0]]) - now, 0.0) if running else 0.0
        need = rest + p.residual[r, stage + 1] if running else p.residual[r, stage]
        slack = float(inst.due_dates[o] - now)
        task = p.task_index(r, stage)
        order.append([slack/tref, need/tref, stage/p.n_stage, float(running), float(not running),
                      (now-inst.arrival_times[o])/tref, len(p.eligible[task])/p.n_machine,
                      (slack-need)/tref])
        earliest = 0.0
        last = None
        for j in range(stage, p.n_stage):
            t = p.task_index(r, j)
            idx = len(op)
            lookup[(int(o), j)] = idx
            is_running = running and j == stage
            op.append([j/p.n_stage, p.min_proc[t]/pref, p.residual[r,j]/tref, slack/tref,
                       float(j == stage and not running), float(is_running), earliest/tref,
                       len(p.eligible[t])/p.n_machine])
            owner.append(k)
            if last is not None:
                prec.append((last, idx))
            last = idx
            for m in p.eligible[t]:
                duration = float(inst.proc_times[t,m])
                elig.append((idx,m))
                edge.append([duration/pref, (slack-p.residual[r,j+1]-duration)/tref,
                             earliest/tref, max(env.machine_free_at[m]-now,0)/tref,
                             len(p.eligible[t])/p.n_machine,
                             float(is_running and m == busy[0])])
                machine[m, 2] += duration / pref / len(p.eligible[t])
                machine[m, 3] += 1.0 / nseen
            if is_running:
                machine[busy[0],4] = slack/tref
            earliest += rest if is_running else p.min_proc[t]
    candidate, features = [], []
    for t,m,o in raw:
        if t < 0:
            candidate.append((-1,-1,-1))
            features.append([0,0,0,0,1,env.wait_interval/tref,0])
        else:
            stage = int(env.stage[o])
            r = int(inst.order_product[o])
            slack = inst.due_dates[o]-now
            need = p.residual[r,stage]
            candidate.append((lookup[(o,stage)],m,local_order[o]))
            features.append([slack/tref,inst.proc_times[t,m]/pref,(slack-need)/tref,
                             need/tref,0,0,(now-inst.arrival_times[o])/tref])
    history_time = max(now-float(inst.arrival_times[arrived].min()), tref)
    rate = max(len(arrived)-1,0)/history_time
    glob = np.asarray([env.n_completed/nseen,env.n_discarded/nseen,len(active)/nseen,
        rate*pref,float(idle.mean()),np.log1p(nseen),env.wait_interval/tref,p.n_stage/10],np.float32)
    return DecisionObservation(np.asarray(order,np.float32), np.asarray(op,np.float32), machine,
        np.asarray(owner,np.int64), np.asarray(prec,np.int64).reshape(-1,2),
        np.asarray(elig,np.int64).reshape(-1,2),np.asarray(edge,np.float32).reshape(-1,EDGE_DIM),
        np.asarray(candidate,np.int64),np.asarray(features,np.float32),glob,
        ActionSet(tuple(Wait() if a[0]<0 else Dispatch(int(state.original_ids[a[2]]),a[1]) for a in raw)), time_ratio=env.p_ref/env.t_ref)



def rule_index(obs, rule='SPT'):
    x=obs.candidate_features;live=np.flatnonzero(x[:,4]==0);wait=np.flatnonzero(x[:,4]!=0)
    if not len(live): return int(wait[0])
    if rule!='SPT': raise ValueError(rule)
    return int(live[np.argmin(x[live,1])])
