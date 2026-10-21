"""Redesigned mechanisms: wait-for-a-machine, contender attention, contested scenario queries, improved-action targets."""
import numpy as np
import pytest
from configs.experiment import config, environment_config
from data.generator import Instance
from environment.public import SchedulingEnv
from environment.interfaces import Dispatch, Wait
from environment.env import dominating_wait
from agent.observation import observe


def shop(proc, due, arrivals=None):
    proc=np.asarray(proc,np.float32);n=len(due)
    i=Instance('shop','test',1,1,(proc.shape[1],),proc,np.zeros(n,np.int64),
               np.zeros(n) if arrivals is None else np.asarray(arrivals,float),np.asarray(due,float),{'DDT':10.})
    return SchedulingEnv(i,environment_config(config(True)))


def test_wait_requires_a_dominating_busy_machine():
    # Machine 1 is fast (2) but busy until 2: 2+2 < 10 dominates the idle slow machine 0 => waiting is admissible.
    e=shop([[10,2]],[50,50]);e.step(Dispatch(0,1))
    assert e.awaited()==(4.,1,0,1,6.) and e.candidate_actions()[-1][0]<0
    # With processing times (3,2) the busy machine would finish at 4 >= 3: no dominance, no wait candidate.
    f=shop([[3,2]],[50,50]);f.step(Dispatch(0,1))
    assert f.awaited() is None and all(a[0]>=0 for a in f.candidate_actions())
    with pytest.raises(ValueError,match='admissible'):f.step(Wait())
    # All machines idle: nothing to wait for.
    g=shop([[10,2]],[50,50]);assert g.awaited() is None and all(a[0]>=0 for a in g.candidate_actions())


def test_wait_wakes_at_the_next_event_and_is_public():
    e=shop([[10,2]],[50,50,50],arrivals=[0,0,1]);e.step(Dispatch(0,1))
    o=observe(e);assert o.global_features[6]==pytest.approx(4/e.t_ref)
    e.step(Wait());assert e.now==1.   # the hidden arrival at 1 precedes the release at 2; the policy decides again
    assert e.awaited()==(4.,1,0,1,7.) and e.stats.held_time==pytest.approx(1.)   # idle finish is now 1+10
    e.step(Wait());assert e.now==2. and e.machine_busy_with[1]<0
    s=e.public_state();v=s.graph_view()
    assert dominating_wait(v.inst,v.problem,v.status,v.stage,v.machine_free_at,v.machine_busy_with,v.now) is None


def contended():
    """Two ready orders: product 0 runs on machine 0 (4) or 1 (8); product 1 only on machine 0 (6). Deadlines 10 and 7."""
    i=Instance('contended','test',2,1,(2,),np.array([[4,8],[6,0]],np.float32),np.array([0,1]),np.array([0.,0.]),np.array([10.,7.]),{'DDT':1.})
    return SchedulingEnv(i,environment_config(config(True)))


def test_contenders_and_late_sets_follow_known_commitments():
    o=observe(contended());by={(int(o.candidate[k,2]),int(o.candidate[k,1])):k for k in range(len(o.candidate))}
    a,b,c=by[0,0],by[0,1],by[1,0]
    # Committing order 0 to machine 0 pushes order 1 (machine 0 only) to finish at 4+6=10 > 7: it becomes a contender and late.
    pairs=np.flatnonzero(o.contender_action==a);assert o.contender_order[pairs].tolist()==[1]
    f=o.contender_features[pairs[0]]
    assert f[0]==pytest.approx(7-10) and f[1]==pytest.approx(7-6) and f[2]==pytest.approx(4) and f[3]==pytest.approx(4)
    assert f[4]==pytest.approx(6) and f[5]==pytest.approx(6/8) and f[6]==pytest.approx(6/8) and f[7]==pytest.approx(7/6)
    assert o.late[a].tolist()==[False,True] and o.late[b].tolist()==[False,False] and o.late[c].tolist()==[False,False]
    assert not len(np.flatnonzero(o.contender_action==b))   # machine 1 is contested by nobody
    assert o.extra[a,11]==pytest.approx(np.log1p(1)) and o.extra[b,11]==0


def test_contender_correction_is_scale_free_and_only_for_contended_candidates():
    import torch
    from dataclasses import replace
    from agent.model import Policy
    c=config(True);o=observe(contended());p=Policy(c)
    with torch.no_grad():
        p.delta.weight.normal_(std=.1);p.gate.weight.zero_()   # frozen gate: the count only enters through it
    by={(int(o.candidate[k,2]),int(o.candidate[k,1])):k for k in range(len(o.candidate))}
    out=p(p.batch([o]),details=True)
    assert out.correction[by[0,1]]==0 and out.correction[by[0,0]]!=0 and out.correction[by[1,0]]!=0
    twice=replace(o,contender_action=np.r_[o.contender_action,o.contender_action],contender_order=np.r_[o.contender_order,o.contender_order],
                  contender_features=np.r_[o.contender_features,o.contender_features])
    torch.testing.assert_close(p(p.batch([twice])).logits,out.logits,atol=1e-6,rtol=1e-5)
