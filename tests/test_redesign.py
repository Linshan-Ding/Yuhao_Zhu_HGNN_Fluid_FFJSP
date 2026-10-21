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
