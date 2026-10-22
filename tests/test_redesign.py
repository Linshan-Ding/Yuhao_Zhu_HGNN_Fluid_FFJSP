"""Redesigned mechanisms: wait-for-a-machine, contender attention, contested scenario queries, improved-action targets."""
import numpy as np
import pytest
from configs.experiment import config, environment_config
from data.generator import Instance
from environment.public import SchedulingEnv
from environment.interfaces import Dispatch, Wait
from environment.env import dominating_wait
from agent.observation import observe
from agent.rules import RulePolicy


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


class FrozenRule:
    """Deterministic SPT continuation standing in for the frozen graph policy."""
    def __init__(self):self.rule=RulePolicy('SPT')
    def act_many(self,observations):return self.rule.act_many(observations)
    def state_dict(self):return {}


def test_adaptive_futures_resolve_the_sign_and_stop_on_ties(monkeypatch):
    import agent.preference as preference
    c=config(True);c['scenario'].update(count=2,max_count=8,reliability_floor=.25)
    e=contended();o=observe(e);by={(int(o.candidate[k,2]),int(o.candidate[k,1])):k for k in range(len(o.candidate))}
    a,b,d=by[0,0],by[0,1],by[1,0]
    # Even futures add nothing; odd futures add an order of product 0 (deadline 9) that only the anchor branch can serve
    # on time, so the challenger's advantage alternates 1,0,1,0,...: the sign is resolved after five futures.
    def future(self,snapshot):
        j=self.calls_made=getattr(self,'calls_made',0)+1
        return (np.array([],np.int64),np.array([]),np.array([])) if j%2 else (np.array([0]),np.array([.5]),np.array([9.]))
    monkeypatch.setattr(preference.ScenarioEvaluator,'future',future)
    label=preference.ScenarioEvaluator(c,1).evaluate(e.public_state(),o,FrozenRule(),10000,'SPT',0,indices=[a,b])
    assert label.complete and label.k_used==5 and label.returns[:,1].tolist()==[2,2,2,2,2] and label.returns[:,0].tolist()==[1,2,1,2,1]
    assert label.mean==pytest.approx(.6) and label.winner=='challenger' and label.reliability>=.5
    tie=preference.ScenarioEvaluator(c,1).evaluate(e.public_state(),o,FrozenRule(),10000,'SPT',0,indices=[b,d])
    assert tie.complete and tie.k_used==2 and tie.mean==0 and tie.winner=='tie' and tie.reliability==0
    # A consistent advantage is resolved with the minimum number of futures.
    monkeypatch.setattr(preference.ScenarioEvaluator,'future',lambda self,s:(np.array([],np.int64),np.array([]),np.array([])))
    sure=preference.ScenarioEvaluator(c,1).evaluate(e.public_state(),o,FrozenRule(),10000,'SPT',0,indices=[a,b])
    assert sure.k_used==2 and sure.mean==1 and sure.reliability==pytest.approx(1/1.25)


def test_improved_action_targets_update_only_from_reliable_wins():
    import torch
    from agent.model import Policy
    from agent.learning import preference_update
    from agent.preference import ScenarioPreference,PreferenceReplay
    c=config(True);c['scenario'].update(reliability_threshold=.5);o=observe(contended());p=Policy(c)
    opt=torch.optim.Adam(p.parameters(),lr=.001);replay=PreferenceReplay(8,2)
    def label(mean,reliability):
        return ScenarioPreference((0,1),np.array([[0.,mean],[0.,mean]]),mean,0.,reliability,.5,1,8,0.,True,'test',[],[],'',[],2,
                                  'tie' if mean==0 else ('challenger' if mean>0 else 'anchor'),[[],[]])
    replay.add(o,label(0.,0.));replay.add(o,label(1.,.3));replay.add(o,label(-1.,.9))
    before=p(p.batch([o])).logp.detach().clone()
    m=preference_update(p,opt,replay,c,np.random.default_rng(1),1)
    assert m['preference_steps']==0 and m['usable_in_replay']==0 and m['replay_labels']==3
    torch.testing.assert_close(p(p.batch([o])).logp,before,rtol=0,atol=0)
    replay.add(o,label(1.,.8))
    m=preference_update(p,opt,replay,c,np.random.default_rng(1),1)
    after=p(p.batch([o])).logp.detach()
    assert m['preference_steps']>=1 and m['usable_in_replay']==1 and after[1]>before[1]   # the challenger (index 1) gains probability


def test_teacher_is_selected_on_the_validation_split_and_recorded(tmp_path):
    import json
    from configs.experiment import resolve_teacher, RULES, matrix
    c=config(True);c['teacher']='auto';d=resolve_teacher(c,tmp_path)
    record=json.loads((tmp_path/'teacher.json').read_text())
    assert d['teacher']==record['rule'] and record['split']=='contested_validation' and set(record['scores'])==set(RULES)
    best=max(record['scores'].values());assert record['scores'][record['rule']]==best
    assert all(record['scores'][r]<best for r in RULES[:RULES.index(record['rule'])])   # ties go to the earlier rule
    assert c['teacher']=='auto' and matrix(d)[0].config['teacher']==record['rule']


def test_online_regimes_draw_loads_and_due_factors_per_regime():
    from data.online import sample
    c=config(True);d=c['data'];d['orders']=[8,8]
    for regime in d['regimes']:regime['weight']=0.
    d['regimes'][0]['weight']=1.;contested=d['regimes'][0];assert contested['by_rho']
    rng=np.random.default_rng(5);draws=[sample(c,rng) for _ in range(20)]
    assert all(contested['rho'][0]<=i.meta['rho_sys']<=contested['rho'][1] and contested['due'][0]<=i.meta['due_factor']<=contested['due'][1] for i in draws)
    assert all(i.meta['sampling_regime']=='contested' for i in draws)
    d['regimes'][0]['weight']=0.;d['regimes'][3]['weight']=1.;legacy=d['regimes'][3]
    draws=[sample(c,rng) for _ in range(10)]
    assert all(legacy['iota'][0]<=i.meta['iota_target']<=legacy['iota'][1] and legacy['due'][0]<=i.meta['due_factor']<=legacy['due'][1] for i in draws)


def test_evaluation_plan_and_groups_follow_the_study():
    from configs.experiment import matrix
    from result.evaluation import evaluation_plan
    from result.statistics import grouped, groups_of
    c=config(True);names=[p['name'] for p in evaluation_plan(c,matrix(c))]
    assert names[:6]==['contested','contested_small','main','small','long_stream','large_shop'] and 'contested_best' in names and 'sensitivity' in names
    assert next(p for p in evaluation_plan(c,matrix(c)) if p['name']=='initialization')['split']=='contested_validation'
    assert 'sensitivity' not in [p['name'] for p in evaluation_plan(c,matrix(c,'pilot'))]
    rows=[dict(rho_target='1.3',due_factor='1.5',orders='48',iota='2.5',rho='1.3'),dict(rho_target='1.0',due_factor='2.0',orders='96',iota='2.0',rho='1.0')]
    assert groups_of(rows)==['all','rho=1','rho=1.3','due=1.5','due=2','orders=48','orders=96']
    assert grouped(rows,'rho=1.3')==rows[:1] and grouped(rows,'orders=96')==rows[1:] and grouped(rows,'all')==rows
    assert groups_of([dict(rho_target='-1.0',iota='1.5',rho='.8')])==['all','frequent','infrequent','overload','underload']


def test_instance_labels_are_readable_and_unique():
    from data.benchmark import number, label, design, cells, parameter_key
    assert [number(x) for x in (1.0,1.15,1.3,2.0,0.6,3.0)]==['1.0','1.15','1.3','2.0','0.6','3.0']
    c=config();rows=design(c);names=[label(s['role'],s,parameter_key(c,s,iota,due,rho,r)) for s,iota,due,rho,r in cells(rows)]
    assert len(names)==len(set(names))==102 and all(set(n)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.') for n in names)
    micro=config(True);assert {label(s['role'],s,parameter_key(micro,s,iota,due,rho,r)) for s,iota,due,rho,r in cells(design(micro))}>={'M8-I1.5-K3.0','C8-R1.3-K1.5-r1','H4-12-I1.5-K3.0'}
