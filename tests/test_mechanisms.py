from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from configs.experiment import config,environment_config
from data.generator import Instance
from data.online import sample
from environment.public import SchedulingEnv,ActionSet
from agent.observation import observe,bounds
from agent.rules import RulePolicy
from agent.model import Policy
from agent.learning import q_bootstrap,preference_update
from agent.preference import ScenarioEvaluator,PreferenceReplay,ScenarioPreference

def tiny(alternative=True):
    i=Instance('tiny','test',2,1,(2,),np.array([[5,5],[6,6 if alternative else 0]],np.float32),
               np.array([0,1]),np.array([0.,0.]),np.array([100.,80.]),{'DDT':10.,'iota':1.,'rho_sys':1.,'due_factor':2.})
    return SchedulingEnv(i,environment_config(config(True)))

def test_known_commitment_alternative_and_selected_operation():
    for alt in (True,False):
        o=observe(tiny(alt));a=next(j for j,(op,m,order) in enumerate(o.candidate) if order==0 and m==0)
        before,finish=bounds(o.base);after,af=bounds(o.base,[a]);other=np.flatnonzero(o.owner==1)[0]
        assert after[0,other]-before[0,other]==pytest.approx(0. if alt else .5)
        selected=o.candidate[a,0];assert after[0,selected]==0 and af[0,selected]==pytest.approx(.5)
        assert o.operation[selected,5]==0 # hypothetical commitment does not mutate public state

@pytest.mark.parametrize('rule',['SPT','FCFS','EDD','MST','CR','LWKR'])
def test_rules_are_independent_of_candidate_enumeration(rule):
    o=observe(tiny());idx=np.arange(len(o.candidate))[::-1]
    b=replace(o.base,candidate=o.candidate[idx],candidate_features=o.candidate_features[idx],actions=ActionSet(tuple(o.actions.actions[i] for i in idx)))
    shuffled=replace(o,base=b,extra=o.extra[idx],keys=o.keys[idx])
    a=RulePolicy(rule).act(o)[0];other=RulePolicy(rule).act(shuffled)[0]
    assert o.actions.actions[a]==shuffled.actions.actions[other]

def test_zero_residual_chunk_batch_and_gradient():
    c=config(True);o=observe(tiny());p=Policy(c);b=p.batch([o]);full=p(b)
    p.residual_enabled=False;base=p(b);torch.testing.assert_close(full.logits,base.logits,rtol=0,atol=0)
    p.residual_enabled=True
    with torch.no_grad():p.delta.weight.fill_(.01)
    p.chunk=1;a=p(p.batch([o])).logits;p.chunk=128;bb=p(p.batch([o,o])).logits
    torch.testing.assert_close(a,bb[:len(a)],atol=2e-6,rtol=1e-5)
    (-p(b).logp[0]).backward();assert p.delta.weight.grad.abs().sum()>0

def test_double_dqn_selection_differs_from_dqn():
    online=torch.tensor([3.,1.,4.,2.]);target=torch.tensor([1.,5.,2.,6.]);offsets=torch.tensor([0,2,4])
    assert q_bootstrap(online,target,offsets,False).tolist()==[5.,6.]
    assert q_bootstrap(online,target,offsets,True).tolist()==[1.,2.]

def test_classic_fast_batch_matches_full_collation():
    from types import SimpleNamespace
    from agent.graph_layers import collate
    c=config(True);c['method']='dqn';p=Policy(c);o=observe(tiny());obs=[o,o]
    old=collate([v.base for v in obs],flags=SimpleNamespace(action=False,dual=False));old.data['extra']=torch.tensor(np.concatenate([v.extra for v in obs]))
    a=p(old);b=p(p.batch(obs));torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0);torch.testing.assert_close(a.value,b.value,rtol=0,atol=0)

def test_node_relabeling_and_hidden_future():
    c=config(True);o=observe(tiny());p=Policy(c)
    with torch.no_grad():p.delta.weight.normal_(std=.05)
    order=np.arange(len(o.order))[::-1];op=np.arange(len(o.operation))[::-1];machine=np.arange(len(o.machine))[::-1]
    io=np.argsort(order);ip=np.argsort(op);im=np.argsort(machine)
    candidate=o.candidate.copy();live=candidate[:,0]>=0
    candidate[live]=np.column_stack((ip[candidate[live,0]],im[candidate[live,1]],io[candidate[live,2]]))
    base=replace(o.base,order=o.order[order],operation=o.operation[op],machine=o.machine[machine],owner=io[o.owner[op]],
                 candidate=candidate,precedence=ip[o.precedence],eligibility=np.column_stack((ip[o.eligibility[:,0]],im[o.eligibility[:,1]])))
    other=replace(o,base=base,local_op=ip[o.local_op])
    torch.testing.assert_close(p(p.batch([o])).logits,p(p.batch([other])).logits,atol=2e-6,rtol=1e-5)
    e=SchedulingEnv(sample(c,np.random.default_rng(71)),environment_config(c));before=observe(e)
    hidden=e.status==0
    # Status constants, not an assumption about the numeric code.
    from environment.env import NOT_ARRIVED
    hidden=e.status==NOT_ARRIVED;e.inst.arrival_times[hidden]+=1000;e.inst.due_dates[hidden]+=5000
    after=observe(e)
    for k in ('extra','keys','condition','local_action','local_op'):np.testing.assert_array_equal(getattr(before,k),getattr(after,k))

def test_scenarios_shared_isolated_and_same_action_zero():
    c=config(True);e=tiny();o=observe(e);p=Policy(c);a=RulePolicy('SPT').act(o)[0]
    before=deepcopy(e.state_dict());ev=ScenarioEvaluator(c,4)
    r=ev.evaluate(e.public_state(),o,p,10000,'SPT',0,[a,a])
    assert r.complete and r.mean==0 and r.reliability==0
    assert all(x[0]==x[1]==h for x,h in zip(r.branch_hashes,r.scene_hashes))
    for k,v in before['dynamic'].items():np.testing.assert_equal(v,e.state_dict()['dynamic'][k])
    ev1=ScenarioEvaluator(c,4);ev2=ScenarioEvaluator(c,4);b=next(i for i in range(len(o.candidate)) if i!=a)
    x=ev1.evaluate(e.public_state(),o,p,10000,'SPT',0,[a,b]);y=ev2.evaluate(e.public_state(),o,p,10000,'SPT',0,[b,a])
    assert x.mean==pytest.approx(-y.mean) and x.scene_hashes==y.scene_hashes


def test_candidates_use_live_policy_and_continuation_stays_frozen():
    c=config(True);e=tiny();o=observe(e);frozen=Policy(c)
    anchor=RulePolicy('SPT').act(o)[0]
    challenger=next(i for i,a in enumerate(o.candidate) if a[0]>=0 and i!=anchor)
    frozen.act=lambda obs:(anchor,0.,0.)
    class CurrentPolicy:
        def act(self,obs):return challenger,0.,0.
    # A one-step allowance suffices to inspect selection; the unfinished branches
    # incur cost but cannot enter replay as valid labels.
    label=ScenarioEvaluator(c,44).evaluate(e.public_state(),o,frozen,1,'SPT',0,candidate_policy=CurrentPolicy())
    assert label.actions==(anchor,challenger)
    assert label.candidate_policy_hash=='CurrentPolicy'
    assert not label.complete and label.steps==1
    assert label.continuations.count('SPT')==label.continuations.count('frozen_policy')==1

def test_preferences_expire_and_rollback(monkeypatch):
    import agent.ppo as ppo  # the transactional step (shared by PPO and preference updates) lives there
    c=config(True);o=observe(tiny());p=Policy(c);opt=torch.optim.Adam(p.parameters(),lr=.001)
    label=ScenarioPreference((0,1),np.array([[0.,1.],[0.,1.]]),1.,0.,.9,.9,1,8,0.,True,'test',[],[],'',[],[])
    replay=PreferenceReplay(4,2);replay.add(o,label);replay.expire(2);assert len(replay.items)==1
    before=deepcopy(p.state_dict());monkeypatch.setattr(ppo,'categorical_kl',lambda *args:torch.tensor(1.))
    m=preference_update(p,opt,replay,c,np.random.default_rng(1),2);assert m['preference_rejected']==1 and not opt.state
    for k,v in before.items():torch.testing.assert_close(v,p.state_dict()[k],rtol=0,atol=0)
    replay.expire(3);assert not replay.items

@pytest.mark.parametrize('method',['hgnn','full','dqn','ddqn','a2c','ppo'])
def test_exact_resume_and_budget(tmp_path,method):
    from agent.training import train
    c=config(True);c['method']=method;c['demonstration'].update(steps=8,epochs=1)
    c['training'].update(milestones=[16,32],evaluate_every=32,rollout_steps=8)
    c['classic'].update(start=8,batch=4,train_every=4)
    torch.set_num_threads(1)
    train(c,tmp_path/'whole',3,32,tmp_path/'data')
    train(c,tmp_path/'part',3,16,tmp_path/'data');train(c,tmp_path/'part',3,32,tmp_path/'data')
    a=torch.load(tmp_path/'whole/checkpoint_last.pt',weights_only=False);b=torch.load(tmp_path/'part/checkpoint_last.pt',weights_only=False)
    assert a['steps']==b['steps']==32
    assert a['real_steps']+a['scenario_steps']+a['demonstrations'].steps+a['lost_upper_bound']==32
    for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
    bad=deepcopy(c);bad['training']['learning_rate']*=2
    with pytest.raises(ValueError,match='incompatible'):train(bad,tmp_path/'part',3,32,tmp_path/'data')


def test_recovery_landing_on_milestone_still_writes_its_checkpoint(tmp_path):
    import json
    from agent.training import train
    c=config(True);c['method']='hgnn';c['demonstration'].update(steps=8,epochs=1)
    c['training'].update(milestones=[16,32],evaluate_every=32,rollout_steps=8);torch.set_num_threads(1)
    run=tmp_path/'run';train(c,run,3,8,tmp_path/'data')
    assert (run/'checkpoint_8.pt').exists()  # budget reached before any update loop
    # Simulate a crash after reserving up to the 16-interaction milestone: the reservation is charged as lost.
    json_path=run/'budget.json';ledger=json.loads(json_path.read_text());ledger['charged_upper_bound']=16;json_path.write_text(json.dumps(ledger))
    train(c,run,3,32,tmp_path/'data')
    milestone=torch.load(run/'checkpoint_16.pt',weights_only=False);last=torch.load(run/'checkpoint_last.pt',weights_only=False)
    assert milestone['steps']==16 and last['lost_upper_bound']==8 and last['steps']==32
    before=torch.load(run/'checkpoint_8.pt',weights_only=False)['model']
    for k in before:torch.testing.assert_close(before[k],milestone['model'][k],rtol=0,atol=0)
