from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from configs.experiment import config, matrix, environment_config, identity
from data.benchmark import prepare, fixtures
from data.generator import Instance
from environment.public import SchedulingEnv
from environment.interfaces import Dispatch, Wait
from agent.observation import observe
from agent.model import Policy
from agent.rules import RulePolicy
from result.recording import Recorder, records, verify, split_time
from result.storage import run_lock, digest


def test_formal_matrix_and_one_factor_sensitivity():
    c=config();jobs=matrix(c)
    assert len(jobs)==73 and sum(j.budget for j in jobs)==64000000
    assert len({j.name for j in jobs})==73
    for j in jobs:
        if j.group=='sensitivity':
            assert sum(j.config['scenario'][k]!=c['scenario'][k] for k in c['scenario'])==1
    other=deepcopy(c);other['runtime']['jobs']=1
    assert identity(c)==identity(other)
    other['training']['learning_rate']*=2
    assert identity(c)!=identity(other)


def test_fifty_unique_cells_shared_prefix_and_alias(tmp_path):
    c=config();m=prepare(c,tmp_path);assert len(m['cases'])==len({r['parameter_hash'] for r in m['cases']})==50
    assert {k:len(v) for k,v in m['splits'].items()}==dict(validation=6,main=12,small=12,long_stream=12,large_shop=8,sensitivity=12)
    assert m['splits']['main']==m['splits']['sensitivity']
    inst=fixtures(c,tmp_path,'main');indexed={(i.order_count,round(i.meta['iota'],1),i.meta['due_factor']):i for i in inst}
    a=indexed[48,.6,1.5];b=indexed[96,.6,1.5];d=indexed[48,1.5,1.5];e=indexed[48,.6,3.]
    np.testing.assert_array_equal(a.proc_times,b.proc_times);np.testing.assert_array_equal(a.order_product,b.order_product[:48])
    np.testing.assert_array_equal(a.arrival_times,b.arrival_times[:48]);np.testing.assert_allclose(a.arrival_times*.6,d.arrival_times*1.5)
    np.testing.assert_array_equal(a.arrival_times,e.arrival_times);np.testing.assert_allclose(e.due_dates-e.arrival_times,2*(a.due_dates-a.arrival_times))
    assert all(.3<=r['rho_sys']<=1.8 for r in m['cases'])
    assert prepare(c,tmp_path)==m
    p=tmp_path/m['cases'][0]['file'];p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):prepare(c,tmp_path)


def test_recorder_partial_abandon_and_reward_reconstruction(tmp_path):
    c=config(True);i=Instance('partial','test',2,2,(1,1),np.array([[2,0],[0,6],[2,0],[0,4]],np.float32),
        np.array([0,1]),np.array([0.,0.]),np.array([8.,20.]),dict(DDT=10,iota=1,rho_sys=1,due_factor=2))
    r=Recorder(tmp_path,c);e=SchedulingEnv(i,environment_config(c)).attach_recorder(r)
    e.step(Dispatch(0,0));assert e.stage[0]==1
    e.step(Wait());assert e.order_outcome[0]==0 and e.machine_busy_time[0]==2
    while not e.done:
        o=observe(e);e.step(o.actions.actions[RulePolicy('SPT').act(o)[0]])
    r.commit(r.snapshot());verify(tmp_path)
    assert sum(x['reward'] for x in records(tmp_path,'decisions'))==e.eta
    events=list(records(tmp_path,'events'))
    assert any(x['kind']=='order_resolved' and x.get('reason')=='hopeless_remaining_route' for x in events)
    starts=[x for x in events if x['kind']=='operation_start'];assert len(starts)==3
    assert len(list(records(tmp_path,'orders')))==2
    restored=SchedulingEnv.from_state_dict(e.state_dict());np.testing.assert_array_equal(restored.machine_busy_time,e.machine_busy_time)


def test_nonpreemption_wait_and_public_isolation():
    c=config(True);i=Instance('running','test',1,1,(2,),np.array([[10,10]],np.float32),np.array([0,0,0]),
        np.array([0.,0.,30.]),np.array([10.,40.,70.]),{'DDT':10})
    e=SchedulingEnv(i,environment_config(c));e.step(Dispatch(0,0));o=observe(e)
    e.step(Wait());assert e.machine_busy_with[0]==0 and e.machine_free_at[0]==10
    before=observe(e);e.inst.arrival_times[2]+=900;e.inst.due_dates[2]+=1000;after=observe(e)
    for key in ('order','operation','machine','candidate','extra','global_features'):np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
    assert not before.order.flags.writeable
    with pytest.raises(ValueError):e.step(Dispatch(1,0))


def test_phase_time_is_split_at_boundaries():
    assert split_time(1,12,10,.2)==[('startup',1.),('arrivals',8.),('drain',2.)]


def test_raw_corruption_and_lock(tmp_path):
    c=config(True);r=Recorder(tmp_path/'raw',c);r.emit('test',dict(x=1));r.commit(r.snapshot());verify(tmp_path/'raw')
    manifest=json.loads((tmp_path/'raw/manifest.json').read_text());p=tmp_path/'raw'/manifest['chunks'][0]['file'];p.write_bytes(b'broken')
    with pytest.raises(ValueError):verify(tmp_path/'raw')
    with run_lock(tmp_path/'job'):
        with pytest.raises(RuntimeError,match='owned'): 
            with run_lock(tmp_path/'job'):pass


def test_ppo_rollout_rollback_restores_logs_and_optimizer(monkeypatch):
    import agent.ppo as ppo
    from data.online import sample
    c=config(True);c['training'].update(epochs=1,minibatch=16)
    e=SchedulingEnv(sample(c,np.random.default_rng(2)),environment_config(c));o=observe(e);p=Policy(c)
    a,lp,value=p.act(o);before=deepcopy(p.state_dict())
    rows=[dict(obs=o,action=a,logp=lp,value=value,reward=.2,stream=0,terminated=True,truncated=False,boundary=True,scenario=None) for _ in range(4)]
    actual=ppo.categorical_kl;calls=[]
    def kl(*args):
        calls.append(1);return torch.tensor(10.) if len(calls)==2 else actual(*args)
    monkeypatch.setattr(ppo,'categorical_kl',kl);opt=torch.optim.Adam(p.parameters(),lr=.0001)
    result=ppo.update(p,opt,rows,{0:0.},c,np.random.default_rng(1),'cpu')
    assert result['rollout_rollback']==1 and result['effective_optimizer_steps']==0 and not opt.state
    for key in before:torch.testing.assert_close(before[key],p.state_dict()[key],rtol=0,atol=0)
    assert result['kl']==pytest.approx(0.,abs=1e-7)


def test_full_size_graph_and_zero_module_boundaries():
    c=config();root=__import__('pathlib').Path(__file__).resolve().parents[1]/'data/instances/fixed'
    i=next(i for i in fixtures(c,root,'large_shop') if i.stage_count==7)
    i=replace(i,arrival_times=np.zeros(i.order_count),due_dates=np.full(i.order_count,100000.))
    e=SchedulingEnv(i,environment_config(c));o=observe(e);p=Policy(c)
    with torch.no_grad():out=p(p.batch([o]));assert torch.isfinite(out.logits).all()
    d=deepcopy(c);d['method']='no_graph';q=Policy(d);assert not q.local
    d['method']='no_scenario';assert Policy(d).local


def test_entry_points_and_code_only_export():
    from configs.experiment import ROOT
    names=['run_00_smoke','run_01_prepare_data','run_02_train_main','run_03_train_comparators','run_04_train_sensitivity',
           'run_05_evaluate','run_06_exact_reference','run_07_statistics','run_08_export','run_formal_all']
    for name in names:assert (ROOT/'scripts'/f'{name}.py').is_file()
    for name in ('result/export.py','scripts/pipeline.py'):
        assert 'Yuhao_Zhu_FFJSP_Order_GNN_Fluid_model' not in (ROOT/name).read_text(encoding='utf-8')


def test_evaluation_cache_reuses_and_detects_corruption(tmp_path):
    from result.evaluation import cached_case
    c=config(True);data=tmp_path/'d';manifest=prepare(c,data);i=fixtures(c,data,'main')[0]
    sha=next(r['sha256'] for r in manifest['cases'] if r['instance_id']==i.instance_id)
    root=tmp_path/'r';policy=RulePolicy('SPT')
    first=cached_case(root,i,policy,'rule:SPT',c,sha,panel=True)
    assert first==cached_case(root,i,policy,'rule:SPT',c,sha,panel=True)
    cache=next((root/'evaluations').iterdir());assert len(cache.name)==24
    (cache/first['panel']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='panel'):cached_case(root,i,policy,'rule:SPT',c,sha,panel=True)


def test_identical_policy_reuses_despite_checkpoint_metadata():
    from result.evaluation import policy_identity
    c=config(True);p=Policy(c);state=dict(cfg=c,model=p.state_dict(),steps=64)
    other=deepcopy(state);other.update(steps=128,eta_validation=.8)
    assert policy_identity(state)==policy_identity(other)
    other['model']['actor.bias']+=1
    assert policy_identity(state)!=policy_identity(other)


def test_hash_is_line_ending_invariant(tmp_path):
    from result.provenance import hash_files
    from configs.experiment import ROOT
    lf=tmp_path/'a.csv';crlf=tmp_path/'b.csv';lf.write_bytes(b'x,y\n1,2\n');crlf.write_bytes(b'x,y\r\n1,2\r\n')
    assert digest(lf)==digest(crlf)
    binary=tmp_path/'a.pt';binary.write_bytes(b'\r\n');other=tmp_path/'b.pt';other.write_bytes(b'\n')
    assert digest(binary)!=digest(other)
    src=ROOT/'.pytest_tmp'/'eol_probe';src.mkdir(parents=True,exist_ok=True)
    try:
        (src/'m.py').write_bytes(b'x=1\r\n');a=hash_files([src/'m.py']);(src/'m.py').write_bytes(b'x=1\n')
        assert hash_files([src/'m.py'])==a
    finally:
        (src/'m.py').unlink();src.rmdir()
