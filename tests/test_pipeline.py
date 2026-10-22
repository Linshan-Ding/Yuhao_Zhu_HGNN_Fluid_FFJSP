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
    c=config()
    with pytest.raises(ValueError,match='resolve_teacher'):matrix(c)
    c['teacher']='SPT';jobs=matrix(c)
    assert len(jobs)==73 and sum(j.budget for j in jobs)==64000000
    assert len({j.name for j in jobs})==73
    pilot=matrix(c,'pilot')
    assert [j.name for j in pilot]==[f'{m}_s{s}' for m in c['pilot']['methods'] for s in (1,2,3)] and sum(j.budget for j in pilot)==9000000
    assert all(j.config['training']['milestones'][-1]==500000 and j.group==('main' if j.method=='full' else 'comparators') for j in pilot)
    ids=[f'CS{n}-R1.3-K{k}-r{r}' for n in (24,36) for k in ('1.5','2.0') for r in (1,2,3)]
    single=matrix(c,'instance',ids)
    assert len(single)==216 and sum(j.budget for j in single)==21600000 and single[0].name=='full_CS24-R1.3-K1.5-r1_s1'
    assert all(j.instance==j.config['instance'] and j.group=='instance' and j.config['training']['evaluate_every']==5000 and
               j.config['training']['milestones']==[10000,30000,50000,100000] and j.config['demonstration']['steps']==2000 for j in single)
    assert all(set(j.name)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._') for j in single)
    with pytest.raises(ValueError,match='fixed instances'):matrix(c,'instance')
    for j in jobs:
        if j.group=='sensitivity':
            assert sum(j.config['scenario'][k]!=c['scenario'][k] for k in c['scenario'])==1
    other=deepcopy(c);other['runtime']['jobs']=1
    assert identity(c)==identity(other)
    other['training']['learning_rate']*=2
    assert identity(c)!=identity(other)


def test_fixed_cells_shared_prefix_replicates_and_alias(tmp_path):
    c=config();m=prepare(c,tmp_path);assert len(m['cases'])==len({r['parameter_hash'] for r in m['cases']})==102
    assert {k:len(v) for k,v in m['splits'].items()}==dict(validation=6,main=12,small=12,long_stream=12,large_shop=8,
        contested_validation=4,contested=36,contested_small=12,sensitivity=36)
    assert m['splits']['contested']==m['splits']['sensitivity']
    contested=[r for r in m['cases'] if r['role']=='contested'];by={(r['rho_target'],r['due_factor'],r['orders'],r['replicate']):r for r in contested}
    assert all(abs(float(r['rho_sys'])-float(r['rho_target']))<1e-9 for r in contested) and len(by)==36
    x=fixtures(c,tmp_path,'contested');inst={i.instance_id:i for i in x}
    a=inst[by[1.0,1.5,48,0]['instance_id']];b=inst[by[1.0,1.5,96,0]['instance_id']];d=inst[by[1.6,1.5,48,0]['instance_id']];e=inst[by[1.0,1.5,48,1]['instance_id']]
    np.testing.assert_array_equal(a.order_product,b.order_product[:48]);np.testing.assert_array_equal(a.arrival_times,b.arrival_times[:48])
    np.testing.assert_allclose(a.arrival_times*1.0,d.arrival_times*1.6);assert not np.array_equal(a.arrival_times,e.arrival_times)
    # Readable ids; legacy cells keep their hash ids as aliases (the published results refer to them) and their files.
    assert by[1.0,1.5,48,0]['instance_id']=='C48-R1.0-K1.5-r1' and by[1.6,2.0,96,2]['instance_id']=='C96-R1.6-K2.0-r3'
    names={r['instance_id'] for r in m['cases']};assert len(names)==102 and {'M48-I1.5-K3.0','S12-I0.6-K1.5','H7-192-I2.5-K1.5','CV72-R1.15-K1.75-r2','CS36-R1.3-K2.0-r1'}<names
    legacy=json.loads((__import__('pathlib').Path(__file__).resolve().parents[1]/'data/instances/fixed/manifest.json').read_text())
    aliases={r['legacy_id']:r for r in m['cases']}
    assert all(i in aliases and aliases[i]['parameter_hash']==x['parameter_hash'] for i,x in ((r['instance_id'],r) for r in legacy['cases']))
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
    c=config(True);i=Instance('partial','test',2,2,(2,1),np.array([[2,10,0],[0,0,6],[2,10,0],[0,0,4]],np.float32),
        np.array([0,1]),np.array([0.,0.]),np.array([8.,7.5]),dict(DDT=10,iota=1,rho_sys=1,due_factor=2))
    r=Recorder(tmp_path,c);e=SchedulingEnv(i,environment_config(c)).attach_recorder(r)
    e.step(Dispatch(0,0));assert e.stage[0]==0 and e.machine_busy_with[0]==0
    # Order 1 may hold for machine 0 (free at 2, then 2 units) instead of the idle slow machine 1 (10 units).
    assert e.awaited()[1:4]==(1,2,0)
    e.step(Wait());assert e.now==2 and e.order_outcome[1]==0 and e.machine_busy_time[0]==2  # 2+6>7.5: holding made order 1 hopeless
    while not e.done:
        o=observe(e);e.step(o.actions.actions[RulePolicy('SPT').act(o)[0]])
    r.commit(r.snapshot());verify(tmp_path)
    assert sum(x['reward'] for x in records(tmp_path,'decisions'))==e.eta
    events=list(records(tmp_path,'events'))
    assert any(x['kind']=='order_resolved' and x.get('reason')=='hopeless_remaining_route' for x in events)
    starts=[x for x in events if x['kind']=='operation_start'];assert len(starts)==2
    assert len(list(records(tmp_path,'orders')))==2
    restored=SchedulingEnv.from_state_dict(e.state_dict());np.testing.assert_array_equal(restored.machine_busy_time,e.machine_busy_time)


def test_nonpreemption_wait_and_public_isolation():
    c=config(True);i=Instance('running','test',1,1,(2,),np.array([[10,2]],np.float32),np.array([0,0,0]),
        np.array([0.,0.,30.]),np.array([10.,40.,70.]),{'DDT':10})
    e=SchedulingEnv(i,environment_config(c));e.step(Dispatch(0,1));o=observe(e)
    # Waiting is offered because machine 1 (free at 2, then 2 units) beats the idle machine 0 (10 units) for order 1.
    assert o.candidate[-1,0]<0 and o.candidate_features[-1,5]==pytest.approx(4/e.t_ref) and o.candidate_features[-1,6]==pytest.approx(6/e.p_ref)
    with pytest.raises(ValueError):e.step(Dispatch(1,1))   # busy machine
    e.step(Wait());assert e.now==2 and e.machine_busy_with[1]<0 and e.n_completed==1   # held until the release, never preempted
    before=observe(e);e.inst.arrival_times[2]+=900;e.inst.due_dates[2]+=1000;after=observe(e)
    for key in ('order','operation','machine','candidate','candidate_features','extra','global_features'):np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
    assert not before.order.flags.writeable
    e.step(Dispatch(1,1));assert e.now==930 and e.n_completed==2   # the (shifted) order 2 arrives with both machines idle
    with pytest.raises(ValueError,match='admissible'):e.step(Wait())


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
    c=config();root=__import__('pathlib').Path(__file__).resolve().parents[1]/'data/instances/fixed2'
    i=next(i for i in fixtures(c,root,'large_shop') if i.stage_count==7)
    i=replace(i,arrival_times=np.zeros(i.order_count),due_dates=np.full(i.order_count,100000.))
    e=SchedulingEnv(i,environment_config(c));o=observe(e);p=Policy(c)
    with torch.no_grad():out=p(p.batch([o]));assert torch.isfinite(out.logits).all()
    d=deepcopy(c);d['method']='no_graph';q=Policy(d);assert not q.local
    d['method']='no_scenario';assert Policy(d).local


def test_entry_points_and_code_only_export():
    from configs.experiment import ROOT
    names=['run_00_smoke','run_01_prepare_data','run_02_train_main','run_03_train_comparators','run_04_train_sensitivity',
           'run_05_evaluate','run_06_exact_reference','run_07_statistics','run_08_export','run_formal_all','run_pilot','run_instance_study']
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
