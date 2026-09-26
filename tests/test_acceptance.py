"""Acceptance failures that must never turn into a reusable success marker."""
from copy import deepcopy
from dataclasses import replace
import json
import multiprocessing as mp
import os
import socket
import time

import numpy as np
import pytest
import torch

from configs.experiment import config, matrix
from result.storage import atomic_json, digest, run_lock


def _count_process(root, count):
    from result.engineering import InteractionCounter
    counter = InteractionCounter(root, 'child')
    for _ in range(count):
        counter.charge()
    counter.close()


def _hold_lock(root, ready, release):
    with run_lock(root):
        ready.set();release.wait(20)


def _train_until_reserved(run,data,ready):
    import agent.training as training
    from result.engineering import counted
    original=training.atomic_json
    def intercept(path,value):
        original(path,value)
        if str(path).endswith('budget.json') and value['committed']>=32 and value['charged_upper_bound']>value['committed']:
            ready.set();time.sleep(60)
    training.atomic_json=intercept
    c=config(True);c['method']='hgnn';c['training']['milestones']=[32,64,128]
    with counted('fault-injection:terminated-child'):
        training.train(c,run,9,128,data)


def test_unlimited_accounting_preserves_history_and_counts_spawn(tmp_path):
    from result.engineering import InteractionCounter, summarize
    old = b'{"limit":50000,"charged_upper_bound":50001,"committed":50000,"sessions":[]}'
    (tmp_path/'interaction_ledger.json').write_bytes(old)
    parent = InteractionCounter(tmp_path, 'parent')
    parent.charge()
    ctx = mp.get_context('spawn')
    children = [ctx.Process(target=_count_process, args=(tmp_path, n)) for n in (131, 259)]
    for child in children: child.start()
    for child in children:
        child.join(30)
        assert child.exitcode == 0
    parent.close()
    result = summarize(tmp_path)
    assert result['limit'] is None
    assert result['committed'] == 50000+1+131+259
    assert result['charged_upper_bound'] == 50001+1+131+259
    assert (tmp_path/'accounting/legacy.json').read_bytes() == old
    assert summarize(tmp_path) == result


def test_unclosed_counter_retains_crash_reservation(tmp_path):
    from result.engineering import InteractionCounter, summarize
    counter = InteractionCounter(tmp_path)
    counter.charge()
    result = summarize(tmp_path)
    assert result['charged_upper_bound'] == 128 and result['committed'] == 0
    counter.close()
    assert summarize(tmp_path)['charged_upper_bound'] == 1


def test_completed_worker_rejects_missing_asset_and_changed_checkpoint(tmp_path):
    from scripts.pipeline import prepare_study, worker
    from result.acceptance import verify_job, verify_assets
    c = config(True)
    spec = matrix(c)[0]
    root, data = tmp_path/'r', tmp_path/'d'
    prepare_study(c, root, data, [spec])
    worker(spec, root, data)
    verify_job(root/'runs'/spec.name, spec)
    cp = root/'runs'/spec.name/f'checkpoint_{spec.budget}.pt'
    state = torch.load(cp, weights_only=False)
    state['model']['actor.bias'] += 1
    torch.save(state, cp)
    with pytest.raises(ValueError, match='changed'):
        worker(spec, root, data)
    assets = tmp_path/'assets'; assets.mkdir()
    p = assets/'plot.png'; p.write_bytes(b'png')
    atomic_json(assets/'manifest.json', dict(assets={'plot.png':digest(p)}))
    verify_assets(assets)
    p.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_assets(assets)


def test_recovery_at_final_budget_commits_lost_charge(tmp_path):
    from agent.training import train
    c=config(True); c['method']='hgnn'; c['training'].update(milestones=[16,32],rollout_steps=8)
    run=tmp_path/'run'
    train(c,run,3,16,tmp_path/'data')
    atomic_json(run/'budget.json',dict(charged_upper_bound=32,committed=16,limit=32))
    out=train(c,run,3,32,tmp_path/'data')
    state=torch.load(run/'checkpoint_last.pt',weights_only=False)
    ledger=json.loads((run/'budget.json').read_text())
    assert out['steps']==state['steps']==ledger['committed']==32
    assert state['lost_upper_bound']==16


def test_stale_lock_recovered_and_foreign_owner_rejected(tmp_path):
    atomic_json(tmp_path/'.lock',dict(pid=2147483647,host=socket.gethostname(),token='stale'))
    with run_lock(tmp_path):
        assert (tmp_path/'abandoned_lock_stale.json').exists()
    atomic_json(tmp_path/'.lock',dict(pid=os.getpid(),host='different-host',token='foreign'))
    with pytest.raises(RuntimeError,match='owned'):
        with run_lock(tmp_path): pass


def test_disk_failure_preserves_existing_checkpoint(tmp_path,monkeypatch):
    from result.storage import disk_check
    import shutil
    cp=tmp_path/'checkpoint.pt';cp.write_bytes(b'durable')
    monkeypatch.setattr(shutil,'disk_usage',lambda p: type('Usage',(),{'free':1})())
    with pytest.raises(OSError,match='Insufficient'):
        disk_check(tmp_path,100)
    assert cp.read_bytes()==b'durable'


def test_five_seed_statistics_known_effects_and_invalid_pairs():
    from result.statistics import interval, paired_values
    rows=[dict(variant=m,seed=str(s),instance_id=i,eta=str(s*.1+(0.2 if m=='full' else 0)))
          for m in ('full','base') for s in range(1,6) for i in ('a','b')]
    seeds,ids,a,b=paired_values(rows,'full','base')
    np.testing.assert_allclose(a-b,.2)
    lo,hi=interval((a-b).mean(1),np.random.default_rng(1),2000)
    assert lo==pytest.approx(.2) and hi==pytest.approx(.2)
    lo,hi=interval(np.zeros(5),np.random.default_rng(1),2000)
    assert lo==hi==0
    rules=[dict(variant='SPT',seed='0',instance_id=i,eta='.2') for i in ids]
    _,_,a,b=paired_values(rows+rules,'full','SPT')
    assert a.shape==(5,2) and b.shape==(1,2)
    for broken in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='Missing|Duplicate|Unmatched'):
            paired_values(broken,'full','base')


def test_spawn_failure_stops_submission_and_can_resume(tmp_path):
    from scripts.pipeline import prepare_study, execute
    c=config(True);specs=matrix(c)[:3]
    broken=deepcopy(specs[1].config);broken['recording']['keep_recovery_copies']=0
    selected=[specs[0],replace(specs[1],config=broken),specs[2]]
    root,data=tmp_path/'r',tmp_path/'d';prepare_study(c,root,data,selected)
    with pytest.raises(ValueError,match='keep_recovery_copies'):
        execute(selected,root,data,2,selected)
    assert not (root/'runs'/specs[2].name/'status.json').exists()
    assert not list((root/'runs').glob('*/.lock'))
    result=execute(specs,root,data,2,specs)
    assert len(result)==3 and all(r['status']=='complete' for r in result)
    assert not any(child.is_alive() for child in mp.active_children())


def test_real_process_lock_conflict_and_terminated_training_recovery(tmp_path):
    from agent.training import train
    from result.recording import verify
    ctx=mp.get_context('spawn');ready=ctx.Event();release=ctx.Event()
    child=ctx.Process(target=_hold_lock,args=(tmp_path/'lock',ready,release));child.start()
    try:
        assert ready.wait(20)
        with pytest.raises(RuntimeError,match='owned'):
            with run_lock(tmp_path/'lock'):pass
    finally:
        release.set();child.join(20)
        if child.is_alive():child.terminate();child.join(10)
    assert child.exitcode==0
    ready=ctx.Event();run=tmp_path/'run';data=tmp_path/'data'
    child=ctx.Process(target=_train_until_reserved,args=(run,data,ready));child.start()
    try:
        assert ready.wait(30)
    finally:
        child.terminate();child.join(10)
    reservation=json.loads((run/'budget.json').read_text())
    prior=torch.load(run/'checkpoint_last.pt',weights_only=False)
    assert reservation['charged_upper_bound']>prior['steps']
    c=config(True);c['method']='hgnn';c['training']['milestones']=[32,64,128]
    out=train(c,run,9,128,data)
    last=torch.load(run/'checkpoint_last.pt',weights_only=False)
    assert out['steps']==last['steps']==128
    assert last['lost_upper_bound']==reservation['charged_upper_bound']-prior['steps']
    verify(run/'raw')


def test_uncommitted_final_model_is_replaced_from_durable_recovery(tmp_path):
    from agent.training import train
    from result.storage import state_hash
    c=config(True);c['method']='hgnn';c['training'].update(milestones=[16,32],rollout_steps=8)
    run=tmp_path/'run';train(c,run,3,16,tmp_path/'data')
    uncommitted=torch.load(run/'checkpoint_16.pt',weights_only=False)
    uncommitted['steps']=32;uncommitted['model']['actor.bias']+=1
    torch.save(uncommitted,run/'checkpoint_32.pt')
    torch.save(uncommitted,run/'checkpoint_best.pt')
    atomic_json(run/'budget.json',dict(charged_upper_bound=32,committed=16,limit=32))
    train(c,run,3,32,tmp_path/'data')
    final=torch.load(run/'checkpoint_32.pt',weights_only=False)
    recovery=torch.load(run/'checkpoint_last.pt',weights_only=False)
    assert state_hash(final['model'])==state_hash(recovery['model'])
    best=torch.load(run/'checkpoint_best.pt',weights_only=False)
    assert state_hash(best['model'])==state_hash(recovery['best_checkpoint']['model'])
    assert list(run.glob('uncommitted_*_checkpoint_32.pt'))


def test_acceptance_paths_separate_runtime_and_concurrency(monkeypatch):
    import scripts.pipeline as pipeline
    from result.provenance import runtime_info
    runtime=runtime_info();monkeypatch.setattr(pipeline,'runtime_info',lambda:runtime)
    first=pipeline.paths(True,1)[0]
    assert first!=pipeline.paths(True,8)[0]
    runtime['packages']['torch']='changed-version'
    assert first!=pipeline.paths(True,1)[0]


def test_bootstrap_honors_configured_threads_over_inherited_environment():
    import subprocess,sys
    from configs.experiment import ROOT
    env=dict(os.environ,OMP_NUM_THREADS='9',MKL_NUM_THREADS='9')
    code="import scripts._bootstrap; import os; assert os.environ['OMP_NUM_THREADS']=='1'; assert os.environ['MKL_NUM_THREADS']=='1'"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)


def test_study_freeze_allows_operational_changes_but_rejects_training_changes(tmp_path):
    from scripts.pipeline import prepare_study
    c=config(True);root,data=tmp_path/'r',tmp_path/'d'
    prepare_study(c,root,data,matrix(c))
    cp=root/'runs/full_s1/checkpoint_last.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'present')
    c['runtime']['jobs']=1
    prepare_study(c,root,data,matrix(c))
    c['training']['learning_rate']*=2
    with pytest.raises(ValueError,match='Frozen'):
        prepare_study(c,root,data,matrix(c))


@pytest.mark.skipif(os.name!='nt',reason='Windows delete-sharing semantics')
def test_atomic_json_retries_transient_windows_reader(tmp_path,monkeypatch):
    import result.storage as storage
    original=storage.os.replace;attempts=[]
    def sharing_violation(source,destination):
        attempts.append(1)
        if len(attempts)<3:
            error=PermissionError('temporarily open reader');error.winerror=32;raise error
        return original(source,destination)
    monkeypatch.setattr(storage.os,'replace',sharing_violation)
    atomic_json(tmp_path/'value.json',{'value':1})
    assert len(attempts)==3 and json.loads((tmp_path/'value.json').read_text())=={'value':1}
