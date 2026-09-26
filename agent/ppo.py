"""PPO with transactional minibatch and whole-rollout KL checks over the full legal-action distribution."""
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from agent.returns import advantages

def categorical_kl(old,new,offsets):
    values=[]
    for a,b in zip(offsets[:-1],offsets[1:]):
        values.append((old[a:b].exp()*(old[a:b]-new[a:b])).sum())
    return torch.stack(values).mean()

def transactional_step(policy,optimizer,loss,max_grad_norm,batch,old_logp,limit):
    """One optimizer step that is rolled back (weights and optimizer state) when the mean full-action KL
    from `old_logp` to the updated distribution is not finite or exceeds `limit`. Returns (accepted, kl, gradient norm)."""
    before=deepcopy(policy.state_dict());before_opt=deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),max_grad_norm));optimizer.step()
    with torch.no_grad():kl=float(categorical_kl(old_logp,policy(batch).logp,batch.offsets))
    accepted=bool(np.isfinite(kl) and kl<=limit)
    if not accepted:policy.load_state_dict(before);optimizer.load_state_dict(before_opt)
    return accepted,kl,norm

def update(policy,optimizer,records,bootstrap,cfg,rng,device):
    tc=cfg['training'];adv,ret=advantages(records,bootstrap,tc['gae_lambda'],tc['gamma'])
    adv=(adv-adv.mean())/max(float(adv.std()),1e-6)
    baseline=deepcopy(policy.state_dict());opt_baseline=deepcopy(optimizer.state_dict())
    full=policy.batch([r['obs'] for r in records])
    with torch.no_grad(): old_full=policy(full).logp.detach()
    old_offsets=full.offsets.cpu().tolist();rows=[];rejected=0;accepted=0;gradient_rows=[];rollout_rollback=0
    policy.train();stop=False
    for _ in range(tc['epochs']):
        for ids in np.array_split(rng.permutation(len(records)),max(1,int(np.ceil(len(records)/tc['minibatch'])))):
            rs=[records[int(i)] for i in ids];batch=policy.batch([r['obs'] for r in rs]);out=policy(batch)
            old_distribution=torch.cat([old_full[old_offsets[i]:old_offsets[i+1]] for i in ids])
            ix=batch.offsets[:-1]+torch.tensor([r['action'] for r in rs],device=device)
            old=out.logp.new_tensor([r['logp'] for r in rs]);logratio=out.logp[ix]-old;ratio=logratio.exp()
            aa=out.logp.new_tensor(adv[ids]);rr=out.value.new_tensor(ret[ids])
            pg=-torch.minimum(ratio*aa,ratio.clamp(1-tc['clip'],1+tc['clip'])*aa).mean()
            vl=.5*F.mse_loss(out.value,rr)
            loss=pg+tc['value_weight']*vl-tc['entropy']*out.entropy.mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            ok,kl,gn=transactional_step(policy,optimizer,loss,tc['max_grad_norm'],batch,old_distribution,tc['target_kl'])
            gradient_rows.append(gn)
            rows.append([float(pg.detach()),float(vl.detach()),kl,float(out.entropy.mean().detach())])
            if not ok:rejected+=1;stop=True;break
            accepted+=1
        if stop: break
    with torch.no_grad():
        final=policy(full);final_kl=float(categorical_kl(old_full,final.logp,full.offsets))
    if not np.isfinite(final_kl) or final_kl>tc['target_kl']:
        policy.load_state_dict(baseline);optimizer.load_state_dict(opt_baseline);rejected+=1;rollout_rollback=1
    with torch.no_grad():
        final=policy(full);final_kl=float(categorical_kl(old_full,final.logp,full.offsets))
        wait=full.candidate_features[:,4]>0
        wait_probability=float(final.logp[wait].exp().sum()/len(records))
    gn_pg=float(np.mean(gradient_rows)) if gradient_rows else None
    policy.eval()
    result=dict(zip(('policy_loss','value_loss','kl_attempted','entropy'),np.mean(rows,axis=0).tolist()))
    result.update(kl=final_kl,rejected_updates=rejected,attempted_minibatches=len(rows),accepted_minibatches=accepted,
        rollout_rollback=rollout_rollback,effective_optimizer_steps=0 if rollout_rollback else accepted,
        gradient_minibatches=len(gradient_rows),ppo_gradient_norm=gn_pg,wait_probability=wait_probability)
    return result
