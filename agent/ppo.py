"""PPO with separate impact supervision and transactional post-update KL checks."""
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

def update(policy,optimizer,records,bootstrap,cfg,rng,device):
    tc=cfg['training'];adv,ret=advantages(records,bootstrap,tc['gae_lambda'],tc['gamma'])
    adv=(adv-adv.mean())/max(float(adv.std()),1e-6)
    labels=0
    baseline=deepcopy(policy.state_dict());opt_baseline=deepcopy(optimizer.state_dict())
    full=policy.batch([r['obs'] for r in records])
    with torch.no_grad(): old_full=policy(full).logp.detach()
    old_offsets=full.offsets.cpu().tolist();rows=[];rejected=0;gn_pg=gn_aux=cosine=0.;rank=[]
    accepted=0;gradient_rows=[];rollout_rollback=0
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
            aux=out.logp.sum()*0
            loss=pg+tc['value_weight']*vl-tc['entropy']*out.entropy.mean()+cfg['scenario']['auxiliary_weight']*aux
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            before=deepcopy(policy.state_dict());before_opt=deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),tc['max_grad_norm']);optimizer.step()
            gradient_rows.append(float(gn))
            with torch.no_grad(): kl=float(categorical_kl(old_distribution,policy(batch).logp,batch.offsets))
            rows.append([float(pg.detach()),float(vl.detach()),float(aux.detach()),kl,float(out.entropy.mean().detach())])
            if not np.isfinite(kl) or kl>tc['target_kl']:
                policy.load_state_dict(before);optimizer.load_state_dict(before_opt);rejected+=1;stop=True;break
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
    result=dict(zip(('policy_loss','value_loss','scenario_loss','kl_attempted','entropy'),np.mean(rows,axis=0).tolist()))
    result.update(kl=final_kl,rejected_updates=rejected,labelled_states=labels,label_fraction=labels/len(records),
        attempted_minibatches=len(rows),accepted_minibatches=accepted,rollout_rollback=rollout_rollback,
        effective_optimizer_steps=0 if rollout_rollback else accepted,gradient_minibatches=len(gradient_rows),
        ppo_gradient_norm=gn_pg,aux_gradient_norm=None,gradient_cosine=None,
        pair_rank_error=None,wait_probability=wait_probability)
    return result
