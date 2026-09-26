"""Standard updates and direct policy preference learning, each with explicit accounting."""
import numpy as np
import torch
from torch.nn import functional as F
from agent.returns import advantages
from agent.ppo import update as ppo_update, transactional_step

def on_policy(policy,optimizer,records,bootstrap,c,rng):
    if c['method']!='a2c':return ppo_update(policy,optimizer,records,bootstrap,c,rng,'cpu')
    adv,ret=advantages(records,bootstrap,c['training']['gae_lambda'],c['training']['gamma'])
    adv=(adv-adv.mean())/max(adv.std(),1e-6);batch=policy.batch([r['obs'] for r in records]);out=policy(batch)
    ix=batch.offsets[:-1]+torch.tensor([r['action'] for r in records])
    pg=-(out.logp[ix]*torch.tensor(adv)).mean();vl=.5*F.mse_loss(out.value,torch.tensor(ret))
    loss=pg+c['training']['value_weight']*vl-c['training']['entropy']*out.entropy.mean()
    optimizer.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),c['training']['max_grad_norm']);optimizer.step()
    return dict(policy_loss=float(pg.detach()),value_loss=float(vl.detach()),gradient_norm=float(gn),effective_optimizer_steps=1)

def preference_update(policy,optimizer,replay,c,rng,version):
    replay.expire(version);metrics=[];rejected=0;sc=c['scenario']
    for _ in range(sc['replay_batches']):
        samples=replay.sample(rng,sc['replay_batch'])
        if not samples:break
        batch=policy.batch([o for o,l in samples]);out=policy(batch);old=out.logp.detach();losses=[]
        for j,(obs,label) in enumerate(samples):
            idx=batch.offsets[j]+torch.tensor(label.actions)
            if c['method']=='impact_only':
                losses.append(F.smooth_l1_loss(out.impact[idx[1]]-out.impact[idx[0]],out.impact.new_tensor(label.mean)))
            else:
                target=out.logp.new_tensor([1-label.probability,label.probability])
                losses.append(label.reliability*F.kl_div(F.log_softmax(out.logits[idx],0),target,reduction='sum'))
        loss=sc['auxiliary_weight']*torch.stack(losses).mean()
        ok,kl,gn=transactional_step(policy,optimizer,loss,c['training']['max_grad_norm'],batch,old,sc['kl'])
        if not ok:rejected+=1;break
        metrics.append((float(loss.detach()),kl,gn))
    return dict(preference_steps=len(metrics),preference_rejected=rejected,preference_loss=float(np.mean([x[0] for x in metrics])) if metrics else None,
                preference_kl=max([x[1] for x in metrics],default=None),preference_gradient=float(np.mean([x[2] for x in metrics])) if metrics else None,
                replay_labels=len(replay.items),label_reuses=replay.reuses)

def q_bootstrap(online,target,offsets,double):
    return torch.stack([target[a:b][online[a:b].argmax()] if double else target[a:b].max() for a,b in zip(offsets[:-1],offsets[1:])])

def q_update(policy,target,optimizer,replay,c,rng):
    size=c['classic']['batch'];ids=rng.choice(len(replay),size,replace=False);rows=[replay[int(i)] for i in ids]
    batch=policy.batch([r[0] for r in rows]);out=policy(batch);ix=batch.offsets[:-1]+torch.tensor([r[1] for r in rows])
    rewards=torch.tensor([r[2] for r in rows],dtype=torch.float32);live=[i for i,r in enumerate(rows) if r[3] is not None]
    with torch.no_grad():
        if live:
            nxt=policy.batch([rows[i][3] for i in live]);on=policy(nxt).logits;tar=target(nxt).logits
            rewards[live]+=c['training']['gamma']*q_bootstrap(on,tar,nxt.offsets,c['method']=='ddqn')
    loss=F.smooth_l1_loss(out.logits[ix],rewards);optimizer.zero_grad();loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),c['training']['max_grad_norm']);optimizer.step()
    return dict(q_loss=float(loss.detach()),q_gradient=float(gn))
