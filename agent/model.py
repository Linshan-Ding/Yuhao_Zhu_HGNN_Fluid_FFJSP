"""A stable HGNN base plus a zero-initialized, gated machine-contender correction."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from agent.graph_layers import (mlp, collate, GraphLayer, GraphBatch, PolicyOutput, ContenderAttention, segment_mean,
    segment_sum, segment_softmax)
from agent.graph import ORDER_DIM, OP_DIM, MACHINE_DIM, EDGE_DIM, ACTION_DIM, GLOBAL_DIM
from agent.observation import EXTRA_DIM, CONTENDER_DIM
from configs.experiment import CLASSIC, uses_graph

class Policy(nn.Module):
    def __init__(self,c):
        super().__init__();self.cfg=c;self.method=c['method'];self.local=uses_graph(self.method);self.residual_enabled=True
        self.classic=self.method in CLASSIC;d=c['classic']['width'] if self.classic else c['network']['width']
        self.width=d
        self.order_encoder=mlp(ORDER_DIM,d,d);self.op_encoder=mlp(OP_DIM,d,d)
        self.machine_encoder=mlp(MACHINE_DIM,d,d);self.edge_encoder=mlp(EDGE_DIM,d,d)
        self.layers=nn.ModuleList([] if self.classic else [GraphLayer(d,self.method!='dual_attention') for _ in range(c['network']['layers'])])
        self.context=mlp(6*d+ACTION_DIM+EXTRA_DIM+GLOBAL_DIM,d,d)
        self.actor=nn.Linear(d,1);self.impact_head=nn.Linear(d,1);self.critic=mlp(3*d+GLOBAL_DIM,d,1)
        nn.init.orthogonal_(self.actor.weight,gain=.01);nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.impact_head.weight);nn.init.zeros_(self.impact_head.bias)
        if self.local:
            # Contention-driven correction: attention over the candidate's contenders, a residual update of the candidate
            # context, a gate conditioned on the contender count and a zero-initialized score offset.
            self.contender=ContenderAttention(d,CONTENDER_DIM);self.correction=mlp(2*d,d,d);self.correction_norm=nn.LayerNorm(d)
            self.delta=nn.Linear(d,1);self.gate=nn.Linear(d+1,1)
            nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2.)

    def batch(self,observations):
        if self.classic:
            # Identical MLP inputs/outputs, without building unused graph or contender incidences.
            device=next(self.parameters()).device;n=len(observations)
            sizes=np.array([[len(o.order),len(o.operation),len(o.machine),len(o.candidate)] for o in observations])
            offsets=np.vstack((np.zeros(4,dtype=int),sizes.cumsum(0)))
            values={key:np.concatenate([getattr(o,key) for o in observations]) for key in
                    ('order','operation','machine','candidate_features','extra')}
            for k,key in enumerate(('order_batch','op_batch','machine_batch','action_batch')):
                values[key]=np.repeat(np.arange(n),sizes[:,k])
            candidate=np.concatenate([o.candidate for o in observations]).copy();live=candidate[:,0]>=0
            candidate[live]+=offsets[values['action_batch'][live]][:,[1,2,0]]
            values.update(candidate=candidate,offsets=offsets[:,3],global_features=np.stack([o.global_features for o in observations]))
            return GraphBatch({k:torch.as_tensor(np.ascontiguousarray(v),device=device) for k,v in values.items()},n)
        flags=SimpleNamespace(action=False,dual=self.method=='dual_attention')
        b=collate([o.base for o in observations],next(self.parameters()).device,flags)
        b.data['extra']=torch.as_tensor(np.concatenate([o.extra for o in observations]),device=b.order.device)
        if self.local:
            arrays={k:[] for k in ('ca','co','cf')};aa=oo=0
            for o in observations:
                arrays['ca'].append(o.contender_action+aa);arrays['co'].append(o.contender_order+oo);arrays['cf'].append(o.contender_features)
                aa+=len(o.candidate);oo+=len(o.order)
            for k,v in arrays.items(): b.data[k]=torch.as_tensor(np.concatenate(v),device=b.order.device)
        return b

    def forward(self,b,include_impact=True,details=False):
        no,np_=len(b.order),len(b.operation)
        x=torch.cat((self.order_encoder(b.order),self.op_encoder(b.operation),self.machine_encoder(b.machine)))
        edge=self.edge_encoder(b.edge_features) if len(self.layers) else None
        for layer in self.layers:x=layer(x,b,edge,dual=self.method=='dual_attention')
        orders,ops,machines=x[:no],x[no:no+np_],x[no+np_:]
        pooled=torch.cat((segment_mean(orders,b.order_batch,b.size),segment_mean(ops,b.op_batch,b.size),
                          segment_mean(machines,b.machine_batch,b.size),b.global_features),-1)
        c=b.candidate;live=(c[:,0]>=0).to(x.dtype)[:,None]
        local=torch.cat((ops[c[:,0].clamp_min(0)],machines[c[:,1].clamp_min(0)],orders[c[:,2].clamp_min(0)]),-1)*live
        h=self.context(torch.cat((local,pooled[b.action_batch],b.candidate_features,b.extra),-1))
        z=self.actor(h).squeeze(-1);base_z=z;impact_h=h;gates=None
        if self.local and self.residual_enabled:
            aggregate,count=self.contender(h,orders,b.ca,b.co,b.cf)
            hc=self.correction_norm(h+self.correction(torch.cat((h,aggregate),-1)))
            gates=self.gate(torch.cat((h,torch.log1p(count)[:,None]),-1)).sigmoid().squeeze(-1)
            # Only contended candidates are corrected; the uncontended ones keep the base score exactly.
            z=z+gates*self.delta(hc).squeeze(-1)*live[:,0]*(count>0).to(z.dtype);impact_h=hc
        p=segment_softmax(z,b.action_batch,b.size);lp=p.clamp_min(1e-30).log()
        out=PolicyOutput(z,self.impact_head(impact_h).squeeze(-1) if include_impact else None,lp,
            self.critic(pooled).squeeze(-1),-segment_sum(p*lp,b.action_batch,b.size),b.offsets)
        if details:
            out.base_logits=base_z;out.correction=z-base_z;out.gates=gates;out.representations=impact_h
        return out

    @torch.no_grad()
    def act_many(self,observations,rng=None,epsilon=0.,return_probabilities=False):
        out=self(self.batch(observations),False);lp=out.logp.cpu().numpy();z=out.logits.cpu().numpy();values=out.value.cpu().numpy()
        results=[];probabilities=[]
        for i,(a,b) in enumerate(zip(out.offsets[:-1],out.offsets[1:])):
            a,b=int(a),int(b);p=np.exp(lp[a:b].astype(np.float64));p/=p.sum()
            if self.method in ('dqn','ddqn'):
                choice=int(rng.integers(b-a)) if rng is not None and rng.random()<epsilon else int(z[a:b].argmax())
            else:choice=int(p.argmax()) if rng is None else int(rng.choice(len(p),p=p))
            results.append((choice,float(lp[a+choice]),float(values[i])));probabilities.append(p)
        return (results,probabilities) if return_probabilities else results
    def act(self,obs,rng=None):return self.act_many([obs],rng)[0]
