"""Ragged graph batches, dual-path attention and sparse action-conditioned updates."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch import nn
from agent.graph import ORDER_DIM, OP_DIM, MACHINE_DIM, EDGE_DIM, ACTION_DIM, GLOBAL_DIM

def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i,h), nn.SiLU(), nn.Linear(h,o))

def segment_sum(x, index, count):
    return x.new_zeros((count,) + x.shape[1:]).index_add_(0,index,x)

def segment_mean(x,index,count):
    den = segment_sum(x.new_ones((len(x),)),index,count).clamp_min(1)
    return segment_sum(x,index,count) / den.reshape((-1,)+(1,)*(x.ndim-1))

def segment_softmax(score,index,count):
    top = score.new_full((count,),-torch.inf)
    top.scatter_reduce_(0,index,score.detach(),reduce="amax",include_self=True)
    ex = (score-top[index]).exp()
    return ex / segment_sum(ex,index,count)[index].clamp_min(1e-12)

@dataclass
class GraphBatch:
    data: dict
    size: int

    def __getattr__(self,name):
        return self.data[name]

def collate(observations, device="cpu", flags=None):
    flags=flags or SimpleNamespace(action=False, dual=False)
    no = sum(len(o.order) for o in observations)
    np_ = sum(len(o.operation) for o in observations)
    arrays = {k:[] for k in ("order","operation","machine","edge_features","candidate_features","global_features",
        "order_batch","op_batch","machine_batch","action_batch","owner","candidate","eligibility",
        "delivery_src","delivery_dst","competition_src","competition_dst","competition_edge",
        "mm_src","mm_dst","mm_op")}
    offsets = [0]; oo=op=mm=aa=ee=0
    for b,o in enumerate(observations):
        n,p,m,a,e = len(o.order),len(o.operation),len(o.machine),len(o.candidate),len(o.eligibility)
        for key in ("order","operation","machine","edge_features","candidate_features"):
            arrays[key].append(getattr(o,key))
        arrays["global_features"].append(o.global_features[None])
        for key,size in (("order_batch",n),("op_batch",p),("machine_batch",m),("action_batch",a)):
            arrays[key].append(np.full(size,b,np.int64))
        arrays["owner"].append(o.owner+oo)
        cand = o.candidate.copy()
        live = cand[:,0] >= 0
        cand[live] += np.array([op,mm,oo])
        arrays["candidate"].append(cand)
        elig = o.eligibility + np.array([op,mm])
        arrays["eligibility"].append(elig)
        src = np.r_[o.owner+oo, np.arange(p)+op+no, o.precedence[:,0]+op+no, o.precedence[:,1]+op+no]
        dst = np.r_[np.arange(p)+op+no, o.owner+oo, o.precedence[:,1]+op+no,o.precedence[:,0]+op+no]
        arrays["delivery_src"].append(src); arrays["delivery_dst"].append(dst)
        es, ed = elig[:,0]+no, elig[:,1]+no+np_
        arrays["competition_src"].append(np.r_[es,ed]); arrays["competition_dst"].append(np.r_[ed,es])
        arrays["competition_edge"].append(np.r_[np.arange(e)+ee,np.arange(e)+ee])
        # Machine competition induced by currently ready operations (dual-attention comparator).
        ms,md,mo = [],[],[]
        for j in (np.flatnonzero(o.operation[:,4]>0) if flags.dual else []):
            machines = o.eligibility[o.eligibility[:,0]==j,1]
            for u in machines:
                for v in machines:
                    if u != v:
                        ms.append(int(u)+mm+no+np_); md.append(int(v)+mm+no+np_); mo.append(int(j)+op+no)
        for key,val in (("mm_src",ms),("mm_dst",md),("mm_op",mo)):
            arrays[key].append(np.asarray(val,np.int64))
        oo+=n; op+=p; mm+=m; aa+=a; ee+=e; offsets.append(aa)
    arrays = {k:np.concatenate(v,axis=0) for k,v in arrays.items()}
    arrays["offsets"] = np.asarray(offsets,np.int64)
    data = {k:torch.as_tensor(np.ascontiguousarray(v),device=device) for k,v in arrays.items()}
    return GraphBatch(data,len(observations))

class Attention(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.q,self.k,self.v = nn.Linear(d,d,bias=False),nn.Linear(d,d,bias=False),nn.Linear(d,d,bias=False)

    def forward(self,x,src,dst,edge=None):
        if len(src)==0:
            return torch.zeros_like(x)
        e = 0 if edge is None else edge
        score = (self.q(x)[dst]*(self.k(x)[src]+e)).sum(-1)/math.sqrt(x.shape[1])
        w = segment_softmax(score,dst,len(x))
        return segment_sum(w[:,None]*(self.v(x)[src]+e),dst,len(x))

class GraphLayer(nn.Module):
    def __init__(self,d,generic=False):
        super().__init__()
        self.generic=generic
        self.delivery=Attention(d)
        self.competition=Attention(d)
        self.fuse=mlp(2*d,d,d)
        self.norm=nn.LayerNorm(d)

    def forward(self,x,b,edge,delivery=True,competition=True,dual=False):
        if self.generic:
            src=torch.cat((b.delivery_src,b.competition_src)); dst=torch.cat((b.delivery_dst,b.competition_dst))
            e=torch.cat((edge.new_zeros((len(b.delivery_src),edge.shape[1])),edge[b.competition_edge]))
            h=self.delivery(x,src,dst,e)
            return self.norm(x+h)
        h1=self.delivery(x,b.delivery_src,b.delivery_dst) if delivery else torch.zeros_like(x)
        cs,cd,ce=b.competition_src,b.competition_dst,edge[b.competition_edge]
        if dual and len(b.mm_src):
            cs=torch.cat((cs,b.mm_src)); cd=torch.cat((cd,b.mm_dst)); ce=torch.cat((ce,x[b.mm_op]))
        h2=self.competition(x,cs,cd,ce) if competition else torch.zeros_like(x)
        return self.norm(x+self.fuse(torch.cat((h1,h2),-1)))

@dataclass
class PolicyOutput:
    logits: torch.Tensor
    impact: torch.Tensor
    logp: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    offsets: torch.Tensor

