"""Canonical rules with explicit, order-independent tie breaking."""
import numpy as np
from configs.experiment import RULES
from agent.graph import rule_index

class RulePolicy:
    def __init__(self,rule):
        if rule not in (*RULES,'SPT-mixed'): raise ValueError(rule)
        self.rule=rule
    def act(self,obs,rng=None):
        if self.rule=='SPT-mixed': return rule_index(obs.base,'SPT'),0.,0.
        live=np.flatnonzero(obs.candidate[:,0]>=0)
        if not len(live): return int(np.flatnonzero(obs.candidate[:,0]<0)[0]),0.,0.
        k=obs.keys
        if self.rule=='SPT': key=lambda i:(k[i,0],k[i,1],k[i,6],k[i,7])
        else:
            col={'FCFS':1,'EDD':2,'MST':3,'CR':4,'LWKR':5}[self.rule]
            key=lambda i:(k[i,col],k[i,1],k[i,6],k[i,0],k[i,7])
        return int(min(live,key=key)),0.,0.
    def act_many(self,obs,rng=None): return [self.act(o,rng) for o in obs]
