"""Bounded engineering validation, independent of the formal training ledger."""
import json
import time
from configs.experiment import ROOT
from result.storage import atomic_json


class InteractionCounter:
    def __init__(self, limit=50000):
        self.path=ROOT/'result/engineering/interaction_ledger.json';self.limit=limit
        if self.path.exists(): state=json.loads(self.path.read_text())
        else: state=dict(charged_upper_bound=24, committed=24, sessions=[], migration_interactions=24)
        self.sessions=state.get('sessions',[])
        self.start=state['charged_upper_bound'];self.spent=self.start;self.reserved=self.start;self.began=time.time()
        self.previous_uncommitted=max(0,self.start-state.get('committed',self.start))

    def persist(self):
        atomic_json(self.path,dict(limit=self.limit,charged_upper_bound=self.reserved,committed=self.spent,
                                  sessions=self.sessions,migration_interactions=24))

    def charge(self):
        if self.spent>=self.limit: raise RuntimeError('Engineering interaction cap of 50000 reached')
        if self.spent>=self.reserved:
            self.reserved=min(self.limit,self.spent+128);self.persist()
        self.spent+=1

    def close(self):
        self.sessions.append(dict(start=self.start,end=self.spent,interactions=self.spent-self.start,
                                  seconds=time.time()-self.began,previous_uncommitted=self.previous_uncommitted))
        self.reserved=self.spent;self.persist()
