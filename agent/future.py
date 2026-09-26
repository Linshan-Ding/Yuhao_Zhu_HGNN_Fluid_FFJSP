"""Future generation from public catalogues and visible history only."""
import numpy as np
from data.generator import capacity_unit_load, route_minimum

class FutureGenerator:
    def __init__(self,cfg,seed):
        self.cfg=cfg; self.rng=np.random.default_rng(seed); self.calls=0
        self.last_cost=cfg['scenario']['minimum_query_cost']  # cost of the last complete query, used for reservations

    def state_dict(self):
        return {"rng":self.rng.bit_generator.state,"calls":self.calls,"last_cost":self.last_cost}

    def load_state_dict(self,state):
        self.rng.bit_generator.state=state["rng"]; self.calls=state["calls"]; self.last_cost=state["last_cost"]

    def future(self,snapshot):
        i=snapshot.instance; c=self.cfg["scenario"]
        minimum=route_minimum(i.proc_times,i.product_count,i.stage_count)
        prior=float(c["prior_count"])
        rate_prior=1.0/capacity_unit_load(i.proc_times,i.product_count,i.stage_count)
        elapsed=max(snapshot.now-float(i.arrival_times.min()),0.0)
        rate=float(self.rng.gamma(prior+max(len(i.arrival_times)-1,0),1/(prior/rate_prior+elapsed)))
        horizon=float(c["horizon_routes"])*float(minimum.mean())
        n=int(self.rng.poisson(rate*horizon))
        arrivals=np.sort(self.rng.uniform(snapshot.now,snapshot.now+horizon,n))
        counts=np.bincount(i.order_product,minlength=i.product_count)+1.0
        products=self.rng.choice(i.product_count,n,p=counts/counts.sum())
        factors=(i.due_dates-i.arrival_times)/minimum[i.order_product]
        sampled=self.rng.choice(factors,n) if len(factors) else np.full(n,2.0)
        deadlines=arrivals+minimum[products]*sampled
        return products,arrivals,deadlines

