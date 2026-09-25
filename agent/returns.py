import numpy as np

def advantages(records,bootstrap,lam=.95,gamma=1.):
    if isinstance(bootstrap,dict):
        a=np.zeros(len(records),np.float32); ret=np.zeros_like(a)
        for stream in bootstrap:
            ix=[k for k,r in enumerate(records) if r.get("stream",0)==stream]
            if ix:
                av,rv=advantages([records[k] for k in ix],bootstrap[stream],lam,gamma)
                a[ix]=av; ret[ix]=rv
        return a,ret
    adv=np.zeros(len(records),np.float32); last=0.
    for t in reversed(range(len(records))):
        r=records[t]
        nxt=bootstrap if t==len(records)-1 else records[t+1]["value"]
        if r["terminated"]:
            nxt=0.
        elif r.get("truncated",False):
            nxt=r["bootstrap"]
        delta=r["reward"]+gamma*nxt-r["value"]
        last=delta+gamma*lam*(0. if r["boundary"] else last)
        adv[t]=last
    ret=adv+np.asarray([r["value"] for r in records],np.float32)
    return adv,ret

