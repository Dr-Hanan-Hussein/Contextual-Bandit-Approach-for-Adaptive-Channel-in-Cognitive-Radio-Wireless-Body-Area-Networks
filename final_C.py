import numpy as np, time
from final_lib import *
t0=time.time(); OUT={}
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
pols = ['ucb1','ucb1_bin','part','linucb','ts','shared','fixctx']
C1 = {}
for pers in [0.2, 0.5, 1.0]:
    c = cfg_with(persist=pers); X,C = make_env(c)
    for p in pols: C1[('persist',pers,p)] = single_collect(run(X,C,c,p), c)
    log(f"persist {pers}")
profiles = {'P1 (default)':(0.6,1.0,0.8,0.7,0.9), 'P2 (strong)':(0.4,0.6,1.0,0.9,0.8), 'P3 (mild)':(0.7,1.0,0.9,0.8,0.9)}
for name,bt in profiles.items():
    c = cfg_with(beta=bt); X,C = make_env(c)
    for p in pols: C1[('beta',name,p)] = single_collect(run(X,C,c,p), c)
    log(f"beta {name}")
OUT['C1'] = C1
save(OUT,'final_C'); log("saved final_C")
