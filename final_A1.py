import numpy as np, time
from final_lib import *
t0=time.time(); OUT={}
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# ---- A1: main single-SU table, T=6000, with curves
cfg = cfg_with(); X,C = make_env(cfg)
pols = ['random','rr','ucb1','ucb1_n','ucb1_bin','qlearn','part','linucb','ts','ts_shared','shared','fixctx']
A1 = {}
for p in pols:
    A1[p] = single_collect(run(X,C,cfg,p), cfg, curves=True); log(f"A1 {p}")
OUT['A1'] = A1

# ---- A2: other horizons
A2 = {}
for T in [3000, 15000]:
    c = cfg_with(T=T); Xt,Ct = make_env(c); A2[T] = {}
    for p in ['ucb1','ucb1_bin','part','linucb','ts','shared','fixctx']:
        A2[T][p] = single_collect(run(Xt,Ct,c,p), c); log(f"A2 T={T} {p}")
OUT['A2'] = A2

# ---- A3: logarithmic growth of the learning regret, T=30000
c = cfg_with(T=30000); X3,C3 = make_env(c); A3 = {}
for label,p,kw in [('shared (kE=0.25)','shared',{}),('shared (kE=1)','shared',dict(kE=1.0)),('part','part',{}),('linucb','linucb',{}),('ucb1_bin','ucb1_bin',{})]:
    c2 = dict(c); c2.update(kw)
    o = run(X3,C3,c2,p); U=U_matrix(c2); Ub=U.max(1)
    ch=o['chan'][:,:,0]; cx=o['ctx'][:,:,0]
    cum = np.cumsum(Ub[cx]-U[cx,ch],1)
    A3[label] = dict(mean=cum.mean(0)[::50], t=np.arange(1,c['T']+1)[::50], gen=np.cumsum(o['regret'][:,:,0],1).mean(0)[::50])
    log(f"A3 {label}")
OUT['A3'] = A3


save(OUT,'final_A1'); log('saved final_A1')
