import numpy as np
from scipy import stats
from crwban_sim import *
from sim_B import run_single2
from sim_B3 import *
beta=[0.6,1.0,0.8,0.7,0.9]
cfg = cfg_with(persist=0.5,kappa=1.0,S=100,seed0=5000,beta=beta)
X,C = make_env(cfg)
ref = metrics(run_single2(X,C,cfg,'fixctx'),cfg)['final'].mean()
variants = [
 ("B0  baseline: per-context stats, raw reward (Eq.4)", dict(variant='ctx_raw')),
 ("S1  per-context, normalised, kE=0.25",              dict(variant='ctx_n', kR=1.0, kE=0.25)),
 ("S1b per-context, normalised, kE=0.5",               dict(variant='ctx_n', kR=1.0, kE=0.5)),
 ("S2  SHARED reliability, kR=kE=1",                   dict(variant='shared', kR=1.0, kE=1.0)),
 ("S3  SHARED, emergency never explores (kE=0)",       dict(variant='shared', kR=1.0, kE=0.0)),
 ("S3b SHARED, emergency kE=0.25",                     dict(variant='shared', kR=1.0, kE=0.25)),
 ("S4a S3 + PU-collision penalty mu=0.25",             dict(variant='shared', kR=1.0, kE=0.0, mu=0.25)),
 ("S4b S3 + PU-collision penalty mu=0.5",              dict(variant='shared', kR=1.0, kE=0.0, mu=0.5)),
 ("S4c S3 + PU-collision penalty mu=1.0",              dict(variant='shared', kR=1.0, kE=0.0, mu=1.0)),
]
print(f"reference (best fixed channel per context): regret {ref:.0f}\n")
print(f"{'variant':52s} {'regret':>7s} {'med':>6s} {'tail%':>6s} {'rout':>6s} {'emerg':>6s} {'PU%':>5s} {'PUr%':>5s} {'PUe%':>5s} {'thr_r':>6s}")
res={}
for name,kw in variants:
    m = metrics(run_variant(X,C,cfg,**kw),cfg); res[name]=m
    x=m['final']
    print(f"{name:52s} {x.mean():7.0f} {np.median(x):6.0f} {100*(x>1.5*ref).mean():5.0f}% {m['reg_r'].mean():6.3f} {m['reg_e'].mean():6.3f} {m['pu'].mean():5.1f} {m['pu_r'].mean():5.1f} {m['pu_e'].mean():5.1f} {m['thr_r'].mean():6.3f}")
b0=res[variants[0][0]]['final']
print("\nPaired tests vs baseline B0 (total regret):")
for name,_ in variants[1:6]:
    x=res[name]['final']; print(f"  {name:52s} {100*(x.mean()/b0.mean()-1):+6.1f}%  p={stats.ttest_rel(x,b0)[1]:.2g}")
