import numpy as np
from scipy import stats
from crwban_sim import *
from sim_B import run_single2
from sim_B3 import *
settings = [
 ("P1 pE=.15", [0.6,1.0,0.8,0.7,0.9], 0.15),
 ("P2 strong ", [0.4,0.6,1.0,0.9,0.8], 0.15),
 ("P3 mild   ", [0.7,1.0,0.9,0.8,0.9], 0.15),
 ("P1 pE=.05", [0.6,1.0,0.8,0.7,0.9], 0.05),
 ("P1 pE=.30", [0.6,1.0,0.8,0.7,0.9], 0.30),
]
print("Fresh seeds (9000+), 100 seeds, T=6000.  Cells: total regret | tail% (>1.5x ref) | emergency regret")
print(f"{'setting':11s} {'ref':>5s} | {'UCB1 rel-only':>20s} | {'B0 baseline':>20s} | {'S1 small':>20s} | {'S3b medium':>20s} | S3b vs B0")
for name,beta,pE in settings:
    cfg = cfg_with(persist=0.5,kappa=1.0,S=100,seed0=9000,beta=beta,pE=pE)
    X,C = make_env(cfg)
    ref = metrics(run_single2(X,C,cfg,'fixctx'),cfg)['final'].mean()
    cells=[]; F={}
    for key,fn in [('ucb',lambda: run_single2(X,C,cfg,'ucb1_bin')),
                   ('B0',lambda: run_variant(X,C,cfg,'ctx_raw')),
                   ('S1',lambda: run_variant(X,C,cfg,'ctx_n',kR=1.0,kE=0.25)),
                   ('S3b',lambda: run_variant(X,C,cfg,'shared',kR=1.0,kE=0.25))]:
        m = metrics(fn(),cfg); F[key]=m['final']
        cells.append(f"{m['final'].mean():6.0f} |{100*(m['final']>1.5*ref).mean():3.0f}% |{m['reg_e'].mean():5.2f}")
    p = stats.ttest_rel(F['S3b'],F['B0'])[1]
    print(f"{name:11s} {ref:5.0f} | "+" | ".join(cells)+f" | {100*(F['S3b'].mean()/F['B0'].mean()-1):+.1f}% p={p:.2g}")
