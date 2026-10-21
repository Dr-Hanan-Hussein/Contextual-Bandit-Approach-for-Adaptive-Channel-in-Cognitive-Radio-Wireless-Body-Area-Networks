import numpy as np, itertools, time, json
from crwban import *
t0=time.time()
base = dict(S=50, seed0=5000, T=6000)
cfg0 = cfg_with(**base); X,C = make_env(cfg0)
res = {}

print("LinUCB (alpha, normalised reward?)  -> mean total regret over 50 calibration seeds")
best=None
for norm in [True, False]:
    for al in [0.1,0.25,0.5,1.0,2.0,4.0]:
        cfg = cfg_with(**base, alpha_lin=al, lin_norm=norm)
        r = summarize(run(X,C,cfg,'linucb'),cfg)['regret'].mean()
        print(f"  norm={norm!s:5s} alpha={al:4.2f}: {r:7.0f}", flush=True)
        if best is None or r<best[0]: best=(r,norm,al)
print("  BEST LinUCB:", best); res['linucb']=dict(alpha_lin=best[2], lin_norm=best[1], regret=best[0])

print("\nTabular Q-learning (alpha, gamma, eps_min, tau)")
best=None
for al,ga,em,tau in itertools.product([0.05,0.1,0.3],[0.0,0.8],[0.02,0.05],[200.0,1000.0]):
    cfg = cfg_with(**base, q_alpha=al, q_gamma=ga, q_eps_min=em, q_tau=tau)
    r = summarize(run(X,C,cfg,'qlearn'),cfg)['regret'].mean()
    print(f"  alpha={al:4.2f} gamma={ga:3.1f} eps_min={em:4.2f} tau={tau:5.0f}: {r:7.0f}", flush=True)
    if best is None or r<best[0]: best=(r,al,ga,em,tau)
print("  BEST Q-learning:", best); res['qlearn']=dict(q_alpha=best[1], q_gamma=best[2], q_eps_min=best[3], q_tau=best[4], regret=best[0])

print("\nNo-tuning policies on the same seeds")
for p in ['ts','ts_shared','rho_rand','shared','part','ucb1','ucb1_bin']:
    r = summarize(run(X,C,cfg0,p),cfg0)['regret'].mean(); print(f"  {p:9s}: {r:7.0f}"); res[p]=dict(regret=r)
json.dump(res, open('results/tuned_baselines.json','w'), indent=1)
print("time", round(time.time()-t0))
