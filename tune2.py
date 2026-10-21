import numpy as np, itertools, json
from crwban import *
base = dict(S=50, seed0=5000, T=6000)
cfg0 = cfg_with(**base); X,C = make_env(cfg0)
res = json.load(open('results/tuned_baselines.json'))
print("LinUCB refinement (normalised reward)")
best=(res['linucb']['regret'],True,1.0)
for al in [0.75,1.0,1.25,1.5]:
    cfg = cfg_with(**base, alpha_lin=al, lin_norm=True)
    r = summarize(run(X,C,cfg,'linucb'),cfg)['regret'].mean(); print(f"  alpha={al}: {r:.0f}", flush=True)
    if r<best[0]: best=(r,True,al)
res['linucb']=dict(alpha_lin=best[2], lin_norm=True, regret=float(best[0]))
print("  BEST:", best)
print("Q-learning extended grid (gamma=0)")
bq=(res['qlearn']['regret'],0.3,0.0,0.02,200.0)
for al,em,tau in itertools.product([0.3,0.5,0.8],[0.01,0.02],[50.0,100.0,200.0]):
    cfg = cfg_with(**base, q_alpha=al, q_gamma=0.0, q_eps_min=em, q_tau=tau)
    r = summarize(run(X,C,cfg,'qlearn'),cfg)['regret'].mean(); print(f"  alpha={al} eps_min={em} tau={tau:5.0f}: {r:.0f}", flush=True)
    if r<bq[0]: bq=(r,al,0.0,em,tau)
res['qlearn']=dict(q_alpha=bq[1], q_gamma=bq[2], q_eps_min=bq[3], q_tau=bq[4], regret=float(bq[0]))
print("  BEST:", bq)
json.dump(res, open('results/tuned_baselines.json','w'), indent=1)
