import numpy as np, itertools
from scipy.optimize import linear_sum_assignment
from scipy import stats
from crwban import *

b,beta = profile(cfg_with()); theta=1-b
delta = np.diff(np.sort(theta)).min(); maxbt=(beta*theta).max()
print(f"delta_min={delta:.2f}  max_n beta_n*theta_n={maxbt:.2f}  -> greedy-optimality condition: wE/wR >= {maxbt/delta:.2f}")
wR=2.5
print("\nExact optimality gap of the greedy priority gateway (true parameters, M=N=5, pE=0.15), and simulated priority gain")
print(f"{'wE/wR':>6s} {'condition':>10s} {'exact gap':>10s} {'subopt patterns':>16s} | {'regret no-prio':>14s} {'regret prio':>12s} {'prio gain':>10s} {'emerg regret/pkt prio vs no-prio':>34s}")
cfgM = cfg_with(M=5); XM,CM = make_env(cfgM)
for ratio in [1,2,4,5.14,6.4,10,20]:
    wE = ratio*wR
    U = np.stack([wR*beta*theta, wE*theta])
    def gv(cls):
        av=list(range(5)); t=0.0
        for m in sorted(range(5), key=lambda i:-cls[i]):
            u=U[1] if cls[m] else U[0]; n=max(av,key=lambda j:u[j]); t+=u[n]; av.remove(n)
        return t
    def ov(cls):
        Wm=np.array([U[1] if cls[m] else U[0] for m in range(5)]); r,c=linear_sum_assignment(-Wm); return Wm[r,c].sum()
    tg=to=0.0; nsub=0
    for cls in itertools.product([0,1],repeat=5):
        p=np.prod([0.15 if x else 0.85 for x in cls]); g_,o_=gv(cls),ov(cls); tg+=p*g_; to+=p*o_; nsub+= (o_-g_>1e-9)
    ok = (wE*delta >= wR*maxbt)
    c2=dict(cfgM); c2['wE']=wE
    rp = summarize(run(XM,CM,c2,'shared',mode='coord'),c2)
    c3=dict(c2); c3['priority']=False
    rn = summarize(run(XM,CM,c3,'shared',mode='coord'),c3)
    print(f"{ratio:6.2f} {('holds' if ok else 'not met'):>10s} {100*(1-tg/to):9.4f}% {nsub:>10d}/32 | {rn['regret'].mean():14.0f} {rp['regret'].mean():12.0f} {100*(1-rp['regret'].mean()/rn['regret'].mean()):9.1f}% {rp['reg_emerg'].mean():10.2f} vs {rn['reg_emerg'].mean():6.2f}", flush=True)

print("\nSingle SU: do the policy decisions depend on the weights? (proposed policy, same seeds, wE varied)")
cfg = cfg_with(); X,C = make_env(cfg)
for ratio in [1,6.4,20]:
    c2=dict(cfg); c2['wE']=ratio*wR
    o = run(X,C,c2,'shared'); print(f"  wE/wR={ratio:5.1f}: channel choices identical to baseline run? ", end='')
    o0 = run(X,C,cfg,'shared'); print(bool((o['chan']==o0['chan']).all()))
