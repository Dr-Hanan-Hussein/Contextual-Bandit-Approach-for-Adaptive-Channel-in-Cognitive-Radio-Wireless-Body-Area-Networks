import numpy as np, time
from scipy import stats
from crwban_sim import *
from sim_B import *
profiles = {
 'P1 independent': [0.6,1.0,0.8,0.7,0.9],
 'P2 strong     ': [0.4,0.6,1.0,0.9,0.8],
 'P3 mild       ': [0.7,1.0,0.9,0.8,0.9],
}
theta = 1-np.array(DEFAULT['busy'])
pols = ['random','fixctx','ucb1','ucb1_n','ucb1_bin','cucb','cucb_n']
for name,beta in profiles.items():
    cfg = cfg_with(persist=0.5,kappa=1.0,S=100,seed0=5000,beta=beta)
    X,C = make_env(cfg)
    print(f"\n=== {name} beta={beta}  routine-best channel={np.argmax(np.array(beta)*theta)}, emergency-best channel={np.argmax(theta)} ; corr(beta,theta)={np.corrcoef(beta,theta)[0,1]:.2f}")
    R={}; O={}
    for p in pols:
        O[p]=run_single2(X,C,cfg,p); R[p]=summarize_single(O[p],cfg)
    print(f"{'policy':9s} {'regret':>7s} {'median':>7s} {'routine':>8s} {'emerg':>7s} {'PU_r%':>6s} {'PU_e%':>6s} {'lat':>6s}")
    for p in pols:
        r=R[p]; pr,pe=pu_by_ctx(O[p]); x=r['final_regret']
        print(f"{p:9s} {x.mean():7.0f} {np.median(x):7.0f} {r['reg_routine'].mean():8.3f} {r['reg_emerg'].mean():7.3f} {pr.mean():6.1f} {pe.mean():6.1f} {np.nanmean(r['lat_mean']):6.3f}")
    for base in ['ucb1','ucb1_n','ucb1_bin']:
        for comp in ['cucb','cucb_n']:
            if comp=='cucb' and base!='ucb1': continue
            if comp=='cucb_n' and base=='ucb1': continue
            for k in ['final_regret','reg_routine','reg_emerg']:
                a,b=R[comp][k],R[base][k]
                print(f"   {comp} vs {base:8s} {k:13s} {100*(a.mean()/b.mean()-1):+6.1f}%  p={stats.ttest_rel(a,b)[1]:.3g}")
