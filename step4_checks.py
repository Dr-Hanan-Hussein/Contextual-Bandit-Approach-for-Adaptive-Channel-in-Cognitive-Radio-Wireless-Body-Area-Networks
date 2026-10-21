import numpy as np, itertools, time
from scipy.optimize import linear_sum_assignment
from scipy import stats
from crwban import *

# ------------------------------------------------------------------ A) pseudo-regret vs ln T
print("=== A) Learning regret vs best fixed channel per context (pseudo-regret), single SU, 25 seeds ===")
cfg = cfg_with(T=30000); X,C = make_env(cfg)
b,beta = profile(cfg); theta = 1-b; w = np.array([cfg['wR'],cfg['wE']])
U = np.stack([w[0]*beta*theta, w[1]*theta])            # expected utility [ctx, channel]
Ubest = U.max(1)
cps = [1000,2000,5000,10000,20000,30000]
res={}
for label,pol,kw in [('shared kE=0.25 (used)','shared',dict(kE=0.25)),('shared kE=1.0 (theory)','shared',dict(kE=1.0)),
                     ('partitioned UCB','part',{}),('LinUCB','linucb',{}),('UCB1 reliability-only','ucb1_bin',{})]:
    c2 = dict(cfg); c2.update(kw)
    o = run(X,C,c2,pol); ch=o['chan'][:,:,0]; cx=o['ctx'][:,:,0]
    pr = Ubest[cx] - U[cx,ch]                          # per-round pseudo-regret
    cum = np.cumsum(pr,axis=1).mean(0)
    res[label]=cum
    vals=[cum[t-1] for t in cps]
    slopes=[(vals[i]-vals[i-1])/(np.log(cps[i])-np.log(cps[i-1])) for i in range(1,len(cps))]
    print(f"{label:26s} R(T) at {cps}: "+" ".join(f"{v:7.0f}" for v in vals))
    print(f"{'':26s} slope dR/dlnT between checkpoints: "+" ".join(f"{s_:7.0f}" for s_ in slopes), flush=True)
# genie regret for contrast
o = run(X,C,cfg,'shared'); g=np.cumsum(o['regret'][:,:,0],axis=1).mean(0)
print(f"{'genie regret (shared)':26s} R(T): "+" ".join(f"{g[t-1]:7.0f}" for t in cps)+"   (linear: ~ %.3f per round)"%((g[-1]-g[9999])/20000))

# ------------------------------------------------------------------ B) greedy priority vs optimal assignment, known parameters
print("\n=== B1) Exact optimality gap with TRUE parameters (M=N=5, every class pattern weighted by its probability) ===")
def greedy_val(cls):
    avail=list(range(5)); tot=0.0
    for m in sorted(range(5), key=lambda i:-cls[i]):         # emergency first
        u = (U[1] if cls[m]==1 else U[0])
        n = max(avail, key=lambda j:u[j]); tot+=u[n]; avail.remove(n)
    return tot
def opt_val(cls):
    Wm = np.array([U[1] if cls[m]==1 else U[0] for m in range(5)])
    r,c = linear_sum_assignment(-Wm); return Wm[r,c].sum()
pE=0.15; tg=to=0.0; worst=0.0; nsub=0; psub=0.0
for cls in itertools.product([0,1],repeat=5):
    p = np.prod([pE if x else 1-pE for x in cls]); g_,o_ = greedy_val(cls),opt_val(cls)
    tg+=p*g_; to+=p*o_
    if o_-g_>1e-9: nsub+=1; psub+=p; worst=max(worst,(o_-g_)/o_)
print(f"expected utility  greedy {tg:.4f}  optimal {to:.4f}  gap {100*(1-tg/to):.3f}%   patterns where greedy is suboptimal: {nsub}/32 (probability {100*psub:.1f}%), worst-case gap {100*worst:.1f}%")
d = np.diff(np.sort(theta)); print("reliability gaps:", np.round(np.sort(d),3), " min gap %.2f  -> lexicographic condition wE/wR >= 1/min_gap = %.2f ; actual wE/wR = %.2f"%(d.min(),1/d.min(),w[1]/w[0]))

print("\n=== B2) In simulation with learned indices (M=N=5, T=6000, 25 seeds) ===")
cfgM = cfg_with(M=5); XM,CM = make_env(cfgM)
t0=time.time()
rg = summarize(run(XM,CM,cfgM,'shared',mode='coord'),cfgM)
c2 = dict(cfgM); c2['assign']='opt'
ro = summarize(run(XM,CM,c2,'shared',mode='coord'),c2)
print(f"(time {time.time()-t0:.0f}s)")
for k in ['regret','reg_emerg','reg_routine','throughput','fail','pu_e','lat_mean']:
    x,y = ro[k],rg[k]
    print(f"  {k:12s} greedy {np.nanmean(rg[k]):9.4f} | optimal {np.nanmean(ro[k]):9.4f} | optimal vs greedy {100*(np.nanmean(x)/np.nanmean(y)-1):+6.2f}%  p={stats.ttest_rel(x,y)[1]:.3g}")
