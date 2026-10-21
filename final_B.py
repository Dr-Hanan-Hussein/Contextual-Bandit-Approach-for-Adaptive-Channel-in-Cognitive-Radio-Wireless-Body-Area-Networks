import numpy as np, itertools, time
from scipy.optimize import linear_sum_assignment
from final_lib import *
t0=time.time(); OUT={}
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

rows = [('No coordination','shared','naive',True),('Decentralised rho-rand','rho_rand','naive',True),
        ('Gateway, no priority','shared','coord',False),('Gateway + priority','shared','coord',True),
        ('Gateway + priority, LinUCB','linucb','coord',True),('Gateway + priority, TS','ts','coord',True)]
# ---- B1/B2: M=N=5 and M=3
B1 = {}
for M in [5,3]:
    cfgM = cfg_with(M=M); XM,CM = make_env(cfgM)
    for name,pol,mode,prio in rows:
        c2 = dict(cfgM); c2['priority']=prio
        B1[(M,name)] = multi_collect(run(XM,CM,c2,pol,mode=mode), c2); log(f"B1 M={M} {name}")
    if M==5:
        c2 = dict(cfgM); c2['assign']='opt'
        B1[(M,'Gateway optimal assignment')] = multi_collect(run(XM,CM,c2,'shared',mode='coord'), c2); log("B1 optimal assignment")
OUT['B1'] = B1

# ---- B3: sweep over M (N=5)
B3 = {}
for M in [1,2,3,4,5,6,8,10]:
    c = cfg_with(M=M); Xm,Cm = make_env(c)
    for mode in (['coord'] if M==1 else ['naive','coord']):
        B3[(M,mode)] = multi_collect(run(Xm,Cm,c,'shared',mode=mode), c)
    log(f"B3 M={M}")
OUT['B3'] = B3

# ---- B4: scalability grid
B4 = {}
for M,N in [(5,5),(8,5),(10,5),(5,10),(10,10),(20,10),(20,20)]:
    c = cfg_with(M=M,N=N,T=3000); Xg,Cg = make_env(c)
    for mode in ['naive','coord']:
        B4[(M,N,mode)] = multi_collect(run(Xg,Cg,c,'shared',mode=mode), c)
    log(f"B4 M={M} N={N}")
OUT['B4'] = B4

# ---- B5: weight sensitivity
b,beta = profile(cfg_with()); theta=1-b
delta = float(np.diff(np.sort(theta)).min()); maxbt = float((beta*theta).max()); wR=2.5
B5 = dict(delta=delta, maxbt=maxbt, rows={})
cfgM = cfg_with(M=5); XM,CM = make_env(cfgM)
for ratio in [1,2,4,6.4,10,20]:
    wE = ratio*wR
    U = np.stack([wR*beta*theta, wE*theta])
    def gv(cls):
        av=list(range(5)); t=0.0
        for m in sorted(range(5), key=lambda i:-cls[i]):
            u=U[1] if cls[m] else U[0]; n=max(av,key=lambda j:u[j]); t+=u[n]; av.remove(n)
        return t
    def ov(cls):
        Wm=np.array([U[1] if cls[m] else U[0] for m in range(5)]); r,c_=linear_sum_assignment(-Wm); return Wm[r,c_].sum()
    tg=to=0.0; nsub=0
    for cls in itertools.product([0,1],repeat=5):
        p=np.prod([0.15 if x else 0.85 for x in cls]); g_,o_=gv(cls),ov(cls); tg+=p*g_; to+=p*o_; nsub+=(o_-g_>1e-9)
    c2=dict(cfgM); c2['wE']=wE
    rp = summarize(run(XM,CM,c2,'shared',mode='coord'),c2)
    c3=dict(c2); c3['priority']=False
    rn = summarize(run(XM,CM,c3,'shared',mode='coord'),c3)
    B5['rows'][ratio] = dict(cond=bool(wE*delta>=wR*maxbt), gap=float(1-tg/to), nsub=int(nsub),
                             reg_prio=float(rp['regret'].mean()), reg_noprio=float(rn['regret'].mean()))
    log(f"B5 ratio={ratio}")
OUT['B5'] = B5
save(OUT,'final_B'); log("saved final_B")
