import numpy as np, time, pickle
from final_lib import *
t0=time.time(); OUT={}
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
A1 = pickle.load(open('results/final_A1.pkl','rb'))['A1']
# ---- A4: emergency-share sweep
A4 = {}
for pE in [0.01,0.05,0.15,0.30]:
    c = cfg_with(pE=pE); Xs,Cs = make_env(c); A4[pE] = {}
    for p in ['ucb1_bin','part','linucb','ts','shared','fixctx']:
        A4[pE][p] = single_collect(run(Xs,Cs,c,p), c)
    log(f"A4 pE={pE}")
OUT['A4'] = A4

# ---- A5: convergence of the emergency class (windows)
wins = [(0,1500),(1500,3000),(3000,6000),(6000,10000),(10000,15000)]
A5 = {}
for pE in [0.05,0.15]:
    c = cfg_with(pE=pE, T=15000); Xc,Cc = make_env(c)
    for p in ['ucb1_bin','part','linucb','shared','fixctx']:
        o = run(Xc,Cc,c,p); reg=o['regret'][:,:,0]; ctx=o['ctx'][:,:,0]; row=[]
        for a,b in wins:
            m = ctx[:,a:b]==1
            row.append(float(((reg[:,a:b]*m).sum(1)/np.maximum(m.sum(1),1)).mean()))
        A5[(pE,p)] = row; log(f"A5 pE={pE} {p}")
OUT['A5'] = dict(windows=wins, rows=A5)

# ---- A6: traffic-class errors
def make_Cp(C, miss, false, seed0=777):
    Cp = C.copy()
    for s in range(C.shape[0]):
        u = np.random.default_rng(seed0+s).random(C.shape[1:])
        Cp[s] = np.where(C[s], u >= miss, u < false)
    return Cp
cfg = cfg_with(); X,C = make_env(cfg); A6 = {'sym':{}, 'grid':{}, 'multi':{}}
for eps in [0.0,0.05,0.10,0.20]:
    Cp = make_Cp(C,eps,eps)
    for p in ['part','linucb','shared']:
        A6['sym'][(eps,p)] = single_collect(run(X,C,cfg,p,Cp=Cp), cfg)
    log(f"A6 sym {eps}")
A6['sym'][(0.0,'ucb1_bin')] = A1['ucb1_bin']
for miss in [0.0,0.05,0.10,0.20]:
    for false in [0.0,0.05,0.10,0.20]:
        Cp = make_Cp(C,miss,false)
        A6['grid'][(miss,false)] = single_collect(run(X,C,cfg,'shared',Cp=Cp), cfg)
    log(f"A6 grid miss={miss}")
cfgM = cfg_with(M=5); XM,CM = make_env(cfgM)
for eps in [0.0,0.05,0.10,0.20]:
    Cp = make_Cp(CM,eps,eps)
    for mode in ['coord','naive']:
        A6['multi'][(eps,mode)] = multi_collect(run(XM,CM,cfgM,'shared',mode=mode,Cp=Cp), cfgM)
    log(f"A6 multi {eps}")
OUT['A6'] = A6
save(OUT,'final_A2'); log('saved final_A2')
