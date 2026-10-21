import numpy as np, time
from crwban_sim import *
targets = dict(ucb1_pu=12.5, cucb_pu=12.0, ucb1_rr=0.298, cucb_rr=0.251, early=20.0)
rows=[]
t0=time.time()
for persist in [0.3, 0.4, 0.5]:
    cfg0 = cfg_with(persist=persist, S=100, seed0=5000)
    X,C = make_env(cfg0)
    for kappa in [0.8, 0.9, 1.0, 1.1, 1.25]:
        cfg = cfg_with(persist=persist, kappa=kappa, S=100, seed0=5000)
        res={}
        for pol in ['ucb1','cucb']:
            o = run_single(X,C,cfg,pol); res[pol]=(summarize_single(o,cfg), o)
        u,c = res['ucb1'][0], res['cucb'][0]
        early = 100*(1-res['cucb'][1]['idle'][:,:100].mean())
        row = dict(persist=persist,kappa=kappa,
                   ucb1_pu=100*u['pu_coll'].mean(), cucb_pu=100*c['pu_coll'].mean(),
                   ucb1_rr=u['reg_routine'].mean(), cucb_rr=c['reg_routine'].mean(),
                   ucb1_re=u['reg_emerg'].mean(), cucb_re=c['reg_emerg'].mean(), early=early)
        row['loss']=sum(((row[k]-v)/v)**2 for k,v in targets.items())
        rows.append(row)
        print(f"p={persist:.1f} k={kappa:.2f} | PU ucb1 {row['ucb1_pu']:5.1f} cucb {row['cucb_pu']:5.1f} | routine ucb1 {row['ucb1_rr']:.3f} cucb {row['cucb_rr']:.3f} | emerg ucb1 {row['ucb1_re']:.2f} cucb {row['cucb_re']:.2f} | first100 {early:4.1f}% | loss {row['loss']:.4f}", flush=True)
print('time', round(time.time()-t0))
for r in sorted(rows,key=lambda r:r['loss'])[:4]: print({k:round(float(v),3) for k,v in r.items()})
