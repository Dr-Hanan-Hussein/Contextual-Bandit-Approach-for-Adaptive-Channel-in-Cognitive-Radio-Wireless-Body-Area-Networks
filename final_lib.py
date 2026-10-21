import numpy as np, pickle, time, os
from scipy import stats
from crwban import *

def U_matrix(cfg):
    b,beta = profile(cfg); th = 1-b; w = np.array([cfg['wR'],cfg['wE']])
    return np.stack([w[0]*beta*th, w[1]*th])

def lat_stats(lat):
    L = np.asarray(lat).ravel(); L = L[~np.isnan(L)]
    return dict(n=int(len(L)), mean=float(L.mean()), p1=float((L==1).mean()), p2=float((L==2).mean()),
                p3p=float((L>=3).mean()), p95=float(np.percentile(L,95)), p99=float(np.percentile(L,99)))

def single_collect(o, cfg, curves=False, ma=100):
    r = summarize(o, cfg)
    U = U_matrix(cfg); Ub = U.max(1)
    ch = o['chan'][:,:,0]; cx = o['ctx'][:,:,0]
    pr = Ub[cx] - U[cx,ch]
    r['learn_regret'] = pr.sum(1)
    r['lat'] = lat_stats(o['lat'])
    if curves:
        S = pr.shape[0]
        cum = np.cumsum(o['regret'][:,:,0], 1)
        r['cum_regret_mean'] = cum.mean(0); r['cum_regret_se'] = cum.std(0,ddof=1)/np.sqrt(S)
        pu = o['pu'][:,:,0].mean(0)
        r['pu_ma'] = np.convolve(pu, np.ones(ma)/ma, mode='valid')
        r['cum_energy_mean'] = np.cumsum(o['energy'][:,:,0],1).mean(0)
        r['cum_learn_mean'] = np.cumsum(pr,1).mean(0)
    return r

def multi_collect(o, cfg):
    r = summarize(o, cfg)
    r['succ_per_su'] = o['succ'].sum(1)
    r['util_per_su'] = o['reward'].sum(1)
    r['lat'] = lat_stats(o['lat'])
    return r

def paired(a, b, key):
    x, y = a[key], b[key]
    return dict(rel=float(np.mean(x)/np.mean(y)-1), p=float(stats.ttest_rel(x,y)[1]), a=float(np.mean(x)), b=float(np.mean(y)))

def save(obj, name):
    os.makedirs('results', exist_ok=True)
    pickle.dump(obj, open(f'results/{name}.pkl','wb'))
