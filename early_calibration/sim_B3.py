"""Option B variants. Same environment/reward as sim_B; different learners."""
import numpy as np
from crwban_sim import make_env, cfg_with, ucb_q

def run_variant(X, C, cfg, variant, kR=1.0, kE=1.0, mu=0.0, seed=7):
    """variant:
       'ctx_raw'   : per-context stats, raw weighted reward, kappa=cfg kappa (= B baseline, Eq.4 as published)
       'ctx_n'     : per-context stats, normalised reward, separate kappa kR / kE
       'shared'    : ONE reliability estimate shared by both contexts; bandwidth known;
                     routine index = (beta+mu)*(theta_up) - mu ; emergency index = theta_up  (kappa kR / kE)
    """
    S, Tk, N = X.shape; T = C.shape[1]; ar = np.arange(S)
    w = np.array([cfg['wR'], cfg['wE']]); beta = np.array(cfg['beta'], float)
    theta = 1 - np.array(cfg['busy'])
    Erew = np.stack([w[0]*beta*theta, w[1]*theta])
    rng = np.random.default_rng(seed)
    c_all = C[:, :, 0].astype(int)
    mean = np.zeros((S, 2, N)); cnt = np.zeros((S, 2, N))      # per-context stats
    th = np.zeros((S, N)); V = np.zeros((S, N))                # shared stats
    out = dict(idle=np.zeros((S, T), bool), reward=np.zeros((S, T)), regret=np.zeros((S, T)),
               ctx=c_all.copy(), chan=np.zeros((S, T), int))
    for t in range(T):
        c = c_all[:, t]
        noise = rng.random((S, N))
        if variant == 'ctx_raw':
            Q = ucb_q(mean[ar, c], cnt[ar, c], cfg['kappa'], noise)
        elif variant == 'ctx_n':
            kap = np.where(c == 1, kE, kR)[:, None]
            Q = ucb_q(mean[ar, c], cnt[ar, c], kap, noise)
        else:  # shared
            tt = np.maximum(V.sum(1, keepdims=True), 1.0)
            b = np.sqrt(2.0 * np.log(tt) / np.maximum(V, 1.0))
            kap = np.where(c == 1, kE, kR)[:, None]
            up = th + kap * b
            Qr = (beta[None, :] + mu) * up - mu
            Qe = up
            Q = np.where((c == 1)[:, None], Qe, Qr)
            unv = V == 0
            if kE == 0:   # emergency never explores: ignore unvisited channels unless none visited
                emer_unv = (c == 1)[:, None] & unv
                allunv = unv.all(1, keepdims=True)
                Q = np.where(emer_unv & ~allunv, -1.0, Q)
                Q = np.where((c == 1)[:, None] & allunv, 1e9 + noise, Q)
                Q = np.where((c == 0)[:, None] & unv, 1e9 + noise, Q)
                Q = np.where(~((c == 1)[:, None] & unv) & ~((c == 0)[:, None] & unv), Q + 1e-9 * noise, Q)
            else:
                Q = np.where(unv, 1e9 + noise, Q + 1e-9 * noise)
        a = Q.argmax(1)
        busy = X[ar, t, a]; idle = ~busy
        r = np.where(c == 0, w[0]*beta[a]*idle, w[1]*idle)
        im = ~X[:, t, :]
        orc = np.where(c == 0, w[0]*(im*beta).max(1), w[1]*im.any(1))
        out['idle'][:, t] = idle; out['reward'][:, t] = r; out['regret'][:, t] = orc - r; out['chan'][:, t] = a
        if variant == 'ctx_raw':
            cnt[ar, c, a] += 1; mean[ar, c, a] += (r - mean[ar, c, a]) / cnt[ar, c, a]
        elif variant == 'ctx_n':
            cnt[ar, c, a] += 1; mean[ar, c, a] += (r / w[c] - mean[ar, c, a]) / cnt[ar, c, a]
        else:
            V[ar, a] += 1; th[ar, a] += (idle.astype(float) - th[ar, a]) / V[ar, a]
    return out

def metrics(out, cfg):
    ctx = out['ctx']; reg = out['regret']; m0 = ctx == 0; m1 = ~m0
    fr = reg.sum(1)
    return dict(final=fr,
                reg_r=(reg*m0).sum(1)/m0.sum(1), reg_e=(reg*m1).sum(1)/m1.sum(1),
                pu=100*(1-out['idle'].mean(1)),
                pu_r=100*(~out['idle'] & m0).sum(1)/m0.sum(1), pu_e=100*(~out['idle'] & m1).sum(1)/m1.sum(1),
                thr_r=(out['reward']*m0).sum(1)/m0.sum(1))
