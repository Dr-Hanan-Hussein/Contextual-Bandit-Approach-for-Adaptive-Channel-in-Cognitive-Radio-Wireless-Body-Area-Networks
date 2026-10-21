"""Option B test: bandwidth enters the routine reward, emergency reward is reliability-only."""
import numpy as np
from crwban_sim import make_env, cfg_with, ucb_q, summarize_single

def run_single2(X, C, cfg, policy, seed=7):
    S, Tk, N = X.shape
    T = C.shape[1]
    ar = np.arange(S)
    w = np.array([cfg['wR'], cfg['wE']])
    beta = np.array(cfg['beta'], float)
    theta = 1 - np.array(cfg['busy'])
    Erew = np.stack([w[0] * beta * theta, w[1] * theta])      # expected reward per context, per channel
    rng = np.random.default_rng(seed)
    c_all = C[:, :, 0].astype(int)
    mean = np.zeros((S, 2, N)); cnt = np.zeros((S, 2, N))
    out = dict(idle=np.zeros((S, T), bool), reward=np.zeros((S, T)), regret=np.zeros((S, T)),
               ctx=c_all.copy(), energy=np.zeros((S, T)), lat=np.full((S, T), np.nan),
               chan=np.zeros((S, T), int))
    K = cfg['K_retry']
    if policy in ('random', 'rr', 'fixctx'):
        kind, mode = policy, None
    else:
        kind = 'cucb' if policy.startswith('cucb') else 'ucb1'
        mode = 'bin' if policy.endswith('_bin') else ('n' if policy.endswith('_n') else 'w')
    for t in range(T):
        c = c_all[:, t]
        Q = None
        if kind == 'random':
            a = rng.integers(0, N, S)
        elif kind == 'rr':
            a = np.full(S, t % N)
        elif kind == 'fixctx':
            Q = Erew[c]; a = Q.argmax(1)
        else:
            k = np.zeros(S, int) if kind == 'ucb1' else c
            Q = ucb_q(mean[ar, k], cnt[ar, k], cfg['kappa'], rng.random((S, N)))
            a = Q.argmax(1)
        busy = X[ar, t, a]; idle = ~busy
        r = np.where(c == 0, w[0] * beta[a] * idle, w[1] * idle)
        idle_mat = ~X[:, t, :]
        orc = np.where(c == 0, w[0] * (idle_mat * beta).max(1), w[1] * idle_mat.any(1))
        out['idle'][:, t] = idle; out['reward'][:, t] = r; out['regret'][:, t] = orc - r
        out['chan'][:, t] = a
        out['energy'][:, t] = cfg['E_sense'] + cfg['E_tx'] + cfg['E_ret'] * busy
        if kind in ('cucb', 'ucb1'):
            lr = r if mode == 'w' else (r / w[c] if mode == 'n' else idle.astype(float))
            cnt[ar, k, a] += 1
            mean[ar, k, a] += (lr - mean[ar, k, a]) / cnt[ar, k, a]
        out['lat'][:, t] = np.where((c == 1) & idle, 1.0, np.nan)
        for s in np.where((c == 1) & busy)[0]:
            if kind == 'random': seq = rng.integers(0, N, K); seq[0] = a[s]
            elif kind == 'rr': seq = (a[s] + np.arange(K)) % N
            else:
                rank = np.argsort(-Q[s]); seq = np.array([rank[i % N] for i in range(K)])
            L = K
            for kk in range(1, K):
                if not X[s, t + kk, seq[kk]]: L = kk + 1; break
            out['lat'][s, t] = L
    return out

def pu_by_ctx(out):
    m0 = out['ctx'] == 0
    return 100 * (~out['idle'] & m0).sum(1) / m0.sum(1), 100 * (~out['idle'] & ~m0).sum(1) / (~m0).sum(1)
