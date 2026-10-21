"""
CR-WBAN contextual-bandit simulator (rebuilt version).

Model (matches Eq. 1-8 of the manuscript, with the reward of Eq. 3 only):
  * N licensed channels. Each channel is a DISCRETE-TIME, STATIONARY two-state
    Markov chain (0 = idle, 1 = busy).  p01 = persist * b_n,  p10 = persist * (1 - b_n),
    so the stationary busy probability is b_n.
  * Each SU has one packet per round. It is emergency with probability pE.
  * Reward (Eq. 3): wR (routine) or wE (emergency) if the chosen channel is idle
    and no other SU uses it; 0 otherwise.
  * Genie oracle: gets w_c whenever at least one channel is idle.
    Regret_t = oracle reward - obtained reward.
  * Energy (Eq. 5): E_sense + E_tx per attempt, plus E_ret on a failed attempt.
  * Latency: attempts needed to deliver an emergency packet. The first attempt is
    the real transmission. If it fails, the packet is retried in the next slots on
    the next-best channels. Retries are measured on the true PU trajectory but
    do not change the learner (shadow retries).

All policies use the SAME random channel trajectories and traffic draws (paired).
"""
import numpy as np

DEFAULT = dict(
    N=5, M=1, T=6000, S=25,
    busy=(0.11, 0.28, 0.42, 0.64, 0.78),
    persist=0.3,          # p01 + p10
    pE=0.15,
    wR=2.5, wE=16.0,
    kappa=1.0,
    E_sense=0.2, E_tx=1.0, E_ret=1.6,
    K_retry=10,
    seed0=1000,
)


def cfg_with(**kw):
    c = dict(DEFAULT)
    c.update(kw)
    return c


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def make_env(cfg):
    S, T, N, M, K = cfg['S'], cfg['T'], cfg['N'], cfg['M'], cfg['K_retry']
    b = np.array(cfg['busy'], dtype=float)
    if len(b) != N:                      # allow N != 5 by interpolating the busy profile
        b = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(cfg['busy'])), cfg['busy'])
    p01 = cfg['persist'] * b
    p10 = cfg['persist'] * (1 - b)
    X = np.zeros((S, T + K, N), dtype=bool)   # True = busy
    C = np.zeros((S, T, M), dtype=bool)       # True = emergency
    for s in range(S):
        rng = np.random.default_rng(cfg['seed0'] + s)
        u = rng.random((T + K, N))
        state = rng.random(N) < b
        for t in range(T + K):
            X[s, t] = state
            flip = np.where(state, u[t] < p10, u[t] < p01)
            state = state ^ flip
        C[s] = rng.random((T, M)) < cfg['pE']
    return X, C


# ----------------------------------------------------------------------------
# UCB helpers
# ----------------------------------------------------------------------------
def ucb_q(mean, cnt, kappa, noise):
    """mean, cnt: [..., N]. Unvisited channels get a huge index (tried first)."""
    Tk = cnt.sum(-1, keepdims=True)
    bonus = kappa * np.sqrt(2.0 * np.log(np.maximum(Tk, 1.0)) / np.maximum(cnt, 1.0))
    Q = mean + bonus
    return np.where(cnt == 0, 1e9 + noise, Q + 1e-9 * noise)


# ----------------------------------------------------------------------------
# Single SU
# ----------------------------------------------------------------------------
def run_single(X, C, cfg, policy, seed=7):
    """policy in: random, rr, ucb1 (mixed weighted reward), ucb1_bin (binary reward),
    cucb (contextual, weighted reward), cucb_bin (contextual, binary reward)."""
    S, Tk, N = X.shape
    T = C.shape[1]
    ar = np.arange(S)
    w = np.array([cfg['wR'], cfg['wE']])
    rng = np.random.default_rng(seed)
    c_all = C[:, :, 0].astype(int)
    mean = np.zeros((S, 2, N))
    cnt = np.zeros((S, 2, N))
    out = dict(idle=np.zeros((S, T), bool), reward=np.zeros((S, T)), regret=np.zeros((S, T)),
               ctx=c_all.copy(), energy=np.zeros((S, T)), lat=np.full((S, T), np.nan),
               chan=np.zeros((S, T), int))
    K = cfg['K_retry']
    for t in range(T):
        c = c_all[:, t]
        Q = None
        if policy == 'random':
            a = rng.integers(0, N, S)
        elif policy == 'rr':
            a = np.full(S, t % N)
        else:
            ctxfree = policy in ('ucb1', 'ucb1_bin')
            k = np.zeros(S, int) if ctxfree else c
            Q = ucb_q(mean[ar, k], cnt[ar, k], cfg['kappa'], rng.random((S, N)))
            a = Q.argmax(1)
        busy = X[ar, t, a]
        idle = ~busy
        r = w[c] * idle
        anyidle = (~X[:, t, :]).any(1)
        out['idle'][:, t] = idle
        out['reward'][:, t] = r
        out['regret'][:, t] = w[c] * anyidle - r
        out['chan'][:, t] = a
        out['energy'][:, t] = cfg['E_sense'] + cfg['E_tx'] + cfg['E_ret'] * busy
        if Q is not None:
            learn_r = idle.astype(float) if policy.endswith('_bin') else r
            cnt[ar, k, a] += 1
            mean[ar, k, a] += (learn_r - mean[ar, k, a]) / cnt[ar, k, a]
        # latency of emergency packets (shadow retries)
        out['lat'][:, t] = np.where((c == 1) & idle, 1.0, np.nan)
        fail_e = np.where((c == 1) & busy)[0]
        for s in fail_e:
            if policy == 'random':
                seq = rng.integers(0, N, K)
                seq[0] = a[s]
            elif policy == 'rr':
                seq = (a[s] + np.arange(K)) % N
            else:
                rank = np.argsort(-Q[s])
                seq = np.array([rank[i % N] for i in range(K)])
            L = K
            for kk in range(1, K):
                if not X[s, t + kk, seq[kk]]:
                    L = kk + 1
                    break
            out['lat'][s, t] = L
    return out


def summarize_single(out, cfg):
    ctx = out['ctx']
    reg = out['regret']
    res = {}
    res['final_regret'] = reg.sum(1)
    res['pu_coll'] = 1.0 - out['idle'].mean(1)
    res['energy'] = out['energy'].sum(1)
    for k, name in ((0, 'reg_routine'), (1, 'reg_emerg')):
        m = ctx == k
        res[name] = (reg * m).sum(1) / np.maximum(m.sum(1), 1)
    res['lat_mean'] = np.nanmean(out['lat'], axis=1)
    return res


# ----------------------------------------------------------------------------
# Multiple SUs
# ----------------------------------------------------------------------------
def run_multi(X, C, cfg, mode, seed=11, policy='cucb'):
    """mode: 'naive' (independent) or 'coord' (gateway, emergency first)."""
    S, Tk, N = X.shape
    T, M = C.shape[1], C.shape[2]
    ar = np.arange(S)
    w = np.array([cfg['wR'], cfg['wE']])
    rng = np.random.default_rng(seed)
    mean = np.zeros((S, M, 2, N))
    cnt = np.zeros((S, M, 2, N))
    Sg, Mg = np.meshgrid(np.arange(S), np.arange(M), indexing='ij')
    out = dict(regret=np.zeros((S, T, M)), pu=np.zeros((S, T, M), bool), su=np.zeros((S, T, M), bool),
               succ=np.zeros((S, T, M), bool), tx=np.zeros((S, T, M), bool),
               energy=np.zeros((S, T, M)), ctx=C.astype(int).copy(), lat=np.full((S, T, M), np.nan),
               chan=np.zeros((S, T, M), int))
    K = cfg['K_retry']
    for t in range(T):
        c = C[:, t, :].astype(int)                      # [S,M]
        mc = np.take_along_axis(mean, c[:, :, None, None], axis=2)[:, :, 0, :]
        cc = np.take_along_axis(cnt, c[:, :, None, None], axis=2)[:, :, 0, :]
        if policy == 'cucb_noctx':   # not used here; kept simple
            mc, cc = mean[:, :, 0, :], cnt[:, :, 0, :]
        Q = ucb_q(mc, cc, cfg['kappa'], rng.random((S, M, N)))   # [S,M,N]
        if mode == 'naive':
            a = Q.argmax(2)
        else:
            key = (1 - c) + rng.random((S, M))          # emergency first, random inside class
            order = np.argsort(key, axis=1)
            reserved = np.zeros((S, N), bool)
            a = np.full((S, M), -1)
            for j in range(M):
                mm = order[:, j]
                q = np.where(reserved, -np.inf, Q[ar, mm])
                aj = q.argmax(1)
                valid = ~reserved.all(1)
                a[ar, mm] = np.where(valid, aj, -1)
                idx = np.where(valid)[0]
                reserved[idx, aj[idx]] = True
        tx = a >= 0
        a_safe = np.where(tx, a, 0)
        occ = np.zeros((S, N), int)
        for m in range(M):
            idx = np.where(tx[:, m])[0]
            np.add.at(occ, (idx, a_safe[idx, m]), 1)
        su = tx & (np.take_along_axis(occ, a_safe, axis=1) >= 2)
        busy = X[ar[:, None], t, a_safe] & tx
        succ = tx & ~busy & ~su
        r = w[c] * succ
        anyidle = (~X[:, t, :]).any(1)
        out['regret'][:, t, :] = w[c] * anyidle[:, None] - r
        out['pu'][:, t, :] = busy
        out['su'][:, t, :] = su
        out['succ'][:, t, :] = succ
        out['tx'][:, t, :] = tx
        out['chan'][:, t, :] = a
        fail = tx & ~succ
        out['energy'][:, t, :] = cfg['E_sense'] + tx * cfg['E_tx'] + cfg['E_ret'] * fail
        # learner update
        if tx.any():
            si, mi = Sg[tx], Mg[tx]
            ci, ai = c[tx], a[tx]
            cnt[si, mi, ci, ai] += 1
            mean[si, mi, ci, ai] += (r[tx] - mean[si, mi, ci, ai]) / cnt[si, mi, ci, ai]
        # emergency latency (shadow retries on PU trajectory only)
        lat = np.full((S, M), np.nan)
        lat[(c == 1) & succ] = 1.0
        fe = np.argwhere((c == 1) & tx & ~succ)
        for s, m in fe:
            rank = np.argsort(-Q[s, m])
            seq = [a[s, m]] + [x for x in rank if x != a[s, m]]
            seq = [seq[i % N] for i in range(K)]
            L = K
            for kk in range(1, K):
                if not X[s, t + kk, seq[kk]]:
                    L = kk + 1
                    break
            lat[s, m] = L
        fe2 = np.argwhere((c == 1) & ~tx)       # deferred emergency packet (M>N): count as K
        for s, m in fe2:
            lat[s, m] = K
        out['lat'][:, t, :] = lat
    return out


def jain(x, axis=-1):
    x = np.asarray(x, float)
    return x.sum(axis) ** 2 / (x.shape[axis] * (x ** 2).sum(axis))


def summarize_multi(out, cfg):
    reg = out['regret']
    per_su = reg.sum(1)                         # [S,M]
    tx = out['tx']
    res = {}
    res['system_regret'] = per_su.sum(1)
    res['su_su'] = out['su'].sum((1, 2)) / tx.sum((1, 2))
    res['su_pu'] = out['pu'].sum((1, 2)) / tx.sum((1, 2))
    res['fail'] = 1 - out['succ'].sum((1, 2)) / tx.sum((1, 2))
    res['jain_regret'] = jain(per_su, axis=1)
    res['range'] = per_su.max(1) - per_su.min(1)
    res['energy'] = out['energy'].sum((1, 2))
    res['lat_mean'] = np.nanmean(np.where(np.isnan(out['lat']), np.nan, out['lat']), axis=(1, 2))
    res['succ_per_su'] = out['succ'].sum(1)     # [S,M]
    res['jain_succ'] = jain(res['succ_per_su'], axis=1)
    return res
