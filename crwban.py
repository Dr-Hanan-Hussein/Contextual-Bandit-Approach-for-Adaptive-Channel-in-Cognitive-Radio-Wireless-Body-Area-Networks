"""
crwban.py - simulator for contextual-bandit channel selection in CR-enabled WBANs.

MODEL
  * N licensed channels. Each channel is a DISCRETE-TIME, STATIONARY two-state Markov chain
    (0 idle, 1 busy): p01 = persist*b_n, p10 = persist*(1-b_n); stationary busy probability = b_n.
  * M secondary users (SUs). Every round each SU has one packet; it is emergency with prob. pE.
  * Channel n has a bandwidth weight beta_n in (0,1], independent of its reliability.
  * Reward of a successful transmission (idle channel, no other SU on it):
        routine   : wR * beta_n        (throughput oriented)
        emergency : wE                 (reliability oriented, bandwidth irrelevant)
    A failed transmission (PU busy or SU-SU collision) gives 0.
  * Genie oracle: best idle channel for the packet's context.  Regret = oracle reward - reward.
  * Energy per attempt: E_sense + E_tx, plus E_ret if the attempt fails.
  * Emergency latency = attempts needed to deliver the packet. Attempt 1 is the real transmission.
    On failure the packet is retried on the next-best channels in the following slots
    (measured on the true PU trajectory; shadow retries do not change the learner).

POLICIES (name -> description)
  random, rr           : non-adaptive baselines
  ucb1                 : context-free UCB1, learns from the weighted reward (as in the first submission)
  ucb1_n               : context-free UCB1, reward normalised by the context weight
  ucb1_bin             : context-free UCB1, reliability only (idle = 1, busy = 0)
  part                 : contextual UCB with SEPARATE statistics per context (first-submission design)
  shared               : PROPOSED contextual UCB: one shared reliability estimate per channel,
                         context-dependent index (routine: beta_n*(theta+bonus); emergency: theta+bonus),
                         context-dependent exploration (kR, kE)
  fixctx               : reference, best fixed channel per context (knows the true statistics)

MULTI-SU MODES
  naive : every SU takes its own best-index channel; two SUs on one channel both fail.
  coord : gateway; emergency SUs first, random order inside a class; each SU gets its best-index
          channel not yet reserved (no SU-SU collisions). If M > N the last SUs wait.
"""
import numpy as np

DEFAULT = dict(
    N=5, M=1, T=6000, S=25,
    busy=(0.11, 0.28, 0.42, 0.64, 0.78),
    beta=(0.6, 1.0, 0.8, 0.7, 0.9),
    persist=0.5,                 # p01 + p10
    pE=0.15,
    wR=2.5, wE=16.0,
    kappa=1.0,                   # exploration coefficient of ucb1 / part
    kR=1.0, kE=0.25,             # exploration coefficients of the proposed policy (routine / emergency)
    mu=0.0,                      # optional PU-collision penalty (sensitivity study only)
    E_sense=0.2, E_tx=1.0, E_ret=1.6,
    K_retry=10,
    seed0=1000,
    priority=True,               # gateway serves emergency SUs first
    # baselines (tuned on calibration seeds 5000+)
    alpha_lin=1.25, lin_norm=True,         # LinUCB exploration; normalised reward
    q_alpha=0.3, q_gamma=0.0, q_eps_min=0.02, q_tau=200.0,   # tabular Q-learning
)


def cfg_with(**kw):
    c = dict(DEFAULT)
    c.update(kw)
    return c


def profile(cfg):
    """Busy probabilities and bandwidths for N channels (N=5 uses the published values;
    other N: busy interpolated over the same 11-78% range, bandwidth pattern repeated)."""
    N = cfg['N']
    if N == len(cfg['busy']):
        return np.array(cfg['busy'], float), np.array(cfg['beta'], float)
    b = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(cfg['busy'])), cfg['busy'])
    beta = np.resize(np.array(cfg['beta'], float), N)
    return b, beta


# -----------------------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------------------
def make_env(cfg):
    S, T, N, M, K = cfg['S'], cfg['T'], cfg['N'], cfg['M'], cfg['K_retry']
    b, _ = profile(cfg)
    p01, p10 = cfg['persist'] * b, cfg['persist'] * (1 - b)
    X = np.zeros((S, T + K, N), dtype=bool)          # True = busy
    C = np.zeros((S, T, M), dtype=bool)              # True = emergency
    for s in range(S):
        rng = np.random.default_rng(cfg['seed0'] + s)
        u = rng.random((T + K, N))
        state = rng.random(N) < b
        for t in range(T + K):
            X[s, t] = state
            state = state ^ np.where(state, u[t] < p10, u[t] < p01)
        C[s] = rng.random((T, M)) < cfg['pE']
    return X, C


# -----------------------------------------------------------------------------------------
# Policies. Each policy returns (q, update); q(c,t,rng) -> index [S,M,N]
# -----------------------------------------------------------------------------------------
def ucb_q(mean, cnt, kappa, noise):
    Tk = cnt.sum(-1, keepdims=True)
    bonus = kappa * np.sqrt(2.0 * np.log(np.maximum(Tk, 1.0)) / np.maximum(cnt, 1.0))
    return np.where(cnt == 0, 1e9 + noise, mean + bonus + 1e-9 * noise)


def make_policy(name, cfg, S, M, N):
    w = np.array([cfg['wR'], cfg['wE']])
    b, beta = profile(cfg)
    theta = 1 - b
    mu = cfg['mu']

    if name == 'random':
        return (lambda c, t, rng: rng.random((S, M, N))), (lambda *a: None)

    if name == 'rr':
        off = np.arange(M)[None, :, None]
        nn = np.arange(N)[None, None, :]
        return (lambda c, t, rng: -((nn - (t + off)) % N) + 1e-9 * rng.random((S, M, N))), (lambda *a: None)

    if name == 'fixctx':
        Erew = np.stack([w[0] * beta * theta, w[1] * theta])
        return (lambda c, t, rng: Erew[c] + 1e-9 * rng.random((S, M, N))), (lambda *a: None)

    if name in ('ucb1', 'ucb1_n', 'ucb1_bin'):
        mean = np.zeros((S, M, N)); cnt = np.zeros((S, M, N))

        def q(c, t, rng):
            return ucb_q(mean, cnt, cfg['kappa'], rng.random((S, M, N)))

        def upd(c, a, tx, succ, r, su):
            si, mi = np.nonzero(tx)
            aa, cc = a[si, mi], c[si, mi]
            lr = {'ucb1': r[si, mi], 'ucb1_n': r[si, mi] / w[cc], 'ucb1_bin': succ[si, mi].astype(float)}[name]
            cnt[si, mi, aa] += 1
            mean[si, mi, aa] += (lr - mean[si, mi, aa]) / cnt[si, mi, aa]
        return q, upd

    if name == 'part':
        mean = np.zeros((S, M, 2, N)); cnt = np.zeros((S, M, 2, N))

        def q(c, t, rng):
            mc = np.take_along_axis(mean, c[:, :, None, None], axis=2)[:, :, 0, :]
            cc = np.take_along_axis(cnt, c[:, :, None, None], axis=2)[:, :, 0, :]
            return ucb_q(mc, cc, cfg['kappa'], rng.random((S, M, N)))

        def upd(c, a, tx, succ, r, su):
            si, mi = np.nonzero(tx)
            aa, cc = a[si, mi], c[si, mi]
            cnt[si, mi, cc, aa] += 1
            mean[si, mi, cc, aa] += (r[si, mi] - mean[si, mi, cc, aa]) / cnt[si, mi, cc, aa]
        return q, upd

    if name == 'shared':
        th = np.zeros((S, M, N)); V = np.zeros((S, M, N))

        def q(c, t, rng):
            tt = np.maximum(V.sum(-1, keepdims=True), 1.0)
            bon = np.sqrt(2.0 * np.log(tt) / np.maximum(V, 1.0))
            kap = np.where(c == 1, cfg['kE'], cfg['kR'])[..., None]
            up = th + kap * bon
            Qr = (beta[None, None, :] + mu) * up - mu
            Q = np.where((c == 1)[..., None], up, Qr)
            return np.where(V == 0, 1e9 + rng.random((S, M, N)), Q + 1e-9 * rng.random((S, M, N)))

        def upd(c, a, tx, succ, r, su):
            si, mi = np.nonzero(tx)
            aa = a[si, mi]
            V[si, mi, aa] += 1
            th[si, mi, aa] += (succ[si, mi].astype(float) - th[si, mi, aa]) / V[si, mi, aa]
        return q, upd

    if name == 'linucb':
        # disjoint LinUCB (Li et al. 2010): one ridge regression per channel, features x = [1, c]
        a11 = np.ones((S, M, N)); a12 = np.zeros((S, M, N)); a22 = np.ones((S, M, N))
        b1 = np.zeros((S, M, N)); b2 = np.zeros((S, M, N))

        def q(c, t, rng):
            cx = c[..., None].astype(float)
            det = a11 * a22 - a12 ** 2
            mean = ((a22 * b1 - a12 * b2) + cx * (-a12 * b1 + a11 * b2)) / det
            var = (a22 - 2 * cx * a12 + cx ** 2 * a11) / det
            return mean + cfg['alpha_lin'] * np.sqrt(np.maximum(var, 0.0)) + 1e-9 * rng.random((S, M, N))

        def upd(c, a, tx, succ, r, su):
            si, mi = np.nonzero(tx)
            aa, cc = a[si, mi], c[si, mi].astype(float)
            rr = r[si, mi] / w[c[si, mi]] if cfg['lin_norm'] else r[si, mi]
            a11[si, mi, aa] += 1.0; a12[si, mi, aa] += cc; a22[si, mi, aa] += cc * cc
            b1[si, mi, aa] += rr; b2[si, mi, aa] += rr * cc
        return q, upd

    if name == 'ts':
        # per-context Beta Thompson sampling (Bernoulli trick for rewards in [0,1])
        al = np.ones((S, M, 2, N)); be = np.ones((S, M, 2, N))
        prng = np.random.default_rng(4242)

        def q(c, t, rng):
            a_ = np.take_along_axis(al, c[:, :, None, None], axis=2)[:, :, 0, :]
            b_ = np.take_along_axis(be, c[:, :, None, None], axis=2)[:, :, 0, :]
            return prng.beta(a_, b_)

        def upd(c, a, tx, succ, r, su):
            si, mi = np.nonzero(tx)
            aa, cc = a[si, mi], c[si, mi]
            y = (prng.random(len(si)) < r[si, mi] / w[cc]).astype(float)
            al[si, mi, cc, aa] += y
            be[si, mi, cc, aa] += 1.0 - y
        return q, upd

    if name == 'ts_shared':
        # Thompson sampling with the proposed structure (shared reliability, known bandwidth)
        al = np.ones((S, M, N)); be = np.ones((S, M, N))
        prng = np.random.default_rng(4243)

        def q(c, t, rng):
            smp = prng.beta(al, be)
            return np.where((c == 1)[..., None], smp, beta[None, None, :] * smp)

        def upd(c, a, tx, succ, r, su):
            si, mi = np.nonzero(tx)
            aa = a[si, mi]
            y = succ[si, mi].astype(float)
            al[si, mi, aa] += y
            be[si, mi, aa] += 1.0 - y
        return q, upd

    if name == 'qlearn':
        # tabular Q-learning, state = (traffic class, last channel, last outcome); epsilon-greedy
        nS = 2 * N * 2
        Qt = np.zeros((S, M, nS, N))
        last_ch = np.zeros((S, M), int); last_out = np.zeros((S, M), int)
        pend = np.zeros((S, M), bool); p_s = np.zeros((S, M), int); p_a = np.zeros((S, M), int); p_r = np.zeros((S, M))
        cur = {}
        Sg, Mg = np.meshgrid(np.arange(S), np.arange(M), indexing='ij')
        al_, ga = cfg['q_alpha'], cfg['q_gamma']
        prng = np.random.default_rng(4244)

        def q(c, t, rng):
            s = c * (2 * N) + last_ch * 2 + last_out
            if pend.any():
                best = Qt[Sg, Mg, s].max(-1)
                si, mi = np.nonzero(pend)
                tgt = p_r[si, mi] + ga * best[si, mi]
                Qt[si, mi, p_s[si, mi], p_a[si, mi]] += al_ * (tgt - Qt[si, mi, p_s[si, mi], p_a[si, mi]])
                pend[:] = False
            idx = Qt[Sg, Mg, s]
            eps = max(cfg['q_eps_min'], np.exp(-t / cfg['q_tau']))
            expl = prng.random((S, M)) < eps
            idx = np.where(expl[..., None], prng.random((S, M, N)), idx + 1e-9 * prng.random((S, M, N)))
            cur['s'] = s
            return idx

        def upd(c, a, tx, succ, r, su):
            p_s[:] = cur['s']; p_a[:] = a
            p_r[:] = r / w[c]
            pend[:] = tx
            last_ch[tx] = a[tx]
            last_out[tx] = succ[tx].astype(int)
        return q, upd

    if name == 'rho_rand':
        # decentralised multi-user bandit (rho-rand, Anandkumar et al. 2011; cf. Liu & Zhao 2010).
        # UCB on channel availability; SU plays the channel of its random rank inside the top-K set;
        # after an SU-SU collision it redraws its rank and does not update its statistics.
        mean = np.zeros((S, M, N)); cnt = np.zeros((S, M, N))
        K = min(M, N)
        prng = np.random.default_rng(4245)
        rank = prng.integers(0, K, (S, M))
        Sg, Mg = np.meshgrid(np.arange(S), np.arange(M), indexing='ij')

        def q(c, t, rng):
            Qu = ucb_q(mean, cnt, cfg['kappa'], rng.random((S, M, N)))
            order = np.argsort(-Qu, axis=2)
            a_ = order[Sg, Mg, rank]
            idx = np.zeros((S, M, N))
            idx[Sg, Mg, a_] = 1.0
            return idx

        def upd(c, a, tx, succ, r, su):
            col = su & tx
            rank[col] = prng.integers(0, K, int(col.sum()))
            ok = tx & ~su
            si, mi = np.nonzero(ok)
            aa = a[si, mi]
            cnt[si, mi, aa] += 1
            mean[si, mi, aa] += (succ[si, mi].astype(float) - mean[si, mi, aa]) / cnt[si, mi, aa]
        return q, upd

    raise ValueError(name)


# -----------------------------------------------------------------------------------------
# Runner (single SU is M = 1)
# -----------------------------------------------------------------------------------------
def run(X, C, cfg, policy, mode='coord', seed=11, Cp=None):
    """Cp: optional PERCEIVED traffic class [S,T,M]. The learner and the gateway use Cp; rewards, regret,
    latency use the true class C. Cp=None means perfect knowledge of the class."""
    S, Tk, N = X.shape
    T, M = C.shape[1], C.shape[2]
    ar = np.arange(S)
    w = np.array([cfg['wR'], cfg['wE']])
    _, beta = profile(cfg)
    K = cfg['K_retry']
    rng = np.random.default_rng(seed)
    q, upd = make_policy(policy, cfg, S, M, N)
    sh = (S, T, M)
    out = dict(regret=np.zeros(sh), reward=np.zeros(sh), pu=np.zeros(sh, bool), su=np.zeros(sh, bool),
               succ=np.zeros(sh, bool), tx=np.zeros(sh, bool), energy=np.zeros(sh),
               ctx=C.astype(int).copy(), lat=np.full(sh, np.nan), chan=np.zeros(sh, int))
    for t in range(T):
        c = C[:, t, :].astype(int)                                  # true class
        cp = c if Cp is None else Cp[:, t, :].astype(int)           # perceived class
        Q = q(cp, t, rng)                                           # [S,M,N]
        if mode == 'naive' or M == 1:
            a = Q.argmax(2)
        elif cfg.get('assign', 'greedy') == 'opt':
            # optimal assignment (Hungarian) on utility-scaled indices, for the optimality-gap study
            from scipy.optimize import linear_sum_assignment
            Wt = np.minimum(Q, 5.0) * w[cp][..., None]
            a = np.full((S, M), -1)
            for s_ in range(S):
                ri, ci = linear_sum_assignment(-Wt[s_])
                a[s_, ri] = ci
        else:
            order = np.argsort((1 - cp) * (1.0 if cfg.get('priority', True) else 0.0) + rng.random((S, M)), axis=1)  # emergency first (if priority), random inside class
            reserved = np.zeros((S, N), bool)
            a = np.full((S, M), -1)
            for j in range(M):
                mm = order[:, j]
                qq = np.where(reserved, -np.inf, Q[ar, mm])
                aj = qq.argmax(1)
                valid = ~reserved.all(1)
                a[ar, mm] = np.where(valid, aj, -1)
                idx = np.where(valid)[0]
                reserved[idx, aj[idx]] = True
        tx = a >= 0
        a_s = np.where(tx, a, 0)
        occ = np.zeros((S, N), int)
        for m in range(M):
            idx = np.where(tx[:, m])[0]
            np.add.at(occ, (idx, a_s[idx, m]), 1)
        su = tx & (np.take_along_axis(occ, a_s, axis=1) >= 2)
        busy = X[ar[:, None], t, a_s] & tx
        succ = tx & ~busy & ~su
        r = np.where(c == 0, w[0] * beta[a_s], w[1]) * succ
        im = ~X[:, t, :]
        orc = np.where(c == 0, w[0] * (im * beta).max(1)[:, None], w[1] * im.any(1)[:, None])
        out['regret'][:, t] = orc - r
        out['reward'][:, t] = r
        out['pu'][:, t] = busy
        out['su'][:, t] = su
        out['succ'][:, t] = succ
        out['tx'][:, t] = tx
        out['chan'][:, t] = a
        fail = tx & ~succ
        out['energy'][:, t] = cfg['E_sense'] + tx * cfg['E_tx'] + cfg['E_ret'] * fail
        r_l = np.where(cp == 0, w[0] * beta[a_s], w[1]) * succ        # reward as seen by the learner
        upd(cp, a_s, tx, succ, r_l, su)
        # emergency latency
        lat = np.full((S, M), np.nan)
        lat[(c == 1) & succ] = 1.0
        for s, m in np.argwhere((c == 1) & tx & ~succ):
            rank = np.argsort(-Q[s, m])
            seq = [a[s, m]] + [x for x in rank if x != a[s, m]]
            seq = [seq[i % N] for i in range(K)]
            L = K
            for kk in range(1, K):
                if not X[s, t + kk, seq[kk]]:
                    L = kk + 1
                    break
            lat[s, m] = L
        lat[(c == 1) & ~tx] = K
        out['lat'][:, t] = lat
    return out


# -----------------------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------------------
def jain(x, axis=-1):
    x = np.asarray(x, float)
    return x.sum(axis) ** 2 / (x.shape[axis] * (x ** 2).sum(axis))


def summarize(out, cfg):
    reg, ctx, tx = out['regret'], out['ctx'], out['tx']
    S, T, M = reg.shape
    m0, m1 = ctx == 0, ctx == 1
    per_su = reg.sum(1)
    r = {}
    r['regret'] = reg.sum((1, 2))                                   # system-wide cumulative regret
    r['regret_per_su'] = per_su
    r['reg_routine'] = (reg * m0).sum((1, 2)) / m0.sum((1, 2))      # per routine packet
    r['reg_emerg'] = (reg * m1).sum((1, 2)) / np.maximum(m1.sum((1, 2)), 1)
    ntx = np.maximum(tx.sum((1, 2)), 1)
    r['pu'] = out['pu'].sum((1, 2)) / ntx
    r['su_su'] = out['su'].sum((1, 2)) / ntx
    r['fail'] = 1 - out['succ'].sum((1, 2)) / ntx
    r['pu_r'] = (out['pu'] & m0).sum((1, 2)) / np.maximum((tx & m0).sum((1, 2)), 1)
    r['pu_e'] = (out['pu'] & m1).sum((1, 2)) / np.maximum((tx & m1).sum((1, 2)), 1)
    r['energy'] = out['energy'].sum((1, 2))
    r['energy_per_su'] = out['energy'].sum((1, 2)) / M
    r['throughput'] = out['reward'].sum((1, 2)) / (T * M)           # mean reward per SU-round
    r['succ_rate'] = out['succ'].sum((1, 2)) / (T * M)
    r['unserved'] = 1 - tx.sum((1, 2)) / (T * M)
    r['jain_regret'] = jain(per_su, axis=1)
    r['jain_succ'] = jain(out['succ'].sum(1), axis=1)
    r['jain_thr'] = jain(out['reward'].sum(1), axis=1)
    r['range_regret'] = per_su.max(1) - per_su.min(1)
    lat = out['lat']
    with np.errstate(all='ignore'):
        r['lat_mean'] = np.nanmean(lat.reshape(S, -1), axis=1)
        r['lat_p95'] = np.nanpercentile(lat.reshape(S, -1), 95, axis=1)
    return r
