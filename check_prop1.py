import numpy as np
from crwban import *
# i.i.d. channels (persist = 1.0 => p01 + p10 = 1, i.e. Bernoulli each round), kappa_R = kappa_E = 1
cfg = cfg_with(persist=1.0, kR=1.0, kE=1.0, S=100, seed0=5000, T=6000)
X,C = make_env(cfg)
b,beta = profile(cfg); theta=1-b; w=np.array([cfg['wR'],cfg['wE']])
U = np.stack([w[0]*beta*theta, w[1]*theta]); T=cfg['T']
o = run(X,C,cfg,'shared'); ch=o['chan'][:,:,0]; cx=o['ctx'][:,:,0]
print("channel idle prob:", theta, " best channel per class:", U.argmax(1), " (0=routine, 1=emergency)")
tot_bound=0.0
print(f"{'class':>6s} {'channel':>8s} {'gap':>7s} {'simulated E[N_c(n)]':>20s} {'bound':>9s} {'ok':>4s}")
for c in [0,1]:
    star=U[c].argmax()
    for n in range(5):
        if n==star: continue
        gap=U[c,star]-U[c,n]
        sim=((ch==n)&(cx==c)).sum(1).mean()
        bound=8*w[c]**2*np.log(T)/gap**2+1+np.pi**2/3
        tot_bound+= 8*w[c]**2*np.log(T)/gap + (1+np.pi**2/3)*gap
        print(f"{c:6d} {n:8d} {gap:7.3f} {sim:20.1f} {bound:9.0f} {'yes' if sim<=bound else 'NO':>4s}")
pr = U.max(1)[cx]-U[cx,ch]
print(f"\nsimulated learning regret at T={T}: {pr.sum(1).mean():.0f}   Proposition-1 bound (Eq. 8): {tot_bound:.0f}")
