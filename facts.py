import pickle, numpy as np
from final_lib import *
A1=pickle.load(open('results/final_A1.pkl','rb')); A2=pickle.load(open('results/final_A2.pkl','rb')); B=pickle.load(open('results/final_B.pkl','rb')); C1=pickle.load(open('results/final_C.pkl','rb'))['C1']
A3=A1['A3']; A2h=A1['A2']; A1=A1['A1']; A4=A2['A4']; A5=A2['A5']; A6=A2['A6']; B1=B['B1']; B3=B['B3']; B4=B['B4']; B5=B['B5']
def P(a,b,k): d=paired(a,b,k); return f"{100*d['rel']:+.1f}% (p={d['p']:.2g})"
print("== A1 paired: proposed vs ...  (genie regret | learning regret | emergency regret | PU | energy | utility)")
for b in ['ucb1','ucb1_bin','qlearn','part','linucb','ts','ts_shared']:
    print(f"{b:9s}", ' | '.join(P(A1['shared'],A1[b],k) for k in ['regret','learn_regret','reg_emerg','pu','energy','throughput']))
print("PU by class (routine/emergency %):", {p:(round(100*A1[p]['pu_r'].mean(),1),round(100*A1[p]['pu_e'].mean(),1)) for p in ['random','ucb1','ucb1_bin','part','linucb','shared','fixctx']})
print("energy saving vs random: proposed %.1f%%, ucb1_bin %.1f%%, ucb1 %.1f%%"%tuple(100*(1-A1[p]['energy'].mean()/A1['random']['energy'].mean()) for p in ['shared','ucb1_bin','ucb1']))
print("PU reduction factor vs random: proposed %.2f, ucb1_bin %.2f"%(A1['random']['pu'].mean()/A1['shared']['pu'].mean(), A1['random']['pu'].mean()/A1['ucb1_bin']['pu'].mean()))
print("genie regret per round slope fixctx:", A1['fixctx']['regret'].mean()/6000)
print("std of genie regret: part %.0f, shared %.0f, linucb %.0f, ucb1 %.0f"%tuple(A1[p]['regret'].std(ddof=1) for p in ['part','shared','linucb','ucb1']))
print("lock-in: share of runs with genie regret > 1.5x fixctx: ", {p: round(100*(A1[p]['regret']>1.5*A1['fixctx']['regret'].mean()).mean()) for p in ['ucb1','part','linucb','shared','ucb1_bin']})
print("\n== A2 horizons: proposed vs  (genie regret | emergency regret)")
for T in [3000,15000]:
    for b in ['ucb1','part','linucb','ucb1_bin']:
        print(T,b, P(A2h[T]['shared'],A2h[T][b],'regret'),'|',P(A2h[T]['shared'],A2h[T][b],'reg_emerg'))
print("T=6000", [ (b,P(A1['shared'],A1[b],'regret')) for b in ['ucb1','part','linucb','ucb1_bin']])
print("\n== A3 slopes per ln T between checkpoints", [1000,2000,5000,10000,20000,30000])
for k,d in A3.items():
    t=d['t']; m=d['mean']; cp=[1000,2000,5000,10000,20000,29950]; idx=[int(np.argmin(abs(t-c))) for c in cp]
    vals=[m[i] for i in idx]; sl=[(vals[i]-vals[i-1])/(np.log(t[idx[i]])-np.log(t[idx[i-1]])) for i in range(1,len(cp))]
    print(f"{k:20s} R(T)={[round(v) for v in vals]} slopes={[round(s) for s in sl]}  genie/round={(d['gen'][-1]-d['gen'][-200])/((t[-1]-t[-200])):.3f}")
print("\n== A4 emergency share: emergency cost vs ideal (%) | total vs proposed (%)")
for q in [0.01,0.05,0.15,0.30]:
    print(q, {p:(round(100*(A4[q][p]['reg_emerg'].mean()/A4[q]['fixctx']['reg_emerg'].mean()-1)), round(100*(A4[q][p]['regret'].mean()/A4[q]['shared']['regret'].mean()-1),1)) for p in ['ucb1_bin','part','linucb','ts','shared']}, " p(shared vs part emerg)=%.2g p(vs linucb total)=%.2g"%(paired(A4[q]['shared'],A4[q]['part'],'reg_emerg')['p'],paired(A4[q]['shared'],A4[q]['linucb'],'regret')['p']))
print("\n== A5 windows (emergency regret per packet) windows",A5['windows'])
for k,v in A5['rows'].items(): print(k,[round(x,3) for x in v])
print("\n== A6 class errors (sym): total regret / emergency regret")
for e in [0.0,0.05,0.10,0.20]:
    print(e, {p:(round(A6['sym'][(e,p)]['regret'].mean()),round(A6['sym'][(e,p)]['reg_emerg'].mean(),2)) for p in ['part','linucb','shared']}, 'shared vs linucb p=%.2g'%paired(A6['sym'][(e,'shared')],A6['sym'][(e,'linucb')],'regret')['p'])
print("one-sided: miss20 only:", round(A6['grid'][(0.2,0.0)]['regret'].mean()), round(A6['grid'][(0.2,0.0)]['reg_emerg'].mean(),2), "| false20 only:", round(A6['grid'][(0.0,0.2)]['regret'].mean()), round(A6['grid'][(0.0,0.2)]['reg_emerg'].mean(),2), "| base:", round(A6['grid'][(0,0)]['regret'].mean()), round(A6['grid'][(0,0)]['reg_emerg'].mean(),2))
print("multi under errors (gateway vs naive sys regret, emergency regret gateway):", {e:(round(A6['multi'][(e,'coord')]['regret'].mean()),round(A6['multi'][(e,'naive')]['regret'].mean()), round(A6['multi'][(e,'coord')]['reg_emerg'].mean(),2)) for e in [0.0,0.05,0.10,0.20]})
print("ucb1_bin regret (unaffected):", round(A1['ucb1_bin']['regret'].mean()))
print("\n== C1 sensitivity (genie regret | emergency regret) for ucb1, part, linucb, shared, fixctx; shared vs linucb")
for key in sorted({(k[0],k[1]) for k in C1}, key=str):
    print(key, {p:(round(C1[key+(p,)]['regret'].mean()),round(C1[key+(p,)]['reg_emerg'].mean(),2)) for p in ['ucb1','ucb1_bin','part','linucb','shared','fixctx']}, 'shared vs linucb %s ; vs ucb1 %s'%(P(C1[key+('shared',)],C1[key+('linucb',)],'regret'),P(C1[key+('shared',)],C1[key+('ucb1',)],'regret')))
print("\n== B1 multi")
for (M,n),r in B1.items():
    print(M,f"{n:28s} sys {r['regret'].mean():7.0f}±{r['regret'].std(ddof=1):4.0f} | emerg {r['reg_emerg'].mean():.2f} | rout {r['reg_routine'].mean():.3f} | PU {100*r['pu'].mean():.1f} (R {100*r['pu_r'].mean():.1f}/E {100*r['pu_e'].mean():.1f}) | SU-SU {100*r['su_su'].mean():.1f} | fail {100*r['fail'].mean():.1f} | succ {100*r['succ_rate'].mean():.1f} | util {r['throughput'].mean():.3f} | Jain succ {r['jain_succ'].mean():.4f} util {r['jain_thr'].mean():.4f} reg {r['jain_regret'].mean():.3f} | lat {r['lat']['mean']:.2f} p95 {r['lat']['p95']:.0f} first {100*r['lat']['p1']:.1f}% | energy/SU {r['energy_per_su'].mean():.0f}")
g5=B1[(5,'Gateway + priority')]
for n in ['No coordination','Decentralised rho-rand','Gateway, no priority']:
    print('proposed vs',n,{k:P(g5,B1[(5,n)],k) for k in ['regret','fail','reg_emerg','throughput','energy_per_su']})
print('optimal vs greedy:',{k:P(B1[(5,'Gateway optimal assignment')],g5,k) for k in ['regret','reg_emerg']})
print('M=3 proposed vs rho-rand:',P(B1[(3,'Gateway + priority')],B1[(3,'Decentralised rho-rand')],'regret'),' vs no coord', P(B1[(3,'Gateway + priority')],B1[(3,'No coordination')],'regret'))
print("jain on regret for modes:", {n: round(B1[(5,n)]['jain_regret'].mean(),3) for n in ['No coordination','Decentralised rho-rand','Gateway, no priority','Gateway + priority']})
print("seed-averaged per-SU regret Jain: ", {n: round(float(jain(B1[(5,n)]['regret_per_su'].mean(0))),4) for n in ['No coordination','Gateway + priority']})
print("\n== B3 sweep: M, mode: PU, SU-SU, succ, unserved, util")
for k,r in sorted(B3.items()): print(k, f"PU {100*r['pu'].mean():.1f} SUSU {100*r['su_su'].mean():.1f} succ {100*r['succ_rate'].mean():.1f} unserved {100*r['unserved'].mean():.1f} util {r['throughput'].mean():.3f}")
print("\n== B4 grid"); 
for k,r in sorted(B4.items()): print(k, f"succ {100*r['succ_rate'].mean():.1f} PU {100*r['pu'].mean():.1f} SUSU {100*r['su_su'].mean():.1f} unserved {100*r['unserved'].mean():.1f} jain {r['jain_succ'].mean():.3f}")
print("\n== B5", {k:v for k,v in B5.items() if k!='rows'}); 
for r,v in B5['rows'].items(): print(r, v, 'gain %.1f%%'%(100*(1-v['reg_prio']/v['reg_noprio'])))
