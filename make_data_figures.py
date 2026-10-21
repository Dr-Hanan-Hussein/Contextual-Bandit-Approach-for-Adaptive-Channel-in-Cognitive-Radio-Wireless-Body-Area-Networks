import numpy as np, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'font.size':9,'axes.labelsize':9,'axes.titlesize':9.5,
                 'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5,
                 'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':450,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavuserif'})
A1 = pickle.load(open('results/final_A1.pkl','rb')); A2 = pickle.load(open('results/final_A2.pkl','rb')); B = pickle.load(open('results/final_B.pkl','rb'))
A3 = A1['A3']; A1 = A1['A1']; A4=A2['A4']; A5=A2['A5']; A6=A2['A6']; B1=B['B1']; B3=B['B3']; B4=B['B4']; B5=B['B5']
import os; os.makedirs('figures/final', exist_ok=True)
OUT='figures/final/'
LAB = {'random':'Random','rr':'Round-robin','qlearn':'Q-learning','ucb1':'UCB1 (mixed reward)','ucb1_n':'UCB1 (normalized)','ucb1_bin':'UCB1 (reliability only)',
       'part':'Partitioned UCB','linucb':'LinUCB','ts':'Thompson (per class)','ts_shared':'Thompson (shared)','shared':'Proposed','fixctx':'Best fixed channel'}
STY = {'random':('#8c8c8c',':',1.2),'rr':('#5c5c5c','--',1.0),'qlearn':('#CC79A7','-.',1.4),'ucb1':('#56B4E9','-',1.4),'ucb1_bin':('#009E73','--',1.4),
       'part':('#E69F00','-',1.4),'linucb':('#0072B2','-.',1.4),'ts':('#D55E00',':',1.6),'ts_shared':('#7a7a00','--',1.2),'shared':('#000000','-',2.2),'fixctx':('#444444',(0,(1,1)),1.2)}
def save(fig,name):
    fig.savefig(OUT+name+'.png',bbox_inches='tight'); fig.savefig(OUT+name+'.pdf',bbox_inches='tight'); plt.close(fig)
def ci(x): x=np.asarray(x); return 1.96*x.std(ddof=1)/np.sqrt(len(x))

# ---- Fig 7: cumulative genie regret (+ excess over best fixed channel)
fig,axs=plt.subplots(1,2,figsize=(6.5,3.3),gridspec_kw={'width_ratios':[1.25,1]}); T=len(A1['random']['cum_regret_mean']); t=np.arange(1,T+1)
for p in ['random','qlearn','ucb1','ucb1_bin','part','linucb','ts','shared','fixctx']:
    c,ls,lw=STY[p]; axs[0].plot(t,A1[p]['cum_regret_mean'],color=c,ls=ls,lw=lw,label=LAB[p])
axs[0].set_xlabel('Round $t$'); axs[0].set_ylabel('Cumulative genie regret'); axs[0].legend(ncol=1,loc='upper left',frameon=False,fontsize=7); axs[0].set_xlim(0,T); axs[0].set_title('(a) All policies',loc='left')
ref=A1['fixctx']['cum_regret_mean']
for p in ['ucb1_bin','part','linucb','ts','ts_shared','shared']:
    c,ls,lw=STY[p]; axs[1].plot(t,A1[p]['cum_regret_mean']-ref,color=c,ls=ls,lw=lw,label=LAB[p])
axs[1].set_xlabel('Round $t$'); axs[1].set_ylabel('Excess over best fixed channel'); axs[1].legend(frameon=False,fontsize=7,loc='upper left'); axs[1].set_xlim(0,T); axs[1].set_title('(b) Close-up',loc='left')
fig.tight_layout(); save(fig,'fig7_genie_regret')

# ---- Fig 8: PU collision moving average
fig,ax=plt.subplots(figsize=(6.5,3.2)); 
for p in ['random','qlearn','ucb1','linucb','shared','ucb1_bin']:
    c,ls,lw=STY[p]; y=100*A1[p]['pu_ma']; ax.plot(np.arange(100,100+len(y)),y,color=c,ls=ls,lw=lw,label=LAB[p])
ax.set_xlabel('Round $t$'); ax.set_ylabel('PU collision rate (%) (100-round average)'); ax.set_ylim(0,55); ax.legend(ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(0.5,1.0))
save(fig,'fig8_pu_collisions')

# ---- Fig 9: emergency latency distribution
fig,axs=plt.subplots(1,2,figsize=(6.5,3.2),gridspec_kw={'width_ratios':[1.35,1]})
def stack(ax,names,labels,get):
    x=np.arange(len(names)); p1=np.array([100*get(n)['p1'] for n in names]); p2=np.array([100*get(n)['p2'] for n in names]); p3=np.array([100*get(n)['p3p'] for n in names])
    ax.bar(x,p1,color='#009E73',label='1 attempt'); ax.bar(x,p2,bottom=p1,color='#E69F00',label='2 attempts'); ax.bar(x,p3,bottom=p1+p2,color='#D55E00',label='3 or more')
    for i in range(len(x)): ax.text(x[i],p1[i]/2,f'{p1[i]:.0f}',ha='center',va='center',color='white',fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha='right'); ax.set_ylim(0,100); ax.grid(axis='x',visible=False)
sn=['random','ucb1','ucb1_bin','qlearn','part','linucb','shared']
stack(axs[0],sn,[LAB[n].replace(' (mixed reward)','').replace(' (reliability only)',' (rel. only)') for n in sn],lambda n:A1[n]['lat']); axs[0].set_ylabel('Emergency packets (%)'); axs[0].set_title('(a) Single SU',loc='left')
mn=['No coordination','Decentralised rho-rand','Gateway, no priority','Gateway + priority']
stack(axs[1],mn,['No coordination','ρ-rand','Gateway,\nno priority','Gateway +\npriority'],lambda n:B1[(5,n)]['lat']); axs[1].set_title('(b) Five SUs',loc='left'); h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,0.99))
save(fig,'fig9_latency')

# ---- Fig 10: per-class regret
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0)); names=['ucb1','ucb1_bin','qlearn','part','linucb','ts','shared','fixctx']
for ax,key,title in [(axs[0],'reg_routine','(a) Routine packets'),(axs[1],'reg_emerg','(b) Emergency packets')]:
    m=[A1[n][key].mean() for n in names]; e=[ci(A1[n][key]) for n in names]
    ax.bar(range(len(names)),m,yerr=e,color=[STY[n][0] for n in names],capsize=2,edgecolor='black',linewidth=0.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LAB[n].replace(' (mixed reward)','').replace(' (reliability only)',' (rel. only)').replace(' (per class)','') for n in names],rotation=45,ha='right')
    ax.set_title(title,loc='left'); ax.set_ylabel('Genie regret per packet'); ax.grid(axis='x',visible=False)
save(fig,'fig10_per_class_regret')

# ---- Fig 11: emergency share sweep
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0)); pEs=[0.01,0.05,0.15,0.30]
for p in ['ucb1_bin','part','linucb','ts','shared']:
    c,ls,lw=STY[p]
    cost=[100*(A4[q][p]['reg_emerg'].mean()/A4[q]['fixctx']['reg_emerg'].mean()-1) for q in pEs]
    tot=[100*(A4[q][p]['regret'].mean()/A4[q]['shared']['regret'].mean()-1) for q in pEs]
    axs[0].plot([100*q for q in pEs],cost,color=c,ls=ls,lw=lw,marker='o',ms=3.5,label=LAB[p]); axs[1].plot([100*q for q in pEs],tot,color=c,ls=ls,lw=lw,marker='o',ms=3.5,label=LAB[p])
for ax,t,yl in [(axs[0],'(a) Emergency cost above the ideal','Extra emergency regret (%)'),(axs[1],'(b) Total regret vs. proposed','Total genie regret vs. proposed (%)')]:
    ax.set_xscale('log'); ax.set_xticks([1,5,15,30]); ax.set_xticklabels(['1','5','15','30']); ax.set_xlabel('Emergency share $p_E$ (%)'); ax.set_title(t,loc='left'); ax.set_ylabel(yl)
axs[0].legend(frameon=False,loc='upper right',fontsize=7)
save(fig,'fig11_emergency_share')

# ---- Fig 12: growth of the learning regret
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0)); sty={'shared (kE=0.25)':('#000000','-',2.0),'shared (kE=1)':('#7a7a7a','--',1.4),'part':STY['part'],'linucb':STY['linucb'],'ucb1_bin':STY['ucb1_bin']}
nm={'shared (kE=0.25)':'Proposed ($\\kappa_E$=0.25)','shared (kE=1)':'Proposed ($\\kappa_E$=1)','part':'Partitioned UCB','linucb':'LinUCB','ucb1_bin':'UCB1 (rel. only)'}
for k in ['shared (kE=0.25)','shared (kE=1)','linucb']:
    c,ls,lw=sty[k]; d=A3[k]; axs[0].plot(d['t'],d['mean'],color=c,ls=ls,lw=lw,label=nm[k])
axs[0].set_xscale('log'); axs[0].set_xlim(10,30000); axs[0].set_ylim(0,900); axs[0].set_xlabel('Round $t$ (log scale)'); axs[0].set_ylabel('Cumulative learning regret'); axs[0].set_title('(a) Logarithmic growth',loc='left'); axs[0].legend(frameon=False,fontsize=7,loc='upper left')
for k in ['part','ucb1_bin']:
    c,ls,lw=sty[k]; d=A3[k]; axs[1].plot(d['t'],d['mean'],color=c,ls=ls,lw=lw,label=nm[k])
d=A3['shared (kE=0.25)']; axs[1].plot(d['t'],d['mean'],color='black',lw=2,label=nm['shared (kE=0.25)'])
axs[1].set_xscale('log'); axs[1].set_xlim(10,30000); axs[1].set_xlabel('Round $t$ (log scale)'); axs[1].set_title('(b) Faster growth',loc='left'); axs[1].legend(frameon=False,fontsize=7,loc='upper left')
fig.tight_layout(); save(fig,'fig12_log_growth')

# ---- Fig 13: class errors
fig,axs=plt.subplots(2,2,figsize=(6.5,5.4)); eps=[0.0,0.05,0.10,0.20]
for p in ['part','linucb','shared']:
    c,ls,lw=STY[p]
    axs[0,0].plot([100*e for e in eps],[A6['sym'][(e,p)]['regret'].mean() for e in eps],color=c,ls=ls,lw=lw,marker='o',ms=3.5,label=LAB[p])
    axs[0,1].plot([100*e for e in eps],[A6['sym'][(e,p)]['reg_emerg'].mean() for e in eps],color=c,ls=ls,lw=lw,marker='o',ms=3.5,label=LAB[p])
for ax,key,lab in [(axs[0,0],'regret','Total genie regret'),(axs[0,1],'reg_emerg','Emergency regret per packet')]:
    ax.axhline(A1['ucb1_bin'][key].mean(),color=STY['ucb1_bin'][0],ls='--',lw=1.2,label='UCB1 (rel. only)'); ax.set_xlabel('Class error rate (%) (misses = false alarms)'); ax.set_ylabel(lab)
axs[0,0].set_title('(a)',loc='left'); axs[0,1].set_title('(b)',loc='left'); axs[0,0].legend(frameon=False,fontsize=7)
g=np.array([[A6['grid'][(m,f)]['regret'].mean() for f in eps] for m in eps]); im=axs[1,0].imshow(g,cmap='YlOrRd',origin='lower')
axs[1,0].set_xticks(range(4)); axs[1,0].set_yticks(range(4)); axs[1,0].set_xticklabels([f'{100*e:.0f}' for e in eps]); axs[1,0].set_yticklabels([f'{100*e:.0f}' for e in eps])
axs[1,0].set_xlabel('False-alarm rate (%)'); axs[1,0].set_ylabel('Missed-emergency rate (%)'); axs[1,0].set_title('(c) Proposed: total regret',loc='left'); axs[1,0].grid(False)
for i in range(4):
    for j in range(4): axs[1,0].text(j,i,f'{g[i,j]:.0f}',ha='center',va='center',fontsize=7.5)
axs[1,1].plot([100*e for e in eps],[A6['multi'][(e,'coord')]['regret'].mean() for e in eps],color='black',lw=2,marker='o',ms=3.5,label='Gateway + priority')
axs[1,1].plot([100*e for e in eps],[A6['multi'][(e,'naive')]['regret'].mean() for e in eps],color='#D55E00',ls='--',lw=1.4,marker='s',ms=3.5,label='No coordination')
axs[1,1].set_xlabel('Class error rate (%)'); axs[1,1].set_ylabel('System regret (5 SUs)'); axs[1,1].set_title('(d) Five SUs',loc='left'); axs[1,1].legend(frameon=False,fontsize=7)
fig.tight_layout(); save(fig,'fig13_class_errors')

# ---- Fig 14: multi-SU summary
fig,axs=plt.subplots(1,3,figsize=(6.5,3.2)); modes=['No coordination','Decentralised rho-rand','Gateway, no priority','Gateway + priority']
short=['No coord.','ρ-rand','Gateway\nno prio.','Gateway\n+ prio.']; cols=['#D55E00','#E69F00','#56B4E9','#000000']
axs[0].bar(range(4),[B1[(5,m)]['regret'].mean() for m in modes],yerr=[ci(B1[(5,m)]['regret']) for m in modes],color=cols,capsize=2,edgecolor='black',linewidth=0.5); axs[0].set_title('(a) System regret',loc='left')
pu=[100*B1[(5,m)]['pu'].mean() for m in modes]; su=[100*B1[(5,m)]['su_su'].mean() for m in modes]
axs[1].bar(range(4),pu,color='#7f7f7f',label='PU collisions',edgecolor='black',linewidth=0.5); axs[1].bar(range(4),su,bottom=pu,color='#D55E00',label='SU–SU collisions',edgecolor='black',linewidth=0.5); axs[1].set_title('(b) Collision rates (%)',loc='left'); axs[1].legend(frameon=False,fontsize=7,loc='upper right')
axs[1].set_ylim(0,78)
axs[2].bar(range(4),[B1[(5,m)]['reg_emerg'].mean() for m in modes],yerr=[ci(B1[(5,m)]['reg_emerg']) for m in modes],color=cols,capsize=2,edgecolor='black',linewidth=0.5); axs[2].set_title('(c) Emergency regret/packet',loc='left')
for ax in axs: ax.set_xticks(range(4)); ax.set_xticklabels(short,fontsize=7); ax.grid(axis='x',visible=False)
fig.tight_layout(); save(fig,'fig14_multi_su_summary')

# ---- Fig 15: sweep over M
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0)); Ms=[1,2,3,4,5,6,8,10]
for mode,c,ls,lab in [('naive','#D55E00','--','No coordination'),('coord','#000000','-','Gateway + priority')]:
    ms=[m for m in Ms if (m,mode) in B3]
    axs[0].plot(ms,[100*B3[(m,mode)]['pu'].mean() for m in ms],color=c,ls=ls,lw=1.8,marker='o',ms=3.5,label=lab)
    axs[1].plot(ms,[100*B3[(m,mode)]['succ_rate'].mean() for m in ms],color=c,ls=ls,lw=1.8,marker='o',ms=3.5,label=lab)
axs[0].axhline(44.6,color='gray',ls=':',lw=1); axs[0].text(7.2,45.6,'mean busy rate 44.6%',fontsize=7,color='gray')
for ax,t,yl in [(axs[0],'(a) PU collision rate','PU collisions (% of transmissions)'),(axs[1],'(b) Success per SU','Delivered packets per SU-round (%)')]:
    ax.axvline(5,color='gray',lw=0.8,ls='-.'); ax.set_xlabel('Number of SUs $M$ ($N=5$ channels)'); ax.set_title(t,loc='left'); ax.set_ylabel(yl)
axs[0].legend(frameon=False,fontsize=7,loc='lower right')
save(fig,'fig15_sweep_M')

# ---- Fig 16: scalability grid
fig,ax=plt.subplots(figsize=(6.5,3.0)); cfgs=[(5,5),(8,5),(10,5),(5,10),(10,10),(20,10),(20,20)]; x=np.arange(len(cfgs)); w=0.38
nv=[100*B4[(m,n,'naive')]['succ_rate'].mean() for m,n in cfgs]; cd=[100*B4[(m,n,'coord')]['succ_rate'].mean() for m,n in cfgs]; un=[100*B4[(m,n,'coord')]['unserved'].mean() for m,n in cfgs]
ax.bar(x-w/2,nv,w,color='#D55E00',label='No coordination',edgecolor='black',linewidth=0.5); ax.bar(x+w/2,cd,w,color='#000000',label='Gateway + priority',edgecolor='black',linewidth=0.5)
for i in range(len(x)):
    ax.text(x[i]+w/2,cd[i]+1.2,f'{cd[i]:.0f}',ha='center',fontsize=7.5)
    if un[i]>0.5: ax.text(x[i]+w/2,cd[i]/2,f'{un[i]:.0f}%\nwait',ha='center',va='center',fontsize=6.5,color='white')
    ax.text(x[i]-w/2,nv[i]+1.2,f'{nv[i]:.0f}',ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([f'{m} SUs\n{n} ch.' for m,n in cfgs],fontsize=7.5); ax.set_ylabel('Delivered packets per SU-round (%)'); ax.set_ylim(0,85); ax.legend(frameon=False,fontsize=7.5,loc='upper right'); ax.grid(axis='x',visible=False)
save(fig,'fig16_scalability')

# ---- Fig 17: fairness
fig,axs=plt.subplots(1,2,figsize=(6.5,3.0),gridspec_kw={'width_ratios':[1,1.2]}); modes=['No coordination','Decentralised rho-rand','Gateway, no priority','Gateway + priority']
data=[B1[(5,m)]['jain_succ'] for m in modes]; bp=axs[0].boxplot(data,patch_artist=True,widths=0.55,medianprops=dict(color='black'))
for patch,c in zip(bp['boxes'],cols): patch.set_facecolor(c); patch.set_alpha(0.8)
axs[0].set_xticklabels(short,fontsize=7); axs[0].set_ylabel("Jain's index on delivered packets"); axs[0].set_title('(a) Fairness per run',loc='left'); axs[0].grid(axis='x',visible=False)
for i,(m,c) in enumerate(zip(modes,cols)):
    s=np.sort(B1[(5,m)]['succ_per_su'],axis=1)[:,::-1].mean(0)/6000*100
    axs[1].plot(range(1,6),s,color=c,marker='o',ms=4,lw=1.6,ls=['--','-.',':','-'][i],label=short[i].replace('\n',' '))
axs[1].set_xlabel('SU rank within a run (best to worst)'); axs[1].set_ylabel('Delivered packets (%)'); axs[1].set_title('(b) Spread between SUs',loc='left'); axs[1].legend(frameon=False,fontsize=7); axs[1].set_xticks(range(1,6))
fig.tight_layout(); save(fig,'fig17_fairness')

# ---- Fig 18: weight sensitivity
fig,ax=plt.subplots(figsize=(4.2,3.0)); rs=sorted(B5['rows']); gain=[100*(1-B5['rows'][r]['reg_prio']/B5['rows'][r]['reg_noprio']) for r in rs]
ax.plot(rs,gain,color='black',marker='o',lw=1.8); ax.axvline(6.4,color='gray',ls='-.',lw=1); ax.text(6.7,5,'used: 6.4',fontsize=7.5,color='gray')
ax.axvspan(B5['maxbt']/B5['delta'],22,color='#56B4E9',alpha=0.12); ax.text(8,48,'sufficient optimality\ncondition holds',fontsize=7,color='#0072B2')
ax.set_xlabel('Weight ratio $\\omega_E/\\omega_R$'); ax.set_ylabel('Regret reduction by priority (%)'); ax.set_xlim(0.5,21); ax.set_ylim(0,55)
save(fig,'fig18_weight_sensitivity')
print('figures written:', sorted(f for f in __import__('os').listdir(OUT) if f.endswith('.png')))
