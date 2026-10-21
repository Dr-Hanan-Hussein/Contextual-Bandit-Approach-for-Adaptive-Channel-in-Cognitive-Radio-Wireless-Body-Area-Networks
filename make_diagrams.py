import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import rcParams
from crwban import *
rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'mathtext.fontset':'dejavuserif','savefig.dpi':450,'pdf.fonttype':42,'ps.fonttype':42})
import os; os.makedirs('figures/final', exist_ok=True)
OUT='figures/final/'
BLUE='#dbe8f7'; BLUE_E='#2f5d8a'; RED='#f7d9d4'; RED_E='#a8321f'; GRN='#d9f0e3'; GRN_E='#1c7a4a'; GRY='#eeeeee'; GRY_E='#555555'; YEL='#fbefcf'; YEL_E='#9a7410'
def save(fig,name): fig.savefig(OUT+name+'.png',bbox_inches='tight'); fig.savefig(OUT+name+'.pdf',bbox_inches='tight'); plt.close(fig)
def box(ax,x,y,w,h,text,fc=BLUE,ec=BLUE_E,fs=8,bold_first=False,ha='center'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2 if ha=='center' else x+1.5,y+h/2,text,ha=ha,va='center',fontsize=fs,linespacing=1.35)
def arrow(ax,p,q,rad=0.0,color='#333333',lw=1.2,text=None,tpos=None,fs=7.5):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,lw=lw,color=color,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))
    if text: ax.text(*(tpos or ((p[0]+q[0])/2,(p[1]+q[1])/2+1.8)),text,ha='center',va='center',fontsize=fs,color=color,style='italic')
def canvas(w,h,xlim,ylim):
    fig,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off'); return fig,ax

# ------------------------------------------------------------------ Fig 4
fig,ax=canvas(6.5,3.7,(0,100),(0,58))
box(ax,1,36,30,16,"1  New packet\nRead its class $c$ (1 bit)\n0 = routine, 1 = emergency",fs=7.6)
box(ax,35,36,30,16,"2  Index of every channel $n$\nif $V_n=0$:  $Q_n=+\\infty$\nelse:  $Q_n=b_c(n)[\\hat\\theta_n+\\kappa_c\\sqrt{2\\ln t/V_n}]$\n(Eq. 6)",fs=7.0)
box(ax,69,36,30,16,"3  Choose\n$a=\\arg\\max_n Q_n$\n(ties broken at random)",fs=7.6)
box(ax,69,8,30,16,"4  Transmit the packet\non channel $a$",fc=GRN,ec=GRN_E,fs=7.6)
box(ax,35,8,30,16,"5  Observe the outcome $y$\n$y=1$: delivered\n$y=0$: PU or SU–SU collision",fc=YEL,ec=YEL_E,fs=7.4)
box(ax,1,8,30,16,"6  Update channel $a$ (Eq. 7)\n$V_a\\leftarrow V_a+1$\n$\\hat\\theta_a\\leftarrow\\hat\\theta_a+(y-\\hat\\theta_a)/V_a$\n$t\\leftarrow t+1$",fs=7.0)
arrow(ax,(31.6,44),(34.4,44)); arrow(ax,(65.6,44),(68.4,44)); arrow(ax,(84,35.5),(84,24.6)); arrow(ax,(68.4,16),(65.6,16)); arrow(ax,(34.4,16),(31.6,16))
arrow(ax,(16,24.6),(16,35.4),text='next round',tpos=(8,30))
ax.text(50,1.5,"Energy of the round (Eq. 9):  $E_{sense}+E_{tx}+E_{ret}\\cdot\\mathbb{1}\\{y=0\\}$",ha='center',fontsize=8,style='italic',color='#333333')
save(fig,'fig4_flowchart')

# ------------------------------------------------------------------ Fig 5
fig,ax=canvas(6.5,3.7,(0,100),(0,58))
box(ax,1,36,30,16,"1  Each SU sends the gateway\nits class bit and its ranked\nchannel list (16 bits, $N=5$)",fs=7.2)
box(ax,35,36,30,16,"2  Form the sets (Eq. 10)\n$\\mathcal{E}=\\{m:c^m=1\\}$\n$\\mathcal{D}=\\{m:c^m=0\\}$\nShuffle each set at random",fs=7.2)
box(ax,69,36,30,16,"3  Priority order (Eq. 11)\n$\\pi=(\\mathcal{E},\\mathcal{D})$\nReserved set $\\mathcal{C}\\leftarrow\\emptyset$",fs=7.4)
box(ax,69,8,30,16,"4  For each SU $m$ in order $\\pi$:\n$a^m=\\arg\\max_{n\\notin\\mathcal{C}}Q^{c^m}_n$\n$\\mathcal{C}\\leftarrow\\mathcal{C}\\cup\\{a^m\\}$ (Eq. 12)",fc=RED,ec=RED_E,fs=7.0)
box(ax,35,8,30,16,"5  If no channel is free\n($M>N$), the SU waits\nfor the next round",fc=YEL,ec=YEL_E,fs=7.4)
box(ax,1,8,30,16,"6  SUs transmit on their\nassigned channels and update\ntheir own statistics",fc=GRN,ec=GRN_E,fs=7.2)
arrow(ax,(31.6,44),(34.4,44)); arrow(ax,(65.6,44),(68.4,44)); arrow(ax,(84,35.5),(84,24.6)); arrow(ax,(68.4,16),(65.6,16)); arrow(ax,(34.4,16),(31.6,16))
arrow(ax,(16,24.6),(16,35.4),text='next round',tpos=(8,30))
ax.text(50,1.5,"No channel is given twice in a round: there are no SU–SU collisions by construction.",ha='center',fontsize=8,style='italic',color='#333333')
save(fig,'fig5_gateway_procedure')

# ------------------------------------------------------------------ Fig 6
fig,ax=canvas(6.5,3.8,(0,100),(0,64))
ax.text(14,61,'Step 1: SUs in this round',ha='center',fontsize=8.5,weight='bold'); ax.text(50,61,'Step 2: priority order',ha='center',fontsize=8.5,weight='bold'); ax.text(86,61,'Step 3: best free channel',ha='center',fontsize=8.5,weight='bold')
sus=[('SU3','emergency',RED,RED_E),('SU1','emergency',RED,RED_E),('SU5','routine',BLUE,BLUE_E),('SU2','routine',BLUE,BLUE_E),('SU4','routine',BLUE,BLUE_E)]
chs=[('Ch 1','$\\theta=0.89$, $\\beta=0.6$','SU3'),('Ch 2','$\\theta=0.72$, $\\beta=1.0$','SU1'),('Ch 3','$\\theta=0.58$, $\\beta=0.8$','SU5'),('Ch 4','$\\theta=0.36$, $\\beta=0.7$','SU2'),('Ch 5','$\\theta=0.22$, $\\beta=0.9$','SU4')]
ys=[50,39,28,17,6]
for (n,c,fc,ec),y in zip(sus,ys): box(ax,3,y,22,8,f"{n}\n({c})",fc=fc,ec=ec,fs=8)
box(ax,38,6,24,48,"Gateway\n\n1. SU3 (emergency)\n2. SU1 (emergency)\n3. SU5 (routine)\n4. SU2 (routine)\n5. SU4 (routine)\n\nEmergency first;\nrandom order inside\neach class",fc='#ffffff',ec='#6a3d9a',fs=7.8)
for (n,th,s),(sn,_,fc,ec),y in zip(chs,sus,ys): 
    box(ax,72,y,26,8,f"{n}  →  {s}\n{th}",fc=(RED if s in ('SU3','SU1') else BLUE),ec=(RED_E if s in ('SU3','SU1') else BLUE_E),fs=7.6)
for y in ys: arrow(ax,(25.8,y+4),(37.4,30),rad=0.0,color='#999999',lw=0.9)
for y in ys: arrow(ax,(62.6,30),(71.4,y+4),rad=0.0,color='#999999',lw=0.9)
ax.text(50,-1.5,"Emergency SUs rank channels by reliability; routine SUs rank by $\\beta_n\\hat\\theta_n$. SU5 prefers Ch 2, then Ch 1.\nBoth are taken, so it receives Ch 3. All five assignments are distinct.",ha='center',va='top',fontsize=7.6,style='italic')
save(fig,'fig6_gateway_example')

# ------------------------------------------------------------------ Fig 1 (fallback)
fig,ax=canvas(6.5,2.3,(0,100),(0,36))
box(ax,1,8,20,22,"On-body sensors\n(ECG, EEG, SpO$_2$,\nblood pressure,\ntemperature)",fc=GRN,ec=GRN_E,fs=7.8)
box(ax,28,8,18,22,"WBAN\ncoordinator\n(sink node)",fs=8)
box(ax,53,8,20,22,"Gateway /\nsmartphone",fc=GRY,ec=GRY_E,fs=8)
box(ax,80,8,19,22,"Hospital server /\nclinician\ndashboard",fc=YEL,ec=YEL_E,fs=7.8)
arrow(ax,(21.7,19),(27.3,19)); arrow(ax,(46.7,19),(52.3,19)); arrow(ax,(73.7,19),(79.3,19))
ax.text(24.5,23.5,'short-range\nlinks',ha='center',fontsize=6.8,style='italic'); ax.text(49.5,23.5,'licensed\nspectrum (CR)',ha='center',fontsize=6.8,style='italic'); ax.text(76.5,23.5,'internet /\ncellular',ha='center',fontsize=6.8,style='italic')
ax.text(11,3,'Body layer',ha='center',fontsize=8,weight='bold'); ax.text(37,3,'Coordinator layer',ha='center',fontsize=8,weight='bold'); ax.text(63,3,'Communication layer',ha='center',fontsize=8,weight='bold'); ax.text(89.5,3,'Back-end layer',ha='center',fontsize=8,weight='bold')
save(fig,'fig1_wban_dataflow')

# ------------------------------------------------------------------ Fig 2 (fallback, from simulated PU activity)
cfg=cfg_with(S=1); X,C=make_env(cfg); o=run(X,C,cfg,'shared'); t0=3000; L=36
fig,ax=plt.subplots(figsize=(6.5,2.9))
for n in range(5):
    for k in range(L):
        busy=X[0,t0+k,n]
        ax.add_patch(Rectangle((k,4-n),0.92,0.86,fc=('#f4c7c0' if busy else '#cdebd8'),ec=('#a8321f' if busy else '#1c7a4a'),lw=0.8,hatch=('////' if busy else None)))
for k in range(L):
    a=o['chan'][0,t0+k,0]; ok=o['succ'][0,t0+k,0]; e=o['ctx'][0,t0+k,0]
    ax.plot(k+0.46,4-a+0.43,marker=('*' if ok else 'X'),color=('#0b3c8a' if ok else 'black'),ms=(9 if ok else 6),mec='white',mew=0.4)
ax.set_xlim(-0.3,L+0.2); ax.set_ylim(-0.3,5.2); ax.set_yticks([4.43-n for n in range(5)]); ax.set_yticklabels([f'Ch {n+1}' for n in range(5)]); ax.set_xlabel('Round'); ax.set_xticks(range(0,L,5)); ax.set_xticklabels([str(t0+k) for k in range(0,L,5)])
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.tick_params(axis='y',length=0)
h=[Rectangle((0,0),1,1,fc='#f4c7c0',ec='#a8321f',hatch='////'),Rectangle((0,0),1,1,fc='#cdebd8',ec='#1c7a4a'),plt.Line2D([],[],marker='*',color='#0b3c8a',ls='',ms=9),plt.Line2D([],[],marker='X',color='black',ls='',ms=6)]
ax.legend(h,['PU busy','PU idle (spectrum hole)','SU delivered','SU collision'],ncol=4,frameon=False,fontsize=7.5,loc='lower center',bbox_to_anchor=(0.5,1.0))
save(fig,'fig2_spectrum_holes')

# ------------------------------------------------------------------ Fig 3 (fallback)
fig,ax=canvas(6.5,3.6,(0,100),(0,56))
box(ax,1,30,17,20,"On-body\nsensors",fc=GRN,ec=GRN_E,fs=8)
box(ax,22,30,20,20,"Class detector\n(rule on sensor)\n$c\\in\\{0,1\\}$",fc=RED,ec=RED_E,fs=7.8)
box(ax,53,26,22,28,"Coordinator (one radio)\nShared-reliability\ncontextual UCB\n$V_n,\\ \\hat\\theta_n,\\ Q^c_n$\nchannel choice $a_t$",fs=7.8)
box(ax,81,26,18,28,"Licensed\nspectrum\nCh 1 … $N$\n(PU busy / idle)",fc=GRY,ec=GRY_E,fs=7.8)
box(ax,53,2,22,14,"Gateway (several WBANs)\npriority order, free channel",fc='#ffffff',ec='#6a3d9a',fs=7.6)
box(ax,81,2,18,14,"Hospital\nserver / EHR",fc=YEL,ec=YEL_E,fs=7.8)
arrow(ax,(18.6,40),(21.4,40)); arrow(ax,(42.6,40),(52.4,40),color=RED_E,lw=1.6,text='class bit',tpos=(47.5,43.6)); arrow(ax,(75.6,40),(80.4,40))
arrow(ax,(60,25.6),(60,16.6),text='class bit,\nranked list',tpos=(54.5,21)); arrow(ax,(68,16.6),(68,25.6),text='assigned\nchannel',tpos=(73.5,21))
arrow(ax,(90,25.6),(90,16.6),color='#555555')
ax.text(24,19,"Emergency ($c=1$): reliability first\nRoutine ($c=0$): bandwidth × reliability",fontsize=7.4,style='italic',ha='center',va='center',color=RED_E)
save(fig,'fig3_architecture')
print('diagrams written')
