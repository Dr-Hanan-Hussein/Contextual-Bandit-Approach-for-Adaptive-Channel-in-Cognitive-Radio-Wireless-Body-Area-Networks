import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import rcParams
rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'mathtext.fontset':'dejavuserif','pdf.fonttype':42,'ps.fonttype':42})
BLUE='#dbe8f7'; BLUE_E='#2f5d8a'; RED='#f7d9d4'; RED_E='#a8321f'; GRN='#d9f0e3'; GRN_E='#1c7a4a'; PUR='#6a3d9a'
import os; os.makedirs('figures/final', exist_ok=True)
fig,ax=plt.subplots(figsize=(5.8,2.9)); ax.set_xlim(0,118); ax.set_ylim(0,58); ax.axis('off')
def box(x,y,w,h,t,fc,ec,fs=7.6,lw=1.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec=ec,lw=lw)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,linespacing=1.3)
def arr(p,q,c='#333333',lw=1.2): ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8,lw=lw,color=c,shrinkA=0,shrinkB=0))
# titles
for x,t in [(18,'Problem'),(60,'Method'),(100,'Results (simulation)')]: ax.text(x,54.5,t,ha='center',fontsize=9,weight='bold')
for x in (38,80): ax.plot([x,x],[3,52],color='#bbbbbb',lw=0.8,ls='--')
# --- problem: three WBANs, five channels
for i,(y,t,fc,ec) in enumerate([(39,'WBAN 1\nalert',RED,RED_E),(29,'WBAN 2\nroutine',BLUE,BLUE_E),(19,'WBAN 3\nroutine',BLUE,BLUE_E)]):
    box(0.5,y,12.5,7.8,t,fc,ec,fs=7.6)
state=[1,0,1,1,0]
for k,b in enumerate(state):
    y=45.5-k*6.4
    ax.add_patch(Rectangle((22,y),8.5,5.0,fc=('#f4c7c0' if b else '#cdebd8'),ec=('#a8321f' if b else '#1c7a4a'),lw=0.8,hatch=('////' if b else None)))
    ax.text(31.6,y+2.5,f'Ch{k+1}',ha='left',va='center',fontsize=7.4)
for y in (43,33,23): arr((13.8,y),(21.2,36),c='#888888',lw=0.9)
ax.text(18,6.5,'Routine data: throughput\nAlerts: reliability\nChannels shared by WBANs',ha='center',va='center',fontsize=7.6,linespacing=1.3)
# --- method
box(42,32,34,16,'Shared-reliability\ncontextual UCB\none reliability estimate per\nchannel; class sets the index',GRN,GRN_E,fs=7.8)
box(42,10,34,16,'Priority gateway\nalerts first, best free\nchannel for each WBAN',RED,RED_E,fs=7.8)
arr((59,31.4),(59,26.8))
# --- results: bar chart + three numbers
bx=[82,92,102]; vals=[73,70,44]; cols=['#D55E00','#E69F00','#000000']
for x,v,c in zip(bx,vals,cols):
    ax.add_patch(Rectangle((x,18),6,v*0.30,fc=c,ec='black',lw=0.5)); ax.text(x+3,18+v*0.30+1,f'{v}',ha='center',fontsize=7.6)
ax.text(85,14.2,'None',ha='center',fontsize=7.2); ax.text(95,14.2,'ρ-rand',ha='center',fontsize=7.2); ax.text(105,14.2,'Gateway',ha='center',fontsize=7.2)
ax.text(98,50,'System regret ($\\times10^3$), 5 users',ha='center',fontsize=7.6)
ax.text(99.5,6.0,'−36% regret vs UCB1 (one user)\nAlert loss ≤ 2% of ideal\nNo user–user collisions',ha='center',va='center',fontsize=7.1,linespacing=1.3)
arr((38.6,29),(41.4,29),lw=1.4); arr((76.6,29),(79.4,29),lw=1.4)
for ext in ('png','pdf'): fig.savefig(f'figures/final/Graphical_abstract.{ext}',dpi=400,bbox_inches='tight')
from PIL import Image
im=Image.open('figures/final/Graphical_abstract.png'); print('size px:',im.size,'width at 13 cm = %.0f dpi'%(im.width/(13/2.54)))
im.convert('RGB').save('figures/final/Graphical_abstract.tiff',compression='tiff_lzw')
im2=im.convert('RGB'); im2.thumbnail((1500,1500)); im2.save('/tmp/ga.jpg',quality=88)
