"""Render reported metrics as normalized score shares with fixed endpoints."""
from pathlib import Path
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=['large','medium','small','tiny'],default='large')
parser.add_argument('--output-dir',type=Path,default=BASE/'assets')
args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
m=json.loads((BASE/f'reports/three_tap_run/{args.variant}/test_metrics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
ours='#119E91'; theirs='#7961C9'; ink='#18273C'; bg='#FAFBFD'
fig,axes=plt.subplots(1,2,figsize=(18,9));fig.patch.set_facecolor(bg)
fig.subplots_adjust(left=.055,right=.955,top=.73,bottom=.24,wspace=.16)
fig.text(.055,.94,'ifmain  /  REPORTED RESULTS',color=ours,fontsize=12,weight='bold')
fig.text(.055,.875,f'DataPoisoning Detector v1 {args.variant.title()} vs. LightShed',fontsize=26,color=ink,weight='bold')
fig.legend(handles=[Patch(color=ours,label=f'Ours · {args.variant.title()} (left)'),Patch(color=theirs,label='LightShed · Table 2 (right)')],loc='upper left',bbox_to_anchor=(.05,.835),frameon=False,ncol=2,fontsize=13)
keys=['nightshade','glaze','mist','metacloak'];labels=['Nightshade¹','Glaze','Mist','MetaCloak']
rows=[]
for ax,kind,other in zip(axes,['Detection recall','Clean-image specificity'],[[96.55,97.26,99.84,91.77],[92.86,97.70,100,84.32]]):
 ax.set_facecolor(bg);ax.set_xlim(-2,102);ax.set_ylim(-.6,3.8)
 ax.set_title(kind+' ↑',loc='left',fontsize=18,color=ink,pad=20,weight='bold')
 for i,(key,label,b) in enumerate(zip(keys,labels,other)):
  y=3-i;a=100*(m['per_method'][key]['recall'] if kind=='Detection recall' else 1-m['fpr'])
  share=100*a/(a+b)
  ax.barh(y,share,height=.39,color=ours);ax.barh(y,100-share,left=share,height=.39,color=theirs)
  ax.plot([share,share],[y-.195,y+.195],color='white',linewidth=2)
  ax.text(50,y+.29,label,ha='center',fontsize=12,color=ink,weight='bold',bbox=dict(facecolor=bg,edgecolor='none',pad=2),zorder=5)
  count=m['per_method'][key]; detail=f" ({round(count['recall']*count['n'])}/{count['n']})" if kind=='Detection recall' else ''
  ax.text(2,y,f'{a:.2f}%'+detail,va='center',ha='left',color='white',fontsize=12,weight='bold' if a>=b else 'normal')
  ax.text(98,y,f'{b:.2f}%',va='center',ha='right',color='white',fontsize=12,weight='bold' if b>=a else 'normal')
  rows.append(dict(metric=kind,protection=key,ours=a,lightshed=b))
 ax.axvline(50,color='#42516A',linewidth=1,linestyle='--',zorder=0)
 ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{abs(x):.0f}'))
 ax.set_xlabel('Normalized score share (%) · 50 = equal reported scores',fontsize=11,color='#607087',labelpad=12)
 ax.set_yticks([]);ax.tick_params(axis='x',colors='#607087',length=0,pad=9)
 for s in ax.spines.values():s.set_visible(False)
fig.text(.055,.15,'Split position = 100 × ours / (ours + LightShed). Labels show actual scores; bold marks the higher score.',fontsize=11,color=ink)
fig.text(.055,.112,'Independent test sets and operating points; this is not a shared-benchmark or preference study.',fontsize=11,color='#607087')
fig.text(.055,.077,f"Our specificity uses the same {m['tn']+m['fp']:,} clean images in every row ({m['tn']:,} accepted).  ¹ LightShed binary comparator.",fontsize=10,color='#607087')
fig.text(.055,.041,'Source: LightShed, USENIX Security 2025, Table 2. Its separate NightShade LPIPS 0.07 result is 99.98% recall / 100% specificity.',fontsize=10,color='#607087')
for ext in ['png','svg']:fig.savefig(args.output_dir/f'published-comparison.{ext}',dpi=180,facecolor=bg)
(args.output_dir/'published-comparison.json').write_text(json.dumps({'model':args.variant,'source':'https://www.usenix.org/system/files/usenixsecurity25-foerster.pdf','rows':rows},indent=2))

