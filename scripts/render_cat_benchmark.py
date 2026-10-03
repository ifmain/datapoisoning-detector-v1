from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
r=json.loads((BASE/'reports/cat_celebahq_benchmark/results.json').read_text())
methods=['advdm+','advdm-','anti-dreambooth','glaze2','metacloak','mist','sds+','sds-','sdsT5']
labels=['AdvDM (+)','AdvDM (-)','Anti-DreamBooth','Glaze 2','MetaCloak','Mist','SDS (+)','SDS (-)','SDST5']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
for v in ['large','medium','small','tiny']:
 fig,ax=plt.subplots(figsize=(13,8));fig.patch.set_facecolor('#FAFBFD');ax.set_facecolor('#FAFBFD')
 fig.subplots_adjust(left=.20,right=.94,top=.76,bottom=.19)
 fig.text(.055,.94,'ifmain / INTERNAL BENCHMARK',fontsize=11,color='#119E91',weight='bold')
 fig.text(.055,.875,f'DataPoisoning Detector v1 · {v.title()}',fontsize=24,color='#18273C',weight='bold')
 fig.text(.055,.815,f"Recall: {r[v]['recall']*100:.2f}% · Clean specificity: {r[v]['specificity']*100:.2f}% (200/200)",fontsize=13,color='#18273C')
 vals=[r[v]['per_method'][m]/2 for m in methods]
 ax.barh(range(9),vals,color='#119E91',height=.58)
 ax.set_yticks(range(9),labels);ax.invert_yaxis();ax.set_xlim(0,126)
 for i,(m,val) in enumerate(zip(methods,vals)):
  ax.text(val+1,i,f"{val:.1f}% ({r[v]['per_method'][m]}/200)",va='center',fontsize=10,color='#18273C')
 ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Detection recall (%) · Higher is better')
 ax.tick_params(length=0);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(False)
 fig.text(.055,.10,'CAT CelebA-HQ: 1,800 protected + 200 clean images. Unchanged thresholds; five-crop inference.',fontsize=10,color='#526078')
 fig.text(.055,.055,'This subset was excluded from training and validation of these checkpoints. Evaluation only.',fontsize=10,color='#526078')
 for ext in ['png','svg']:fig.savefig(BASE/f'assets/cat-celebahq-{v}.{ext}',dpi=170,facecolor=fig.get_facecolor())
 plt.close(fig)
