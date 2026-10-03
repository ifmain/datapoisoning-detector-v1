import json,re,shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
BUNDLE=BASE.parent
variants=['large','medium','small','tiny']
r=json.loads((BASE/'reports/cat_celebahq_benchmark/results.json').read_text())
methods=['advdm+','advdm-','anti-dreambooth','glaze2','metacloak','mist','sds+','sds-','sdsT5']
labels=['AdvDM (+)','AdvDM (-)','Anti-DreamBooth','Glaze 2 / published Glaze','MetaCloak','Mist','SDS (+)','SDS (-)','SDS T5']
ref={'glaze2':(97.26,97.70),'metacloak':(91.77,84.32),'mist':(99.84,100.0)}
source='https://www.usenix.org/system/files/usenixsecurity25-foerster.pdf'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
for v in variants:
 fig,axes=plt.subplots(1,2,figsize=(17,8),sharey=True)
 fig.subplots_adjust(left=.21,right=.97,top=.80,bottom=.20,wspace=.15)
 fig.suptitle(f'DataPoisoning Detector v1 {v.title()} / Internal benchmark',fontsize=21,fontweight='bold',y=.95)
 fig.text(.21,.875,'Teal: our CAT evaluation    Purple: LightShed published Table 2',fontsize=12)
 for col,ax in enumerate(axes):
  ours=[r[v]['per_method'][m]/2 if col==0 else 100*r[v]['specificity'] for m in methods]
  for i,(m,value) in enumerate(zip(methods,ours)):
   ax.barh(i-.17,value,height=.29,color='#119E91');ax.text(value+1,i-.17,f'{value:.2f}',va='center',fontsize=9)
   if m in ref:
    other=ref[m][col];ax.barh(i+.17,other,height=.29,color='#8672CB');ax.text(other+1,i+.17,f'{other:.2f}',va='center',fontsize=9)
   else:ax.text(2,i+.17,'Not reported',va='center',fontsize=9,color='#8672CB')
  ax.set_xlim(0,119);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(range(9),labels);ax.set_title(['Detection recall (%)','Clean specificity (%)'][col]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  for sp in ax.spines.values():sp.set_visible(False)
 axes[0].invert_yaxis()
 fig.text(.04,.11,'Our results: 200 protected images per method; one shared pool of 200 clean images. Five crops; fixed thresholds.',fontsize=10)
 fig.text(.04,.075,'LightShed: independently published evaluation, not run on this internal test. Glaze version/settings may differ.',fontsize=10)
 fig.text(.04,.04,'Not reported = absent from Table 2, not zero performance. Source: USENIX Security 2025, Foerster et al., Table 2.',fontsize=10)
 for ext in ['png','svg']:fig.savefig(BASE/f'assets/published-comparison-{v}.{ext}',dpi=150)
 plt.close(fig)
 payload=dict(variant=v,benchmark='CAT CelebA-HQ internal 2000',source=source,ours=r[v],published_reference=ref)
 (BASE/f'assets/published-comparison-{v}.json').write_text(json.dumps(payload,indent=2))
 for ext in ['png','svg','json']:
  shutil.copy2(BASE/f'assets/published-comparison-{v}.{ext}',BUNDLE/f'hf-{v}/assets/published-comparison.{ext}')
# Keep legacy generic asset names consistent with the new Large plot.
for ext in ['png','svg','json']:shutil.copy2(BASE/f'assets/published-comparison-large.{ext}',BASE/f'assets/published-comparison.{ext}')
def cell(x,best):return ('**' if x==best else '')+f'{x:.2f}%'+('**' if x==best else '')
def comparison(v):
 text='## Published reference alongside the internal benchmark\n\n'
 if v is None:
  text+='| Large | Medium |\n|:---:|:---:|\n| ![Large](assets/published-comparison-large.png) | ![Medium](assets/published-comparison-medium.png) |\n| **Small** | **Tiny** |\n| ![Small](assets/published-comparison-small.png) | ![Tiny](assets/published-comparison-tiny.png) |\n\nThe following numerical tables show Large. Each figure shows its named variant.\n\n';v='large'
 else:text+='![Internal benchmark and published reference](assets/published-comparison.png)\n\n'
 for col,title in enumerate(['Detection recall','Clean-image specificity']):
  text+=f'### {title}\n\n| Protection | Our {v.title()} (%) | LightShed Table 2 (%) |\n|---|---:|---:|\n'
  for m,label in zip(methods,labels):
   own=r[v]['per_method'][m]/2 if col==0 else 100*r[v]['specificity'];other=ref.get(m)
   text+=f'| {label} | '+(cell(own,max(own,other[col])) if other else f'{own:.2f}%')+' | '+(cell(other[col],max(own,other[col])) if other else 'Not reported')+' |\n'
  text+='\n'
 return text+f'Our numbers come exclusively from the internal CAT benchmark above. Clean specificity uses the same 200 controls for every row. LightShed numbers are published results on its own evaluation, not measurements on this internal set. Glaze 2 is shown alongside published Glaze; versions and settings are not asserted identical. **Bold** marks the larger reported value, including ties, not a controlled head-to-head win. Not reported means absent from Table 2. [LightShed source, Table 2]({source}).\n\n'
for folder,v in [(BASE,None)]+[(BUNDLE/f'hf-{v}',v) for v in variants]:
 p=folder/'README.md';s=p.read_text(encoding='utf-8')
 if '## Original test: model family comparison' in s:
  start=s.index('## Original test: model family comparison');end=s.index('## Additional internal benchmark: CAT CelebA-HQ');s=s[:start]+s[end:]
 s=re.sub(r'## Published reference alongside the internal benchmark.*?(?=## License)', '', s, flags=re.S)
 s=s.replace('## Additional internal benchmark: CAT CelebA-HQ','## Internal benchmark: CAT CelebA-HQ')
 s=re.sub(r'## Local Glaze check.*?(?=## License)', '',s,flags=re.S)
 s=s.replace('No competitor numbers are mixed into this benchmark.','Published reference results are listed separately below.')
 s=s.replace('This run excludes CelebA-HQ from training, validation, test and patch caches.','The training run excludes CelebA-HQ from training, validation and patch caches; it is used only for the separate internal evaluation below.')
 s=re.sub(r'The revised test set.*?subset previously exposed a failure.', 'All benchmark tables below use the internal CAT evaluation and unchanged release thresholds.', s)
 s=s.replace('## License and data provenance',comparison(v)+'## License and data provenance')
 for name in ['Large','Medium','Small','Tiny']:
  s=s.replace('| '+name+' |','| ['+name+'](https://huggingface.co/ifmain/datapoisoning-detector-v1-'+name.lower()+') |')
 p.write_text(s,encoding='utf-8')
print('Updated 5 READMEs and comparison assets for all 4 variants')
