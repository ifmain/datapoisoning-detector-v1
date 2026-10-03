from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import shutil
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'assets'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
INK='#17263C'; MUTED='#607087'; BLUE='#E8F1FF'; TEAL='#DDF5EF'; PURPLE='#EEE8FF'
def canvas(title, subtitle):
    fig,ax=plt.subplots(figsize=(18,10)); fig.patch.set_facecolor('#FAFBFD'); ax.set_facecolor('#FAFBFD')
    ax.set_xlim(0,180); ax.set_ylim(0,100); ax.axis('off'); fig.subplots_adjust(0,0,1,1)
    ax.text(8,94,'ifmain  /  RESEARCH',fontsize=12,color='#356AC3',weight='bold')
    ax.text(8,87,title,fontsize=27,color=INK,weight='bold')
    ax.text(8,81,subtitle,fontsize=13,color=MUTED)
    return fig,ax

def box(ax,x,y,w,h,title,sub='',color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=1.5',facecolor=color,edgecolor='#C7D2E1',linewidth=1.2))
    ax.text(x+w/2,y+h/2+(1.7 if sub else 0),title,ha='center',va='center',fontsize=13,weight='bold',color=INK)
    if sub: ax.text(x+w/2,y+h/2-2.5,sub,ha='center',va='center',fontsize=10,color=MUTED)
def arrow(ax,a,b,color='#71839B'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,linewidth=1.5,color=color,connectionstyle='arc3',shrinkA=9,shrinkB=9))
def save(fig,name):
    for ext in ['png','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)

fig,ax=canvas('One image. Four feature levels. One decision.','DataPoisoning Detector v1  /  Multi-level FLUX.2 VAE features')
ax.text(8,74,'01  FROZEN ENCODER',fontsize=11,color='#356AC3',weight='bold')
for x,t,s in [(8,'RGB crop','128 × 128 pixels'),(41,'Encoder block 1','128 × 128 × 128'),(74,'Encoder block 2','256 × 64 × 64'),(107,'Encoder block 3','512 × 32 × 32'),(140,'Final posterior','Mean + 2× packing')]:
    box(ax,x,58,28,12,t,s)
for x in [36,69,102,135]: arrow(ax,(x,64),(x+5,64))
ax.text(151,54,'128 × 8 × 8',ha='right',fontsize=10,color=MUTED)
for x,t,s in [(41,'Project E1','4 spatial reductions'),(74,'Project E2','3 spatial reductions'),(107,'Project E3','2 spatial reductions'),(140,'Project Z','Linear projection')]:
    arrow(ax,(x+14,58),(x+14,48)); box(ax,x,36,28,12,t,s,TEAL); arrow(ax,(x+14,36),(x+14,29))
ax.text(8,43,'02  TRAINABLE\nPROJECTIONS',fontsize=11,color='#198676',weight='bold',linespacing=1.8)
box(ax,41,19,127,10,'Fuse four streams + three early-to-final differences','concat(E1, E2, E3, Z, E1 − Z, E2 − Z, E3 − Z) → Linear + LayerNorm',PURPLE)
ax.text(8,22,'Each stream:\n64 tokens × D',fontsize=12,color=MUTED,linespacing=1.7)
arrow(ax,(105,19),(105,13))
ax.text(105,9,'64 fused tokens + CLS  →  N transformer blocks  →  classifier',ha='center',fontsize=14,color=INK,weight='bold')
ax.text(8,3,'Feature dimensions: channels × height × width. Intermediate encoder operations are omitted for clarity.',fontsize=10,color=MUTED)
save(fig,'architecture')

fig,ax=canvas('From local evidence to an image-level score.','Inference and distillation  /  The same four-stream architecture across the model family')
ax.text(8,73,'INFERENCE',fontsize=12,color='#356AC3',weight='bold')
for x,w,t,s in [(8,31,'Five RGB crops','Corners + center · 128 × 128'),(48,34,'Four-stream detector','Frozen VAE + trainable head'),(91,34,'Pool + classify','CLS + central 6 × 6 tokens'),(134,37,'Image decision','Mean of five logits ≥ threshold')]: box(ax,x,53,w,14,t,s)
for a,b in [(39,48),(82,91),(125,134)]: arrow(ax,(a,60),(b,60))
ax.text(8,47,'96 × 96 active center; surrounding tokens supply context. Scores are raw logits, not calibrated probabilities.',fontsize=12,color=MUTED)
ax.text(8,38,'SOFT-TARGET DISTILLATION',fontsize=12,color='#198676',weight='bold')
box(ax,8,15,36,14,'Shared frozen VAE','Identical encoder features',BLUE)
box(ax,57,25,40,11,'Large teacher','Frozen detector head',PURPLE)
box(ax,57,7,40,11,'Student head','Medium / Small / Tiny',TEAL)
arrow(ax,(44,26),(57,30)); arrow(ax,(44,19),(57,13))
box(ax,111,17,60,17,'Soft targets + supervised learning','50% Bernoulli KL (T = 2) + 50% supervised objective',TEAL)
arrow(ax,(97,30),(111,29)); arrow(ax,(97,13),(111,21))
ax.text(111,11,'Supervised objective: classification + paired ranking.',fontsize=10,color=MUTED)
ax.text(8,2,'Teacher and encoder stay frozen. Only the student head is optimized during distillation.',fontsize=11,color=MUTED)
save(fig,'inference-distillation')
print('Created original PNG + SVG architecture figures in',OUT)
