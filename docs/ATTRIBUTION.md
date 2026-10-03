# Research and source attribution

The public bibliography focuses on protection and dataset references. Technical implementation sources are identified below and in source comments.

| Location / use | Basis | Implementation provenance |
|---|---|---|
| src/dpdetector/model.py frozen encoder | BFL FLUX.2 VAE; Diffusers; Kingma & Welling | Licensed dependency and pretrained encoder; retain Apache notices |
| Transformer attention | Vaswani et al., 2017 | Original implementation using PyTorch |
| LayerScale | Touvron et al., 2021 | Original implementation of the published technique |
| GroupNorm / LayerNorm | Wu & He, 2018; Ba et al., 2016 | PyTorch layers |
| src/dpdetector/distillation.py | Hinton et al., 2015 | Original Bernoulli KL implementation; optional, no students released |
| scripts/train.py AdamW | Loshchilov & Hutter | PyTorch optimizer |
| CAT / robust-style-mimicry / XAI | Dataset distributions | Image pairs only; no purification or attack code copied |
| IMPRESS | Helen distribution | Clean images only; no purification code used |
| Open Images / DiffVax / local data | Clean controls and user pairs | Training data, not code dependencies |
| LightShed | Published comparison | Literature only; no code, gated data, or weights |

The current training run excludes CelebA/CelebA-HQ. Retained underlying image lineage includes VGGFace2, WikiArt and Helen. Those sources must be cited and reviewed separately from repository code licenses.
