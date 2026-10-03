# DataPoisoning Detector v1

<!-- architecture-figures -->
## Architecture at a glance

![Four-level VAE feature fusion](assets/architecture.png)

![Image-level inference and soft-target distillation](assets/inference-distillation.png)

Original diagrams for this release. Editable SVG versions: [architecture](assets/architecture.svg) · [inference and distillation](assets/inference-distillation.svg).
<!-- /architecture-figures -->


[Code and training](https://github.com/ifmain/datapoisoning-detector-v1) · [Evaluation reports](https://github.com/ifmain/datapoisoning-detector-v1/tree/main/reports/three_tap_run)

**Hugging Face models:** [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) · [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) · [Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small) · [Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny)

Install with `pip install -e .` (or `pip install -r requirements.txt`). For figure rendering, install `pip install ".[plots]"`. See [installation verification](docs/INSTALLATION.md). Run `python scripts/predict.py image.png --model ../hf-large --device cuda`. Training: `python scripts/train.py --help`; optional distillation uses `--teacher`, `--kd-weight 0.5`, `--temperature 2`.

The detector compares encoder blocks 1, 2 and 3 with the final posterior-mean representation of a frozen FLUX.2 VAE. Each stream is projected to an 8 × 8 token grid. The three early-to-final differences are fused with all four feature streams before transformer classification. Inputs are native-resolution 128 × 128 RGB crops with a central 96 × 96 active region. The VAE decoder and generative FLUX transformer are not used.

This run excludes CelebA-HQ from training, validation, test and patch caches. It contains 21,464 training, 1,257 validation and 1,446 test records. Large starts with newly initialized trainable parameters and the licensed frozen VAE encoder. Students learn from this new Large through temperature-2 Bernoulli KL soft-target distillation: 50% distillation and 50% supervised classification plus paired ranking. Checkpoint selection uses validation recall at a target false-positive rate of 1%, with ROC-AUC as the tie-breaker. A decline receives two additional recovery epochs unless a sustained overfitting signal is observed.

Image-level inference averages five crop logits. Scores are not calibrated probabilities. The revised test set is a filtered subset of the previous experiment, not a fresh blind holdout. These results must not be presented as the old experiment's measurements. Local Glaze performance is reported separately below because that subset previously exposed a failure.

| Variant | Total parameters | Trainable parameters | Status |
|---|---:|---:|---:|
| [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) | 296,264,065 | 261,838,209 | Evaluated |
| [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) | 182,722,433 | 148,296,577 | Evaluated |
| [Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small) | 82,635,137 | 48,209,281 | Evaluated |
| [Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny) | 43,280,257 | 8,854,401 | Evaluated |

## Original test: model family comparison

### Detection recall

Percentage of protected images correctly detected. Higher is better.

| Protection | [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) | [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) | [Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small) | [Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny) |
|---|---:|---:|---:|---:|
| Nightshade | 97.92% (47/48) | 95.83% (46/48) | **100.00% (48/48)** | 95.83% (46/48) |
| Glaze | **95.45% (63/66)** | 93.94% (62/66) | **95.45% (63/66)** | **95.45% (63/66)** |
| Mist | **100.00% (22/22)** | **100.00% (22/22)** | **100.00% (22/22)** | **100.00% (22/22)** |
| MetaCloak | **100.00% (4/4)** | **100.00% (4/4)** | **100.00% (4/4)** | **100.00% (4/4)** |

### Clean-image specificity

Percentage of clean images correctly accepted as clean. Higher is better. Each model uses its own validation-selected threshold on the same clean-control pool.

| Evaluation pool | [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) | [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) | [Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small) | [Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny) |
|---|---:|---:|---:|---:|
| Shared clean controls | 99.1129% (1229/1240) | **99.2742% (1231/1240)** | 99.0323% (1228/1240) | 99.1129% (1229/1240) |

### Distillation quality

All students use the newly trained three-tap Large teacher. No previous detector checkpoint is used. Teacher agreement measures matching decisions, not correctness.

| Metric | [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) | [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) | [Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small) | [Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny) |
|---|---:|---:|---:|---:|
| Accuracy | 98.8935% | 98.7552% | 98.8935% | 98.8243% |
| Recall | 97.5728% | 95.6311% | 98.0583% | 97.0874% |
| Precision | 94.8113% | 95.6311% | 94.3925% | 94.7867% |
| False positive rate | 0.8871% | 0.7258% | 0.9677% | 0.8871% |
| ROC-AUC | 0.998555 | 0.997851 | 0.999010 | 0.996336 |
| Teacher decision agreement | 100% (self) | 99.4467% | 99.5851% | 99.5159% |
| Image-logit MAE (lower is better) | 0 (self) | 0.658140 | 0.638014 | 0.683860 |
| Recall change vs. teacher (pp) | 0 (self) | -1.941748 | 0.485437 | -0.485437 |

## Original test: published comparison

| [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) | [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) |
|:---:|:---:|
| [![Large vs. LightShed](assets/published-comparison-large.png)](assets/published-comparison-large.png) | [![Medium vs. LightShed](assets/published-comparison-medium.png)](assets/published-comparison-medium.png) |
| **[Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small)** | **[Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny)** |
| [![Small vs. LightShed](assets/published-comparison-small.png)](assets/published-comparison-small.png) | [![Tiny vs. LightShed](assets/published-comparison-tiny.png)](assets/published-comparison-tiny.png) |

Click a figure to view it at full size. The numerical tables below report the **Large** model.

### Detection recall

Percentage of protected images correctly detected. Higher is better.

| Protection | Our recall (detected/positive) | LightShed Table 2 recall |
|---|---:|---:|
| Nightshade (binary comparator) | **97.92% (47/48)** | 96.55% |
| Glaze | 95.45% (63/66) | **97.26%** |
| Mist | **100.00% (22/22)** | 99.84% |
| MetaCloak | **100.00% (4/4)** | 91.77% |

### Clean-image specificity

Percentage of clean images correctly accepted as clean. Higher is better. Large uses one shared clean-control pool.

| Comparison condition | Our clean specificity | LightShed Table 2 specificity |
|---|---:|---:|
| Nightshade (binary comparator) | **99.11%** | 92.86% |
| Glaze | **99.11%** | 97.70% |
| Mist | 99.11% | **100.00%** |
| MetaCloak | **99.11%** | 84.32% |

Published independent evaluations; bold marks the higher value in each row. LightShed reports method-specific operating points; ours uses a fixed validation-selected threshold. Its NightShade LPIPS 0.07 condition separately reports 99.98% recall / 100% specificity. [Source: LightShed, Table 2](https://www.usenix.org/conference/usenixsecurity25/presentation/foerster). No LightShed code, weights or gated dataset was used.

## Additional internal benchmark: CAT CelebA-HQ

All 2,000 eligible images in the archived subset: 200 per protection method and 200 clean controls. Noisy baseline excluded. **Original five-crop inference and unchanged release thresholds**, not experimental sliding-window calibration.

| Large | Medium |
|:---:|:---:|
| ![Large](assets/cat-celebahq-large.png) | ![Medium](assets/cat-celebahq-medium.png) |
| **Small** | **Tiny** |
| ![Small](assets/cat-celebahq-small.png) | ![Tiny](assets/cat-celebahq-tiny.png) |

### Detection recall

Percentage of protected images correctly detected. Higher is better.

| Protection | Large | Medium | Small | Tiny |
|---|---:|---:|---:|---:|
| advdm- | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |
| advdm+ | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |
| anti-dreambooth | 3.00% (6/200) | **38.00% (76/200)** | 23.00% (46/200) | 1.50% (3/200) |
| glaze2 | 99.00% (198/200) | 99.00% (198/200) | **99.50% (199/200)** | **99.50% (199/200)** |
| metacloak | 0.50% (1/200) | **29.00% (58/200)** | 12.50% (25/200) | 2.50% (5/200) |
| mist | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |
| sds- | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |
| sds+ | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |
| sdsT5 | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |

### Clean-image specificity

Percentage of clean images correctly accepted as clean. Higher is better. All models use the same 200 clean controls.

| Evaluation pool | Large | Medium | Small | Tiny |
|---|---:|---:|---:|---:|
| Shared clean controls | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** | **100.00% (200/200)** |

### Overall results

| Metric | Large | Medium | Small | Tiny |
|---|---:|---:|---:|---:|
| Accuracy | 80.2500% | 86.6000% | 83.5000% | 80.3500% |
| Protected-image recall | 78.0556% | 85.1111% | 81.6667% | 78.1667% |
| Clean-image specificity | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| False positive rate | 0.00% | 0.00% | 0.00% | 0.00% |
| True positives | 1405 | 1532 | 1470 | 1407 |
| False negatives | 395 | 268 | 330 | 393 |
| False positives | 0 | 0 | 0 | 0 |
| True negatives | 200 | 200 | 200 | 200 |
| Decision threshold (logit) | -1.5164062976837156 | -1.1406249999999998 | -0.7234374880790709 | -1.8132812976837156 |

Bold marks the highest per-method recall or specificity, including ties. No competitor numbers are mixed into this benchmark.


**Evaluation-only source:** [CAT](https://github.com/senp98/CAT), CelebA-HQ subset. This subset was excluded from training and validation of the currently released checkpoints and was not used to fit their thresholds. The project author confirms having reviewed the source terms and permission for this evaluation use. Images are not redistributed. Protection methods share underlying originals.

[Detailed protocol and results](https://github.com/ifmain/datapoisoning-detector-v1/blob/main/reports/cat_celebahq_benchmark/BENCHMARK.md).

## Local Glaze check

| Variant | Detected / positive |
|---|---:|
| [Large](https://huggingface.co/ifmain/datapoisoning-detector-v1-large) | 2/2 |
| [Medium](https://huggingface.co/ifmain/datapoisoning-detector-v1-medium) | 1/2 |
| [Small](https://huggingface.co/ifmain/datapoisoning-detector-v1-small) | 2/2 |
| [Tiny](https://huggingface.co/ifmain/datapoisoning-detector-v1-tiny) | 2/2 |

## License and data provenance

The author licenses the code and the rights they hold in the new model weights under Apache-2.0. The frozen BFL VAE component is Apache-2.0. This grant does not establish third-party rights in training images. Retained sources include VGGFace2 and WikiArt through CAT, robust-style-mimicry, XAI, Helen, Open Images, DiffVax and user-supplied data; their commercial permissions have not all been verified. See [the provenance report](https://github.com/ifmain/datapoisoning-detector-v1/blob/main/reports/three_tap_run/provenance.md).

## Citation

If you find our work helpful, feel free to give us a cite.
```bibtex
@misc{ifmain2026datapoisoningdetector,
    title = {DataPoisoning Detector v1: Multi-Level FLUX VAE Features for Protective-Perturbation Detection},
    url = {https://github.com/ifmain/datapoisoning-detector-v1},
    author = {{ifmain}},
    month = {October},
    year = {2026}
}
```

Selected protection and dataset references: [REFERENCES.bib](https://github.com/ifmain/datapoisoning-detector-v1/blob/main/REFERENCES.bib).

## Component layout

Each model repository stores the trained head in `detector/` and the frozen VAE encoder in `vae/`. Load the repository root with `dpdetector.load_detector`. See [component documentation](docs/COMPONENTS.md).
