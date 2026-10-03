# DataPoisoning Detector v1 — CAT CelebA-HQ benchmark

Additional internal benchmark of the four published checkpoints, reported separately from the original test set.

## Evaluation protocol

- Source: [CAT](https://github.com/senp98/CAT), archived CelebA-HQ subset.
- All 2,000 eligible images: 1,800 protected (200 per protection method) and 200 clean controls. The additional noisy baseline is excluded.
- Native-resolution five-crop inference, 128 × 128 per crop; image score is the mean crop logit. This is the original release protocol, not sliding-window inference.
- Each checkpoint uses its unchanged published validation-selected threshold. No training or threshold fitting on this benchmark.
- The frozen VAE encoder is identical across checkpoints (verified tensor equality), so its features are computed once. Each model applies its own feature projections, transformer and classifier independently.
- Counts are images, not independent identities: methods reuse underlying originals. These measurements are descriptive and do not establish generalization to unrelated datasets.
- The project author confirms having reviewed the source terms and permission for this evaluation use. The CAT CelebA-HQ subset was excluded from training and validation of these released checkpoints and was not used to fit their thresholds. Images are not redistributed.

## Model family comparison

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

Raw measurements: [results.json](results.json). Image-level predictions: [predictions.json](predictions.json). Source list: [manifest.json](manifest.json).
