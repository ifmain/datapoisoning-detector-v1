# Training and distillation

Install the package with `pip install -e .` and a CUDA-compatible PyTorch build.
Training requires a CUDA GPU; there is no automatic CPU fallback.

## Data preparation

Supply `detector_train.jsonl`, `detector_validation.jsonl` and
`detector_test.jsonl` in your data directory. Each line describes an image:

```json
{"pair_id":"example-protected","group_id":"original-example","clean_pixel_hash":"SHA256_OF_CLEAN_RGB_PIXELS","source":"own-data","method":"glaze","label":1,"path":"/data/protected.png","in_path":"/data/protected.png","out_path":"/data/original.png"}
```

Use `label: 0` and `method: "clean"` for clean controls. Paths must resolve on
the training machine. Protected training records need aligned clean targets.
Split by original image and subject/artist before extracting patches. Keep all
variants of an original in one split. Include clean controls as well as protected
images. Source images are not distributed with the code.

```bash
python scripts/prepare_cache.py --data-dir /data/detector
python scripts/download_vae.py --output-dir /models/flux2
```

Cache preparation preserves pixels and extracts aligned 128x128 crops. Existing
cache files are not overwritten. The downloaded encoder is in `/models/flux2/vae`.

## Train Large

```bash
python scripts/train.py --config configs/large.json --data-dir /data/detector --vae-dir /models/flux2/vae --output-dir /runs/large
python scripts/export_checkpoint.py --checkpoint /runs/large/best.pt --output-dir /models/detector-large
```

The FLUX encoder is frozen. Trainable heads fuse features from encoder blocks
1, 2 and 3 and the final posterior mean. The configuration files define the
Large, Medium, Small and Tiny architectures and optimizer settings.

## Distill students

```bash
python scripts/train.py --config configs/medium.json --data-dir /data/detector --vae-dir /models/flux2/vae --output-dir /runs/medium --teacher /models/detector-large --kd-weight 0.5 --temperature 2
```

Use `configs/small.json` or `configs/tiny.json` with a separate output directory
for the other students. Each student uses Large directly as its teacher.
Distillation combines temperature-scaled Bernoulli KL with supervised
classification and paired ranking; it does not replace soft targets with hard
teacher labels. Teacher parameters stay frozen.

## Selection, stopping and resumption

The best checkpoint maximizes validation recall at a target FPR of 1%, breaking
ties with ROC-AUC. Each model selects its own threshold. After the first epoch
without a new best, two additional recovery epochs are allowed. A conservative
sustained-overfitting heuristic can stop the run earlier. The configured base
budget is 12 epochs, with at most two recovery epochs beyond that budget.

Create `pause.request` in the run directory to pause after an epoch checkpoint.
Resume with the same command plus `--resume`; model, optimizer, sampler and RNG
states are restored from `latest.pt`. Keep the configuration and manifests fixed.
Export uses `best.pt`, not the final epoch. Test evaluation uses the selected
checkpoint and its validation-selected threshold.

## Results and release layout

See [the README](../README.md) for completed test results and
[evaluation reports](../reports/three_tap_run) for metrics, validation histories,
distillation agreement and export verification. Unfinished variants are marked
Coming soon. Validation snapshots must not be mixed with final test metrics.
The [article](ARTICLE.md) describes the architecture and findings.

Code is released under [Apache-2.0](../LICENSE). See [data provenance](LICENSING.md)
for the separate status of training-image rights.
