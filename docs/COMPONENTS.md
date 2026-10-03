# Model components

Each Hugging Face repository has the following layout:

```text
README.md
LICENSE
model_index.json
detector/
  config.json
  model.safetensors
vae/
  config.json
  model.safetensors
assets/
licenses/
```

`detector` holds the trained early-feature projections, latent projection,
normalization, fusion, transformer and classifier. `vae` holds the frozen
FLUX.2 encoder and quantization convolution. Its tensor names are relative to the
encoder component; the detector loader restores the `backbone.` prefix when
assembling the complete model. There are no VAE decoder weights in this release.

```python
from dpdetector import load_detector, predict_image

model, config = load_detector("ifmain/datapoisoning-detector-v1-small", device="cuda")
result = predict_image(model, config, "image.png")
```

Local loading accepts either the package root (`../hf-small`) or its
`detector/` directory. Remote loading downloads both components. The loader also
supports older single-file releases. `model_index.json` documents this custom
package layout; this release does not implement a Diffusers pipeline.

The frozen encoder is the same architecture across all four variants. Tiny is a
smaller detection head, not a separately compressed VAE. Future restoration work
can reuse the encoder independently, but needs its own restoration network and
decoder. The classifier head is not an image decoder.

Export using `scripts/export_checkpoint.py`; it writes the component layout.
The local release folders are `hf-large`, `hf-medium`, `hf-small` and `hf-tiny`.
