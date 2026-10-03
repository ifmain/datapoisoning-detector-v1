# SPDX-License-Identifier: Apache-2.0
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True)
args = parser.parse_args()
snapshot_download('black-forest-labs/FLUX.2-klein-base-4B',
                  revision='a3b4f4849157f664bdbc776fd7453c2783562f4d',
                  allow_patterns=['vae/config.json','vae/diffusion_pytorch_model.safetensors','LICENSE*','README.md'],
                  local_dir=args.output_dir)
