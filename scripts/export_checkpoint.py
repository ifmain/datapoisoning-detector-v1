# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,required=True,help='A trusted local best.pt checkpoint from this trainer')
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--vae-config',type=Path,default=Path(__file__).resolve().parents[1]/'vae_config.json')
args = parser.parse_args()
state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
cfg = state['config']
args.output_dir.mkdir(parents=True,exist_ok=True)
head_dir = args.output_dir/'detector'
encoder_dir = args.output_dir/'vae'
head_dir.mkdir(exist_ok=True)
encoder_dir.mkdir(exist_ok=True)
save_file({k:v.contiguous() for k,v in state['model'].items() if not k.startswith('backbone.')},str(head_dir/'model.safetensors'),metadata={'format':'pt'})
save_file({k.removeprefix('backbone.'):v.contiguous() for k,v in state['model'].items() if k.startswith('backbone.')},str(encoder_dir/'model.safetensors'),metadata={'format':'pt'})
config = dict(model_type='datapoisoning_detector',architectures=['Detector'],width=cfg['width'],depth=cfg['depth'],
              heads=cfg.get('heads',16),variant=cfg.get('variant','large'),parameters=state['parameters'],
              patch_size=128,active_size=96,decision_threshold=state['validation']['threshold'],
              pooling='mean_of_five_crop_logits',selected_epoch=state['validation']['epoch'],
              encoder_taps=[1,2,3], architecture_revision='three-early-taps-v2',
              license='apache-2.0', license_status='author_grant; retained_dataset_rights_not_fully_verified')
(head_dir/'config.json').write_text(json.dumps(config,indent=2))
(encoder_dir/'config.json').write_text(args.vae_config.read_text())

(args.output_dir/'model_index.json').write_text(json.dumps(dict(format='dpdetector-components-v1',loader='dpdetector.load_detector',detector='detector',vae='vae'),indent=2))
