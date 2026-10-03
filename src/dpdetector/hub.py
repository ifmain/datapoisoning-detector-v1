# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from .model import Detector
from .data import five_crops


def load_detector(model_id_or_path, device='cpu', token=None, revision=None):
    directory = Path(model_id_or_path)
    if not directory.is_dir():
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(str(model_id_or_path), token=token, revision=revision,
                         allow_patterns=['detector/*', 'vae/*', 'config.json', 'vae_config.json', 'model.safetensors']))
    if (directory/'detector/config.json').is_file():
        directory = directory/'detector'
    config = json.loads((directory/'config.json').read_text())
    encoder_directory = directory.parent/'vae'
    split_components = (encoder_directory/'model.safetensors').is_file()
    vae_config = json.loads((encoder_directory/'config.json' if split_components else directory/'vae_config.json').read_text())
    model = Detector(width=config['width'], depth=config['depth'], heads=config['heads'],
                     vae_config=vae_config, load_vae=False)
    state = load_file(str(directory/'model.safetensors'))
    if split_components:
        encoder = load_file(str(encoder_directory/'model.safetensors'))
        state.update({'backbone.' + key: value for key, value in encoder.items()})
    model.load_state_dict(state, strict=True)
    if model.parameter_counts() != config['parameters']:
        raise ValueError('Parameter count does not match the release configuration')
    model.to(device).eval()
    return model, config


@torch.no_grad()
def predict_image(model, config, image):
    device = next(model.parameters()).device
    crops = five_crops({'path':str(image)})
    x = torch.from_numpy(crops.transpose(0,3,1,2).copy()).to(device=device,dtype=torch.float32).div_(255)
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        logits = model(x).float()
    score = logits.mean().item()
    threshold = config['decision_threshold']
    return dict(score=score,score_type='mean_crop_logit',threshold=threshold,
                flagged=score>=threshold,crop_logits=logits.cpu().tolist())
