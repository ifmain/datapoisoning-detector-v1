# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import torch
from dpdetector import load_detector, predict_image

parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('--model',default='ifmain/datapoisoning-detector-v1-large')
parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
torch.set_num_threads(6)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model,config = load_detector(args.model,device=args.device)
print(json.dumps(predict_image(model,config,args.image),indent=2))
