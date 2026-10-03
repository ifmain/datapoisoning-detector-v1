# SPDX-License-Identifier: Apache-2.0
"""Train or distill a configured detector with resumable GPU checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / 'src'))
import numpy as np
import torch
from torch.nn import functional as F
from dpdetector.model import Detector
from dpdetector.hub import load_detector
from dpdetector.distillation import binary_distillation_loss
from dpdetector.data import PairedTraining, evaluation_cache, read_rows


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def metric_summary(labels, scores, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp, fp = int((pred & labels).sum()), int((pred & ~labels).sum())
    fn, tn = int((~pred & labels).sum()), int((~pred & ~labels).sum())
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp/max(1,tp+fn), fpr=fp/max(1,fp+tn), precision=tp/max(1,tp+fp),
                accuracy=(tp+tn)/len(labels), roc_auc=float(roc_auc_score(labels,scores)),
                average_precision=float(average_precision_score(labels,scores)))


@torch.no_grad()
def evaluate(model, rows, cache):
    model.eval()
    scores = []
    for start in range(0, len(rows), 2):
        images = np.asarray(cache[start:start+2]).reshape(-1,128,128,3).transpose(0,3,1,2).copy()
        x = torch.from_numpy(images).to('cuda',dtype=torch.float32).div_(255)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        scores.extend(logits.float().reshape(-1,5).mean(1).cpu().tolist())
    return np.asarray(scores)


def check_split_integrity(directory, metadata):
    splits = {s: read_rows(directory / f'detector_{s}.jsonl') for s in ['train','validation','test']}
    result = {s:len(rows) for s,rows in splits.items()}
    for field in ['pair_id', 'group_id', 'clean_pixel_hash']:
        sets = {s:{r[field] for r in rows if r.get(field)} for s,rows in splits.items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f'Split overlap {field}: {a}/{b}: {len(overlap)}')
    train_ids = {r['pair_id'] for r in splits['train']}
    assert all(m['pair_id'] in train_ids for m in metadata)
    result['manifest_sha256'] = {s:hashlib.sha256((directory/f'detector_{s}.jsonl').read_bytes()).hexdigest() for s in splits}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/large.json')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--vae-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--teacher', help='Optional teacher release directory or HF repository ID')
    parser.add_argument('--kd-weight', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=2.0)
    args = parser.parse_args()
    if not 0 <= args.kd_weight <= 1 or args.temperature <= 0:
        parser.error('kd-weight must be in [0,1] and temperature must be positive')
    config = json.loads(args.config.read_text())
    config.update(teacher=args.teacher, kd_weight=args.kd_weight, temperature=args.temperature)
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), 'GPU is required; CPU fallback is disabled.'
    torch.set_num_threads(6)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    directory = args.data_dir
    data = PairedTraining(directory, config['seed'])
    audit = check_split_integrity(directory, data.metadata)
    if args.resume and (run/'data_audit.json').exists():
        previous_audit = json.loads((run/'data_audit.json').read_text())
        if audit['manifest_sha256'] != previous_audit['manifest_sha256']:
            raise ValueError('Dataset manifests changed since the saved training run')
    write_json(run/'data_audit.json', audit)
    print('Loading FLUX VAE and creating configured detector...', flush=True)
    model = Detector(args.vae_dir, config['width'], config['depth'], heads=config['heads']).cuda()
    counts = model.parameter_counts()
    assert counts == config['parameters'], (counts, config['parameters'])
    report = dict(config=config, parameters=counts, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  precision='VAE float32; trainable head autocast bfloat16; optimizer float32', pid=os.getpid())
    write_json(run/'run_config.json', report)
    print(json.dumps(report), flush=True)
    # Research basis: AdamW, Loshchilov and Hutter (2019), arXiv:1711.05101; PyTorch implementation.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'], fused=True)
    teacher = load_detector(args.teacher, device='cuda')[0].requires_grad_(False).eval() if args.teacher else None
    if teacher is not None:
        student_encoder, teacher_encoder = model.backbone.state_dict(), teacher.backbone.state_dict()
        if student_encoder.keys() != teacher_encoder.keys() or not all(
                torch.equal(student_encoder[k], teacher_encoder[k]) for k in student_encoder):
            raise ValueError('Shared-feature distillation requires identical frozen encoders')
        del student_encoder, teacher_encoder
    labels = torch.tensor([1.,0.,0.,0.], device='cuda')
    positive_weight = torch.tensor(3., device='cuda')

    def update(step, accumulation):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_steps = config['epochs'] * config['steps_per_epoch']
        warmup = min(1., (step+1)/config['warmup_steps'])
        progress = min(1., max(0., (step-config['warmup_steps'])/max(1,total_steps-config['warmup_steps'])))
        lr = config['learning_rate'] * warmup * (0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        losses = []
        for _ in range(accumulation):
            x = torch.from_numpy(data.batch()).cuda().float().div_(255)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = model.backbone(x)
                logits = model.forward_features(features)
                classification = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=positive_weight)
                ranking = F.softplus(1. - (logits[0].float() - logits[1].float()))
                loss = classification + config['pair_loss_weight'] * ranking
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher.forward_features(features)
                    loss = (1-args.kd_weight)*loss + args.kd_weight*binary_distillation_loss(logits,teacher_logits,args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at {step}')
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        return float(np.mean(losses)), float(norm), lr

    if args.smoke:
        t = time.monotonic()
        loss, norm, lr = update(0, 1)
        torch.cuda.synchronize()
        x = torch.from_numpy(data.batch()).cuda().float().div_(255)
        features = model.backbone(x[:1])
        smoke = dict(parameters=counts, loss=loss, gradient_norm=norm, seconds=time.monotonic()-t,
                     peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                     feature_shapes=[list(f.shape) for f in features],
                     frozen_encoder_has_grad=any(p.grad is not None for p in model.backbone.parameters()),
                     trainable_parameters_with_grad=sum(p.numel() for p in model.parameters() if p.grad is not None))
        write_json(run/'smoke_test.json', smoke)
        print(json.dumps(smoke), flush=True)
        return

    start_epoch, step, best, stale = 0, 0, (-1.,-1.), 0
    if args.resume:
        state = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)
        if state.get('training_config', config) != config:
            raise ValueError('Resume configuration differs from the checkpoint configuration')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, step, best, stale = state['epoch'], state['step'], tuple(state['best']), state['stale']
        data.rng.bit_generator.state = state['numpy_sampler_state']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(f'Resumed completed epoch {start_epoch}, optimizer step {step}', flush=True)
    history = [json.loads(p.read_text()) for p in sorted(run.glob('validation_epoch_*.json'))
               if int(p.stem.rsplit('_', 1)[-1]) <= start_epoch]
    stop_reason = 'epoch_limit'
    val_rows, val_cache = evaluation_cache(directory, 'validation', run/'evaluation_cache')
    y_val = np.asarray([r['label'] for r in val_rows])
    t_start = time.monotonic()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, config['epochs'] + config['recovery_epochs']):
        losses = []
        for iteration in range(config['steps_per_epoch']):
            loss, grad, lr = update(step, config['accumulation'])
            losses.append(loss)
            step += 1
            if iteration % 10 == 0:
                status = dict(status='training', epoch=epoch+1, iteration=iteration+1, global_step=step,
                              loss=float(np.mean(losses[-10:])), learning_rate=lr, gradient_norm=grad,
                              elapsed_seconds=time.monotonic()-t_start,
                              gpu_allocated_gib=torch.cuda.memory_allocated()/2**30)
                write_json(run/'status.json', status)
                print(json.dumps(status), flush=True)
        write_json(run/'status.json', dict(status='validation',epoch=epoch+1,global_step=step))
        scores = evaluate(model, val_rows, val_cache)
        clean_scores = np.sort(scores[y_val == 0])[::-1]
        max_fp = int(config['target_validation_fpr'] * len(clean_scores))
        threshold = np.nextafter(clean_scores[max_fp], np.inf)
        metrics = metric_summary(y_val, scores, threshold)
        metrics.update(epoch=epoch+1, training_loss=float(np.mean(losses)),
                       validation_bce=float(np.mean(np.logaddexp(0, scores) - y_val * scores)))
        history.append(metrics)
        # A conservative trend heuristic, not a proof of overfitting: both validation
        # ranking and recall worsen while training loss improves, twice in succession.
        def deteriorating(a, b):
            return (b['training_loss'] < a['training_loss'] * 0.95
                    and b['recall'] < a['recall'] - 0.005
                    and b['roc_auc'] < a['roc_auc'] - 0.001
                    and ('validation_bce' not in a or b['validation_bce'] > a['validation_bce']))
        overfitting_signal = len(history) >= 3 and all(deteriorating(a,b) for a,b in zip(history[-3:-1],history[-2:]))
        metrics['overfitting_signal'] = overfitting_signal
        write_json(run/f'validation_epoch_{epoch+1:02d}.json', metrics)
        print('VALIDATION ' + json.dumps(metrics), flush=True)
        key = (metrics['recall'], metrics['roc_auc'])
        if key > best:
            best, stale = key, 0
            # Portable inference checkpoint includes only the used frozen encoder, not the decoder.
            torch.save(dict(model=model.state_dict(), config=config, parameters=counts, validation=metrics), run/'best.pt.tmp')
            (run/'best.pt.tmp').replace(run/'best.pt')
            write_json(run/'best_validation.json', metrics)
        else:
            stale += 1
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), training_config=config, epoch=epoch+1, step=step,
                        best=best, stale=stale, numpy_sampler_state=data.rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), run/'latest.pt.tmp')
        (run/'latest.pt.tmp').replace(run/'latest.pt')
        if (run/'pause.request').exists():
            (run/'pause.request').replace(run/f'pause_acknowledged_epoch_{epoch+1}.txt')
            write_json(run/'status.json',dict(status='paused',completed_epoch=epoch+1,resume_epoch=epoch+2,global_step=step))
            return
        if overfitting_signal:
            stop_reason = 'sustained_overfitting_signal'
            break
        if stale >= 1 + config['recovery_epochs']:
            stop_reason = 'no_recovery_after_two_additional_epochs'
            break
        if epoch + 1 >= config['epochs'] and stale == 0:
            break
    write_json(run/'stopping_decision.json',dict(reason=stop_reason,completed_epoch=epoch+1,stale_epochs=stale,
                                                recovery_epochs=config['recovery_epochs']))
    best_state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model'])
    threshold = best_state['validation']['threshold']
    del best_state, optimizer
    test_rows, test_cache = evaluation_cache(directory,'test',run/'evaluation_cache')
    write_json(run/'status.json', dict(status='test_evaluation',global_step=step))
    scores = evaluate(model,test_rows,test_cache)
    labels_test = np.asarray([r['label'] for r in test_rows])
    result = metric_summary(labels_test,scores,threshold)
    result['per_method'] = {method:dict(n=int(mask.sum()), recall=float((scores[mask]>=threshold).mean()))
                            for method in sorted({r['method'] for r in test_rows if r['label']})
                            if (mask:=np.array([r['method']==method and r['label']==1 for r in test_rows])).any()}
    write_json(run/'test_metrics.json',result)
    write_json(run/'test_predictions.json', [dict(pair_id=r['pair_id'],label=r['label'],method=r['method'],score=float(s)) for r,s in zip(test_rows,scores)])
    write_json(run/'status.json',dict(status='complete',global_step=step,test=result,elapsed_seconds=time.monotonic()-t_start))
    print('TEST ' + json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
