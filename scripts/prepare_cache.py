# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from dpdetector.data import read_rows


def pixels(path):
    with Image.open(path) as image:
        x = np.array(image.convert('RGB'))
    h,w = x.shape[:2]
    return np.pad(x,((0,max(0,128-h)),(0,max(0,128-w)),(0,0)),mode='reflect')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261002)
    args = parser.parse_args()
    rows = read_rows(args.data_dir/'detector_train.jsonl')
    counts = [int(r.get('crops_per_record',8 if r['label'] else 2)) for r in rows]
    if min(counts)<1:
        raise ValueError('crops_per_record must be positive')
    for name in ['input_patches.npy','clean_positive_patches.npy','patch_meta.json']:
        if (args.data_dir/name).exists():
            raise FileExistsError(f'{name} already exists; use a new cache directory')
    total_positive = sum(n for r,n in zip(rows,counts) if r['label'])
    if not total_positive or all(r['label'] for r in rows):
        raise ValueError('Both protected and clean records are required')
    images = np.lib.format.open_memmap(args.data_dir/'input_patches.npy',mode='w+',dtype='uint8',shape=(sum(counts),128,128,3))
    clean = np.lib.format.open_memmap(args.data_dir/'clean_positive_patches.npy',mode='w+',dtype='uint8',shape=(total_positive,128,128,3))
    rng = np.random.default_rng(args.seed)
    metadata = []
    index = reference = 0
    for row,n in zip(rows,counts):
        x = pixels(row['in_path'])
        target = pixels(row['out_path']) if row['label'] else None
        if target is not None and x.shape != target.shape:
            raise ValueError(f"Pair is not spatially aligned: {row['pair_id']}")
        h,w = x.shape[:2]
        for _ in range(n):
            y,left = int(rng.integers(h-127)),int(rng.integers(w-127))
            images[index] = x[y:y+128,left:left+128]
            ref = -1
            if target is not None:
                clean[reference] = target[y:y+128,left:left+128]
                ref,reference = reference,reference+1
            metadata.append(dict(label=row['label'],method=row['method'],source=row['source'],pair_id=row['pair_id'],reference=ref))
            index += 1
    images.flush()
    clean.flush()
    (args.data_dir/'patch_meta.json').write_text(json.dumps(metadata))
    print(f'Prepared {index} input crops and {reference} aligned clean crops')


if __name__ == '__main__':
    main()
