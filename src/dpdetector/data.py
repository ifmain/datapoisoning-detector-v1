# SPDX-License-Identifier: Apache-2.0
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image


class PairedTraining:
    def __init__(self, directory, seed=20261002):
        self.images = np.load(directory / 'input_patches.npy', mmap_mode='r')
        self.clean = np.load(directory / 'clean_positive_patches.npy', mmap_mode='r')
        self.metadata = json.loads((directory / 'patch_meta.json').read_text())
        self.positive = {}
        self.negative = {}
        for i, row in enumerate(self.metadata):
            table, key = (self.positive, row['method']) if row['label'] else (self.negative, row['source'])
            table.setdefault(key, []).append(i)
        self.rng = np.random.default_rng(seed)
        self.methods = sorted(self.positive)
        self.sources = sorted(self.negative)

    def batch(self):
        # A protected crop, its aligned clean target, and two independent clean controls.
        method = self.rng.choice(self.methods)
        pos = int(self.rng.choice(self.positive[method]))
        reference = self.metadata[pos]['reference']
        assert reference >= 0
        crops = [np.array(self.images[pos]), np.array(self.clean[reference])]
        for _ in range(2):
            source = self.rng.choice(self.sources)
            crops.append(np.array(self.images[int(self.rng.choice(self.negative[source]))]))
        # Identical dihedral augmentation of each pair, no resampling or compression.
        rotation, flip = int(self.rng.integers(4)), bool(self.rng.integers(2))
        x = np.rot90(np.stack(crops), rotation, axes=(1, 2))
        if flip:
            x = x[:, :, ::-1]
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def read_rows(path):
    with open(path, encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def five_crops(row):
    with Image.open(row['path']) as image:
        image = image.convert('RGB')
        w, h = image.size
        if min(w, h) < 128:
            # Preserve pixels; reflect-pad small inputs instead of upsampling.
            arr = np.array(image)
            arr = np.pad(arr, ((0, max(0, 128-h)), (0, max(0, 128-w)), (0, 0)), mode='reflect')
            image = Image.fromarray(arr)
            w, h = image.size
        xy = [(0, 0), (w-128, 0), (0, h-128), (w-128, h-128), ((w-128)//2, (h-128)//2)]
        return np.stack([np.array(image.crop((x, y, x+128, y+128))) for x, y in xy])


def evaluation_cache(directory, split, cache_directory):
    rows = read_rows(directory / f'detector_{split}.jsonl')
    cache_directory.mkdir(parents=True, exist_ok=True)
    target = cache_directory / f'{split}_five_crops.npy'
    if not target.exists():
        partial = target.with_suffix('.partial.npy')
        out = np.lib.format.open_memmap(partial, mode='w+', dtype='uint8', shape=(len(rows), 5, 128, 128, 3))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, value in enumerate(pool.map(five_crops, rows)):
                out[i] = value
                if (i + 1) % 200 == 0:
                    print(f'Caching {split}: {i+1}/{len(rows)}', flush=True)
        out.flush()
        del out
        partial.replace(target)
    return rows, np.load(target, mmap_mode='r')
