#!/usr/bin/env python3
"""Build a compact inventory for local data artifacts."""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def describe_array(path):
    array = np.load(path, mmap_mode='r')
    return {
        'path': str(path.relative_to(ROOT)),
        'size_bytes': path.stat().st_size,
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }


manifest = {
    'splits': [],
    'arrays': [],
    'esm2': {},
    'models': [],
}

for path in sorted((ROOT / 'data' / 'splits').glob('*.csv')):
    manifest['splits'].append({
        'path': str(path.relative_to(ROOT)),
        'size_bytes': path.stat().st_size,
    })

for path in sorted((ROOT / 'data' / 'features').glob('*/*.npy')):
    manifest['arrays'].append(describe_array(path))

esm_paths = list((ROOT / 'data' / 'features' / 'esm2').glob('*.pt'))
manifest['esm2'] = {
    'path': 'data/features/esm2',
    'file_count': len(esm_paths),
    'size_bytes': sum(path.stat().st_size for path in esm_paths),
    'sample_shape': (
        list(torch.load(esm_paths[0], weights_only=True).shape)
        if esm_paths else None
    ),
}

for path in sorted((ROOT / 'models').glob('*.pth')):
    manifest['models'].append({
        'path': str(path.relative_to(ROOT)),
        'size_bytes': path.stat().st_size,
    })

output = ROOT / 'data' / 'MANIFEST.json'
output.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True) + '\n',
    encoding='utf-8',
)
print(output)
