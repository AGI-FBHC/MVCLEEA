#!/usr/bin/env python3
"""Summarize a completed ablation suite as JSON, CSV and Markdown."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

rows = []
pattern = f'ablation_{args.split}_*_seed{args.seed}.json'
for path in sorted((ROOT / 'results').glob(pattern)):
    payload = json.loads(path.read_text(encoding='utf-8'))
    row = {
        'ablation': payload['ablation'],
        'experiment_name': payload['experiment_name'],
    }
    for level in range(1, 5):
        metrics = payload['results'][f'L{level}']
        row[f'L{level}_precision'] = metrics['precision']
        row[f'L{level}_recall'] = metrics['recall']
        row[f'L{level}_f1'] = metrics['f1']
        row[f'L{level}_auc'] = metrics['auc']
    rows.append(row)

rows.sort(key=lambda row: (row['ablation'] != 'full', row['ablation']))
stem = ROOT / 'results' / f'ablation_{args.split}_seed{args.seed}_summary'

with stem.with_suffix('.json').open('w', encoding='utf-8') as handle:
    json.dump(rows, handle, indent=2)
    handle.write('\n')

with stem.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

full = next(row for row in rows if row['ablation'] == 'full')
lines = [
    f'# {args.split} ablation results (seed {args.seed})',
    '',
    '| Ablation | L1 F1 | L2 F1 | L3 F1 | L4 F1 | Delta L4 |',
    '|---|---:|---:|---:|---:|---:|',
]
for row in rows:
    lines.append(
        f"| {row['ablation']} | "
        f"{row['L1_f1']:.4f} | {row['L2_f1']:.4f} | "
        f"{row['L3_f1']:.4f} | {row['L4_f1']:.4f} | "
        f"{row['L4_f1'] - full['L4_f1']:+.4f} |"
    )
stem.with_suffix('.md').write_text(
    '\n'.join(lines) + '\n',
    encoding='utf-8',
)
print(stem.with_suffix('.md'))
