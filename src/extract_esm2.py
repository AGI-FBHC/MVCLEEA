#!/usr/bin/env python3
"""Extract length-aware ESM-2 embeddings without special/padding tokens."""

import argparse
import csv
import os
import time
from pathlib import Path

import torch

import esm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_sequences(csv_path):
    records = []
    with open(csv_path) as handle:
        reader = csv.reader(handle, delimiter='\t')
        next(reader)
        for row in reader:
            records.append((row[0], row[2].strip()))
    return records


def extract_dataset(model, alphabet, records, output_dir, batch_size, device):
    converter = alphabet.get_batch_converter()
    pending = [
        record for record in records
        if not os.path.exists(os.path.join(output_dir, f'{record[0]}.pt'))
    ]
    # Similar lengths in a batch reduce padding and GPU memory.
    pending.sort(key=lambda item: len(item[1]))
    started = time.time()

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        _, _, tokens = converter(batch)
        tokens = tokens.to(device)
        with torch.inference_mode():
            representations = model(tokens, repr_layers=[33])['representations'][33]

        for index, (protein_id, sequence) in enumerate(batch):
            # Position 0 is BOS; sequence residues occupy [1, length].
            embedding = representations[index, 1:len(sequence) + 1].mean(dim=0)
            torch.save(embedding.cpu(), os.path.join(output_dir, f'{protein_id}.pt'))

        done = min(start + batch_size, len(pending))
        if done % 100 == 0 or done == len(pending):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f'Processed {done}/{len(pending)} '
                f'({done / elapsed:.2f} proteins/s)',
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default=PROJECT_ROOT / 'data' / 'splits',
        type=Path,
    )
    parser.add_argument(
        '--output_dir',
        default=PROJECT_ROOT / 'data' / 'features' / 'esm2',
        type=Path,
    )
    parser.add_argument('--datasets', nargs='+', default=['split10', 'new'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)

    for dataset in args.datasets:
        records = load_sequences(os.path.join(args.data_dir, f'{dataset}.csv'))
        print(f'{dataset}: {len(records)} proteins', flush=True)
        extract_dataset(
            model, alphabet, records, args.output_dir, args.batch_size, device
        )


if __name__ == '__main__':
    main()
