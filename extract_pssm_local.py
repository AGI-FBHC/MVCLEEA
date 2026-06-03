#!/usr/bin/env python3
"""
PSSM Feature Extraction Script for Local Machine (Memory-Optimized)

Uses np.memmap for incremental writing - no large temporary storage needed.

Usage:
    python extract_pssm_local.py --dataset main --num-workers 8
"""

import os
import sys
import argparse
import pickle
import subprocess
import tempfile
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gc


BLAST_DB = 'pssm/uniprot_sprot.fasta'
BLAST_DIR = 'pssm/ncbi-blast-2.17.0+/bin'
MAX_LENGTH = 1024


def parse_pssm(pssm_path: str, max_length: int = 1024) -> np.ndarray:
    pssm = np.zeros((max_length, 20), dtype=np.float32)
    try:
        with open(pssm_path, 'r') as f:
            lines = f.readlines()
        row = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 22 and parts[0].isdigit():
                if row >= max_length:
                    break
                for col in range(20):
                    pssm[row, col] = float(parts[col + 2])
                row += 1
        pssm = pssm[:max_length]
        pssm = 1.0 / (1.0 + np.exp(-pssm))
    except Exception:
        pssm = np.ones((max_length, 20), dtype=np.float32) * 0.05
    return pssm


def extract_single_pssm(args):
    seq, db_path, psiblast_path, max_length = args
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, 'query.fasta')
        pssm_path = os.path.join(tmpdir, 'query.pssm')

        with open(fasta_path, 'w') as f:
            f.write(f">query\n{seq}\n")

        cmd = [
            psiblast_path,
            '-query', fasta_path,
            '-db', db_path,
            '-num_iterations', '3',
            '-evalue', '0.001',
            '-out_ascii_pssm', pssm_path,
            '-save_pssm_after_last_round',
            '-hspsep_bundlesize', '1',
            '-num_threads', '1',
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=180)
            pssm = parse_pssm(pssm_path, max_length)
        except Exception:
            pssm = np.ones((max_length, 20), dtype=np.float32) * 0.05

    return pssm


def main():
    parser = argparse.ArgumentParser(description='PSSM extraction')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--dataset', type=str, default='main',
                        choices=['main', 'new', 'price'])
    parser.add_argument('--blast-db', type=str, default=BLAST_DB)
    parser.add_argument('--blast-dir', type=str, default=BLAST_DIR)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--batch-size', type=int, default=1000)

    args = parser.parse_args()

    psiblast_path = os.path.join(args.blast_dir, 'psiblast')
    if not os.path.exists(psiblast_path):
        print(f"Error: psiblast not found at {psiblast_path}")
        return
    if not os.path.exists(args.blast_db):
        print(f"Error: BLAST database not found at {args.blast_db}")
        return

    suffix = '' if args.dataset == 'main' else f'_{args.dataset}'
    seq_file = os.path.join(args.data_dir, f'id_sequence{suffix}.pkl')
    if not os.path.exists(seq_file):
        print(f"Error: Sequence file not found at {seq_file}")
        return

    print(f"Loading sequences from {seq_file}...")
    with open(seq_file, 'rb') as f:
        id_sequence = pickle.load(f)

    sequences = list(id_sequence.values())
    total = len(sequences)
    print(f"Total proteins: {total}")
    print(f"Workers: {args.num_workers}, Batch size: {args.batch_size}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'pssm_{args.dataset}.npy')

    memmap_file = output_path + '.dat'
    if os.path.exists(memmap_file):
        existing = np.load(memmap_file, mmap_mode='r')
        start_idx = existing.shape[0]
        del existing
        print(f"Resuming from index {start_idx}")
    else:
        start_idx = 0

    mm = np.memmap(memmap_file, dtype=np.float32, mode='w+',
                    shape=(total, args.max_length, 20))

    worker_args = [
        (seq, args.blast_db, psiblast_path, args.max_length)
        for seq in sequences
    ]

    for batch_start in tqdm(range(start_idx, total, args.batch_size), desc="Batches"):
        batch_end = min(batch_start + args.batch_size, total)
        batch_args = worker_args[batch_start:batch_end]

        batch_results = []
        with Pool(args.num_workers) as pool:
            for pssm in pool.imap(extract_single_pssm, batch_args):
                batch_results.append(pssm)
                if len(batch_results) % 100 == 0:
                    print(f"  Processed {len(batch_results)}/{batch_end - batch_start}")

        for i, pssm in enumerate(batch_results):
            mm[batch_start + i] = pssm

        mm.flush()
        del batch_results
        gc.collect()

        print(f"Batch {batch_start}-{batch_end} done, progress: {batch_end}/{total} ({100*batch_end/total:.1f}%)")

    del mm
    os.rename(memmap_file, output_path)
    print(f"\nDone! Saved to {output_path}")


if __name__ == '__main__':
    main()
