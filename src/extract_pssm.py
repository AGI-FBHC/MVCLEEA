#!/usr/bin/env python3
"""Generate sigmoid-normalized PSSM features with NCBI PSI-BLAST."""

import argparse
import csv
import os
import pickle
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_records(csv_path):
    records = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            protein_id = row[0].strip()
            sequence = row[2].strip().upper()
            records.append((protein_id, sequence))
    return records


def parse_pssm(path, max_length):
    pssm = np.zeros((max_length, 20), dtype=np.float32)
    rows = 0
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 22 or not parts[0].isdigit():
                continue
            if rows >= max_length:
                break
            pssm[rows] = np.asarray(parts[2:22], dtype=np.float32)
            rows += 1

    if rows == 0:
        raise ValueError(f"No PSSM rows parsed from {path}")
    return 1.0 / (1.0 + np.exp(-pssm))


def extract_one(task):
    (
        protein_id,
        sequence,
        psiblast,
        blast_db,
        num_iterations,
        evalue,
        max_length,
    ) = task

    with tempfile.TemporaryDirectory() as temporary_directory:
        fasta_path = os.path.join(temporary_directory, "query.fasta")
        pssm_path = os.path.join(temporary_directory, "query.pssm")

        with open(fasta_path, "w") as handle:
            handle.write(f">{protein_id}\n{sequence}\n")

        command = [
            psiblast,
            "-query",
            fasta_path,
            "-db",
            blast_db,
            "-num_iterations",
            str(num_iterations),
            "-evalue",
            str(evalue),
            "-out_ascii_pssm",
            pssm_path,
            "-save_pssm_after_last_round",
            "-num_threads",
            "1",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PSI-BLAST failed for {protein_id}: {result.stderr.strip()}"
            )
        return parse_pssm(pssm_path, max_length)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1024x20 sigmoid-normalized PSSM features."
    )
    parser.add_argument(
        "--data_dir",
        default=PROJECT_ROOT / "data" / "datasets",
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        default=PROJECT_ROOT / "data" / "features" / "pssm",
        type=Path,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["new", "price", "new-815"],
    )
    parser.add_argument("--psiblast", required=True, type=Path)
    parser.add_argument("--blast_db", required=True, type=Path)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only extract the first N records for a quick test.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        csv_path = args.data_dir / f"{dataset}.csv"
        records = load_records(csv_path)
        if args.limit is not None:
            records = records[:args.limit]

        ids = [protein_id for protein_id, _ in records]
        tasks = [
            (
                protein_id,
                sequence,
                str(args.psiblast),
                str(args.blast_db),
                args.num_iterations,
                args.evalue,
                args.max_length,
            )
            for protein_id, sequence in records
        ]

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            features = np.stack(list(pool.map(extract_one, tasks)))

        output_path = args.output_dir / f"{dataset}_pssm.npy"
        ids_path = args.output_dir / f"{dataset}_ids.pkl"
        np.save(output_path, features)
        with open(ids_path, "wb") as handle:
            pickle.dump(ids, handle)

        print(
            f"{dataset}: saved {features.shape} {features.dtype} "
            f"to {output_path}"
        )


if __name__ == "__main__":
    main()
