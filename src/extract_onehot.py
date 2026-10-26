#!/usr/bin/env python3
"""Generate padded residue one-hot features from TSV split files."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
UNKNOWN_INDEX = len(AMINO_ACIDS)


def load_records(csv_path):
    records = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            records.append((row[0].strip(), row[2].strip().upper()))
    return records


def encode_sequence(sequence, max_length):
    encoded = np.zeros((max_length, len(AMINO_ACIDS) + 1), dtype=np.float32)
    for position, residue in enumerate(sequence[:max_length]):
        encoded[position, AMINO_ACID_TO_INDEX.get(residue, UNKNOWN_INDEX)] = 1.0
    return encoded


def main():
    parser = argparse.ArgumentParser(
        description="Generate padded 1024x21 residue one-hot features."
    )
    parser.add_argument(
        "--data_dir",
        default=PROJECT_ROOT / "data" / "datasets",
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        default=PROJECT_ROOT / "data" / "features" / "onehot",
        type=Path,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["new", "price", "new-815"],
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only encode the first N records for a quick test.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        csv_path = args.data_dir / f"{dataset}.csv"
        records = load_records(csv_path)
        if args.limit is not None:
            records = records[:args.limit]

        ids = [protein_id for protein_id, _ in records]
        features = np.stack(
            [encode_sequence(sequence, args.max_length) for _, sequence in records]
        )

        output_path = args.output_dir / f"{dataset}_onehot.npy"
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
