"""
Data Preparation Script for MVCLEEA

Reads protein ID → sequence and ID → EC mappings from data/*.pkl,
extracts ESM2 / PSSM / One-hot features, and saves as .npy arrays.

Usage:
    # Full pipeline (ESM2 + PSSM + One-hot)
    python prepare_data.py --data-dir data --output-dir data/processed

    # Skip ESM2 (if embeddings already exist elsewhere)
    python prepare_data.py --data-dir data --output-dir data/processed --skip-esm2

    # Skip PSSM (if PSI-BLAST database not available)
    python prepare_data.py --data-dir data --output-dir data/processed --skip-pssm

    # Process evaluation set
    python prepare_data.py --data-dir data --output-dir data/processed --dataset new
"""
import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm


def load_mappings(data_dir: str, dataset_name: str):
    """Load ID→EC and ID→sequence mappings."""
    suffix = '' if dataset_name == 'main' else f'_{dataset_name}'

    with open(os.path.join(data_dir, f'id_ec{suffix}.pkl'), 'rb') as f:
        id_ec = pickle.load(f)
    with open(os.path.join(data_dir, f'id_sequence{suffix}.pkl'), 'rb') as f:
        id_sequence = pickle.load(f)

    return id_ec, id_sequence


def build_ec_vocabulary(id_ec: dict) -> list:
    """Build sorted EC number vocabulary."""
    return sorted(set(ec for ecs in id_ec.values() for ec in ecs))


def prepare_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load mappings
    id_ec, id_sequence = load_mappings(args.data_dir, args.dataset)
    all_ecs = build_ec_vocabulary(id_ec)
    ec_to_idx = {ec: i for i, ec in enumerate(all_ecs)}
    num_classes = len(all_ecs)

    # Filter to proteins with known sequences
    protein_ids = [pid for pid in id_ec.keys() if pid in id_sequence]
    print(f"Proteins with EC labels: {len(id_ec)}")
    print(f"Proteins with sequences: {len(protein_ids)}")
    print(f"EC classes: {num_classes}")

    sequences = [id_sequence[pid] for pid in protein_ids]

    # --- ESM2 Features ---
    esm2_path = os.path.join(args.output_dir, f'esm2_{args.dataset}.npy')
    if args.skip_esm2 and os.path.exists(esm2_path):
        print(f"Skipping ESM2, loading existing {esm2_path}")
        esm2_features = np.load(esm2_path)
    elif args.skip_esm2:
        print("Skipping ESM2 extraction.")
        esm2_features = None
    else:
        from features import ESM2Extractor
        print(f"Extracting ESM2 features (model: {args.esm2_model})...")
        extractor = ESM2Extractor(
            model_name=args.esm2_model,
            device=args.device,
            max_length=args.max_length,
        )
        embeddings = []
        for seq in tqdm(sequences, desc="ESM2"):
            emb = extractor.extract(seq)
            embeddings.append(emb.numpy())
        esm2_features = np.stack(embeddings)
        np.save(esm2_path, esm2_features)
        print(f"Saved ESM2: {esm2_features.shape}")

    # --- PSSM Features ---
    pssm_path = os.path.join(args.output_dir, f'pssm_{args.dataset}.npy')
    if args.skip_pssm and os.path.exists(pssm_path):
        print(f"Skipping PSSM, loading existing {pssm_path}")
        try:
            pssm_features = np.load(pssm_path)
        except Exception:
            print("  PSSM file has raw format, loading with fromfile...")
            pssm_features = np.fromfile(pssm_path, dtype=np.float32).reshape(len(protein_ids), 1024, 20)
    elif args.skip_pssm:
        print("Skipping PSSM extraction.")
        pssm_features = None
    else:
        from features import PSSMExtractor
        print(f"Extracting PSSM features (PSI-BLAST db: {args.blast_db})...")
        extractor = PSSMExtractor(
            db_path=args.blast_db,
            blast_dir=args.blast_dir,
            max_length=args.max_length,
        )
        pssm_list = []
        for seq in tqdm(sequences, desc="PSSM"):
            pssm = extractor.extract(seq)
            pssm_list.append(pssm)
        pssm_features = np.stack(pssm_list)
        np.save(pssm_path, pssm_features)
        print(f"Saved PSSM: {pssm_features.shape}")

    # --- One-hot Features ---
    onehot_path = os.path.join(args.output_dir, f'onehot_{args.dataset}.npy')
    if not args.skip_onehot:
        from features import OneHotExtractor
        print("Extracting One-hot features...")
        extractor = OneHotExtractor(max_length=args.max_length)
        onehot_features = extractor.extract_batch(sequences)
        np.save(onehot_path, onehot_features)
        print(f"Saved One-hot: {onehot_features.shape}")
    elif os.path.exists(onehot_path):
        onehot_features = np.load(onehot_path)
    else:
        onehot_features = None

    # --- Labels ---
    labels_path = os.path.join(args.output_dir, f'labels_{args.dataset}.npy')
    label_list = []
    for pid in protein_ids:
        label_vec = np.zeros(num_classes, dtype=np.float32)
        for ec in id_ec[pid]:
            if ec in ec_to_idx:
                label_vec[ec_to_idx[ec]] = 1.0
        label_list.append(label_vec)
    labels = np.stack(label_list)
    np.save(labels_path, labels)

    # --- Metadata ---
    ec_list_path = os.path.join(args.output_dir, 'ec_list.pkl')
    with open(ec_list_path, 'wb') as f:
        pickle.dump(all_ecs, f)

    pid_path = os.path.join(args.output_dir, f'protein_ids_{args.dataset}.pkl')
    with open(pid_path, 'wb') as f:
        pickle.dump(protein_ids, f)

    print(f"\nSaved to {args.output_dir}/:")
    if esm2_features is not None:
        print(f"  esm2_{args.dataset}.npy    {esm2_features.shape}")
    if pssm_features is not None:
        print(f"  pssm_{args.dataset}.npy    {pssm_features.shape}")
    if onehot_features is not None:
        print(f"  onehot_{args.dataset}.npy  {onehot_features.shape}")
    print(f"  labels_{args.dataset}.npy  {labels.shape}")
    print(f"  ec_list.pkl                ({num_classes} EC classes)")


def main():
    parser = argparse.ArgumentParser(description='Prepare MVCLEEA dataset')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--dataset', type=str, default='main',
                        choices=['main', 'new', 'price'])

    # Feature extraction options
    parser.add_argument('--skip-esm2', action='store_true',
                        help='Skip ESM2 extraction')
    parser.add_argument('--skip-pssm', action='store_true',
                        help='Skip PSSM extraction (requires PSI-BLAST)')
    parser.add_argument('--skip-onehot', action='store_true',
                        help='Skip One-hot extraction')

    # ESM2 options
    parser.add_argument('--esm2-model', type=str,
                        default='facebook/esm2_t33_650M_UR50D',
                        help='ESM2 model name from HuggingFace')

    # PSSM options
    parser.add_argument('--blast-db', type=str, default='',
                        help='Path to BLAST database for PSI-BLAST')
    parser.add_argument('--blast-dir', type=str, default='',
                        help='Path to BLAST+ bin directory')

    # General
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:3',
                        help='Device for ESM2 inference (cpu/cuda)')

    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == '__main__':
    main()
