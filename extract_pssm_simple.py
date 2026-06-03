import os
import subprocess
import tempfile
import numpy as np
import pickle
from tqdm import tqdm


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
    except:
        pssm = np.ones((max_length, 20), dtype=np.float32) * 0.05
    return pssm


def extract_single_pssm(seq, db_path, psiblast_path, num_iters, evalue, max_length, threads):
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, 'query.fasta')
        pssm_path = os.path.join(tmpdir, 'query.pssm')

        with open(fasta_path, 'w') as f:
            f.write(f">query\n{seq}\n")

        cmd = [
            psiblast_path,
            '-query', fasta_path,
            '-db', db_path,
            '-num_iterations', str(num_iters),
            '-evalue', str(evalue),
            '-out_ascii_pssm', pssm_path,
            '-num_threads', str(threads),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            pssm = parse_pssm(pssm_path, max_length)
        except:
            pssm = np.ones((max_length, 20), dtype=np.float32) * 0.05

    return pssm


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PSSM extraction')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--dataset', type=str, default='main')
    parser.add_argument('--blast-db', type=str,
                        default='/root/autodl-tmp/MVCLEEA/pssm/uniprot_sprot.fasta')
    parser.add_argument('--blast-dir', type=str,
                        default='/root/autodl-tmp/MVCLEEA/pssm/ncbi-blast-2.17.0+/bin')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=1024)
    args = parser.parse_args()

    print(f"Using {args.threads} threads")

    suffix = '' if args.dataset == 'main' else f'_{args.dataset}'
    with open(os.path.join(args.data_dir, f'id_sequence{suffix}.pkl'), 'rb') as f:
        id_sequence = pickle.load(f)

    sequences = list(id_sequence.values())
    print(f"Total proteins: {len(sequences)}")

    psiblast_path = os.path.join(args.blast_dir, 'psiblast')
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'pssm_{args.dataset}.npy')

    all_pssm = []
    for i in tqdm(range(len(sequences)), desc="PSSM"):
        seq = sequences[i]
        pssm = extract_single_pssm(seq, args.blast_db, psiblast_path, 1, 0.001, args.max_length, args.threads)
        all_pssm.append(pssm)

    pssm_array = np.stack(all_pssm)
    print(f"PSSM array shape: {pssm_array.shape}")

    np.save(output_path, pssm_array)
    print(f"Saved PSSM to {output_path}")


if __name__ == '__main__':
    main()