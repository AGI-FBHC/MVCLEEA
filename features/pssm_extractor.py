import os
import subprocess
import numpy as np
import tempfile
from typing import Optional


# Standard 20 amino acids (order used in PSSM)
AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class PSSMExtractor:
    """
    PSSM Feature Extractor

    Generates Position-Specific Scoring Matrix from protein sequences
    using PSI-BLAST against a reference database.

    The PSSM is then Sigmoid-normalized to [0, 1] and padded/truncated
    to (1024, 20).

    Args:
        db_path:     Path to BLAST database (e.g., UniRef90)
        blast_dir:   Path to BLAST+ bin directory
        evalue:      E-value threshold for PSI-BLAST iterations
        num_iters:   Number of PSI-BLAST iterations
        max_length:  Maximum sequence length
    """

    def __init__(self, db_path: str, blast_dir: str = '',
                 evalue: float = 0.001, num_iters: int = 3,
                 max_length: int = 1024):
        self.db_path = db_path
        self.psiblast = os.path.join(blast_dir, 'psiblast')
        self.evalue = evalue
        self.num_iters = num_iters
        self.max_length = max_length

    def extract(self, sequence: str) -> np.ndarray:
        """
        Generate PSSM for a single sequence.

        Args:
            sequence: Amino acid sequence string
        Returns:
            pssm: (1024, 20) float32 array, Sigmoid-normalized
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, 'query.fasta')
            pssm_path = os.path.join(tmpdir, 'query.pssm')

            with open(fasta_path, 'w') as f:
                f.write(f">query\n{sequence}\n")

            cmd = [
                self.psiblast,
                '-query', fasta_path,
                '-db', self.db_path,
                '-num_iterations', str(self.num_iters),
                '-evalue', str(self.evalue),
                '-out_ascii_pssm', pssm_path,
                '-num_threads', '1',
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True,
                               timeout=300)
                pssm = self._parse_pssm(pssm_path)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Fallback: uniform PSSM if PSI-BLAST fails
                pssm = np.ones((self.max_length, 20), dtype=np.float32) * 0.05

        return pssm

    def _parse_pssm(self, pssm_path: str) -> np.ndarray:
        """Parse PSI-BLAST ASCII PSSM output file."""
        pssm = np.zeros((self.max_length, 20), dtype=np.float32)

        with open(pssm_path, 'r') as f:
            lines = f.readlines()

        # Skip header lines (first 3 lines)
        data_started = False
        row = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 22 and parts[0].isdigit():
                data_started = True
                if row >= self.max_length:
                    break
                # Columns 2-21 are the PSSM values for 20 amino acids
                for col in range(20):
                    pssm[row, col] = float(parts[col + 2])
                row += 1

        # Truncate to actual length, pad rest with zeros
        pssm = pssm[:self.max_length]

        # Sigmoid normalization to [0, 1]
        pssm = 1.0 / (1.0 + np.exp(-pssm))

        return pssm

    def extract_from_pssm_file(self, pssm_file: str) -> np.ndarray:
        """
        Load a pre-computed PSSM from an existing file.

        Args:
            pssm_file: Path to PSI-BLAST PSSM output file
        Returns:
            pssm: (1024, 20) float32 array, Sigmoid-normalized
        """
        pssm = self._parse_pssm(pssm_file)
        return pssm
