import numpy as np

# 20 standard amino acids + unknown 'X'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class OneHotExtractor:
    """
    One-Hot Encoding Feature Extractor

    Encodes protein sequences as one-hot matrices.
    20 standard amino acids + 'X' (unknown) = 21 channels.

    Input:  Amino acid sequence string
    Output: (1024, 21) float32 matrix

    Args:
        max_length: Maximum sequence length (pad/truncate to this)
    """

    def __init__(self, max_length: int = 1024):
        self.max_length = max_length

    def extract(self, sequence: str) -> np.ndarray:
        """
        One-hot encode a single protein sequence.

        Args:
            sequence: Amino acid sequence string (e.g., "MKWVTFISLLFLFSSAYS...")
        Returns:
            onehot: (1024, 21) float32 array
        """
        onehot = np.zeros((self.max_length, 21), dtype=np.float32)

        for i, aa in enumerate(sequence[:self.max_length]):
            idx = AA_TO_IDX.get(aa.upper(), AA_TO_IDX['X'])
            onehot[i, idx] = 1.0

        # If sequence is shorter than max_length, remaining rows stay zero

        return onehot

    def extract_batch(self, sequences: list) -> np.ndarray:
        """
        One-hot encode a batch of sequences.

        Args:
            sequences: List of amino acid sequence strings
        Returns:
            onehot: (N, 1024, 21) float32 array
        """
        return np.stack([self.extract(seq) for seq in sequences])
