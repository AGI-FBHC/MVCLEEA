import torch
import numpy as np
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    Protein Dataset for MVCLEEA

    Loads pre-assembled numpy arrays produced by prepare_data.py.

    Args:
        esm2_features: (N, 1280) float32
        pssm:          (N, 1024, 20) float32, Sigmoid-normalized
        onehot:        (N, 1024, 21) float32
        labels:        (N, C) float32, multi-hot EC annotations
        augment:       Whether to apply data augmentation
        esm2_noise_std: Gaussian noise std for ESM2 features
        pssm_dropout:  Random zero-out probability for PSSM
        onehot_dropout: Random zero-out probability for One-hot
    """

    def __init__(self, esm2_features, pssm, onehot, labels,
                 augment=False, esm2_noise_std=0.01,
                 pssm_dropout=0.05, onehot_dropout=0.05):
        self.esm2_features = torch.FloatTensor(esm2_features)
        self.pssm = torch.FloatTensor(pssm)
        self.onehot = torch.FloatTensor(onehot)
        self.labels = torch.FloatTensor(labels)
        self.augment = augment
        self.esm2_noise_std = esm2_noise_std
        self.pssm_dropout = pssm_dropout
        self.onehot_dropout = onehot_dropout

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        esm2 = self.esm2_features[idx]
        pssm = self.pssm[idx]
        onehot = self.onehot[idx]

        if self.augment:
            # Gaussian noise on ESM2 features
            if self.esm2_noise_std > 0:
                noise = torch.randn_like(esm2) * self.esm2_noise_std
                esm2 = esm2 + noise

            # Random dropout on PSSM (zero out entire positions)
            if self.pssm_dropout > 0:
                mask = torch.rand(pssm.shape[0]) > self.pssm_dropout
                pssm = pssm * mask.unsqueeze(-1).float()

            # Random dropout on One-hot (zero out entire positions)
            if self.onehot_dropout > 0:
                mask = torch.rand(onehot.shape[0]) > self.onehot_dropout
                onehot = onehot * mask.unsqueeze(-1).float()

        return {
            'esm2': esm2,
            'pssm': pssm,
            'onehot': onehot,
            'labels': self.labels[idx],
        }


class FilteredProteinDataset(Dataset):
    """
    Filtered Protein Dataset with EC class-aware negative sampling.

    For each protein, keeps all positive EC labels and samples negatives
    from the same EC main class (first level), filtering out unrelated classes.

    Example: If protein has EC 3.4.11.1, keep negatives from EC 3.x.x.x only.
    """

    def __init__(self, esm2_features, pssm, onehot, labels, ec_list, neg_per_pos=5, seed=42):
        """
        Args:
            esm2_features: (N, 1280) float32
            pssm: (N, 1024, 20) float32
            onehot: (N, 1024, 21) float32
            labels: (N, C) float32
            ec_list: list of EC strings corresponding to label columns
            neg_per_pos: number of negatives to keep per positive (default: 5)
            seed: random seed for reproducibility
        """
        self.esm2_features = torch.FloatTensor(esm2_features)
        self.pssm = torch.FloatTensor(pssm)
        self.onehot = torch.FloatTensor(onehot)
        self.labels = torch.FloatTensor(labels)
        self.neg_per_pos = neg_per_pos
        self.rng = np.random.RandomState(seed)

        # Pre-compute EC main class mapping as integers for speed
        main_class_ids = [int(ec.split('.')[0]) for ec in ec_list]
        self.main_class_of = np.array(main_class_ids, dtype=np.int32)  # (C,)
        self.unique_main_classes = np.unique(self.main_class_of)

        # For each main class, pre-compute which column indices belong to it
        self.main_class_to_cols = {}
        for mc in self.unique_main_classes:
            self.main_class_to_cols[mc] = np.where(self.main_class_of == mc)[0]

        self.filtered_labels = self._create_filtered_labels_fast()

    def _create_filtered_labels_fast(self):
        """Vectorised filtered-label creation using numpy."""
        labels_np = self.labels.numpy().astype(np.int32)
        n_samples, n_classes = labels_np.shape
        filtered = np.zeros_like(labels_np)

        for i in range(n_samples):
            row = labels_np[i]
            pos_idx = np.where(row == 1)[0]
            if len(pos_idx) == 0:
                continue

            # Mark all positives
            filtered[i, pos_idx] = 1

            # Find main classes of this sample's positives
            pos_main = np.unique(self.main_class_of[pos_idx])

            # Gather all same-main-class columns, then remove the positives
            same_class_cols = np.concatenate([
                self.main_class_to_cols[mc] for mc in pos_main
            ])
            neg_candidates = np.setdiff1d(same_class_cols, pos_idx)

            # Sample negatives
            n_neg = min(len(pos_idx) * self.neg_per_pos, len(neg_candidates))
            if n_neg > 0:
                chosen = self.rng.choice(neg_candidates, size=n_neg, replace=False)
                filtered[i, chosen] = 1

        return torch.from_numpy(filtered.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'esm2': self.esm2_features[idx],
            'pssm': self.pssm[idx],
            'onehot': self.onehot[idx],
            'labels': self.filtered_labels[idx],
        }

    def get_positive_rate(self):
        """Calculate positive sample rate after filtering."""
        return self.filtered_labels.sum().item() / self.filtered_labels.numel()

    def get_stats(self):
        """Get statistics about the filtered dataset."""
        pos_per_sample = self.filtered_labels.sum(dim=1)
        return {
            'total_samples': len(self.filtered_labels),
            'mean_pos_per_sample': pos_per_sample.mean().item(),
            'max_pos_per_sample': pos_per_sample.max().item(),
            'min_pos_per_sample': pos_per_sample.min().item(),
            'filtered_positive_rate': self.get_positive_rate(),
        }


def collate_fn(batch):
    return {
        'esm2': torch.stack([item['esm2'] for item in batch]),
        'pssm': torch.stack([item['pssm'] for item in batch]),
        'onehot': torch.stack([item['onehot'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }
