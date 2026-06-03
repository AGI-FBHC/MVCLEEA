"""
Evaluate MVCLEEA by EC hierarchy level (Level 1-4).

The model predicts 5236 Level-4 EC classes. We aggregate predictions
to coarser levels and evaluate each level independently.

Usage:
    python test_by_ec_level.py --checkpoint checkpoints/best_model.pth --device cuda:2
"""
import argparse
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from models import MVCLEEA
from losses import JointFocalLoss
from utils import ProteinDataset, collate_fn


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MVCLEEA(
        num_classes=config['model']['num_classes'],
        esm2_dim=config['model']['esm2_dim'],
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def get_ec_level(ec_str, level):
    """Extract EC number at specified level.
    level=1: '3'
    level=2: '3.4'
    level=3: '3.4.11'
    level=4: '3.4.11.1'
    """
    parts = ec_str.split('.')
    return '.'.join(parts[:level])


def build_level_mapping(ec_list, level):
    """Build mapping from Level-4 index to level-N group index."""
    groups = {}
    for i, ec in enumerate(ec_list):
        group = get_ec_level(ec, level)
        if group not in groups:
            groups[group] = []
        groups[group].append(i)
    return groups


@torch.no_grad()
def get_predictions(model, esm2, pssm, onehot, device, batch_size=256):
    """Run model and return sigmoid probabilities."""
    dataset = ProteinDataset(esm2, pssm, onehot,
                             np.zeros((len(esm2), 1), dtype=np.float32),
                             augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True,
    )
    all_preds = []
    for batch in tqdm(loader, desc="Forward pass", leave=False):
        outputs = model(
            batch['esm2'].to(device),
            batch['pssm'].to(device),
            batch['onehot'].to(device),
        )
        all_preds.append(torch.sigmoid(outputs['logits']).cpu())
    return torch.cat(all_preds, dim=0).numpy()


def aggregate_labels_to_level(labels_np, ec_list, level):
    """Aggregate Level-4 binary labels to level-N.
    A sample is positive at level-N if it has ANY positive Level-4 child.
    Returns: (N, n_groups) binary array, group_names list
    """
    groups = build_level_mapping(ec_list, level)
    group_names = sorted(groups.keys())
    n_samples = labels_np.shape[0]
    aggregated = np.zeros((n_samples, len(group_names)), dtype=np.int32)
    for gi, group in enumerate(group_names):
        child_indices = groups[group]
        aggregated[:, gi] = (labels_np[:, child_indices].sum(axis=1) > 0).astype(np.int32)
    return aggregated, group_names


def aggregate_preds_to_level(preds_np, ec_list, level, method='max'):
    """Aggregate Level-4 predictions to level-N.
    method='max': take max probability among children
    method='sum': take sum (then clip to 1)
    """
    groups = build_level_mapping(ec_list, level)
    group_names = sorted(groups.keys())
    n_samples = preds_np.shape[0]
    aggregated = np.zeros((n_samples, len(group_names)), dtype=np.float32)
    for gi, group in enumerate(group_names):
        child_indices = groups[group]
        if method == 'max':
            aggregated[:, gi] = preds_np[:, child_indices].max(axis=1)
        else:
            aggregated[:, gi] = np.clip(preds_np[:, child_indices].sum(axis=1), 0, 1)
    return aggregated, group_names


def evaluate_level(labels_np, preds_np, level_name, n_balanced_classes=100, max_per_class=200):
    """Per-class balanced evaluation at one EC level."""
    rng = np.random.RandomState(42)
    n_samples, n_classes = labels_np.shape

    col_pos = labels_np.sum(axis=0)
    eligible = np.where(col_pos >= 5)[0]
    if len(eligible) == 0:
        return None

    if len(eligible) > n_balanced_classes:
        eval_classes = rng.choice(eligible, n_balanced_classes, replace=False)
    else:
        eval_classes = eligible

    # Pre-sample balanced indices
    K = len(eval_classes)
    M = max_per_class * 2
    y_true_all = np.zeros((K, M), dtype=np.int32)
    y_score_all = np.zeros((K, M), dtype=np.float32)
    valid_mask = np.zeros(K, dtype=bool)

    for ci, c in enumerate(eval_classes):
        pos_idx = np.where(labels_np[:, c] == 1)[0]
        neg_idx = np.where(labels_np[:, c] == 0)[0]
        n = min(len(pos_idx), len(neg_idx), max_per_class)
        if n < 3:
            continue
        sp = rng.choice(pos_idx, n, replace=False)
        sn = rng.choice(neg_idx, n, replace=False)
        idx = np.concatenate([sp, sn])
        m = len(idx)
        y_true_all[ci, :m] = labels_np[idx, c]
        y_score_all[ci, :m] = preds_np[idx, c]
        valid_mask[ci] = True

    y_true_all = y_true_all[valid_mask]
    y_score_all = y_score_all[valid_mask]
    K = y_true_all.shape[0]

    if K == 0:
        return None

    # Threshold search
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.10, 0.81, 0.05):
        y_pred = (y_score_all > thresh).astype(np.int32)
        tp = ((y_pred == 1) & (y_true_all == 1)).sum(axis=1).astype(float)
        fp = ((y_pred == 1) & (y_true_all == 0)).sum(axis=1).astype(float)
        fn = ((y_pred == 0) & (y_true_all == 1)).sum(axis=1).astype(float)
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1.mean() > best_f1:
            best_f1 = f1.mean()
            best_thresh = thresh

    lo, hi = max(0.01, best_thresh - 0.04), min(0.99, best_thresh + 0.04)
    for thresh in np.arange(lo, hi, 0.01):
        y_pred = (y_score_all > thresh).astype(np.int32)
        tp = ((y_pred == 1) & (y_true_all == 1)).sum(axis=1).astype(float)
        fp = ((y_pred == 1) & (y_true_all == 0)).sum(axis=1).astype(float)
        fn = ((y_pred == 0) & (y_true_all == 1)).sum(axis=1).astype(float)
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1.mean() > best_f1:
            best_f1 = f1.mean()
            best_thresh = thresh

    # Final metrics
    y_pred_best = (y_score_all > best_thresh).astype(np.int32)
    tp = ((y_pred_best == 1) & (y_true_all == 1)).sum(axis=1).astype(float)
    fp = ((y_pred_best == 1) & (y_true_all == 0)).sum(axis=1).astype(float)
    fn = ((y_pred_best == 0) & (y_true_all == 1)).sum(axis=1).astype(float)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    aucs = []
    auc_idx = rng.choice(K, min(50, K), replace=False) if K > 50 else range(K)
    for ci in auc_idx:
        yt, ys = y_true_all[ci], y_score_all[ci]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            pass

    return {
        'f1': float(f1.mean()),
        'precision': float(prec.mean()),
        'recall': float(rec.mean()),
        'auc': float(np.mean(aucs)) if aucs else 0.0,
        'threshold': best_thresh,
        'n_classes': K,
        'n_total_groups': n_classes,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate by EC level')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='main_test',
                        choices=['main_test', 'new', 'price'],
                        help='Which dataset to evaluate')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, device)

    # Load EC vocabulary
    ec_list_path = os.path.join(config['data']['processed_dir'], 'ec_list.pkl')
    with open(ec_list_path, 'rb') as f:
        ec_list = pickle.load(f)
    print(f"  EC classes (Level 4): {len(ec_list)}")

    # Show EC level statistics
    for level in [1, 2, 3, 4]:
        groups = build_level_mapping(ec_list, level)
        print(f"  Level {level}: {len(groups)} groups")

    processed_dir = config['data']['processed_dir']

    # Load data based on dataset choice
    if args.dataset == 'main_test':
        print(f"\nLoading main test split...")
        esm2 = np.load(os.path.join(processed_dir, 'esm2_main.npy'))
        pssm_path = os.path.join(processed_dir, 'pssm_main_fixed.npy')
        if not os.path.exists(pssm_path):
            pssm_path = os.path.join(processed_dir, 'pssm_main.npy')
        pssm = np.load(pssm_path)
        onehot = np.load(os.path.join(processed_dir, 'onehot_main.npy'))
        labels_path = os.path.join(processed_dir, 'labels_filtered.npy')
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
        else:
            labels = np.load(os.path.join(processed_dir, 'labels_main.npy'))
        n = len(labels)
        test_start = int(n * 0.9)
        esm2, pssm, onehot, labels = esm2[test_start:], pssm[test_start:], onehot[test_start:], labels[test_start:]
        print(f"  Test samples: {len(labels)}")
    else:
        print(f"\nLoading {args.dataset} dataset...")
        suffix = f'_{args.dataset}'
        esm2 = np.load(os.path.join(processed_dir, f'esm2{suffix}.npy'))
        pssm = np.load(os.path.join(processed_dir, f'pssm{suffix}.npy'))
        onehot = np.load(os.path.join(processed_dir, f'onehot{suffix}.npy'))
        labels = np.load(os.path.join(processed_dir, f'labels{suffix}.npy'))
        print(f"  Samples: {len(labels)}")

    # Get model predictions (at Level 4)
    print("\nRunning model inference...")
    preds_np = get_predictions(model, esm2, pssm, onehot, device)
    print(f"  Predictions shape: {preds_np.shape}")

    # Evaluate at each level
    print(f"\n{'='*70}")
    print(f"PER-LEVEL EVALUATION ({args.dataset})")
    print(f"{'='*70}")
    print(f"{'Level':<8} {'Groups':>8} {'Eval':>6} {'F1':>8} {'P':>8} {'R':>8} {'AUC':>8} {'Thresh':>8}")
    print(f"{'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for level in [1, 2, 3, 4]:
        # Aggregate labels and predictions to this level
        labels_level, group_names = aggregate_labels_to_level(labels, ec_list, level)
        preds_level, _ = aggregate_preds_to_level(preds_np, ec_list, level, method='max')

        result = evaluate_level(labels_level, preds_level, f"Level {level}")

        if result:
            print(f"Level {level:<3} {result['n_total_groups']:>8} {result['n_classes']:>6} "
                  f"{result['f1']:>8.4f} {result['precision']:>8.4f} {result['recall']:>8.4f} "
                  f"{result['auc']:>8.4f} {result['threshold']:>8.2f}")
        else:
            print(f"Level {level:<3} {len(group_names):>8} {'N/A':>6} — insufficient data")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
