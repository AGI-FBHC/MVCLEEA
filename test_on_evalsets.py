"""
Evaluate trained MVCLEEA model on independent test sets.

Test sets:
  - main_test: held-out split from main training set
  - new: New-392 independent evaluation set
  - price: Price-149 independent evaluation set

Usage:
    python test_on_evalsets.py --checkpoint checkpoints/best_model.pth --device cuda:2
    python test_on_evalsets.py --checkpoint checkpoints/best_model.pth --device cuda:2 --skip-esm2
"""
import argparse
import os
import pickle
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

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
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val_f1_balanced={ckpt.get('val_f1_balanced', 'N/A')}")
    return model, config


def ensure_test_features(dataset_name, data_dir, output_dir, ec_list,
                         esm2_model='facebook/esm2_t33_650M_UR50D',
                         device='cpu', max_length=1024, skip_esm2=False):
    """Ensure ESM2, PSSM, One-hot, and labels exist for a test dataset."""
    suffix = f'_{dataset_name}'

    # Load mappings
    with open(os.path.join(data_dir, f'id_ec{suffix}.pkl'), 'rb') as f:
        id_ec = pickle.load(f)
    with open(os.path.join(data_dir, f'id_sequence{suffix}.pkl'), 'rb') as f:
        id_sequence = pickle.load(f)

    # Filter to proteins with known sequences
    protein_ids = [pid for pid in id_ec.keys() if pid in id_sequence]
    sequences = [id_sequence[pid] for pid in protein_ids]
    print(f"  {dataset_name}: {len(protein_ids)} proteins with sequences")

    # Build labels using training EC vocabulary
    ec_to_idx = {ec: i for i, ec in enumerate(ec_list)}
    labels = np.zeros((len(protein_ids), len(ec_list)), dtype=np.float32)
    matched, unmatched = 0, 0
    for i, pid in enumerate(protein_ids):
        for ec in id_ec[pid]:
            if ec in ec_to_idx:
                labels[i, ec_to_idx[ec]] = 1.0
                matched += 1
            else:
                unmatched += 1
    print(f"  Labels: {matched} matched, {unmatched} unseen EC numbers")

    labels_path = os.path.join(output_dir, f'labels{suffix}.npy')
    np.save(labels_path, labels)

    # ESM2
    esm2_path = os.path.join(output_dir, f'esm2{suffix}.npy')
    if os.path.exists(esm2_path):
        print(f"  ESM2: loading from {esm2_path}")
        esm2_data = np.load(esm2_path)
    elif skip_esm2:
        print(f"  ESM2: skipped (--skip-esm2)")
        esm2_data = None
    else:
        print(f"  ESM2: extracting with {esm2_model}...")
        from features import ESM2Extractor
        extractor = ESM2Extractor(model_name=esm2_model, device=device, max_length=max_length)
        embeddings = []
        for seq in tqdm(sequences, desc=f"ESM2 ({dataset_name})"):
            embeddings.append(extractor.extract(seq).numpy())
        esm2_data = np.stack(embeddings)
        np.save(esm2_path, esm2_data)
        print(f"  ESM2: saved {esm2_data.shape}")

    # PSSM
    pssm_path = os.path.join(output_dir, f'pssm{suffix}.npy')
    if os.path.exists(pssm_path):
        print(f"  PSSM: loading from {pssm_path}")
        pssm_data = np.load(pssm_path)
    else:
        print(f"  PSSM: NOT FOUND at {pssm_path}")
        pssm_data = None

    # One-hot
    onehot_path = os.path.join(output_dir, f'onehot{suffix}.npy')
    if os.path.exists(onehot_path):
        print(f"  One-hot: loading from {onehot_path}")
        onehot_data = np.load(onehot_path)
    else:
        print(f"  One-hot: generating...")
        from features import OneHotExtractor
        extractor = OneHotExtractor(max_length=max_length)
        onehot_data = extractor.extract_batch(sequences)
        np.save(onehot_path, onehot_data)
        print(f"  One-hot: saved {onehot_data.shape}")

    return esm2_data, pssm_data, onehot_data, labels


@torch.no_grad()
def evaluate_balanced(model, esm2, pssm, onehot, labels, device,
                      n_eval_classes=200, max_per_class=200, batch_size=256):
    """Per-class balanced evaluation."""
    dataset = ProteinDataset(esm2, pssm, onehot, labels, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=True,
    )

    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Forward pass", leave=False):
        outputs = model(
            batch['esm2'].to(device),
            batch['pssm'].to(device),
            batch['onehot'].to(device),
        )
        all_preds.append(torch.sigmoid(outputs['logits']).cpu())
        all_labels.append(batch['labels'])

    preds_np = torch.cat(all_preds, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy().astype(np.int32)

    # Per-class balanced sampling
    rng = np.random.RandomState(42)
    col_pos = labels_np.sum(axis=0)
    eligible = np.where(col_pos >= 10)[0]
    if len(eligible) > n_eval_classes:
        eval_classes = rng.choice(eligible, n_eval_classes, replace=False)
    else:
        eval_classes = eligible

    K = len(eval_classes)
    M = max_per_class * 2
    y_true_all = np.zeros((K, M), dtype=np.int32)
    y_score_all = np.zeros((K, M), dtype=np.float32)
    valid_mask = np.zeros(K, dtype=bool)

    for ci, c in enumerate(eval_classes):
        pos_idx = np.where(labels_np[:, c] == 1)[0]
        neg_idx = np.where(labels_np[:, c] == 0)[0]
        n = min(len(pos_idx), len(neg_idx), max_per_class)
        if n < 5:
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

    aucs, auprs = [], []
    auc_idx = rng.choice(K, min(50, K), replace=False) if K > 50 else range(K)
    for ci in auc_idx:
        yt, ys = y_true_all[ci], y_score_all[ci]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            pass
        try:
            auprs.append(average_precision_score(yt, ys))
        except ValueError:
            pass

    return {
        'f1_balanced': float(f1.mean()),
        'precision_balanced': float(prec.mean()),
        'recall_balanced': float(rec.mean()),
        'auc_balanced': float(np.mean(aucs)) if aucs else 0.0,
        'aupr_balanced': float(np.mean(auprs)) if auprs else 0.0,
        'best_threshold': best_thresh,
        'n_eval_classes': K,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate MVCLEEA on test sets')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--skip-esm2', action='store_true',
                        help='Skip ESM2 extraction (use pre-existing features)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['main_test', 'new', 'price'],
                        help='Which datasets to evaluate')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, device)

    # Load EC vocabulary
    ec_list_path = os.path.join(config['data']['processed_dir'], 'ec_list.pkl')
    with open(ec_list_path, 'rb') as f:
        ec_list = pickle.load(f)
    print(f"  EC classes: {len(ec_list)}")

    processed_dir = config['data']['processed_dir']
    data_dir = 'data'

    results = {}

    # ── Main test split ──
    if 'main_test' in args.datasets:
        print(f"\n{'='*60}")
        print("Evaluating on main test split...")
        print(f"{'='*60}")

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

        # Use same split as training (last 10%)
        n = len(labels)
        test_start = int(n * 0.9)
        esm2_test = esm2[test_start:]
        pssm_test = pssm[test_start:]
        onehot_test = onehot[test_start:]
        labels_test = labels[test_start:]
        print(f"  Test samples: {len(labels_test)}")

        metrics = evaluate_balanced(model, esm2_test, pssm_test, onehot_test,
                                    labels_test, device)
        results['main_test'] = metrics
        print(f"  Results: F1={metrics['f1_balanced']:.4f}, "
              f"P={metrics['precision_balanced']:.4f}, "
              f"R={metrics['recall_balanced']:.4f}, "
              f"AUC={metrics['auc_balanced']:.4f}, "
              f"AUPR={metrics['aupr_balanced']:.4f}, "
              f"Thresh={metrics['best_threshold']:.2f}")

    # ── Independent test sets ──
    for ds_name in ['new', 'price']:
        if ds_name not in args.datasets:
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name} dataset...")
        print(f"{'='*60}")

        esm2, pssm, onehot, labels = ensure_test_features(
            ds_name, data_dir, processed_dir, ec_list,
            device=args.device if not args.skip_esm2 else 'cpu',
            skip_esm2=args.skip_esm2,
        )

        if esm2 is None or pssm is None or onehot is None:
            print(f"  SKIPPING {ds_name}: missing features")
            continue

        metrics = evaluate_balanced(model, esm2, pssm, onehot, labels, device)
        results[ds_name] = metrics
        print(f"  Results: F1={metrics['f1_balanced']:.4f}, "
              f"P={metrics['precision_balanced']:.4f}, "
              f"R={metrics['recall_balanced']:.4f}, "
              f"AUC={metrics['auc_balanced']:.4f}, "
              f"AUPR={metrics['aupr_balanced']:.4f}, "
              f"Thresh={metrics['best_threshold']:.2f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'F1':>8} {'P':>8} {'R':>8} {'AUC':>8} {'AUPR':>8} {'Thresh':>8} {'Classes':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in results.items():
        print(f"{name:<15} {m['f1_balanced']:>8.4f} {m['precision_balanced']:>8.4f} "
              f"{m['recall_balanced']:>8.4f} {m['auc_balanced']:>8.4f} "
              f"{m['aupr_balanced']:>8.4f} {m['best_threshold']:>8.2f} "
              f"{m['n_eval_classes']:>8d}")


if __name__ == '__main__':
    main()
