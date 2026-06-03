import argparse
import os
import sys
import time
import pickle
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

from models import MVCLEEA
from losses import JointFocalLoss
from utils import ProteinDataset, FilteredProteinDataset, collate_fn


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class TeeLogger:
    def __init__(self, log_path: str):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log = open(log_path, 'w', buffering=1)

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def stratified_multilabel_split(labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Simple multilabel stratified split based on most frequent class per sample."""
    np.random.seed(seed)
    n = len(labels)
    # Assign each sample to its most frequent positive class for stratification
    primary_class = np.argmax(labels, axis=1)
    classes = np.unique(primary_class)

    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(primary_class == c)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(0, int(len(idx) * val_ratio))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, file=sys.stdout)
    for batch in pbar:
        esm2 = batch['esm2'].to(device)
        pssm = batch['pssm'].to(device)
        onehot = batch['onehot'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(esm2, pssm, onehot)

        loss, loss_focal, L_ES, L_PS, L_OS = criterion(
            outputs['logits'], labels,
            outputs['Z_E'], outputs['Z_P'], outputs['Z_O'], outputs['Z_S']
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_focal += loss_focal.item()
        all_preds.append(torch.sigmoid(outputs['logits']).detach().cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}", focal=f"{loss_focal.item():.4f}")

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    avg_focal = total_focal / len(dataloader)

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds_binary = (preds > 0.5).astype(np.int32)

    train_f1_micro = f1_score(labels, preds_binary, average='micro', zero_division=0)
    train_f1_weighted = f1_score(labels, preds_binary, average='weighted', zero_division=0)

    return avg_loss, avg_focal, train_f1_micro, train_f1_weighted


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, n_eval_classes=200, max_per_class=100):
    """
    Evaluate with per-class balanced sampling, fully vectorised.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False, file=sys.stdout)
    for batch in pbar:
        esm2 = batch['esm2'].to(device)
        pssm = batch['pssm'].to(device)
        onehot = batch['onehot'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(esm2, pssm, onehot)
        loss, _, _, _, _ = criterion(
            outputs['logits'], labels,
            outputs['Z_E'], outputs['Z_P'], outputs['Z_O'], outputs['Z_S']
        )

        total_loss += loss.item()
        all_preds.append(outputs['logits'].cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    preds_np = torch.sigmoid(preds).numpy()
    labels_np = labels.numpy().astype(np.int32)

    # ── Pre-compute balanced sample indices per class ──
    rng = np.random.RandomState(42)
    col_pos = labels_np.sum(axis=0)
    eligible = np.where(col_pos >= 10)[0]
    if len(eligible) > n_eval_classes:
        eval_classes = rng.choice(eligible, n_eval_classes, replace=False)
    else:
        eval_classes = eligible

    # Pre-sample balanced indices and pack into arrays
    K = len(eval_classes)
    M = max_per_class * 2  # pos + neg per class
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

    y_true_all = y_true_all[valid_mask]   # (K', M)
    y_score_all = y_score_all[valid_mask]
    K = y_true_all.shape[0]

    # ── Fully vectorised threshold search ──
    # For each (class, threshold), compute TP, FP, FN then F1
    thresholds = np.arange(0.10, 0.81, 0.05)
    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_score_all > thresh).astype(np.int32)  # (K, M)
        tp = ((y_pred == 1) & (y_true_all == 1)).sum(axis=1).astype(np.float32)
        fp = ((y_pred == 1) & (y_true_all == 0)).sum(axis=1).astype(np.float32)
        fn = ((y_pred == 0) & (y_true_all == 1)).sum(axis=1).astype(np.float32)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-8)
        mean_f1 = f1_arr.mean()
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thresh = thresh

    # Fine search
    lo = max(0.01, best_thresh - 0.04)
    hi = min(0.99, best_thresh + 0.04)
    for thresh in np.arange(lo, hi, 0.01):
        y_pred = (y_score_all > thresh).astype(np.int32)
        tp = ((y_pred == 1) & (y_true_all == 1)).sum(axis=1).astype(np.float32)
        fp = ((y_pred == 1) & (y_true_all == 0)).sum(axis=1).astype(np.float32)
        fn = ((y_pred == 0) & (y_true_all == 1)).sum(axis=1).astype(np.float32)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-8)
        mean_f1 = f1_arr.mean()
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thresh = thresh

    # ── Final metrics at best threshold ──
    y_pred_best = (y_score_all > best_thresh).astype(np.int32)
    tp = ((y_pred_best == 1) & (y_true_all == 1)).sum(axis=1).astype(np.float32)
    fp = ((y_pred_best == 1) & (y_true_all == 0)).sum(axis=1).astype(np.float32)
    fn = ((y_pred_best == 0) & (y_true_all == 1)).sum(axis=1).astype(np.float32)
    prec_arr = tp / (tp + fp + 1e-8)
    rec_arr = tp / (tp + fn + 1e-8)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)

    # AUC/AUPR per class (subsample for speed)
    class_aucs, class_auprs = [], []
    auc_indices = rng.choice(K, min(50, K), replace=False) if K > 50 else range(K)
    for ci in auc_indices:
        yt = y_true_all[ci]
        ys = y_score_all[ci]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            class_aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            pass
        try:
            class_auprs.append(average_precision_score(yt, ys))
        except ValueError:
            pass

    metrics = {
        'f1_balanced': float(f1_arr.mean()),
        'precision_balanced': float(prec_arr.mean()),
        'recall_balanced': float(rec_arr.mean()),
        'auc_balanced': np.mean(class_aucs) if class_aucs else 0.0,
        'aupr_balanced': np.mean(class_auprs) if class_auprs else 0.0,
        'best_threshold': best_thresh,
        'n_eval_classes': K,
    }

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train MVCLEEA Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Setup logging
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or 'train'
    log_path = os.path.join(log_dir, f"{run_name}_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"Log file: {log_path}")
    print(f"Device: {device}")

    # ── Load data ──
    data_cfg = config['data']
    processed_dir = data_cfg.get('processed_dir', 'data/processed')

    print(f"Loading preprocessed data from {processed_dir}/...")
    esm2_data = np.load(os.path.join(processed_dir, 'esm2_main.npy'))

    pssm_fixed = os.path.join(processed_dir, 'pssm_main_fixed.npy')
    pssm_raw = os.path.join(processed_dir, 'pssm_main.npy')
    if os.path.exists(pssm_fixed):
        pssm_data = np.load(pssm_fixed)
    else:
        try:
            pssm_data = np.load(pssm_raw)
        except Exception:
            pssm_data = np.fromfile(pssm_raw, dtype=np.float32).reshape(227177, 1024, 20)
            np.save(pssm_fixed, pssm_data)

    onehot_path = os.path.join(processed_dir, 'onehot_main.npy')
    if os.path.exists(onehot_path):
        onehot_data = np.load(onehot_path)
    else:
        print("WARNING: onehot_main.npy not found, generating...")
        from features import OneHotExtractor
        with open(os.path.join(processed_dir, '..', 'id_sequence.pkl'), 'rb') as f:
            id_sequence = pickle.load(f)
        protein_ids_path = os.path.join(processed_dir, 'protein_ids_main.pkl')
        if os.path.exists(protein_ids_path):
            with open(protein_ids_path, 'rb') as f:
                protein_ids = pickle.load(f)
        else:
            with open(os.path.join(processed_dir, '..', 'id_ec.pkl'), 'rb') as f:
                id_ec = pickle.load(f)
            protein_ids = [pid for pid in id_ec.keys() if pid in id_sequence]
        sequences = [id_sequence[pid] for pid in protein_ids]
        onehot_data = OneHotExtractor(max_length=1024).extract_batch(sequences)
        np.save(onehot_path, onehot_data)

    labels_path = os.path.join(processed_dir, 'labels_filtered.npy')
    if os.path.exists(labels_path):
        labels_data = np.load(labels_path)
    else:
        labels_data = np.load(os.path.join(processed_dir, 'labels_main.npy'))

    # Load EC list for filtered dataset
    ec_list_path = os.path.join(processed_dir, 'ec_list.pkl')
    with open(ec_list_path, 'rb') as f:
        ec_list = pickle.load(f)

    print(f"  ESM2: {esm2_data.shape}, PSSM: {pssm_data.shape}, "
          f"One-hot: {onehot_data.shape}, Labels: {labels_data.shape}")
    print(f"  EC classes: {len(ec_list)}")

    # ── Create dataset ──
    use_filtered = data_cfg.get('use_filtered_dataset', True)
    aug_cfg = data_cfg.get('augmentation', {})

    if use_filtered:
        print(f"\nUsing FilteredProteinDataset (neg_per_pos={data_cfg.get('neg_per_pos', 5)})...")
        dataset = FilteredProteinDataset(
            esm2_data, pssm_data, onehot_data, labels_data, ec_list,
            neg_per_pos=data_cfg.get('neg_per_pos', 5), seed=42
        )
        stats = dataset.get_stats()
        print(f"  Filtered stats:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        active_labels = dataset.filtered_labels.numpy()
    else:
        dataset = ProteinDataset(
            esm2_data, pssm_data, onehot_data, labels_data,
            augment=True,
            esm2_noise_std=aug_cfg.get('esm2_noise_std', 0.01),
            pssm_dropout=aug_cfg.get('pssm_dropout', 0.05),
            onehot_dropout=aug_cfg.get('onehot_dropout', 0.05),
        )
        active_labels = labels_data

    # Non-augmented copy for validation
    val_dataset_raw = ProteinDataset(esm2_data, pssm_data, onehot_data, labels_data, augment=False)

    # ── Split ──
    pos_rate = active_labels.sum() / active_labels.size
    neg_rate = 1 - pos_rate
    pos_weight = neg_rate / pos_rate
    print(f"\n  Active positive rate: {pos_rate*100:.4f}%")
    print(f"  pos_weight: {pos_weight:.2f}")

    train_idx, val_idx, test_idx = stratified_multilabel_split(
        active_labels,
        train_ratio=data_cfg['train_ratio'],
        val_ratio=data_cfg['val_ratio'],
        seed=42,
    )
    print(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    if use_filtered:
        train_set = Subset(dataset, train_idx)
    else:
        train_set = Subset(dataset, train_idx)
    val_set = Subset(val_dataset_raw, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=config['training']['batch_size'],
        shuffle=True, collate_fn=collate_fn,
        num_workers=data_cfg.get('num_workers', 4), pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config['training']['batch_size'],
        shuffle=False, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Build model ──
    model = MVCLEEA(
        num_classes=config['model']['num_classes'],
        esm2_dim=config['model']['esm2_dim'],
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # ── Build loss ──
    loss_cfg = config['loss']
    criterion = JointFocalLoss(
        num_classes=config['model']['num_classes'],
        lambda_val=loss_cfg.get('lambda_val', 0.02),
        temperature=loss_cfg.get('temperature', 0.07),
        gamma=loss_cfg.get('gamma', 5.0),
        pos_weight=pos_weight,
        label_smoothing=loss_cfg.get('label_smoothing', 0.05),
    )

    # ── Build optimizer + scheduler ──
    train_cfg = config['training']
    optimizer = optim.Adam(
        model.parameters(), lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )
    warmup_epochs = train_cfg.get('warmup_epochs', 10)
    total_epochs = train_cfg['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs)
    early_stopper = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 10))

    # ── Print config ──
    print(f"\n{'='*70}")
    print(f"AGGRESSIVE TRAINING CONFIG:")
    print(f"  Epochs: {total_epochs}, Batch: {train_cfg['batch_size']}, LR: {train_cfg['learning_rate']}")
    print(f"  Lambda: {loss_cfg['lambda_val']}, Gamma: {loss_cfg['gamma']}, Temp: {loss_cfg['temperature']}")
    print(f"  Label Smoothing: {loss_cfg.get('label_smoothing', 0.05)}")
    print(f"  pos_weight: {pos_weight:.2f}")
    print(f"  Warmup: {warmup_epochs}, Early Stop: {train_cfg.get('early_stopping_patience', 10)}")
    print(f"  Filtered Dataset: {use_filtered}")
    print(f"{'='*70}\n")

    save_dir = config['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        train_loss, train_focal, train_f1_micro, train_f1_weighted = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scheduler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        progress = epoch / total_epochs * 100
        best_marker = " *" if val_metrics['f1_balanced'] > best_f1 else ""

        print(f"Epoch {epoch:3d}/{total_epochs} ({progress:5.1f}%) | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s | Total: {total_elapsed/60:.1f}min")
        print(f"  Train — Loss: {train_loss:.4f}, Focal: {train_focal:.4f}, "
              f"F1 μ: {train_f1_micro:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, "
              f"F1 bal: {val_metrics['f1_balanced']:.4f}, "
              f"P bal: {val_metrics['precision_balanced']:.4f}, "
              f"R bal: {val_metrics['recall_balanced']:.4f}")
        print(f"  Val   — AUC: {val_metrics['auc_balanced']:.4f}, "
              f"AUPR: {val_metrics['aupr_balanced']:.4f}, "
              f"Thresh: {val_metrics['best_threshold']:.2f}, "
              f"Classes: {val_metrics['n_eval_classes']}{best_marker}")

        if val_metrics['f1_balanced'] > best_f1:
            best_f1 = val_metrics['f1_balanced']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_f1_balanced': best_f1,
                'val_metrics': val_metrics,
                'config': config,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  >> Best model saved (F1 bal: {best_f1:.4f})")

        if early_stopper.step(val_metrics['f1_balanced']):
            print(f"\nEarly stopping at epoch {epoch}")
            break
        print()

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete in {total_time/60:.1f} min. Best balanced F1: {best_f1:.4f}")
    print(f"{'='*70}")

    sys.stdout = logger.stdout
    sys.stderr = logger.stderr
    logger.close()
    print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
