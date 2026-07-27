#!/usr/bin/env python3
"""MVCLEEA v2: mask-aware ESM-2 + PSSM + OneHot."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
import sys
import argparse
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, normalize
import pickle

from components import (
    PLMViewEncoder, CrossAttentionPooling,
    ProjectionHead, get_ec_level
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Encoders
# ============================================================

class ResidueEncoder(nn.Module):
    """Encode residue features without convolving across amino-acid columns."""

    def __init__(self, input_dim, output_dim=256, hidden_dim=128):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x, mask):
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        x = self.input_proj(self.input_norm(x)) * mask_f
        residual = x
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = (x + residual) * mask_f

        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attentive_mean = (x * weights).sum(dim=1)

        masked_x = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
        max_pool = masked_x.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.output(torch.cat([attentive_mean, max_pool], dim=-1))


class MultiLabelHierarchicalSupConLoss(nn.Module):
    """Treat proteins sharing any EC prefix as positives at each level."""

    def __init__(self, temperature=0.07, level_weights=(0.1, 0.2, 0.3, 0.4)):
        super().__init__()
        self.temperature = temperature
        self.level_weights = level_weights

    def forward(self, embeddings, labels_per_level):
        z = F.normalize(embeddings, dim=1)
        logits = z @ z.t() / self.temperature
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        total = embeddings.sum() * 0.0
        total_weight = 0.0
        for weight, labels in zip(self.level_weights, labels_per_level):
            positives = (labels.float() @ labels.float().t()) > 0
            positives &= ~eye
            valid = positives.any(dim=1)
            if not valid.any():
                continue
            per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) /
                           positives.sum(dim=1).clamp_min(1))
            total = total + weight * per_anchor[valid].mean()
            total_weight += weight
        return total / max(total_weight, 1e-8)


def teacher_alignment_loss(teacher, students):
    """Align each biological view to ESM while preserving ESM as the anchor."""
    target = F.normalize(teacher.detach(), dim=1)
    return torch.stack([
        (1.0 - (F.normalize(student, dim=1) * target).sum(dim=1)).mean()
        for student in students
    ]).mean()


# ============================================================
# 3-View Model: ESM-2 + PSSM + OneHot
# ============================================================

class DynamicFeatureFusion(nn.Module):
    """Predict per-protein view weights instead of global scalar weights."""

    def __init__(self, feature_dim=256, num_views=4):
        super().__init__()
        self.num_views = num_views
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * num_views, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_views),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, *views):
        weights = torch.softmax(self.gate(torch.cat(views, dim=1)), dim=1)
        stacked = torch.stack(views, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.norm(fused + views[0]), weights


class MVCLEEA_ESM2_PSSM_OneHotV2(nn.Module):
    """Mask-aware ESM-2 + PSSM + OneHot model."""

    def __init__(self, esm_dim=1280,
                 feature_dim=256, hidden_dim=512, out_dim=128, num_heads=8):
        super().__init__()
        self.esm_encoder = PLMViewEncoder(esm_dim, feature_dim)
        self.pssm_encoder = ResidueEncoder(20, feature_dim)
        self.onehot_encoder = ResidueEncoder(21, feature_dim)
        self.cross_attention = CrossAttentionPooling(feature_dim, num_heads)
        self.fusion = DynamicFeatureFusion(feature_dim, num_views=4)
        self.projection = ProjectionHead(feature_dim, hidden_dim, out_dim)

    def forward(self, esm, pssm, onehot):
        mask = onehot.abs().sum(dim=-1) > 0
        # Mask inconsistent PSSM padding before any normalization or convolution.
        pssm = pssm.masked_fill(~mask.unsqueeze(-1), 0.0)
        Z_E = self.esm_encoder(esm)
        Z_P = self.pssm_encoder(pssm, mask)
        Z_O = self.onehot_encoder(onehot, mask)
        Z_S, attn_weights = self.cross_attention(Z_E, Z_P, Z_O)
        Z_F, fusion_weights = self.fusion(Z_E, Z_P, Z_O, Z_S)
        embedding = self.projection(Z_F)
        return embedding, Z_E, Z_P, Z_O, Z_S, attn_weights, fusion_weights

    def get_fused_embedding(self, esm, pssm, onehot):
        with torch.no_grad():
            embedding, _, _, _, _, _, _ = self.forward(esm, pssm, onehot)
        return embedding


# ============================================================
# Data Loading
# ============================================================

def load_ec_mappings(path):
    id_ec, ec_id = {}, {}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            id_ec[row[0]] = row[1].split(';')
            for ec in row[1].split(';'):
                ec_id.setdefault(ec, set()).add(row[0])
    return id_ec, ec_id


def load_esm2_embeddings(ids, esm2_dir):
    emb_list = []
    missing = []
    for pid in ids:
        path = f'{esm2_dir}/{pid}.pt'
        if os.path.exists(path):
            emb_list.append(torch.load(path, weights_only=False))
        else:
            missing.append(pid)
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} ESM-2 embeddings in {esm2_dir}; '
            f'first missing ID: {missing[0]}'
        )
    return torch.stack(emb_list)


def load_data(ids, data_dir, split_name):
    """Load ESM-2, PSSM, OneHot for given IDs."""
    esm = load_esm2_embeddings(ids, esm2_dir)

    # PSSM
    pssm_path = f'{data_dir}/pssm/{split_name}_pssm.npy'
    pssm_ids_path = f'{data_dir}/pssm/{split_name}_ids.pkl'
    if os.path.exists(pssm_path) and os.path.exists(pssm_ids_path):
        pssm_data = np.load(pssm_path)
        with open(pssm_ids_path, 'rb') as f:
            pssm_ids = pickle.load(f)
        id_to_idx = {pid: i for i, pid in enumerate(pssm_ids)}
        pssm_list = []
        for pid in ids:
            if pid in id_to_idx:
                pssm_list.append(torch.tensor(pssm_data[id_to_idx[pid]], dtype=torch.float32))
            else:
                pssm_list.append(torch.zeros(1024, 20))
        pssm = torch.stack(pssm_list)
    else:
        pssm = torch.zeros(len(ids), 1024, 20)

    # OneHot
    onehot_path = f'{data_dir}/onehot/{split_name}_onehot.npy'
    onehot_ids_path = f'{data_dir}/onehot/{split_name}_ids.pkl'
    if os.path.exists(onehot_path):
        onehot_data = np.load(onehot_path)
        if os.path.exists(onehot_ids_path):
            with open(onehot_ids_path, 'rb') as f:
                onehot_ids = pickle.load(f)
        elif os.path.exists(pssm_ids_path):
            # Legacy OneHot arrays were generated in the PSSM ID order.
            with open(pssm_ids_path, 'rb') as f:
                onehot_ids = pickle.load(f)
        else:
            raise FileNotFoundError(f'No ID mapping found for {onehot_path}')
        id_to_idx = {pid: i for i, pid in enumerate(onehot_ids)}
        onehot_list = []
        for pid in ids:
            if pid in id_to_idx:
                onehot_list.append(torch.tensor(onehot_data[id_to_idx[pid]], dtype=torch.float32))
            else:
                onehot_list.append(torch.zeros(1024, 21))
        onehot = torch.stack(onehot_list)
    else:
        onehot = torch.zeros(len(ids), 1024, 21)

    return esm, pssm, onehot


# ============================================================
# Training
# ============================================================

def train(train_ids, id_ec, ec_id, data_dir, split_name,
          epochs=100, lr=0.0005, batch_size=128, temperature=0.07,
          alpha=0.1, device='cpu', save_path='data/model/esm2_pssm_onehot_v2.pth'):
    """Train ESM-2 + PSSM + OneHot model."""

    print('Loading data...', flush=True)
    esm_tr, pssm_tr, onehot_tr = load_data(train_ids, data_dir, split_name)

    ec_list = sorted(list(ec_id.keys()))
    ec_to_idx_per_level = {}
    for level in range(1, 5):
        level_ecs = sorted(set(get_ec_level(ec, level) for ec in ec_list))
        ec_to_idx_per_level[level] = {ec: i for i, ec in enumerate(level_ecs)}

    id_ec_ordered = {pid: id_ec[pid] for pid in train_ids}
    label_tensors = {}
    for level in range(1, 5):
        labels = np.zeros(
            (len(train_ids), len(ec_to_idx_per_level[level])),
            dtype=np.bool_,
        )
        for i, (pid, ecs) in enumerate(id_ec_ordered.items()):
            for ec in ecs:
                le = get_ec_level(ec, level)
                if le in ec_to_idx_per_level[level]:
                    labels[i, ec_to_idx_per_level[level][le]] = True
        label_tensors[level] = torch.from_numpy(labels)

    model = MVCLEEA_ESM2_PSSM_OneHotV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    supcon_criterion = MultiLabelHierarchicalSupConLoss(temperature=temperature)

    print(f'Training for {epochs} epochs...', flush=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(train_ids))

        for i in range(0, len(train_ids), batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < 2:
                continue

            esm_batch = esm_tr[batch_idx].to(device)
            pssm_batch = pssm_tr[batch_idx].to(device)
            onehot_batch = onehot_tr[batch_idx].to(device)
            hier_label_batch = [
                label_tensors[level][batch_idx].to(device)
                for level in range(1, 5)
            ]

            embedding, Z_E, Z_P, Z_O, Z_S, _, _ = model(esm_batch, pssm_batch, onehot_batch)

            loss_supcon = supcon_criterion(embedding, hier_label_batch)
            loss_crossview = teacher_alignment_loss(Z_E, (Z_P, Z_O))

            loss = (1 - alpha) * loss_supcon + alpha * loss_crossview

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_ids) // batch_size, 1)

        if (epoch + 1) % 10 == 0:
            supcon_val = loss_supcon.item() if hasattr(loss_supcon, 'item') else float(loss_supcon)
            crossview_val = loss_crossview.item() if hasattr(loss_crossview, 'item') else float(loss_crossview)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} '
                  f'(SupCon: {supcon_val:.4f}, CrossView: {crossview_val:.4f})', flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    print(f'Training completed! Best loss: {best_loss:.4f}', flush=True)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, train_ids, test_ids, id_ec, test_id_ec, ec_id,
                   data_dir, train_split, test_split, device='cpu'):
    """Evaluate model with KNN."""
    print('Evaluating...', flush=True)
    train_data = load_data(train_ids, data_dir, train_split)
    test_data = load_data(test_ids, data_dir, test_split)

    model.eval()
    with torch.no_grad():
        train_emb_list = []
        for i in range(0, len(train_ids), 256):
            batch = [d[i:i+256].to(device) for d in train_data]
            emb = model.get_fused_embedding(*batch)
            train_emb_list.append(emb.cpu())
        train_emb = torch.cat(train_emb_list)

        test_emb_list = []
        for i in range(0, len(test_ids), 256):
            batch = [d[i:i+256].to(device) for d in test_data]
            emb = model.get_fused_embedding(*batch)
            test_emb_list.append(emb.cpu())
        test_emb = torch.cat(test_emb_list)

    def cosine_sim(a, b): return np.dot(normalize(a), normalize(b).T)
    sim = cosine_sim(test_emb.numpy(), train_emb.numpy())
    train_labels = [id_ec[pid] for pid in train_ids]

    predictions = {}
    for i in range(len(test_ids)):
        idx = np.argsort(sim[i])[-10:]
        ec_votes = Counter()
        for j in idx:
            for ec in train_labels[j]: ec_votes[ec] += sim[i, j]
        final_ecs = [ec for ec, score in ec_votes.items() if score >= 0.3]
        if not final_ecs and ec_votes: final_ecs = [ec_votes.most_common(1)[0][0]]
        predictions[i] = final_ecs

    all_train_ecs = sorted(list(ec_id.keys()))
    results = {}

    for level in range(1, 5):
        true_level = [[get_ec_level(e, level) for e in test_id_ec.get(pid, [])] for pid in test_ids]
        pred_level = [[get_ec_level(e, level) for e in predictions.get(i, [])] for i in range(len(test_ids))]
        all_level = sorted(list(set(ec for ecs in true_level for ec in ecs) |
                                set(ec for ecs in pred_level for ec in ecs) |
                                set(get_ec_level(ec, level) for ec in all_train_ecs)))
        mlb = MultiLabelBinarizer(classes=all_level)
        yt = mlb.fit_transform(true_level)
        yp = mlb.transform(pred_level)
        p = precision_score(yt, yp, average='weighted', zero_division=0)
        r = recall_score(yt, yp, average='weighted', zero_division=0)
        f1 = f1_score(yt, yp, average='weighted', zero_division=0)
        try: auc = roc_auc_score(yt, yp, average='weighted')
        except: auc = 0.0
        results[level] = {'precision': p, 'recall': r, 'f1': f1, 'auc': auc}

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MVCLEEA v2 ESM-2 + PSSM + OneHot')
    parser.add_argument('--train_data', type=str, default='split10')
    parser.add_argument('--test_data', type=str, default='new')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--project_root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    project_root = args.project_root.resolve()
    split_dir = project_root / 'data' / 'splits'
    feature_dir = project_root / 'data' / 'features'
    id_ec, ec_id = load_ec_mappings(split_dir / f'{args.train_data}.csv')
    test_id_ec, _ = load_ec_mappings(split_dir / f'{args.test_data}.csv')
    train_ids = list(id_ec.keys())
    test_ids = list(test_id_ec.keys())

    global esm2_dir, data_dir
    esm2_dir = str(feature_dir / 'esm2')
    data_dir = str(feature_dir)

    model_path = project_root / 'models' / f'mvcleea_v2_{args.train_data}.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = MVCLEEA_ESM2_PSSM_OneHotV2().to(device)

    if args.retrain or not os.path.exists(model_path):
        model = train(
            train_ids, id_ec, ec_id, data_dir, args.train_data,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            alpha=args.alpha,
            device=device, save_path=model_path
        )
    else:
        print(f'Loading model from {model_path}', flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Evaluate
    results = evaluate_model(
        model, train_ids, test_ids, id_ec, test_id_ec, ec_id,
        data_dir, args.train_data, args.test_data, device=device
    )

    print(f'\n{"="*60}', flush=True)
    print(f'ESM-2 + PSSM + OneHot v2 Results', flush=True)
    print(f'{"="*60}', flush=True)
    header = f'{"Level":>6s} | {"Precision":>10s} | {"Recall":>10s} | {"F1":>10s} | {"AUC":>10s}'
    print(header, flush=True)
    print('-'*60, flush=True)
    for level in range(1, 5):
        r = results[level]
        line = f'{level:>6d} | {r["precision"]:10.4f} | {r["recall"]:10.4f} | {r["f1"]:10.4f} | {r["auc"]:10.4f}'
        print(line, flush=True)
    print(f'{"="*60}', flush=True)


if __name__ == '__main__':
    main()
