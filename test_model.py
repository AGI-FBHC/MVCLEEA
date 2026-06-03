"""
MVCLEEA Forward Pass Verification Script

Verifies tensor dimension correctness across all modules using dummy tensors.
"""
import torch
import sys

sys.path.insert(0, '.')

from models import (
    SemanticViewNetwork,
    EvolutionaryViewNetwork,
    SyntacticViewNetwork,
    CrossAttentionPooling,
    AdaptiveFeatureFusion,
    HierarchicalClassifier,
    MVCLEEA,
)
from losses import JointLoss


def test_individual_modules():
    """Test each module independently."""
    batch_size = 4
    device = torch.device('cpu')

    print("=" * 60)
    print("Testing Individual Modules")
    print("=" * 60)

    # --- Semantic View ---
    print("\n[1/6] Semantic View Network (Z_E)")
    semantic_net = SemanticViewNetwork(input_dim=1280, output_dim=128)
    esm2_input = torch.randn(batch_size, 1280)
    Z_E = semantic_net(esm2_input)
    print(f"  Input:  {esm2_input.shape}")
    print(f"  Output: {Z_E.shape}")
    assert Z_E.shape == (batch_size, 128), f"Expected (4, 128), got {Z_E.shape}"
    print("  PASS ✓")

    # --- Evolutionary View ---
    print("\n[2/6] Evolutionary View Network (Z_P)")
    evol_net = EvolutionaryViewNetwork(output_dim=128)
    pssm_input = torch.sigmoid(torch.randn(batch_size, 1, 1024, 20))
    Z_P = evol_net(pssm_input)
    print(f"  Input:  {pssm_input.shape}")
    print(f"  Output: {Z_P.shape}")
    assert Z_P.shape == (batch_size, 128), f"Expected (4, 128), got {Z_P.shape}"
    print("  PASS ✓")

    # --- Syntactic View ---
    print("\n[3/6] Syntactic View Network (Z_O)")
    synt_net = SyntacticViewNetwork(output_dim=128)
    onehot_input = torch.zeros(batch_size, 1, 1024, 21)
    for b in range(batch_size):
        positions = torch.randint(0, 1024, (500,))
        amino_acids = torch.randint(0, 21, (500,))
        onehot_input[b, 0, positions, amino_acids] = 1.0
    Z_O = synt_net(onehot_input)
    print(f"  Input:  {onehot_input.shape}")
    print(f"  Output: {Z_O.shape}")
    assert Z_O.shape == (batch_size, 128), f"Expected (4, 128), got {Z_O.shape}"
    print("  PASS ✓")

    # --- Cross-Attention Pooling ---
    print("\n[4/6] Cross-Attention Pooling → Shared View (Z_S)")
    cross_attn = CrossAttentionPooling(feature_dim=128, num_heads=4)
    Z_S = cross_attn(Z_E, Z_P, Z_O)
    print(f"  Inputs: Z_E={Z_E.shape}, Z_P={Z_P.shape}, Z_O={Z_O.shape}")
    print(f"  Output: {Z_S.shape}")
    assert Z_S.shape == (batch_size, 128), f"Expected (4, 128), got {Z_S.shape}"
    print("  PASS ✓")

    # --- Adaptive Fusion ---
    print("\n[5/6] Adaptive Feature Fusion → Z_F")
    fusion = AdaptiveFeatureFusion(num_views=4)
    Z_F = fusion(Z_E, Z_P, Z_O, Z_S)
    print(f"  Inputs: Z_E={Z_E.shape}, Z_P={Z_P.shape}, Z_O={Z_O.shape}, Z_S={Z_S.shape}")
    print(f"  Output: {Z_F.shape}")
    print(f"  Fusion weights (softmax): {torch.softmax(fusion.alpha, dim=0).data}")
    assert Z_F.shape == (batch_size, 128), f"Expected (4, 128), got {Z_F.shape}"
    print("  PASS ✓")

    # --- Hierarchical Classifier ---
    print("\n[6/6] Hierarchical Classifier → Logits")
    num_classes = 588
    classifier = HierarchicalClassifier(input_dim=128, num_classes=num_classes)
    logits = classifier(Z_F)
    print(f"  Input:  {Z_F.shape}")
    print(f"  Output: {logits.shape}")
    assert logits.shape == (batch_size, num_classes)
    print("  PASS ✓")


def test_full_model():
    """Test the complete MVCLEEA model end-to-end."""
    batch_size = 4
    num_classes = 588

    print("\n" + "=" * 60)
    print("Testing Full MVCLEEA Model (End-to-End)")
    print("=" * 60)

    model = MVCLEEA(num_classes=num_classes)
    model.eval()

    # Dummy inputs (mimicking real data shapes)
    esm2_features = torch.randn(batch_size, 1280)
    pssm = torch.sigmoid(torch.randn(batch_size, 1024, 20))
    onehot = torch.zeros(batch_size, 1024, 21)
    for b in range(batch_size):
        positions = torch.randint(0, 1024, (500,))
        amino_acids = torch.randint(0, 21, (500,))
        onehot[b, positions, amino_acids] = 1.0

    print(f"\n  ESM2 input:   {esm2_features.shape}")
    print(f"  PSSM input:   {pssm.shape}")
    print(f"  One-hot input: {onehot.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(esm2_features, pssm, onehot)

    print(f"\n  logits:  {outputs['logits'].shape}")
    print(f"  Z_E:     {outputs['Z_E'].shape}")
    print(f"  Z_P:     {outputs['Z_P'].shape}")
    print(f"  Z_O:     {outputs['Z_O'].shape}")
    print(f"  Z_S:     {outputs['Z_S'].shape}")
    print(f"  Z_F:     {outputs['Z_F'].shape}")

    # Verify all shapes
    assert outputs['logits'].shape == (batch_size, num_classes)
    assert outputs['Z_E'].shape == (batch_size, 128)
    assert outputs['Z_P'].shape == (batch_size, 128)
    assert outputs['Z_O'].shape == (batch_size, 128)
    assert outputs['Z_S'].shape == (batch_size, 128)
    assert outputs['Z_F'].shape == (batch_size, 128)

    print("\n  All output dimensions verified. PASS ✓")


def test_joint_loss():
    """Test the joint loss function."""
    batch_size = 4
    num_classes = 588

    print("\n" + "=" * 60)
    print("Testing Joint Loss Function")
    print("=" * 60)

    criterion = JointLoss(num_classes=num_classes, lambda_val=0.15)

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    Z_E = torch.randn(batch_size, 128, requires_grad=True)
    Z_P = torch.randn(batch_size, 128, requires_grad=True)
    Z_O = torch.randn(batch_size, 128, requires_grad=True)
    Z_S = torch.randn(batch_size, 128, requires_grad=True)

    total_loss, loss_bce, L_ES, L_PS, L_OS = criterion(
        logits, targets, Z_E, Z_P, Z_O, Z_S
    )

    print(f"  Total Loss:   {total_loss.item():.4f}")
    print(f"  BCE Loss:     {loss_bce.item():.4f}")
    print(f"  L_ES:         {L_ES.item():.4f}")
    print(f"  L_PS:         {L_PS.item():.4f}")
    print(f"  L_OS:         {L_OS.item():.4f}")

    # Verify loss formula: L = (1-λ)*L_BCE + (λ/3)*(L_ES + L_PS + L_OS)
    lambda_val = 0.15
    expected = (1 - lambda_val) * loss_bce + (lambda_val / 3) * (L_ES + L_PS + L_OS)
    assert abs(total_loss.item() - expected.item()) < 1e-5, \
        f"Loss formula mismatch: {total_loss.item()} vs {expected.item()}"

    # Verify gradient flow
    total_loss.backward()
    print("\n  Gradient flow verified. PASS ✓")


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("MVCLEEA Forward Pass Verification")
    print("=" * 60)

    test_individual_modules()
    test_full_model()
    test_joint_loss()

    # Model parameter count
    model = MVCLEEA(num_classes=588)
    total, trainable = count_parameters(model)
    print("\n" + "=" * 60)
    print(f"Model Parameters: {total:,} total, {trainable:,} trainable")
    print("=" * 60)

    print("\nAll tests passed.")
