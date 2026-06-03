import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic_view import SemanticViewNetwork
from .evolutionary_view import EvolutionaryViewNetwork
from .syntactic_view import SyntacticViewNetwork
from .cross_attention import CrossAttentionPooling
from .adaptive_fusion import AdaptiveFeatureFusion
from .classifier import HierarchicalClassifier


class MVCLEEA(nn.Module):
    """
    MVCLEEA: Multi-View Contrastive Learning for Enzyme Function Annotation

    A multi-view neural network for hierarchical enzyme function prediction
    (EC number classification).

    Architecture Overview:
        1. Three independent feature extraction branches:
           - Semantic View (Z_E): MLP on ESM2 embeddings
           - Evolutionary View (Z_P): CNN on PSSM matrices
           - Syntactic View (Z_O): CNN on One-hot encoded sequences

        2. Cross-Attention Pooling → Shared View (Z_S)

        3. Adaptive Feature Fusion → Z_F

        4. Hierarchical Classifier → EC number probabilities

    Inputs:
        esm2_features: Pre-extracted ESM2 embeddings, (batch, 1280)
        pssm:          PSSM matrix (Sigmoid-normalized), (batch, 1024, 20)
        onehot:        One-hot encoded sequence, (batch, 1024, 21)

    Outputs:
        logits:  Class probabilities, (batch, num_classes)
        Z_E:     Semantic features, (batch, 128)
        Z_P:     Evolutionary features, (batch, 128)
        Z_O:     Syntactic features, (batch, 128)
        Z_S:     Shared view features, (batch, 128)
        Z_F:     Fused features, (batch, 128)
    """

    def __init__(self, num_classes: int = 588, esm2_dim: int = 1280,
                 feature_dim: int = 128, num_heads: int = 4):
        super().__init__()

        # Multi-view feature extractors
        self.semantic_view = SemanticViewNetwork(
            input_dim=esm2_dim, output_dim=feature_dim
        )
        self.evolutionary_view = EvolutionaryViewNetwork(
            output_dim=feature_dim
        )
        self.syntactic_view = SyntacticViewNetwork(
            output_dim=feature_dim
        )

        # Cross-attention pooling for shared view
        self.cross_attention = CrossAttentionPooling(
            feature_dim=feature_dim, num_heads=num_heads
        )

        # Adaptive feature fusion
        self.fusion = AdaptiveFeatureFusion(num_views=4)

        # Hierarchical classifier
        self.classifier = HierarchicalClassifier(
            input_dim=feature_dim, num_classes=num_classes
        )

    def forward(self, esm2_features: torch.Tensor,
                pssm: torch.Tensor,
                onehot: torch.Tensor):
        """
        Forward pass of MVCLEEA.

        Args:
            esm2_features: ESM2 embeddings, (batch, 1280)
            pssm:          PSSM matrix,      (batch, 1024, 20)
            onehot:        One-hot matrix,    (batch, 1024, 21)

        Returns:
            dict with keys:
                'logits':  Class probabilities, (batch, num_classes)
                'Z_E':     Semantic features
                'Z_P':     Evolutionary features
                'Z_O':     Syntactic features
                'Z_S':     Shared view features
                'Z_F':     Fused features
        """
        # 1. Multi-view feature extraction
        Z_E = self.semantic_view(esm2_features)

        # Add channel dim for CNN: (batch, 1, 1024, 20)
        pssm_input = pssm.unsqueeze(1)
        Z_P = self.evolutionary_view(pssm_input)

        # Add channel dim for CNN: (batch, 1, 1024, 21)
        onehot_input = onehot.unsqueeze(1)
        Z_O = self.syntactic_view(onehot_input)

        # 2. Cross-attention pooling → Shared View
        Z_S = self.cross_attention(Z_E, Z_P, Z_O)

        # 3. Adaptive feature fusion
        Z_F = self.fusion(Z_E, Z_P, Z_O, Z_S)

        # 4. Hierarchical classification
        logits = self.classifier(Z_F)

        return {
            'logits': logits,
            'Z_E': Z_E,
            'Z_P': Z_P,
            'Z_O': Z_O,
            'Z_S': Z_S,
            'Z_F': Z_F,
        }
