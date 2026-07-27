# Method

MVCLEEA v2 combines three complementary protein representations.

1. ESM-2 embeddings are mean-pooled over real residues only. BOS, EOS and
   batch padding are excluded.
2. PSSM and OneHot arrays are encoded along the sequence axis with residual
   one-dimensional convolutions.
3. A OneHot-derived mask removes padded positions from PSSM, convolution,
   attention pooling and max pooling.
4. A global-token cross-attention module aggregates the three views.
5. A per-protein gate weights ESM-2, PSSM, OneHot and cross-attention
   features, followed by an ESM residual connection.
6. Multi-label hierarchical supervised contrastive learning treats proteins
   sharing any EC prefix as positives. L1-L4 weights are 0.1, 0.2, 0.3 and
   0.4.
7. PSSM and OneHot views are weakly aligned to a detached ESM-2 teacher.
8. The final 128-dimensional embeddings are classified by cosine KNN.

The training objective is:

```text
loss = 0.9 * hierarchical_multi_label_contrastive_loss
     + 0.1 * teacher_alignment_loss
```
