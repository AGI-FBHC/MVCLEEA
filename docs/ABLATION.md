# Ablation study

## Protocol

All ablations use the same split, test set, optimizer and KNN evaluation.
The controlled split10 suite uses:

- random seed: 42;
- epochs: 100;
- batch size: 128;
- learning rate: 0.0005 with cosine decay;
- test set: `new`;
- GPU: CUDA 2;
- one changed factor per experiment.

The fixed-seed Full run is the only valid reference for these ablations.
The earlier split10 checkpoint was trained without a recorded seed and reached
L4 F1 0.3363, whereas the fixed-seed Full run reached 0.3043. This difference
shows that split10 has meaningful run-to-run variance.

## split10 results

| Ablation | L1 F1 | L2 F1 | L3 F1 | L4 F1 | Delta L4 |
|---|---:|---:|---:|---:|---:|
| Full | 0.8498 | 0.8307 | 0.6815 | 0.3043 | 0.0000 |
| w/o padding mask | 0.8177 | 0.7817 | 0.6863 | 0.2993 | -0.0050 |
| single-label hierarchy | 0.8363 | 0.7968 | 0.7035 | 0.3266 | +0.0222 |
| L4 supervision only | 0.5843 | 0.5374 | 0.5990 | 0.2979 | -0.0065 |
| w/o dynamic fusion | 0.8114 | 0.8028 | 0.6616 | 0.2944 | -0.0099 |
| w/o cross-attention | 0.8512 | 0.8168 | 0.7030 | 0.3107 | +0.0064 |
| w/o teacher alignment | 0.7854 | 0.7703 | 0.6619 | 0.3217 | +0.0173 |
| w/o PSSM view | 0.8820 | 0.8347 | 0.7631 | 0.3135 | +0.0092 |
| w/o OneHot view | 0.8086 | 0.7896 | 0.6723 | 0.2890 | -0.0153 |

## Interpretation

### Supported contributions

The padding mask improves L1, L2 and L4. Its largest effects are at L1 and
L2, consistent with removing train-test padding artifacts that otherwise
distort broad class structure.

Dynamic fusion improves every level. Uniform averaging reduces L4 by 0.0099
and L1 by 0.0384, supporting per-protein view weighting over fixed equal
weights.

OneHot is useful even though its learned global fusion weight was small in the
original run. Removing it causes the largest L4 decrease among single-view
ablations (-0.0153). A small average gate weight therefore does not imply zero
causal contribution.

Multi-level supervision is essential for coarse hierarchy. Training with only
L4 labels collapses L1 and L2 by 0.2655 and 0.2933, respectively. Its direct
L4 benefit on split10 is smaller, indicating that its strongest role is to
organize the embedding space hierarchically.

### Contributions not supported on split10

Removing cross-attention slightly improves L3 and L4. The global-token module
may be redundant with dynamic fusion at this data scale.

Removing teacher alignment improves L4 but harms L1-L3. Alignment therefore
appears to trade coarse hierarchical organization against fine-grained
discrimination rather than providing a uniform gain.

Removing PSSM improves all four F1 values in this fixed-seed split10 run.
PSSM may be noisy or redundant in the small-data regime, despite receiving a
large learned gate weight in the earlier checkpoint. Gate weights should not
be interpreted as causal importance.

Single-label supervision improves L4 but harms L1 and L2. Multi-label overlap
may produce overly broad positive sets in small batches, while single-label
training creates sharper fine-grained clusters. Larger split50 and split100
experiments are required before changing the final method.

## Limitations

This suite uses one controlled seed because the full split50 and split100
matrix is computationally expensive. Small positive or negative differences
on split10 may be stochastic. Conclusions should be considered stable only
when their direction persists on larger splits or across additional seeds.

Detailed logs are in `logs/ablations/`; machine-readable per-run metrics and
summary files are in `results/`.
