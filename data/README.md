# Paper datasets

This directory contains only the independent test datasets used in the
MVCLEEA paper. Training splits are intentionally not included.

| File | Dataset | Proteins | Size |
|---|---|---:|---:|
| `datasets/new.csv` | New-392 | 392 | 186,949 bytes |
| `datasets/price.csv` | Price-149 | 149 | 61,952 bytes |
| `datasets/new-815.csv` | New-815 | 815 | 354,722 bytes |

Each file is TAB-separated and uses the following columns:

```text
Entry    EC number    Sequence
```

## Provenance

- New-392 and Price-149 follow the data distributed with
  [CLEAN](https://github.com/tttianhao/CLEAN).
- New-815 follows the
  [MAPred](https://github.com/Rongdingyi/MAPred) evaluation data.

These datasets contain protein identifiers, EC annotations, and amino-acid
sequences derived from public resources. Users are responsible for complying
with the terms of the original data providers and repositories.

## SHA-256

```text
33e1ff5356c206b99449b3b2a3903bff41b1f30317be057f66f821d32534e259  new.csv
670924baf57826eafb43a3264ba2bc01a74caa72ced73279f1cc64f13ef2adc8  price.csv
b472ccb166d494e842b76b2272dd4e464073e9e782da384c183a86b9a7f7d1f5  new-815.csv
```
