# Phase 6 FEM reference, study p6_fem3

Gates (proposed, not confirmed by the user): mesh change <= 0.5% of rise, domain change <= 2.0% of rise.

| ladder | case | Tmax per level C | last change % of rise | mesh converged | Tmax wide domain C | domain change % of rise | domain insensitive |
|---|---|---|---|---|---|---|---|
| V3 | xlpe_single | 36.4819, 36.8187, 36.8525, 36.8619 | 0.056 | True | 36.9048 | 0.31 | True |
| V4 | xlpe_backfill | 33.3384, 33.5088, 33.5250, 33.5297 | 0.035 | True | 33.5917 | 0.49 | True |
| V4 | xlpe_discrete_layers | 36.9014, 37.3364, 37.3713, 37.3808 | 0.055 | True | 37.3735 | 0.01 | True |
| V4 | aras_flat | 50.0326, 50.0541, 50.0576, 50.0578 | 0.001 | True | 51.1035 | 3.48 | False |
| V5 | kim_v5_summer | 49.2033, 49.2060, 49.2062 | 0.001 | True | 53.2443 | 11.87 | False |
