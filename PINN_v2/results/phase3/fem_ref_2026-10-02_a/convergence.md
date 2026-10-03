# Phase 3 FEM reference, study fem_ref_2026-10-02_a

## Manufactured cases (unit square, P2)

| case | level | dofs | L2 error K | max nodal K | balance % | L2 order |
|---|---|---|---|---|---|---|
| mms_constant | 0 | 1089 | 2.062e-03 | 4.322e-04 | 4.763e-01 |  |
| mms_constant | 1 | 4225 | 2.580e-04 | 2.707e-05 | 1.201e-01 | 3.00 |
| mms_constant | 2 | 16641 | 3.226e-05 | 1.693e-06 | 3.010e-02 | 3.00 |
| mms_constant | 3 | 66049 | 4.033e-06 | 1.055e-07 | 7.529e-03 | 3.00 |
| mms_interface | 0 | 1089 | 3.709e-04 | 2.700e-04 | 3.163e-01 |  |
| mms_interface | 1 | 4225 | 4.633e-05 | 3.296e-05 | 7.978e-02 | 3.00 |
| mms_interface | 2 | 16641 | 5.789e-06 | 4.161e-06 | 2.002e-02 | 3.00 |
| mms_interface | 3 | 66049 | 7.235e-07 | 5.322e-07 | 5.013e-03 | 3.00 |
| mms_robin | 0 | 1089 | 1.449e-04 | 2.654e-04 | 1.517e-03 |  |
| mms_robin | 1 | 4225 | 1.817e-05 | 3.375e-05 | 4.911e-04 | 3.00 |
| mms_robin | 2 | 16641 | 2.275e-06 | 4.255e-06 | 1.374e-04 | 3.00 |
| mms_robin | 3 | 66049 | 2.847e-07 | 5.340e-07 | 3.623e-05 | 3.00 |

## Concentric cable (exact radial solution)

| case | level | dofs | rmse K | Tmax error K | balance % |
|---|---|---|---|---|---|
| coaxial_full | 0 | 10229 | 3.408e-04 | 7.117e-04 | 4.679e-01 |
| coaxial_full | 1 | 38906 | 4.634e-05 | 5.598e-05 | 1.184e-01 |
| coaxial_full | 2 | 152951 | 6.217e-06 | 3.677e-06 | 2.704e-02 |
| coaxial_bc_dirichlet | 0 | 10229 | 3.408e-04 | 7.117e-04 | 4.679e-01 |
| coaxial_bc_dirichlet | 1 | 38906 | 4.634e-05 | 5.598e-05 | 1.184e-01 |
| coaxial_bc_dirichlet | 2 | 152951 | 6.217e-06 | 3.677e-06 | 2.704e-02 |
| coaxial_robin | 0 | 10229 | 3.480e-04 | 7.592e-04 | 4.769e-01 |
| coaxial_robin | 1 | 38906 | 4.687e-05 | 5.937e-05 | 1.208e-01 |
| coaxial_robin | 2 | 152951 | 6.223e-06 | 3.890e-06 | 2.763e-02 |

## Symmetry cross-check (half domain with q.n = 0 side against full domain)

| level | source ratio | top flux ratio | symmetry-side flux W/m | mirror dT mean K | half-full dT mean K |
|---|---|---|---|---|---|
| 0 | 2.000000 | 1.995253 | 1.18e-01 | 1.32e-01 | 1.93e-02 |
| 1 | 2.000000 | 1.998964 | -1.82e-02 | 2.72e-04 | 1.12e-02 |
| 2 | 2.000000 | 2.010089 | 4.96e-04 | 1.75e-03 | -9.42e-04 |
