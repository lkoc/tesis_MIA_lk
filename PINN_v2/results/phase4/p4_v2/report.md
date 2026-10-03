# Phase 4 PINN baselines, study p4_v2

Acceptance criteria: proposed, not confirmed by the user.

| level | case | arch | seed | status | NRMSE % | dT error % | Tmax err K | worst region % | balance % | [T] max K | [q.n] rms % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V2 | coaxial_full | M0 | 0 | completed | 13.6 | 100 | 12 | 100 | 100 | 0 | 6.67 | rejected |
| V2 | coaxial_full | M0 | 1 | completed | 13.5 | 99.9 | 12 | 99.8 | 96.7 | 0 | 6.64 | rejected |
| V2 | coaxial_full | M1 | 0 | completed | 0.0133 | 0.0293 | 0.00352 | 0.0293 | 0.0146 | 0.00304 | 0.035 | accepted |
| V2 | coaxial_full | M1 | 1 | completed | 0.0126 | 0.0131 | 0.00157 | 0.0159 | 0.00626 | 0.00364 | 0.0246 | accepted |
| V2 | coaxial_full | M2 | 0 | completed | 0.049 | 0.0507 | 0.00607 | 0.056 | 0.111 | 0.00638 | 0.0402 | accepted |
| V2 | coaxial_full | M2 | 1 | completed | 0.0531 | 0.022 | 0.00264 | 0.0532 | 0.0615 | 0.00452 | 0.0164 | accepted |
| V2 | coaxial_robin | M0 | 0 | completed | 18.4 | 99.6 | 12.9 | 99.6 | 95.4 | 0 | 6.6 | rejected |
| V2 | coaxial_robin | M0 | 1 | completed | 23.4 | 109 | 14.2 | 109 | 105 | 0 | 11.4 | rejected |
| V2 | coaxial_robin | M1 | 0 | completed | 0.00814 | 0.0202 | 0.00262 | 0.0202 | 0.013 | 0.00307 | 0.0366 | accepted |
| V2 | coaxial_robin | M1 | 1 | completed | 0.0167 | 0.00459 | 0.000596 | 0.0168 | 0.00307 | 0.00388 | 0.0245 | accepted |
| V2 | coaxial_robin | M2 | 0 | completed | 0.0541 | 0.0345 | 0.00448 | 0.0542 | 0.231 | 0.00585 | 0.0509 | accepted |
| V2 | coaxial_robin | M2 | 1 | completed | 0.0705 | 0.0402 | 0.00521 | 0.0707 | 0.00979 | 0.00476 | 0.02 | accepted |
| V2 | coaxial_bc_dirichlet | M0 | 0 | completed | 13.6 | 100 | 12 | 100 | 100 | 0 | 6.67 | rejected |
| V2 | coaxial_bc_dirichlet | M0 | 1 | completed | 13.5 | 99.9 | 12 | 99.8 | 96.7 | 0 | 6.64 | rejected |
| V2 | coaxial_bc_dirichlet | M1 | 0 | completed | 0.0133 | 0.0293 | 0.00352 | 0.0293 | 0.0146 | 0.00304 | 0.035 | accepted |
| V2 | coaxial_bc_dirichlet | M1 | 1 | completed | 0.0126 | 0.0131 | 0.00157 | 0.0159 | 0.00626 | 0.00364 | 0.0246 | accepted |
| V2 | coaxial_bc_dirichlet | M2 | 0 | completed | 0.049 | 0.0507 | 0.00607 | 0.056 | 0.111 | 0.00638 | 0.0402 | accepted |
| V2 | coaxial_bc_dirichlet | M2 | 1 | completed | 0.0531 | 0.022 | 0.00264 | 0.0532 | 0.0615 | 0.00452 | 0.0164 | accepted |
