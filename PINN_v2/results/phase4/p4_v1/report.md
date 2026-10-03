# Phase 4 PINN baselines, study p4_v1

Acceptance criteria: proposed, not confirmed by the user.

| level | case | arch | seed | status | NRMSE % | dT error % | Tmax err K | worst region % | balance % | [T] max K | [q.n] rms % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | mms_interface | M0 | 0 | completed | 6.22 | 3.26 | 0.407 | 7.16 | 3.13 | 0 | 17.1 | rejected |
| V1 | mms_interface | M0 | 1 | completed | 6.31 | 3.55 | 0.443 | 7.24 | 2.94 | 0 | 17.7 | rejected |
| V1 | mms_interface | M0 | 2 | completed | 6.45 | 3.34 | 0.417 | 7.41 | 3.22 | 0 | 17.5 | rejected |
| V1 | mms_interface | M1 | 0 | completed | 0.0116 | 0.0103 | 0.00129 | 0.0136 | 0.00604 | 0.00754 | 0.0249 | accepted |
| V1 | mms_interface | M1 | 1 | completed | 0.00656 | 0.0211 | 0.00264 | 0.00739 | 0.00173 | 0.0046 | 0.0224 | accepted |
| V1 | mms_interface | M1 | 2 | completed | 0.0108 | 0.0372 | 0.00465 | 0.011 | 0.00653 | 0.00581 | 0.0242 | accepted |
| V1 | mms_interface | M2 | 0 | completed | 0.0947 | 0.000937 | 0.000117 | 0.107 | 0.0222 | 0.018 | 0.0493 | accepted |
| V1 | mms_interface | M2 | 1 | completed | 0.0653 | 0.0449 | 0.00561 | 0.0702 | 0.0295 | 0.00891 | 0.0345 | accepted |
| V1 | mms_interface | M2 | 2 | completed | 0.126 | 0.0138 | 0.00172 | 0.141 | 0.05 | 0.0188 | 0.0481 | accepted |
