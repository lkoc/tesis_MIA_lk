# phase6 PINN study, study p6_pinn_m1

Acceptance criteria: proposed, not confirmed by the user.

| level | case | arch | seed | status | NRMSE % | dT error % | Tmax err K | worst region % | balance % | [T] max K | [q.n] rms % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V3 | xlpe_single | M1 | 0 | completed | 0.558 | 0.885 | 0.149 | 0.885 | 0.203 | 0.0265 | 0.0981 | accepted |
| V4 | xlpe_backfill | M1 | 0 | completed | 2.14 | 7.23 | 0.978 | 7.23 | 7.31 | 0.107 | 0.559 | rejected |
| V4 | xlpe_discrete_layers | M1 | 0 | completed | 0.53 | 2.42 | 0.42 | 2.42 | 1.81 | 0.612 | 1.57 | rejected |
| V4 | aras_flat | M1 | 0 | completed | 0.981 | 1.21 | 0.362 | 4.83 | 0.552 | 0.106 | 0.187 | rejected |
| V5 | kim_v5_summer | M1 | 0 | completed | 18.7 | 65.6 | 22.3 | 80.6 | 164 | 0.666 | 1.34 | rejected |
