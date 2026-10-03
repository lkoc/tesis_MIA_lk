# phase5 PINN study, study p5_v2

Acceptance criteria: proposed, not confirmed by the user.

| level | case | arch | seed | status | NRMSE % | dT error % | Tmax err K | worst region % | balance % | [T] max K | [q.n] rms % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V2 | coaxial_full | CUSP | 0 | completed | 13.3 | 77.8 | 9.31 | 77.7 | 48.1 | 3.55e-15 | 23.7 | rejected |
| V2 | coaxial_full | DCSNN | 0 | completed | 13.6 | 100 | 12 | 100 | 66.5 | 0.0149 | 33.3 | rejected |
| V2 | coaxial_full | FBPINN | 0 | completed | 14.6 | 102 | 12.3 | 102 | 88.5 | 0 | 28.6 | rejected |
| V2 | coaxial_full | M1+rba | 0 | completed | 0.0456 | 0.0175 | 0.0021 | 0.0555 | 0.0702 | 0.0134 | 0.135 | accepted |
