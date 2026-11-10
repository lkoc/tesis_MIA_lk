# Inferencia preliminar de campaña adaptativa

Fuente: `results/adaptive_all_cases_20261026c/summary.json`.
Estas métricas provienen de los últimos bloques L-BFGS guardados; no son todavía la campaña completa.

## Annulus

| Modelo | Semillas completas | NRMSE medio | NRMSE std | Error Tmax medio (K) | Balance medio (%) |
|---|---:|---:|---:|---:|---:|
| CUSPR | 3 | 11.352 % | 0.172 % | 17.745 | 0.623 |
| M0R | 2 | 111.147 % | 95.247 % | 18.080 | 42.287 |
| M2 | 3 | 17.201 % | 0.309 % | 20.037 | 107.499 |

## Estado

- `M1`: las entradas anulares iniciales fallaron antes de ejecutar por la interfaz rectangular del adaptador; la corrección `inner/outer` ya está aplicada y debe reintentarse.
- `M0R`: falta una semilla y la variabilidad observada es muy alta.
- `CUSPR`: estable entre las tres semillas, pero aún lejos del criterio estricto.
- `M2`: temperatura relativamente consistente, pero balance energético inaceptable.
- El resto de los casos todavía no tiene inferencia estadística completa en esta campaña.
