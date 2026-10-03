# Fase 7 - Resumen agregado de los estudios PINN

Generado por `pinn_v2.experiments.aggregate` a partir de los `summary.json` existentes.
Los criterios de aceptacion son propuestos, no confirmados por el usuario.

| estudio | caso | arq. | sem. | NRMSE % | dT % | Tmax K | balance % | veredicto | gates fallidos |
|---|---|---|---|---|---|---|---|---|---|
| p4_v0 | mms_constant | M0 | 0 | 0.00322 | 0.002509 | 0.0007524 | 0.004561 | accepted |  |
| p4_v0 | mms_constant | M0 | 1 | 0.005139 | 0.0009784 | 0.0002934 | 0.0072 | accepted |  |
| p4_v0 | mms_constant | M0 | 2 | 0.002756 | 0.0007724 | 0.0002317 | 0.004921 | accepted |  |
| p4_v0 | mms_constant | M1 | 0 | 0.00322 | 0.002509 | 0.0007524 | 0.004561 | accepted |  |
| p4_v0 | mms_constant | M1 | 1 | 0.005139 | 0.0009784 | 0.0002934 | 0.0072 | accepted |  |
| p4_v0 | mms_constant | M1 | 2 | 0.002756 | 0.0007724 | 0.0002317 | 0.004921 | accepted |  |
| p4_v0 | mms_constant | M2 | 0 | 0.04188 | 0.04357 | 0.01307 | 0.006836 | accepted |  |
| p4_v0 | mms_constant | M2 | 1 | 0.03988 | 0.03162 | 0.009482 | 0.01036 | accepted |  |
| p4_v0 | mms_constant | M2 | 2 | 0.06249 | 0.1414 | 0.04242 | 0.01488 | accepted |  |
| p4_v0 | mms_robin | M0 | 0 | 0.007972 | 0.006478 | 0.0009697 | 0.008115 | accepted |  |
| p4_v0 | mms_robin | M0 | 1 | 0.004094 | 0.003962 | 0.000593 | 0.008359 | accepted |  |
| p4_v0 | mms_robin | M0 | 2 | 0.0152 | 0.001025 | 0.0001535 | 0.009122 | accepted |  |
| p4_v0 | mms_robin | M1 | 0 | 0.007972 | 0.006478 | 0.0009697 | 0.008115 | accepted |  |
| p4_v0 | mms_robin | M1 | 1 | 0.004094 | 0.003962 | 0.000593 | 0.008359 | accepted |  |
| p4_v0 | mms_robin | M1 | 2 | 0.0152 | 0.001025 | 0.0001535 | 0.009122 | accepted |  |
| p4_v0 | mms_robin | M2 | 0 | 0.05058 | 0.008486 | 0.00127 | 0.03922 | accepted |  |
| p4_v0 | mms_robin | M2 | 1 | 0.06232 | 0.02603 | 0.003897 | 0.02308 | accepted |  |
| p4_v0 | mms_robin | M2 | 2 | 0.05441 | 0.03485 | 0.005216 | 0.04248 | accepted |  |
| p4_v1 | mms_interface | M0 | 0 | 6.224 | 3.262 | 0.4072 | 3.126 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v1 | mms_interface | M0 | 1 | 6.309 | 3.552 | 0.4434 | 2.941 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v1 | mms_interface | M0 | 2 | 6.449 | 3.344 | 0.4175 | 3.22 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v1 | mms_interface | M1 | 0 | 0.01161 | 0.01031 | 0.001287 | 0.00604 | accepted |  |
| p4_v1 | mms_interface | M1 | 1 | 0.006564 | 0.02111 | 0.002636 | 0.001727 | accepted |  |
| p4_v1 | mms_interface | M1 | 2 | 0.01076 | 0.03723 | 0.004648 | 0.006529 | accepted |  |
| p4_v1 | mms_interface | M2 | 0 | 0.09473 | 0.0009375 | 0.000117 | 0.0222 | accepted |  |
| p4_v1 | mms_interface | M2 | 1 | 0.06535 | 0.04492 | 0.005608 | 0.02954 | accepted |  |
| p4_v1 | mms_interface | M2 | 2 | 0.1257 | 0.01376 | 0.001718 | 0.05 | accepted |  |
| p4_v2 | coaxial_full | M0 | 0 | 13.64 | 100 | 11.98 | 100 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v2 | coaxial_full | M0 | 1 | 13.53 | 99.85 | 11.96 | 96.68 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v2 | coaxial_full | M1 | 0 | 0.01333 | 0.02934 | 0.003515 | 0.01457 | accepted |  |
| p4_v2 | coaxial_full | M1 | 1 | 0.01258 | 0.01313 | 0.001573 | 0.006257 | accepted |  |
| p4_v2 | coaxial_full | M2 | 0 | 0.04905 | 0.05068 | 0.006071 | 0.1112 | accepted |  |
| p4_v2 | coaxial_full | M2 | 1 | 0.05307 | 0.02203 | 0.002639 | 0.06145 | accepted |  |
| p4_v2 | coaxial_robin | M0 | 0 | 18.4 | 99.63 | 12.93 | 95.45 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v2 | coaxial_robin | M0 | 1 | 23.42 | 109.4 | 14.19 | 105 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v2 | coaxial_robin | M1 | 0 | 0.008145 | 0.02017 | 0.002617 | 0.013 | accepted |  |
| p4_v2 | coaxial_robin | M1 | 1 | 0.01675 | 0.00459 | 0.0005955 | 0.003071 | accepted |  |
| p4_v2 | coaxial_robin | M2 | 0 | 0.05409 | 0.0345 | 0.004476 | 0.2306 | accepted |  |
| p4_v2 | coaxial_robin | M2 | 1 | 0.07048 | 0.04015 | 0.00521 | 0.009791 | accepted |  |
| p4_v2 | coaxial_bc_dirichlet | M0 | 0 | 13.64 | 100 | 11.98 | 100 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v2 | coaxial_bc_dirichlet | M0 | 1 | 13.53 | 99.85 | 11.96 | 96.68 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p4_v2 | coaxial_bc_dirichlet | M1 | 0 | 0.01333 | 0.02934 | 0.003515 | 0.01457 | accepted |  |
| p4_v2 | coaxial_bc_dirichlet | M1 | 1 | 0.01258 | 0.01313 | 0.001573 | 0.006257 | accepted |  |
| p4_v2 | coaxial_bc_dirichlet | M2 | 0 | 0.04905 | 0.05068 | 0.006071 | 0.1112 | accepted |  |
| p4_v2 | coaxial_bc_dirichlet | M2 | 1 | 0.05307 | 0.02203 | 0.002639 | 0.06145 | accepted |  |
| p5_v1 | mms_interface | CUSP | 0 | 0.006617 | 0.003864 | 0.0004823 | 0.003027 | accepted |  |
| p5_v1 | mms_interface | CUSP | 1 | 0.007166 | 0.007115 | 0.0008882 | 0.00288 | accepted |  |
| p5_v1 | mms_interface | CUSP | 2 | 0.004807 | 0.005235 | 0.0006535 | 0.0008509 | accepted |  |
| p5_v1 | mms_interface | DCSNN | 0 | 0.03847 | 0.1165 | 0.01454 | 0.0007495 | accepted |  |
| p5_v1 | mms_interface | DCSNN | 1 | 0.02034 | 0.03199 | 0.003994 | 0.01389 | accepted |  |
| p5_v1 | mms_interface | DCSNN | 2 | 0.0198 | 0.01338 | 0.00167 | 0.008087 | accepted |  |
| p5_v1 | mms_interface | FBPINN | 0 | 6.019 | 2.965 | 0.3701 | 2.816 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | FBPINN | 1 | 6.331 | 3.219 | 0.4018 | 3.005 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | FBPINN | 2 | 6.507 | 3.007 | 0.3753 | 3.41 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M0+rba | 0 | 6.409 | 3.661 | 0.4571 | 1.074 | rejected | interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M0+rba | 1 | 6.318 | 3.497 | 0.4365 | 1.204 | rejected | interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M0+rba | 2 | 6.324 | 3.978 | 0.4967 | 1.063 | rejected | interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M0+sa | 0 | 6.479 | 3.785 | 0.4725 | 1.978 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M0+sa | 1 | 6.503 | 3.851 | 0.4807 | 2.228 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M0+sa | 2 | 6.519 | 3.864 | 0.4824 | 2.11 | rejected | balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v1 | mms_interface | M1+rba | 0 | 0.1701 | 0.5805 | 0.07246 | 0.02152 | accepted |  |
| p5_v1 | mms_interface | M1+rba | 1 | 0.1291 | 0.4604 | 0.05748 | 0.03519 | accepted |  |
| p5_v1 | mms_interface | M1+rba | 2 | 0.1338 | 0.3566 | 0.04452 | 0.01432 | accepted |  |
| p5_v1 | mms_interface | M1+sa | 0 | 0.1549 | 0.5833 | 0.07282 | 0.06993 | accepted |  |
| p5_v1 | mms_interface | M1+sa | 1 | 0.1198 | 0.465 | 0.05805 | 0.07374 | accepted |  |
| p5_v1 | mms_interface | M1+sa | 2 | 0.1491 | 0.3998 | 0.04991 | 0.00407 | accepted |  |
| p5_v2 | coaxial_full | CUSP | 0 | 13.28 | 77.75 | 9.315 | 48.06 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v2 | coaxial_full | DCSNN | 0 | 13.6 | 99.97 | 11.98 | 66.55 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v2 | coaxial_full | FBPINN | 0 | 14.57 | 102.4 | 12.27 | 88.55 | rejected | Tmax_rise_pct,balance_pct,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p5_v2 | coaxial_full | M1+rba | 0 | 0.04564 | 0.01752 | 0.002098 | 0.07024 | accepted |  |
| p6_pinn_kim_long | kim_v5_summer | M1 | 0 | 13.36 | 54.17 | 18.42 | 167.2 | rejected | Tmax_rise_pct,balance_pct,interface_T_jump_max_K,nrmse_pct,region_rmse_rise_pct |
| p6_pinn_m1 | xlpe_single | M1 | 0 | 0.5575 | 0.8851 | 0.1492 | 0.2032 | accepted |  |
| p6_pinn_m1 | xlpe_backfill | M1 | 0 | 2.139 | 7.226 | 0.9776 | 7.314 | rejected | Tmax_rise_pct,balance_pct,interface_T_jump_max_K,region_rmse_rise_pct |
| p6_pinn_m1 | xlpe_discrete_layers | M1 | 0 | 0.5299 | 2.418 | 0.4202 | 1.806 | rejected | interface_T_jump_max_K,interface_flux_rms_pct |
| p6_pinn_m1 | aras_flat | M1 | 0 | 0.9814 | 1.205 | 0.3623 | 0.5523 | rejected | interface_T_jump_max_K |
| p6_pinn_m1 | kim_v5_summer | M1 | 0 | 18.68 | 65.6 | 22.31 | 163.6 | rejected | Tmax_rise_pct,balance_pct,interface_T_jump_max_K,interface_flux_rms_pct,nrmse_pct,region_rmse_rise_pct |
| p6_pinn_m1_long | xlpe_backfill | M1 | 0 | 2.427 | 1.793 | 0.2426 | 4.556 | rejected | balance_pct,interface_T_jump_max_K |
| p6_pinn_m1_long | xlpe_discrete_layers | M1 | 0 | 0.407 | 0.7729 | 0.1343 | 0.2486 | rejected | interface_T_jump_max_K |
| p6_pinn_m1_long | aras_flat | M1 | 0 | 0.5022 | 1.975 | 0.5937 | 0.09634 | accepted |  |

Corridas: 79; aceptadas: 51.
