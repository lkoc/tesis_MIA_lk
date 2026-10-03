# phase5 PINN study, study p5_v1

Acceptance criteria: proposed, not confirmed by the user.

| level | case | arch | seed | status | NRMSE % | dT error % | Tmax err K | worst region % | balance % | [T] max K | [q.n] rms % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | mms_interface | CUSP | 0 | completed | 0.00662 | 0.00386 | 0.000482 | 0.00826 | 0.00303 | 0 | 0.0248 | accepted |
| V1 | mms_interface | CUSP | 1 | completed | 0.00717 | 0.00712 | 0.000888 | 0.00914 | 0.00288 | 0 | 0.04 | accepted |
| V1 | mms_interface | CUSP | 2 | completed | 0.00481 | 0.00523 | 0.000653 | 0.00564 | 0.000851 | 0 | 0.011 | accepted |
| V1 | mms_interface | DCSNN | 0 | completed | 0.0385 | 0.116 | 0.0145 | 0.041 | 0.000749 | 0.0258 | 0.0849 | accepted |
| V1 | mms_interface | DCSNN | 1 | completed | 0.0203 | 0.032 | 0.00399 | 0.0264 | 0.0139 | 0.0141 | 0.0437 | accepted |
| V1 | mms_interface | DCSNN | 2 | completed | 0.0198 | 0.0134 | 0.00167 | 0.0266 | 0.00809 | 0.0133 | 0.0564 | accepted |
| V1 | mms_interface | FBPINN | 0 | completed | 6.02 | 2.97 | 0.37 | 6.97 | 2.82 | 0 | 16.8 | rejected |
| V1 | mms_interface | FBPINN | 1 | completed | 6.33 | 3.22 | 0.402 | 7.28 | 3.01 | 0 | 17.6 | rejected |
| V1 | mms_interface | FBPINN | 2 | completed | 6.51 | 3.01 | 0.375 | 7.46 | 3.41 | 0 | 17.5 | rejected |
| V1 | mms_interface | M0+rba | 0 | completed | 6.41 | 3.66 | 0.457 | 7.11 | 1.07 | 0 | 20 | rejected |
| V1 | mms_interface | M0+rba | 1 | completed | 6.32 | 3.5 | 0.437 | 7.04 | 1.2 | 0 | 19.8 | rejected |
| V1 | mms_interface | M0+rba | 2 | completed | 6.32 | 3.98 | 0.497 | 7.08 | 1.06 | 0 | 19.7 | rejected |
| V1 | mms_interface | M0+sa | 0 | completed | 6.48 | 3.78 | 0.472 | 7.32 | 1.98 | 0 | 19.2 | rejected |
| V1 | mms_interface | M0+sa | 1 | completed | 6.5 | 3.85 | 0.481 | 7.35 | 2.23 | 0 | 19.3 | rejected |
| V1 | mms_interface | M0+sa | 2 | completed | 6.52 | 3.86 | 0.482 | 7.36 | 2.11 | 0 | 19.3 | rejected |
| V1 | mms_interface | M1+rba | 0 | completed | 0.17 | 0.58 | 0.0725 | 0.214 | 0.0215 | 0.0229 | 0.75 | accepted |
| V1 | mms_interface | M1+rba | 1 | completed | 0.129 | 0.46 | 0.0575 | 0.17 | 0.0352 | 0.0384 | 0.527 | accepted |
| V1 | mms_interface | M1+rba | 2 | completed | 0.134 | 0.357 | 0.0445 | 0.147 | 0.0143 | 0.0527 | 0.435 | accepted |
| V1 | mms_interface | M1+sa | 0 | completed | 0.155 | 0.583 | 0.0728 | 0.202 | 0.0699 | 0.0273 | 0.74 | accepted |
| V1 | mms_interface | M1+sa | 1 | completed | 0.12 | 0.465 | 0.058 | 0.156 | 0.0737 | 0.0302 | 0.522 | accepted |
| V1 | mms_interface | M1+sa | 2 | completed | 0.149 | 0.4 | 0.0499 | 0.169 | 0.00407 | 0.0678 | 0.398 | accepted |
