# Revisión de física explícita y efecto piel

La revisión metodológica iniciada el 14 de septiembre de 2026 precede a estos cambios. Se implementó la ecuación de calor en todas las capas para todas las variantes activas; la verificación posterior no cambia retroactivamente los resultados reducidos.

## Evidencia nueva

| Corrida | Variante | Tmax, °C | Error Tmax, K | Balance, % | Estado |
|---|---|---:|---:|---:|---|
| [Benchmarks/explicit_results/coaxial_full/dc_temperature/workflow/subdomain/seed11/pinn_seed11.json](../Benchmarks/explicit_results/coaxial_full/dc_temperature/workflow/subdomain/seed11/pinn_seed11.json) | subdomain | 32.575572 | 0.003990 | 0.010321 | Aceptada |
| [Benchmarks/explicit_results/coaxial_full/fixed/mixed/seed11/pinn_seed11.json](../Benchmarks/explicit_results/coaxial_full/fixed/mixed/seed11/pinn_seed11.json) | mixed | 31.972583 | 0.007189 | 0.002369 | Rechazada |
| [Benchmarks/explicit_results/coaxial_full/fixed/mixed_verified/seed11/pinn_seed11.json](../Benchmarks/explicit_results/coaxial_full/fixed/mixed_verified/seed11/pinn_seed11.json) | mixed | 31.911381 | 0.068392 | 0.058240 | Aceptada |
| [Benchmarks/explicit_results/coaxial_full/fixed/subdomain/seed11/pinn_seed11.json](../Benchmarks/explicit_results/coaxial_full/fixed/subdomain/seed11/pinn_seed11.json) | subdomain | 31.976627 | 0.003146 | 0.006289 | Aceptada |
| [Benchmarks/explicit_results/xlpe_single/fixed/subdomain/seed11/pinn_seed11.json](../Benchmarks/explicit_results/xlpe_single/fixed/subdomain/seed11/pinn_seed11.json) | subdomain | 36.544649 | 0.307839 | 0.392917 | Aceptada |

Los pilotos aceptados cumplen también los saltos de temperatura/flujo y el balance por material registrados en sus JSON. El piloto mixto rechazado se conserva. Este informe corresponde a los pilotos previos. La confirmación independiente y la extensión de ampacidad se documentan en [la revisión integral](REVISION_INTEGRAL_TESIS.md), sin sumar estos pilotos a sus réplicas.

Pruebas automatizadas: 362 ejecutadas; 0 fallos y 0 errores. Tres pruebas lentas fueron excluidas por la configuración existente. La primera ejecución conjunta detectó contaminación de la precisión global de PyTorch entre pruebas; se aisló el estado de cada prueba y se repitió la suite completa.

## Efecto piel

Los datos del catálogo se tratan como DC para la estimación; su etiqueta R20 no prueba la naturaleza del dato original. Antes de concluir sobre un cable real, documentar construcción, resistencia, frecuencia y temperatura de la fuente primaria. La corrección no debe repetirse si el dato ya es AC a 60 Hz.

La estimación de conductor circular macizo aislado arroja aumento de pérdida de 0,195–0,317 % para XLPE y 25,578–37,136 % para los casos grandes, al comparar 90 y 20 °C. A igual pérdida, el diagnóstico radial cambia Tmax en aproximadamente 0,00001 K, 0,004 K y 0,0016 K para XLPE, Aras y Kim. Son resultados condicionales calculados, no valores certificados de esos cables. [Datos y supuestos](auditoria/metodologia_2026-09-14/skin_effect.json).

El contraste FEM 2D adicional del caso Aras, con cuatro mallas por fuente y potencia AC igualada por integración, dio cambio de Tmax de **-0.004045 K**, máximo cambio de campo muestreado 0.004045 K y diferencia relativa de potencia 1.55e-13 %. [Comparación completa](../Benchmarks/explicit_results/skin_aras_60Hz/equal_power/comparison.json). La normalización elimina la pequeña diferencia de potencia debida a interpolar el perfil eléctrico; sus factores están registrados. Las mallas se refinan para evaluar la estabilidad de la diferencia, no solo la temperatura absoluta.

Este ensayo apoya que la redistribución radial es secundaria para la temperatura estacionaria de ese caso, conservando el aumento de pérdida AC. No demuestra lo mismo para proximidad, conductores segmentados, pantallas con pérdidas o transitorios.

La ecuación de calor y el perfil volumétrico se conservan en el código incluso cuando la sensibilidad sea pequeña. El acoplamiento DC usa conductividad eléctrica local; el acoplamiento electromagnético AC con temperatura no uniforme sigue pendiente. [Especificación, fuentes primarias y comandos](../Benchmarks/FULL_DOMAIN.md).

## Depuración y trazabilidad

Se retiraron 59 archivos obsoletos (24,746,696 bytes). Los dos respaldos LaTeX únicos y la copia PDF del plan están en un [archivo verificado](historico/plan_respaldos_previos_2026-09-14.zip). [Manifiesto de limpieza](auditoria/metodologia_2026-09-14/cleanup.json).

La referencia coaxial inicial tenía un defecto de interpolación al seleccionar una celda del material vecino; se retiraron sus archivos numéricos y se preservaron los metadatos del diagnóstico. `reference_verified` contiene la referencia corregida: error de Tmax de 0,00000368 K frente a la solución analítica. No se borraron campañas históricas ni cambios previos del usuario.

El plan fuente incluye el precisamiento térmico y eléctrico. Los PDF y tablas anteriores aún representan la formulación histórica y requieren recompilación y actualización de resultados cuando se cierre la campaña nueva.

