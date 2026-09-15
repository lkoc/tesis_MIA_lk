# Evidencia de la revisión metodológica del 14 de septiembre de 2026

La [metodología propuesta](../../METODOLOGIA_PROPUESTA_2026-09-14.md) se definió y aprobó técnicamente antes de revisar la implementación. La [auditoría posterior](../../AUDITORIA_POST_METODOLOGIA_2026-09-14.md) separa aceptación del protocolo, verificación del modelo reducido y brechas respecto de capas 2D explícitas.

| Evidencia | Archivo | Alcance |
|---|---|---|
| Campos y métricas seleccionados | [evidence.json](evidence.json) | 90 registros, huellas de casos/referencias/fuentes y RMSE recalculado |
| CPU/XPU | [compute_probe.json](compute_probe.json) | Hardware y microbenchmark de derivadas; no rendimiento hasta solución aceptada |
| Entrenamiento repetido | [repeat_evidence.json](repeat_evidence.json) | Dos reinicios completos de semilla 11; diferencias de campo y Tmáx de 0 K |
| Configuraciones y trazas | `*_repeat_configuration.json`, `*_repeat.log` | Comandos e insumos de las repeticiones |
| Resultados nuevos | `retrained/` | Pesos, campos, fuentes y nubes; preserva campañas anteriores |

Desde la raíz del repositorio, para revisar métricas sin reentrenar:

```powershell
python docs/auditoria/metodologia_2026-09-14/collect_evidence.py
```

Para repetir los diagnósticos de cómputo, sin entrenamientos simultáneos:

```powershell
python docs/auditoria/metodologia_2026-09-14/probe_compute.py
```

Para reiniciar las dos corridas con sus configuraciones archivadas:

```powershell
python docs/auditoria/metodologia_2026-09-14/repeat_selected.py
```

Los scripts vuelven a generar sus resúmenes en esta carpeta. El solucionador conserva los resultados previos de una repetición en `run_history` cuando corresponde. Para conservar también resúmenes y trazas de cada sesión, copiar esta carpeta a otro identificador antes de repetirla.

Se ejecutaron las 29 pruebas existentes de `Benchmarks/tests`, todas aprobadas. Las comprobaciones nuevas no implementan la candidata mixta ni una referencia multicapa. La conclusión de esta revisión es aceptar el protocolo metodológico y restringir el valor probatorio del artefacto actual a su modelo DC reducido, con los controles de interfaz pendientes descritos en el informe.
