# Evaluación y criterios de aceptación

Después de entrenar se hace una **auditoría independiente**: puntos densos con una semilla (909091) que el
entrenamiento nunca usa. Se reporta, por corrida:

| Medida | Qué mide |
|---|---|
| `pde_rms` | residuo de la ecuación por región |
| `constitutive_rms` | residuo de la ley constitutiva (M2) |
| `interfaces` | salto de temperatura y de flujo normal en cada interfaz |
| `boundary` | error en cada tramo de frontera, con su unidad |
| `balance` | balance de energía global y por región, relativo a la potencia total |

Si hay referencia (exacta o suministrada) se añade:

| Medida | Qué mide |
|---|---|
| `nrmse_pct` | error RMS ponderado por área, como porcentaje del aumento de temperatura |
| `error_rise_pct` | error de Tmax como porcentaje del aumento de temperatura |
| `region_rmse_rise_pct` | error RMS en cada región |
| `hotspot_distance_m` | distancia entre el punto caliente predicho y el de referencia |
| `conductor_mean_error_K` | error de la temperatura media de cada conductor |

## Compuertas

| Compuerta | Umbral |
|---|---|
| NRMSE | 5 % |
| Error de Tmax (aumento) | 5 % |
| Error RMS por región (aumento) | 5 % |
| Balance de energía (global y por región) | 2 % |
| Salto máximo de T en interfaces | 0.1 K |
| Salto RMS de flujo en interfaces | 1 % |

Para el error RMS regional se admite una tolerancia de **menos de 1 punto porcentual** por encima del umbral nominal: por ejemplo, 5.32 % se informa como `accepted_with_tolerance`, mientras que 6.00 % se rechaza. `raw_gates` conserva el resultado contra el umbral estricto de 5 % y `gates` indica el veredicto efectivo. No se aplica tolerancia a las demás compuertas.

Veredictos: `accepted` (todos los objetivos nominales), `accepted_with_tolerance` (solo el exceso regional está dentro de la tolerancia), `rejected` o `no_reference` (sin referencia no se puede aceptar).

Los umbrales nominales y la tolerancia están en `pinn_v2/evaluation/metrics.py` (`CRITERIA` y `ACCEPTANCE_TOLERANCE`).

## Qué NO es una referencia

Los valores de un artículo y los resultados FEM sirven para **evaluar**. No se usan como etiquetas.
Cada resultado dice con qué referencia se comparó (`reference_label`).