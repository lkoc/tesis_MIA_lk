# Auditoría y consolidación de la tesis

Fecha de inicio: 10 de septiembre de 2026.

## Secuencia de ejecución

1. Inventariar fuentes, conservar el estado inicial y leer guías UNI y locales.
2. Contrastar Plan, entrega de Google Drive, bibliografía y catálogo de casos.
3. Auditar ecuaciones, datos, entrenamiento, métricas y resultados históricos.
4. Construir `Benchmarks` con especificación común para PINN y FEniCS, soluciones manufacturadas, casos de cables y estudios de convergencia.
5. Ejecutar pruebas, referencias FEM, entrenamientos y evaluación independiente; registrar también resultados rechazados.
6. Integrar metodología, resultados, discusión, conclusiones, resumen y anexos en la tesis, conservando el estilo académico del proyecto.
7. Compilar y revisar referencias, tablas, figuras y PDF; actualizar el informe de auditoría y las limitaciones reales.

## Criterios de trabajo

- El Plan y las entregas anteriores son antecedentes trazables; no se sustituyen sus archivos históricos.
- Los cambios previos del usuario se conservan. El estado inicial se registra antes de editar.
- Una coincidencia de temperatura con un artículo no demuestra equivalencia de modelos ni validación experimental.
- Cada caso nuevo se resuelve también con FEniCS. No se atribuyen a FEniCS soluciones calculadas con otra biblioteca.
- La evidencia distingue implementación, verificación numérica, contraste bibliográfico y validación experimental.
- Las conclusiones se limitan a ejecuciones comprobadas y declaran objetivos alcanzados parcialmente.

## Estado

| Etapa | Estado y evidencia |
|---|---|
| 1–2. Documentos y fuentes | Revisados Plan, entrega Drive, guías UNI y bibliografía; decisiones en `AUDITORIA_TESIS.md`. |
| 3. Código | Corregidos operador, contornos, interfaces, conductividad y pérdidas; pruebas de física ejecutadas. |
| 4. Formato y batería | 19 JSON físicos, fórmulas continuas, estratos exactos y entrada común para ambos métodos. |
| 5. Evaluación | Completadas campañas, estudio de resolución, propagación y auditoría de archivos; fallos conservados. |
| 6. Integración | Completados metodología, resultados, discusión, conclusiones, resumen y anexos desde evidencia calculada. |
| 7. Documento final | Cuadernos ejecutados, PDF compilado y revisión de referencias, tablas, figuras y trazabilidad completada. |

La corrección solicitada sobre pérdidas amplía la campaña: la operación usa
R(Tc) individual y los ensayos a R20 fija quedan identificados como controles.
El informe interno incluye todos los intentos. La prueba de reinstalación
Python usa un entorno aislado y conserva su registro de comandos.
