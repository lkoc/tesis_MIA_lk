# Enmienda metodológica antes de la confirmación: calibración C2

Registrada durante B, antes de ejecutar C o D y antes de observar cualquier entrenamiento de confirmación. Complementa el protocolo A–D original, que se conserva sin editar. Es una iteración formativa DSR autorizada por el objetivo de optimizar el entrenamiento antes de producción.

## Motivo y dictamen técnico

El aumento de colocaciones 1x–4x en los estratos deja saltos máximos de temperatura del orden de 0,46–0,65 K, superiores al límite 0,1 K, aun cuando el campo, la máxima y el balance pueden cumplir sus tolerancias. Mantener indefinidamente un peso de continuidad elegido en el piloto no permite distinguir falta de capacidad de una competencia entre términos de pérdida. La bibliografía de gradientes PINN motiva estudiar ese equilibrio, sin cambiar la física ni relajar los criterios.

Se aprueba técnicamente una calibración limitada **después de C y antes de D**. La ampliación no se presenta como preespecificada al inicio de A: se declara que responde a los diagnósticos de desarrollo. Ninguna observación de D interviene en la decisión.

## Diseño fijado

- Conservar arquitectura, colocaciones, política, tasa y presupuestos elegidos en C.
- Conservar exactamente las ecuaciones, normalizaciones y puertas de aceptación FEM/PINN.
- Comparar el peso de continuidad térmica 100, 1000 y 10000. La referencia 100 reutiliza las cuatro ejecuciones de la tasa elegida en C; no se cuenta como cuatro entrenamientos nuevos. Los otros dos valores generan ocho ejecuciones adicionales.
- Mantener pesos PDE=1, constitutiva=1, flujo=10 y energía=10. El único factor modificado en C2 es el peso de continuidad de temperatura, común a todas las interfaces.
- Usar XLPE homogéneo y estratos, inicializaciones 11 y 23 y muestreos 101 y 103. Conservar todos los resultados, incluidos rechazos.
- Priorizar una configuración que acepte las cuatro ejecuciones. Entre varias completamente aceptadas, elegir el menor peso; si ninguna lo consigue, elegir más aceptaciones y luego menor máxima violación, conservando el carácter provisional.
- Congelar la receta completa resultante antes de D. D conserva siete problemas y las semillas 71, 83 y 97; no se modifica después de ver sus resultados.
- La extensión de corriente límite conserva también el peso seleccionado en C2, junto con el resto de la receta y el doble de presupuesto ya registrado. No se ajusta a partir de las corridas de ampacidad.

El recuento único queda en 101 entrenamientos de selección/confirmación (36 A, 24 B, 12 C, 8 C2 nuevos y 21 D), más nueve de corriente límite. Esta búsqueda sigue siendo preliminar: no optimiza simultáneamente todos los pesos, activaciones y arquitecturas, ni convierte aceptación de desarrollo en certificación operativa.

Fundamento: [Wang, Teng y Perdikaris](https://arxiv.org/abs/2001.04536). La campaña ensaya pesos fijos; no afirma implementar su algoritmo de adaptación de gradientes.
