# Paquete de casos y candidatos PINN_v2

Versión de esquema: `pinn-v2-case-1`. Fecha del inventario: 2026-10-10.

El ZIP incluye 21 JSON activos copiados desde `PINN_v2/pinn_v2/cases/v2/`; cada uno fue cargado con `CaseDefinition.load` al generar el paquete. La hoja de cálculo asociada informa la procedencia por grupo: que una definición cargue no significa que todos sus valores sean publicados; varios registros heredados mantienen `supuesto`.

La carpeta `candidatos_no_ejecutables/` contiene el índice de las 16 entradas del informe previo, registros COES y 139 menciones textuales MINEM de `Subterráneo` sin deduplicar. Estos JSON reutilizan la envoltura y el `schema_version` de PINN_v2, pero dejan los grupos requeridos como `faltante`; por diseño no pasan el cargador ni deben usarse para entrenar/simular. Para varios artículos, `reported_data` resume lo que la fuente sí publica; `faltante` significa que no está normalizado en el bloque físico completo, no necesariamente que el artículo carezca de ese dato. No se inventan dimensiones, profundidad, capas de suelo ni fronteras.

El proyecto COES de 220 kV se clasifica como planificado según el diagnóstico 2023. Los registros MINEM son inventario de líneas y no contienen geometría de zanja ni propiedades térmicas del suelo. El PDF separado COES/DP-02-2022 no pudo descargarse desde la URL citada (respuesta vacía); el caso COES se sustenta en el diagnóstico local consultado.
