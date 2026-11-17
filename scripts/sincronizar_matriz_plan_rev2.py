"""Sincroniza la matriz rev2 con el protocolo del plan del 17-11-2026."""

from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "matriz_consistencia/Matriz_de_Consistencia_Tesis_Cables_Heterogeneidad_Termica_rev2.xlsx"
SOURCE = ROOT / "plan_v2/capitulos/anexos/anexo_1_matriz_consistencia.tex"

UPDATES = {
    "Matriz_de_Consistencia": {
        "B2": "PG. En el arreglo de cables-entorno térmico, la heterogeneidad espacial del suelo nativo, el lecho de asiento y el suelo de relleno puede modificar T(x,y), Tmax e Imax respecto de una representación homogénea equivalente (Enescu et al., 2021; CIGRÉ, 2025). Sin embargo, resulta difícil cuantificar esas diferencias de forma reproducible y establecer cuándo son relevantes para las decisiones de diseño y operación. Las representaciones homogéneas equivalentes son rápidas, aunque no describen de forma explícita la conductividad variable k(x,y); los modelos FEM sí pueden representarla, pero exigen construir, mallar y verificar cada escenario. En consecuencia, persiste la necesidad de una formulación computacional verificada y trazable que represente k(x,y), estime el campo térmico y documente la exactitud, la consistencia física y la robustez del análisis.",
        "I3": "Requisitos críticos cubiertos. Umbral: cobertura mayor que 90 % de los requisitos críticos identificados, con trazabilidad requisito-prueba.",
        "H4": "HE2. La especificación trazable y referencias válidas permitirán producir un artefacto aceptado en F2: E_T < 0,1 K, NRMSE menor que 1 % y desequilibrio no mayor a 2 % en cada una de cinco semillas.",
        "I4": "E_T = |Tmax_PINN - Tmax_ref|: error absoluto frente a la referencia, menor que 0,1 K. NRMSE: error cuadrático medio normalizado por elevación térmica, menor que 1 %. Cada una de cinco semillas debe cumplir estos umbrales y los controles físicos de la sección Protocolo reproducible de la PINN. No hay aceptación alternativa del 5 %.",
        "J4": "Parámetros físicos coincidentes entre artefacto y referencia. Controles: dominio, malla y tolerancia FEM; redes de tres capas de 32 neuronas por subdominio, tanh, float64; Adam 20000 y L-BFGS hasta 200 iteraciones. Semillas de confirmación: 101, 211, 307, 401 y 503; colocación: semilla + 10000. Pesos congelados tras piloto; w_d=0; FEM no interviene en entrenamiento, selección ni parada. Versiones y causas de fallo registradas.",
        "H5": "HE3. A pérdidas iguales se espera aumento de Tmax ante baja k cercana y reducción ante relleno de alta k. Se contrastará el sentido y la relevancia; se admitirán casos sin efecto o inconclusos.",
        "I5": "E_T: error absoluto frente a FEM, menor que 0,5 K; NRMSE menor que 2,5 % por semilla. Delta Tmax = Tmax_het - Tmax_hom: efecto físico, no error numérico. Relevancia: delta_T = 1 K y envolventes conservadoras de cinco semillas según Relevancia y reglas de decisión. Desequilibrio no mayor a 2 % y controles físicos. F3 no sustituye F2.",
        "J5": "Parámetros físicos del escenario, no VI: mapa k(x,y), C_k, f_h y d_h/D. Controles: pérdidas prescritas iguales en VD3, profundidad, disposición, T_amb, dominio, contornos y configuración aceptada; semillas pareadas y referencia FEM propia para cada miembro. Cruce del límite térmico: ensayo adicional con acoplamiento electrotermal activo.",
        "H6": "HE4. Con promedio por área del entorno se espera sobreestimación homogénea de Imax ante baja k cercana. Se contrastarán reducción y sensibilidad al contraste, extensión y proximidad; la expectativa podrá refutarse.",
        "I6": "Ampacidad Imax (A). Error frente a FEM menor que 1 % por semilla, distinto de Delta I (%) = 100 (I_het - I_hom) / I_hom. Relevancia: delta_I = 2 % y envolventes de cinco semillas. Margen térmico (K). Calidad de la optimización por juicio experto: escala de 1 a 10, protocolo preliminar del Anexo 2, sin umbral definitivo; no sustituye la verificación numérica.",
        "J6": "Parámetros heredados mediante VD3. k_eq = suma(A_m k_m) / suma(A_m), solo en el entorno externo a los cables; no ajustado a resultados. Misma geometría, contornos, T_lim y ley R(T), no las mismas pérdidas en ampacidad. Búsqueda independiente PINN/FEM: intervalo relativo <= 0,001 y defecto térmico <= 0,05 K, hasta 60 bisecciones y 12 ampliaciones; intervalos ampliados por incertidumbre según metodología. Pérdidas DC o AC efectiva excluyentes; sin pérdidas dieléctricas, de pantalla ni armadura. Sin validación de campo.",
    },
    "Operacionalización": {
        "C7": "Paquete compuesto por VD3 concluyente y un catálogo nuevo C_4. Comparador principal: media de conductividad ponderada por área del entorno externo al cable, conservando sus capas. Comparadores conservadores u optimistas solo como sensibilidad adicional identificada.",
        "D8": "Estado de VD1; cobertura superior al 90 % de requisitos críticos identificados; trazabilidad requisito-prueba; unidades, supuestos y exclusiones.",
        "D9": "Estado VD2; F2: E_T < 0,1 K, NRMSE < 1 % y desequilibrio <= 2 % en cada una de cinco semillas; MAE, controles físicos y dominio. Referencias independientes del entrenamiento y la parada.",
        "D10": "Estado VD3; F3: E_T < 0,5 K, NRMSE < 2,5 % y desequilibrio <= 2 % por semilla. T(x,y), Tmax, Delta Tmax = Tmax_het - Tmax_hom, gradiente y punto caliente. Relevancia delta_T = 1 K con envolventes de cinco semillas.",
        "D11": "Estado VD4; Imax con error relativo frente a FEM < 1 % por semilla; Delta I (%) = 100 (I_het - I_hom) / I_hom, margen y cobertura de pares. Relevancia delta_I = 2 %; propagación de intervalos de corriente. Juicio experto complementario, no puerta numérica.",
        "C13": "Criterios aplicados a VD1-VD4 y a la síntesis DSR posterior. F2 y F3 son acumulativas, sin puerta alternativa del 5 %. Cinco semillas completas; fallos e incertidumbre que impida clasificar se conservan como inconclusos. No constituyen una quinta VD.",
        "D13": "Para efecto pareado z_s: J_z = [min(z_s)-b_z, max(z_s)+b_z]. Exigir (max(z_s)-min(z_s))/2 + b_z < delta_z. Sin efecto relevante si J_z queda dentro de [-delta_z, delta_z]; relevante si queda completamente fuera por un lado; cruce de límite o fallo: inconcluso. b_z incorpora errores y estimaciones de referencia; no es intervalo de confianza ni cota rigurosa.",
        "D14": "Red por subdominio: 3 capas de 32, tanh, Xavier uniforme y float64. 512 puntos interiores, 256 por interfaz y 128 por tramo exterior. Adam 20000 (lr=0,003) y L-BFGS hasta 200. w_f=w_b=w_e=1; w_d=0; w_i en {1,10,100} elegido en piloto. Semillas 101,211,307,401,503; colocación +10000; control +20000 y comprobación +30000. Configuración congelada, versiones y fallos registrados.",
        "C15": "Incertidumbre documental y error numérico estimado que se registran para interpretar la evidencia. Las semillas de confirmación están prefijadas y no se sustituyen. No confundir variabilidad de ejecución con incertidumbre física de una instalación.",
        "D18": "Catálogo mínimo: A1 analítico; M1 y M2 manufacturados; C0 plano y C1 trébol, adaptaciones/supuestos explícitos con referencia FEM. 24 escenarios heterogéneos o de control para las dos disposiciones, comparadores por área y cinco semillas. Véase Catálogo mínimo y fichas de los casos del capítulo 4 y hoja Diseño de casos.",
    },
    "Diseño de casos": {
        "D3": "Mínimo: base homogénea, inclusión rectangular de baja conductividad, relleno rectangular de mejora y estrato horizontal en C0 plano y C1 trébol. Zona seca solo como representación térmica prescrita; combinado multimaterial o publicado adicional: extensión identificada con ficha completa.",
        "E3": "24 escenarios heterogéneos o de control para dos disposiciones; barridos de un parámetro a la vez desde la base, no factorial completo. Cada heterogéneo tiene comparador por área, cinco semillas y FEM de ambos miembros. Ver Catálogo mínimo y fichas de los casos.",
        "D6": "Principal: heterogéneo 2D frente a homogéneo con k_eq = suma(A_m k_m)/suma(A_m) en el entorno externo a los cables. Alternativas armónica, conservadora u optimista solo como sensibilidad separada.",
        "E6": "Calcular áreas y k_eq antes de resolver; no calibrar a temperaturas ni ampacidad. Conservar dominio, cable, contornos y T_lim. El promedio no garantiza resistencia térmica equivalente ni cota conservadora.",
        "D8": "k_h/k_nativo = 0,5 y 0,25 para baja conductividad; 2 y 4 para mejora. Base: 0,5 o 2 según el material. Variación de un parámetro por vez.",
        "D9": "Inclusiones: altura D=0,1067 m y anchura w/D en {2,4}. Estrato: espesor 2D y ancho del dominio. f_h = área de región heterogénea / área del entorno externo al cable, fijada por geometría.",
        "D10": "Separación libre d_h/D en {0,5; 2}, respecto de envolvente del cable; D=0,1067 m. Inclusiones sobre conductor central de C0 o superior de C1, sin intersectar los cables; estrato con cara inferior a esas separaciones.",
        "D11": "Mismo dominio, cable, geometría, contornos, T_amb, T_lim, semillas y configuración congelada. Solo cambia la representación k del entorno. VD3: pérdidas lineales prescritas iguales; VD4: misma ley R(T), con corriente y pérdidas recalculadas independientemente.",
        "E11": "k_eq por área antes del cálculo. VD3 a pérdidas fijas sin actualizar R(T); cruce de límite a corriente común como ensayo electrotermal adicional. VD4: bisección independiente y verificación FEM de extremos/punto medio PINN, sin transferir campos al entrenamiento.",
        "D12": "Desarrollo: 7, 11 y 23. Confirmación: 101, 211, 307, 401 y 503 en todos los casos, sin selección posterior; semilla de colocación = red + 10000. Mismas semillas en ambos miembros del par.",
        "E12": "Reportar mediana, mínimo, máximo, aceptadas y fallos; envolventes de cinco semillas con error numérico estimado. Fallo o cruce de límite de clasificación: inconcluso; no sustituir ejecuciones ni relajar tolerancias.",
    },
}


def signature(workbook):
    return [
        (
            sheet.title, sheet.sheet_state, sheet.max_row, sheet.max_column,
            tuple(str(item) for item in sheet.merged_cells.ranges),
            [(cell.coordinate, copy(cell._style)) for row in sheet for cell in row],
            [(key, dict(value), copy(value._style)) for key, value in sheet.row_dimensions.items()],
            [(key, dict(value), copy(value._style)) for key, value in sheet.column_dimensions.items()],
            copy(sheet.page_setup), copy(sheet.page_margins),
            str(sheet.print_area), sheet.print_title_rows, sheet.print_title_cols,
        )
        for sheet in workbook
    ]


def verify(workbook):
    for sheet_name, updates in UPDATES.items():
        for coordinate, expected in updates.items():
            if workbook[sheet_name][coordinate].value != expected:
                raise ValueError(f"Celda no sincronizada: {sheet_name}!{coordinate}")
    if "5 %" in workbook["Matriz_de_Consistencia"]["H4"].value:
        raise ValueError("HE2 conserva el umbral anterior")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    for anchor in (r"E_T<\SI{0.1}{\kelvin}", r"\delta_T=\SI{1}{\kelvin}", r"\delta_I=2\,\pct"):
        if anchor not in source:
            raise ValueError(f"La fuente no corresponde al protocolo: {anchor}")
    workbook = load_workbook(WORKBOOK)
    if args.check:
        verify(workbook)
        print("OK: las 3 hojas estan sincronizadas con el protocolo del plan.")
        return
    before = signature(workbook)
    old_values = {(sheet.title, cell.coordinate): cell.value for sheet in workbook for row in sheet for cell in row}
    backup_dir = Path(tempfile.mkdtemp(prefix="matriz_pre_sync_"))
    backup = backup_dir / WORKBOOK.name
    shutil.copy2(WORKBOOK, backup)
    for sheet_name, updates in UPDATES.items():
        for coordinate, value in updates.items():
            workbook[sheet_name][coordinate] = value
    temporary = WORKBOOK.with_suffix(".tmp.xlsx")
    workbook.save(temporary)
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Libro ZIP invalido")
    reloaded = load_workbook(temporary)
    verify(reloaded)
    if signature(reloaded) != before:
        raise ValueError("Se altero la estructura o el formato del libro")
    for sheet in reloaded:
        for row in sheet:
            for cell in row:
                if cell.coordinate not in UPDATES.get(sheet.title, {}) and cell.value != old_values[(sheet.title, cell.coordinate)]:
                    raise ValueError(f"Cambio no autorizado: {sheet.title}!{cell.coordinate}")
    os.replace(temporary, WORKBOOK)
    print(json.dumps({
        "fecha_local": datetime.now().isoformat(timespec="seconds"),
        "archivo": str(WORKBOOK), "backup": str(backup),
        "celdas_alineadas": sum(map(len, UPDATES.values())),
        "estructura_y_estilos": "conservados", "hojas": workbook.sheetnames,
        "sha256": hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(),
        "fuente_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()