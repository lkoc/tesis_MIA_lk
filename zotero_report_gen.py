"""
Genera ANALISIS_BIBLIOGRAFICO_ZOTERO.md con campo "Aporte al proyecto" (~50 palabras)
para cada paper, en relación al tema: PINNs para cálculo térmico de cables enterrados.
"""
import json
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# GENERADOR DE "APORTE AL PROYECTO" por categoría + contenido del paper
# ────────────────────────────────────────────────────────────────────────────

def has(text, kws):
    return any(k in text for k in kws)

def aporte(it, cat):
    """Genera ~50 palabras describiendo el aporte del paper al proyecto de tesis."""
    title  = (it.get("title") or "").lower()
    abstr  = (it.get("abstractNote") or "").lower()
    tags   = " ".join(it.get("tags") or []).lower()
    full   = f"{title} {abstr} {tags}"
    au     = (it.get("authors") or ["?"])[0].split()[-1].lower()

    # ── PINN ─────────────────────────────────────────────────────────────────
    if cat == "PINN":
        # Papers que llegaron a PINN por colección pero son realmente sobre DTR/cables
        if has(full, ("dynamic thermal rating","dynamic rating","thermal rating","dtr")) and not has(full, ("pinn","physics-informed neural")):
            return ("Revisión de métodos para clasificación térmica dinámica de cables subterráneos. "
                    "Define el marco DTR en que se inscribe la tesis: el PINN ofrece una alternativa "
                    "computacionalmente eficiente a FEM iterativo para evaluación continua de la "
                    "temperatura máxima del conductor en tiempo real.")
        if has(full, ("thermal assessment","thermal impact","cable current rating","overview")) and "cable" in full and not has(tags, ("pinn","physics-informed")):
            return ("Revisa métodos de evaluación térmica de cables de potencia y su impacto en la "
                    "clasificación de corriente. Establece el estado del arte que la tesis amplía, "
                    "posicionando el enfoque PINN como alternativa de mayor precisión frente a los "
                    "modelos analíticos y de elementos finitos existentes.")
        if has(full, ("fluctuant","fluctuat","continuously","time-var","cyclic load")) and not has(tags, ("pinn","physics-informed")):
            return ("Analiza la temperatura de cables bajo carga fluctuante con suelo variable usando "
                    "IEC 60853. Referencia directa para la extensión transitoria del PINN de la tesis, "
                    "donde los pesos temporales de la función de pérdida capturan la inercia térmica "
                    "del sistema cable-suelo.")
        if "raissi" in au or "foundational" in tags:
            return ("Presenta el marco original de las PINNs: redes neuronales que incorporan "
                    "ecuaciones diferenciales parciales como función de pérdida. Es el fundamento "
                    "matemático directo de la arquitectura empleada en la tesis para resolver "
                    "la ecuación de calor en el dominio 2D del cable enterrado.")
        if "review" in tags or "systematic" in tags or "survey" in full:
            return ("Revisión sistemática del estado del arte en PINNs. Sirve como mapa "
                    "metodológico para la tesis: inventaría variantes de función de pérdida, "
                    "estrategias de entrenamiento y dominios de aplicación, orientando la "
                    "selección de arquitectura residual y pesos adaptativos utilizados.")
        if has(full, ("inverse","estimation","reconstruct","unknown parameter")):
            return ("Demuestra el uso de PINNs en problemas inversos de conducción de calor: "
                    "identificación de conductividad o fuentes térmicas con datos escasos. "
                    "Informa la extensión de la tesis hacia estimación de k_suelo y k_backfill "
                    "en instalaciones reales sin medición directa.")
        if has(full, ("cable","underground cable","ampacity")):
            return ("Aplica aprendizaje profundo físicamente informado directamente a cables "
                    "de potencia enterrados. Valida la viabilidad del enfoque PINN en el dominio "
                    "de la tesis y ofrece arquitecturas y métricas de comparación frente "
                    "a FEM e IEC 60287.")
        if has(full, ("porous","geothermal","subsurface","heterogeneous")):
            return ("Aplica PINNs a medios porosos o subsuperficiales con conductividad variable "
                    "espacialmente, análogo al suelo estratificado de la tesis. Aporta técnicas "
                    "de muestreo adaptativo y ponderación de residuos en dominios heterogéneos "
                    "relevantes para la interfaz PAC/suelo.")
        if has(full, ("domain decomposition","xpinn","cpinn","parallel")):
            return ("Introduce descomposición de dominio en PINNs (xPINNs/cPINNs) para "
                    "geometrías complejas con subdominios múltiples. Aplicable a la tesis para "
                    "manejar la interfaz discontinua cable-backfill y zonas PAC con propiedades "
                    "térmicas distintas en un solo entrenamiento.")
        if has(full, ("3d","anisotropic","three-dimensional")):
            return ("Extiende PINNs a conducción de calor 3D y medios anisotrópicos. Referencia "
                    "para escalar la arquitectura 2D de la tesis a configuraciones "
                    "tridimensionales de cables en zanja y evaluar tensores de conductividad "
                    "variables en el suelo real.")
        if has(full, ("transient","nonlinear","time-dependent")):
            return ("Aborda conducción de calor no lineal o transitoria con PINNs, incluyendo "
                    "conductividad dependiente de temperatura. Extiende la capacidad del modelo "
                    "de tesis hacia régimen transitorio, necesario para clasificación dinámica "
                    "bajo perfiles de carga variables.")
        if has(full, ("temperature field","reconstruct","boundary condition")):
            return ("Reconstruye campo de temperatura mediante PINN con condiciones de contorno "
                    "parciales. Aporta técnicas de pérdida en contorno y regularización "
                    "aplicables al modelo residual de la tesis para mejorar precisión "
                    "en la superficie del conductor y la interfaz suelo-aire.")
        if has(full, ("cnn","lstm","condition monitoring","fault")):
            return ("Combina redes físicamente mejoradas (CNN-LSTM) con modelos de física "
                    "para monitoreo de cables enterrados en tiempo real. Representa la "
                    "evolución hacia aplicaciones industriales del enfoque PINN de la tesis, "
                    "integrando detección de fallas y estimación de temperatura.")
        if has(full, ("pso","optimization","ampacity","backfill")):
            return ("Optimiza la ampacidad de cables de alta tensión ajustando parámetros de "
                    "backfill térmico mediante PSO. Complementa la tesis mostrando que el "
                    "modelo PINN puede integrarse en bucles de optimización para diseño "
                    "óptimo de instalaciones de cable.")
        return ("Contribuye al estado del arte de PINNs para resolver problemas de transferencia "
                "de calor con condiciones heterogéneas. Aporta técnicas de arquitectura, "
                "función de pérdida o estrategia de entrenamiento transferibles al modelo "
                "de la tesis para calcular el campo térmico del cable enterrado.")

    # ── Norma ─────────────────────────────────────────────────────────────────
    if cat == "Norma":
        if "60287" in full:
            return ("Define el método analítico oficial IEC para calcular temperatura máxima "
                    "y ampacidad de cables en régimen permanente. Es la línea base normativa "
                    "de la tesis: los resultados PINN se validan directamente contra esta norma "
                    "en los benchmarks Aras 2005 y Kim 2025.")
        if "60853" in full or "cyclic" in full:
            return ("Establece el procedimiento IEC para clasificación cíclica y de emergencia "
                    "de cables. Provee el marco normativo para extender la tesis a cargas "
                    "variables, donde el PINN debe predecir temperaturas bajo perfiles "
                    "cíclicos sin re-entrenar por cada ciclo.")
        if "60502" in full or "construction" in full:
            return ("Especifica los parámetros constructivos de cables XLPE de alta tensión: "
                    "capas, materiales y tolerancias geométricas. Fuente directa de los radios "
                    "y conductividades de cada capa usados en el modelo multicapa "
                    "de la tesis para construir la temperatura de fondo T_bg.")
        if "442" in full or "thermal resistivity" in full:
            return ("Guía IEEE para medir resistividad térmica del suelo in situ. Define el "
                    "protocolo de obtención del parámetro k_soil que alimenta el PINN; su "
                    "incertidumbre de medición impacta directamente el error de predicción "
                    "de temperatura del conductor en la tesis.")
        if "astm" in full:
            return ("Norma ASTM para medición de conductividad térmica de suelo y roca. "
                    "Complementa la guía IEEE para caracterizar backfill y capa PAC; sus valores "
                    "de k_eff son parámetros de entrada críticos para el dominio heterogéneo "
                    "modelado con la función sigmoidea del PINN de la tesis.")
        if "835" in full or "ampacity tables" in full:
            return ("Tablas estándar IEEE de ampacidad para cables de potencia. Referencia "
                    "de validación cruzada para los resultados PINN en instalaciones típicas; "
                    "verifica que el modelo reproduce valores tabulados antes de evaluar "
                    "configuraciones no estándar como el backfill PAC.")
        return ("Norma de referencia que rige el diseño o medición del sistema de cables. "
                "Provee parámetros normativos y criterios de aceptación que la tesis emplea "
                "para validar y contextualizar los resultados del PINN frente a los "
                "métodos analíticos oficiales vigentes.")

    # ── FEM/Numérico ──────────────────────────────────────────────────────────
    if cat == "FEM/Numérico":
        if has(full, ("cable","underground","ampacity")):
            return ("Aplica FEM directamente al cálculo térmico de cables enterrados, generando "
                    "campos de temperatura de referencia. Actúa como benchmark primario de la "
                    "tesis: los errores PINN vs FEM (< 1 K en los casos validados) demuestran "
                    "la equivalencia computacional del método propuesto.")
        if has(full, ("composite","inhomogeneous","heterogeneous","two-phase")):
            return ("Modela numéricamente conducción de calor en materiales compuestos o medios "
                    "no homogéneos, análogo al dominio cable-backfill-suelo de la tesis. Informa "
                    "la discretización espacial y los esquemas de interfaz usados para validar "
                    "la transición PAC/suelo con el PINN.")
        if has(full, ("b1.56","b1.87","cigre","rating examples","verification")):
            return ("Documento CIGRE con casos de verificación de herramientas de clasificación "
                    "de cables por FEM y métodos analíticos. Proporciona geometrías y resultados "
                    "tabulados que la tesis usa como casos de prueba independientes para "
                    "validar el flujo de trabajo PINN implementado.")
        return ("Referencia de métodos numéricos clásicos para conducción de calor. Fundamenta "
                "el rol de los métodos de referencia en la tesis y provee formulaciones de "
                "diferencias o elementos finitos contra las cuales se compara la solución "
                "PINN en precisión y costo computacional.")

    # ── Fundamentos TC ────────────────────────────────────────────────────────
    if cat == "Fundamentos TC":
        if "carslaw" in au or "conduction of heat in solids" in title:
            return ("Texto clásico de conducción de calor en sólidos, incluyendo la solución "
                    "analítica de Kennelly para cilindros en semiplano. Base de la función "
                    "T_bg que la tesis usa como inicialización del modelo residual PINN "
                    "para acelerar la convergencia del entrenamiento.")
        if "heat equation" in title or "diffusion equation" in title:
            return ("Derivación y propiedades de la ecuación de calor/difusión. Marco "
                    "matemático de la función de pérdida PDE del PINN de la tesis; define "
                    "la forma de la ecuación de Poisson estacionaria resuelta en el "
                    "dominio 2D cable-backfill-suelo.")
        if "green" in title:
            return ("Funciones de Green para la ecuación de calor en geometrías canónicas. "
                    "Informa la construcción de T_bg del modelo residual de la tesis, "
                    "particularmente para la distribución de fuentes de calor en el "
                    "conductor y la capa dieléctrica XLPE.")
        if has(au, ("çengel","cengel","engel")):
            return ("Texto de ingeniería en transferencia de calor con enfoque práctico. "
                    "Provee correlaciones y propiedades térmicas de materiales del cable "
                    "(XLPE, cobre, PVC) y del suelo usados como parámetros de entrada "
                    "en el modelo PINN de la tesis.")
        if "patankar" in au:
            return ("Referencia fundamental de transferencia de calor y flujo numérico "
                    "(volúmenes finitos). Informa la comprensión de los esquemas discretos "
                    "usados como baseline e ilustra la ventaja del PINN al evitar la "
                    "discretización explícita del dominio.")
        return ("Fundamento teórico de la conducción de calor en sólidos. Provee la ecuación "
                "gobernante, condiciones de contorno y propiedades térmicas que conforman "
                "el problema físico resuelto por el PINN en el dominio 2D "
                "cable-backfill-suelo de la tesis.")

    # ── Cable/Ampacidad ───────────────────────────────────────────────────────
    if cat == "Cable/Ampacidad":
        if "neher" in au or "mcgrath" in au:
            return ("Paper seminal (1957) que establece el modelo térmico de cables como "
                    "circuito de resistencias térmicas y fuentes de calor. Antecedente directo "
                    "de IEC 60287; sus ecuaciones de T_bg multicapa son la base analítica del "
                    "modelo residual PINN de la tesis.")
        if "anders" in au and "unfavorable" in title:
            return ("Trata la ampacidad en entornos térmicamente desfavorables: suelo seco, "
                    "backfill heterogéneo y zonas de PAC. Complementa el benchmark Kim 2025, "
                    "donde el PAC modifica localmente k_eff y el PINN captura la corrección "
                    "de temperatura sobre la solución IEC de fondo.")
        if "anders" in au:
            return ("Referencia completa de cálculo de ampacidad para cables de transmisión "
                    "y distribución. Formaliza el modelo IEC 60287 con ejemplos numéricos que "
                    "la tesis usa para verificar T_bg analítica y la generación de calor "
                    "en conductor, pantalla y dieléctrico XLPE.")
        if "aras" in au:
            return ("Evalúa métodos de cálculo de ampacidad comparando IEC 60287, FEM ANSYS "
                    "y modelos analíticos para cables XLPE 154 kV. Es el benchmark principal "
                    "de la tesis (formación trefoil y plana), con temperatura objetivo 90 °C "
                    "reproducida por el PINN con error < 0.1 K.")
        if "kim" in au:
            return ("Analiza transferencia de calor en cable con backfill PAC (material de "
                    "alta conductividad k ≈ 2 W/m·K). Segundo benchmark central de la tesis: "
                    "154 kV enterrado en Corea con T_FEM = 70.6 °C, reproducida por el PINN "
                    "con curriculum training con error +0.6 K.")
        if has(full, ("duct","improvement","mixed filler","phase change")):
            return ("Estudia la mejora de ampacidad en ductos mediante rellenos de alta "
                    "conductividad y materiales de cambio de fase. Aporta evidencia de la "
                    "sensibilidad de temperatura a k_backfill, validando la importancia "
                    "de ese parámetro en el modelo PINN de la tesis.")
        if has(full, ("climate","seasonal","long term","network planning")):
            return ("Cuantifica el impacto de variaciones climáticas y estacionales en la "
                    "ampacidad de cables. Contextualiza la extensión futura de la tesis "
                    "hacia clasificación dinámica dependiente de temperatura del suelo "
                    "variable con la estación y el cambio climático.")
        if has(full, ("xlpe","nanocomposite","insulation","dielectric")):
            return ("Revisa propiedades dieléctricas y térmicas de la XLPE y sus "
                    "nanocompuestos. Actualiza los valores de k_XLPE y capacidad calorífica "
                    "que alimentan el modelo de capas del cable en la arquitectura "
                    "residual PINN de la tesis.")
        return ("Analiza cálculo de ampacidad y temperatura en cables de potencia "
                "subterráneos. Aporta el marco físico del dominio de aplicación y valores "
                "de referencia para contrastar la predicción PINN con métodos estándar "
                "IEC y FEM en las configuraciones estudiadas.")

    # ── Suelo/Backfill ────────────────────────────────────────────────────────
    if cat == "Suelo/Backfill":
        if "farouki" in au:
            return ("Referencia clásica de propiedades térmicas de suelos naturales: "
                    "conductividad, capacidad calorífica y efecto de la humedad. Define "
                    "el rango k_soil (0.5–2.5 W/m·K) y los modelos de mezcla usados para "
                    "parametrizar el dominio de suelo heterogéneo del PINN de la tesis.")
        if "vries" in au:
            return ("Modelo De Vries para conductividad térmica efectiva de suelos según "
                    "humedad y textura. Base teórica del parámetro k_eff en la zona de "
                    "secado crítica alrededor del cable, cuya variación espacial captura "
                    "la función sigmoidea del dominio PINN de la tesis.")
        if has(au, ("ocłoń","oclon")):
            return ("Optimiza la geometría del lecho de backfill alrededor del cable "
                    "mediante PSO con modelo FEM. Benchmark de optimización que valida "
                    "los valores de k_backfill óptimos (PAC ≈ 2 W/m·K) usados en el "
                    "caso Kim 2025 de la tesis.")
        if "khumalo" in au:
            return ("Evaluación crítica de métodos de clasificación bajo condiciones de "
                    "secado del suelo. Cuantifica el error del IEC 60287 cuando k_soil "
                    "disminuye por secado, motivando el modelo PINN de la tesis para "
                    "capturar k_soil(x,y) espacialmente variable.")
        if "kolawole" in au:
            return ("Evalúa k_soil bajo diferentes salinidades, minerales y humedad en "
                    "infraestructura enterrada. Aporta evidencia experimental de la "
                    "variabilidad espacial de k_soil que el dominio heterogéneo del PINN "
                    "representa con la función de transición sigmoidea.")
        if "malmedal" in au:
            return ("Mide estabilidad térmica y resistividad del suelo bajo cable. Define "
                    "el concepto de zona crítica de secado, cuya geometría y parámetros "
                    "alimentan la malla de muestreo adaptativo de puntos de colocación "
                    "en el entrenamiento del PINN de la tesis.")
        return ("Caracteriza propiedades térmicas del suelo o backfill alrededor del cable. "
                "Provee valores de conductividad y capacidad calorífica que parametrizan el "
                "dominio heterogéneo del PINN de la tesis para modelar correctamente "
                "la disipación de calor hacia el entorno enterrado.")

    # ── DTR/Dinámico ──────────────────────────────────────────────────────────
    if cat == "DTR/Dinámico":
        if "klaasen" in au:
            return ("Metodología para identificar cuellos de botella térmicos en redes de "
                    "cables con perfiles de carga variables. Caso de aplicación directa donde "
                    "el PINN de la tesis puede sustituir cálculos FEM iterativos para "
                    "clasificación en tiempo real de segmentos críticos.")
        if "enescu" in au and "dynamic" in full:
            return ("Revisión de métodos para clasificación térmica dinámica de cables "
                    "subterráneos. Define el marco DTR en el que se inscribe la tesis: el "
                    "PINN ofrece una alternativa eficiente a FEM iterativo para evaluación "
                    "continua de la temperatura máxima del conductor.")
        if has(au, ("hegg","heggå")):
            return ("Modelo de temperatura transitoria para clasificación dinámica de cables "
                    "basado en el circuito RC térmico IEC. Punto de comparación para la "
                    "extensión transitoria de la tesis: el PINN debe superar la precisión "
                    "del modelo RC en cargas no periódicas.")
        if "kwak" in au:
            return ("Evalúa capacidad térmica y ampacidad de cables XLPE 24 kV en aire bajo "
                    "cargas variables. Aporta datos de validación para la extensión del PINN "
                    "de la tesis a instalaciones en aire y confirma la metodología de "
                    "circuito RC equivalente como referencia.")
        if has(full, ("climate","seasonal","long term")):
            return ("Analiza el impacto de variaciones climáticas estacionales en ampacidad "
                    "de cables. Motiva la integración de condiciones de contorno variables "
                    "(T_suelo, T_ambiente) en el PINN de la tesis para clasificación "
                    "dinámica a largo plazo.")
        if has(full, ("fluctuant","fluctuat","continuously","time-var")):
            return ("Analiza temperatura de cables bajo carga fluctuante con suelo variable "
                    "usando IEC 60853. Referencia para la extensión transitoria del PINN, "
                    "donde los pesos temporales de la función de pérdida deben capturar "
                    "la inercia térmica del sistema cable-suelo.")
        if "fariz" in au:
            return ("Mapeo sistemático del estado del arte en DTR para cables subterráneos "
                    "(2026). Posiciona la contribución de la tesis en el panorama actual: "
                    "los PINNs representan la frontera metodológica para clasificación "
                    "dinámica eficiente en redes eléctricas de alta tensión.")
        return ("Aborda la clasificación térmica dinámica o el análisis de cargas cíclicas "
                "en cables de potencia. Contextualiza la aplicación práctica del PINN "
                "de la tesis en redes eléctricas donde la carga varía continuamente "
                "y se requiere evaluación térmica en tiempo real.")

    # ── Otro ──────────────────────────────────────────────────────────────────
    return ("Referencia de soporte para el proyecto de tesis. Aporta contexto complementario "
            "sobre metodología o dominio relacionado con el cálculo térmico de cables "
            "enterrados mediante métodos avanzados de aprendizaje automático "
            "informado por física.")

with open("zotero_library.json", encoding="utf-8") as f:
    raw = json.load(f)

# Deduplicar
seen = {}
for it in raw:
    key = (it.get("title") or "").strip().lower()[:80]
    yr  = (it.get("date") or "")[:4]
    uid = f"{key}|{yr}"
    if uid not in seen:
        seen[uid] = it
    else:
        existing = seen[uid]
        existing["collections"] = list(set(
            (existing.get("collections") or []) + (it.get("collections") or [])))
        existing["tags"] = list(set(
            (existing.get("tags") or []) + (it.get("tags") or [])))
        if len(it.get("abstractNote") or "") > len(existing.get("abstractNote") or ""):
            existing["abstractNote"] = it["abstractNote"]
        if not existing.get("DOI") and it.get("DOI"):
            existing["DOI"] = it["DOI"]

items = list(seen.values())

def categorize(it):
    tags_str  = " ".join(it.get("tags") or []).lower()
    title_str = (it.get("title") or "").lower()
    colls     = " ".join(it.get("collections") or []).lower()
    combined  = f"{tags_str} {title_str} {colls}"
    if "pinn" in combined or "physics-informed" in combined:
        return "PINN"
    if ("iec 60287" in combined or "iec 60853" in combined or "ieee std" in combined
            or "astm" in combined or "iec 60502" in combined):
        return "Norma"
    if ("fem" in combined or "finite element" in combined or "fdm" in combined
            or "numerical heat" in combined or "coleccion-02" in combined):
        return "FEM/Numérico"
    if ("coleccion-01" in combined or "fundamentos transferencia" in combined
            or "heat conduction" in title_str or "heat equation" in title_str
            or "heat transfer handbook" in title_str):
        return "Fundamentos TC"
    if ("backfill" in combined or "bedding" in combined or "suelos" in combined
            or "coleccion-04" in combined or "soil thermal" in combined
            or "soil moisture" in combined or "thermal resistivity" in combined
            or "thermal properties of soils" in title_str):
        return "Suelo/Backfill"
    if ("dtr" in combined or "dynamic thermal" in combined or "cyclic" in combined
            or "transient" in combined or "coleccion-05" in combined):
        return "DTR/Dinámico"
    if ("ampacity" in combined or "cable rating" in combined or "coleccion-03" in combined
            or "underground power cable" in combined or "xlpe" in combined):
        return "Cable/Ampacidad"
    return "Otro"

CATEGORY_ROLES = {
    "PINN": {
        "title": "🟣 PINNs (Physics-Informed Neural Networks)",
        "role": "NÚCLEO METODOLÓGICO",
        "desc": "Papers fundacionales y aplicados de PINNs. Base matemática y computacional de la tesis.",
    },
    "FEM/Numérico": {
        "title": "🟠 Métodos Numéricos FEM/FDM",
        "role": "MÉTODO DE REFERENCIA / VALIDACIÓN",
        "desc": "Herramientas numéricas clásicas usadas como benchmark y comparación con PINNs.",
    },
    "Fundamentos TC": {
        "title": "🔴 Fundamentos de Transferencia de Calor",
        "role": "BASE TEÓRICA",
        "desc": "Marco teórico para la ecuación de calor, conducción, y modelos continuos.",
    },
    "Cable/Ampacidad": {
        "title": "🟡 Cables de Potencia & Ampacidad",
        "role": "DOMINIO DE APLICACIÓN",
        "desc": "Literatura técnica sobre cables XLPE, cálculo de ampacidad y operación.",
    },
    "Suelo/Backfill": {
        "title": "🟢 Suelo, Backfill & Humedad",
        "role": "PARÁMETROS CLAVE DEL PROBLEMA",
        "desc": "Caracterización térmica del entorno enterrado: conductividad, secado, materiales.",
    },
    "DTR/Dinámico": {
        "title": "🔵 DTR & Cargas Cíclicas",
        "role": "APLICACIÓN / EXTENSIÓN",
        "desc": "Clasificación dinámica térmica y operación en regímenes variables.",
    },
    "Norma": {
        "title": "⚙️ Normas & Estándares",
        "role": "REFERENCIA NORMATIVA",
        "desc": "Normas IEC 60287, IEC 60853, IEEE, ASTM que gobiernan el cálculo oficial.",
    },
    "Otro": {
        "title": "⚪ Otros",
        "role": "REFERENCIA GENERAL",
        "desc": "Papers de soporte sin categoría principal.",
    },
}

def get_importance(it):
    tags = " ".join(it.get("tags") or []).lower()
    if "thesis-core" in tags:    return "⭐⭐⭐ NÚCLEO TESIS"
    if "foundational" in tags:   return "⭐⭐ FUNDACIONAL"
    if "systematic" in tags or "review" in tags: return "⭐⭐ REVISIÓN SISTEMÁTICA"
    if "pinn" in tags:           return "⭐ Relevante PINN"
    if "ampacity" in tags:       return "⭐ Relevante Ampacidad"
    return ""

# Organizar por categoría
by_cat = {}
for it in items:
    cat = categorize(it)
    by_cat.setdefault(cat, []).append(it)

# Ordenar cada categoría por año
for cat in by_cat:
    by_cat[cat].sort(key=lambda x: (x.get("date") or "0000")[:4], reverse=True)

lines = []
lines.append("# Análisis Bibliográfico — Tesis MIA")
lines.append("## PINNs para Cálculo Térmico de Cables de Potencia Enterrados")
lines.append(f"\n**Total papers únicos:** {len(items)}  |  **Base de datos:** Zotero (QU1267)  |  **Fecha análisis:** Mayo 2026\n")
lines.append("---\n")

for cat_key in ["PINN","FEM/Numérico","Fundamentos TC","Cable/Ampacidad","Suelo/Backfill","DTR/Dinámico","Norma","Otro"]:
    if cat_key not in by_cat:
        continue
    cat_items = by_cat[cat_key]
    info = CATEGORY_ROLES[cat_key]
    lines.append(f"\n## {info['title']}")
    lines.append(f"**Rol en la tesis:** {info['role']}  |  **{len(cat_items)} papers**")
    lines.append(f"> {info['desc']}\n")

    for it in cat_items:
        yr      = (it.get("date") or "s.f.")[:4]
        title   = it.get("title") or "Sin título"
        authors = it.get("authors") or []
        first_a = authors[0] if authors else "Anónimo"
        all_a   = "; ".join(authors[:3]) + (" et al." if len(authors)>3 else "")
        doi     = it.get("DOI") or ""
        url     = it.get("url") or ""
        link    = f"https://doi.org/{doi}" if doi else url
        pub     = it.get("publicationTitle") or ""
        abstr   = (it.get("abstractNote") or "")[:300]
        tags    = [t for t in (it.get("tags") or [])
                   if not t.startswith("#") and t not in ("nosource","reference")]
        imp     = get_importance(it)

        aporte_txt = aporte(it, cat_key)

        lines.append(f"### {first_a.split()[-1]} ({yr}) — {title[:80]}{'...' if len(title)>80 else ''}")
        if imp:
            lines.append(f"**{imp}**")
        lines.append(f"- **Autores:** {all_a}")
        if pub:
            lines.append(f"- **Publicación:** {pub}")
        if link:
            lines.append(f"- **DOI/URL:** {link}")
        if abstr:
            lines.append(f"- **Abstract:** {abstr}{'...' if len(it.get('abstractNote',''))>300 else ''}")
        if tags:
            key_tags = [t for t in tags if t not in ("nosource","#nosource") and len(t)<40][:8]
            lines.append(f"- **Keywords:** {', '.join(key_tags)}")
        lines.append(f"- **Aporte al proyecto:** {aporte_txt}")
        lines.append("")

report = "\n".join(lines)
Path("ANALISIS_BIBLIOGRAFICO_ZOTERO.md").write_text(report, encoding="utf-8")
print(f"✅ Reporte: ANALISIS_BIBLIOGRAFICO_ZOTERO.md")
print(f"   {len(items)} papers únicos en {len(by_cat)} categorías")
for cat_key in ["PINN","FEM/Numérico","Fundamentos TC","Cable/Ampacidad","Suelo/Backfill","DTR/Dinámico","Norma","Otro"]:
    n = len(by_cat.get(cat_key,[]))
    if n: print(f"   {cat_key:20s}: {n:3d} papers")
