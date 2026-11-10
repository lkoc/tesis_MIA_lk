"""
extract_figures.py
==================
Dos fases:
  1) --explore : genera thumbnails de TODAS las paginas de los PDFs clave
                 -> graficos_ppt/explore/<pdf_alias>_p{N:03d}.png  (72 dpi)
  2) --extract : extrae paginas especificas + recorta y guarda PNG final (150 dpi)
                 -> graficos_ppt/<nombre>.png
  3) --copy    : copia los PNGs propios del workspace -> graficos_ppt/
Ejecutar en orden:
    python extract_figures.py --copy --explore
    # (revisar graficos_ppt/explore/*.png para confirmar números de página)
    python extract_figures.py --extract
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                             # temas_cursoi_Tesis_1/
WORKSPACE    = SCRIPT_DIR.parent                                  # tesis_MIA_lk/
GRAFICOS     = SCRIPT_DIR / "graficos_ppt"
EXPLORE_DIR  = GRAFICOS / "explore"
ZOTERO_STOR  = Path(r"C:\Users\QU1267\Zotero\storage")

# ---------------------------------------------------------------------------
# Figuras propias del workspace a copiar
# ---------------------------------------------------------------------------
WORKSPACE_COPIES = [
    (
        WORKSPACE / "examples/aras_2005_154kv/results_research/geometry.png",
        GRAFICOS  / "own_aras_geometry.png",
    ),
    (
        WORKSPACE / "examples/aras_2005_154kv/results_research/temperature_field.png",
        GRAFICOS  / "own_aras_T_field.png",
    ),
    (
        WORKSPACE / "examples/kim_2024_154kv_bedding/results_multilayer_research/k_field_case_A.png",
        GRAFICOS  / "own_kim_k_field_A.png",
    ),
    (
        WORKSPACE / "examples/kim_2024_154kv_bedding/results_multilayer_research/temperature_field_case_A.png",
        GRAFICOS  / "own_kim_T_field_A.png",
    ),
    (
        WORKSPACE / "examples/kim_2024_154kv_bedding/results_pac_research/comparison_bar.png",
        GRAFICOS  / "own_kim_comparison_bar.png",
    ),
    (
        WORKSPACE / "examples/kim_2024_154kv_optim_C/results_fem/fig01_T_field_zoom.png",
        GRAFICOS  / "own_kim_fem_zoom.png",
    ),
]

# ---------------------------------------------------------------------------
# PDFs de Zotero para explorar / extraer
# ---------------------------------------------------------------------------
PDF_SOURCES = {
    "thue": ZOTERO_STOR / "HD2QM5YC" / "thue1999electrical__Electrical_Power_Cable_Engineering.pdf",
    "kim2025": ZOTERO_STOR / "N3AFJI3S" / "kimEffectAmbientAir2025__Kim_2024.pdf",
    "khumalo": ZOTERO_STOR / "7XD4PDPD" / "khumaloCriticalAssessmentCable2025__Khumalo_Critical_2025.pdf",
    "oclon": ZOTERO_STOR / "4DLMJ8KZ" / "oclon2015optimizing__Oclon_cable_bedding_2005.pdf",
    "aras": ZOTERO_STOR / "D6HD9TH9" / "aras2005assessment__Aras_ampacitiy_2005.pdf",
    "enescu": ZOTERO_STOR / "988TAG4G" / "Enescu et al. - 2021 - Concepts and methods to assess the dynamic thermal.pdf",
}

# ---------------------------------------------------------------------------
# Páginas a extraer (0-based) — ajustar tras exploración
# Formato: (alias_pdf, pagina_0based, nombre_destino, dpi)
# ---------------------------------------------------------------------------
EXTRACTIONS = [
    # Kim 2025 – Fig.1: configuración cables en bedding (3 capas suelo + PAC)
    ("kim2025", 2,  "kim2025_geometry.png",        150),
    # Kim 2025 – Fig.2: sección transversal cable XLPE (7 capas rotuladas)
    ("kim2025", 3,  "kim2025_cable_xsection.png",  150),
    # Kim 2025 – Fig.7: condiciones de contorno del dominio (suelo 3 capas, BCs)
    ("kim2025", 7,  "kim2025_BC.png",              150),
    # Kim 2025 – Fig.19: distribución de temperatura UPCS (arena, verano/invierno)
    ("kim2025", 14, "kim2025_T_contour.png",        150),
    # Khumalo – Fig.2: circuito resistivo térmico del cable (capas conductor→suelo)
    ("khumalo", 4,  "khumalo_thermal_circuit.png", 150),
    # Enescu 2021 – Fig.1: cable enterrado en backfill, sección zanja (diagram limpio)
    ("enescu",  4,  "enescu_trench_xsection.png",  150),
    # Enescu 2021 – Fig.2: sección transversal cable + modelo RC termoeléctrico
    ("enescu",  6,  "enescu_thermal_circuit.png",  150),
    # Aras 2005 – Fig.1: vista transversal cable 154 kV (XLPE/CONDUCTOR/SCREEN/COVER)
    ("aras",    5,  "aras_cable_xsection.png",     150),
    # Aras 2005 – Fig.2: dominio 2D (18×10m), cable a 1.2m de prof., BCs T=20°C
    ("aras",    6,  "aras_domain_BC.png",          150),
    # Ocloń 2015 – Fig.1: instalación flat/trefoil + Fig.2: sección cable 400 kV
    ("oclon",   2,  "oclon_install_types.png",     150),
]


# ===========================================================================
def do_copy():
    """Copia los PNGs propios del workspace a graficos_ppt/."""
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    print("=== COPY: PNGs propios del workspace ===")
    for src, dst in WORKSPACE_COPIES:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  OK  {dst.name}")
        else:
            print(f"  MISS  {src}")


def do_explore():
    """Genera thumbnails de todas las páginas de los PDFs clave."""
    try:
        import fitz
    except ImportError:
        sys.exit("ERROR: PyMuPDF no instalado. Ejecuta: pip install pymupdf")

    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== EXPLORE: thumbnails de PDFs ===")

    for alias, pdf_path in PDF_SOURCES.items():
        if not pdf_path.exists():
            print(f"  MISS  {alias}: {pdf_path}")
            continue

        doc = fitz.open(str(pdf_path))
        n   = doc.page_count
        print(f"  {alias:10s}  {n:3d} paginas  ->  explore/{alias}_p*.png")

        mat = fitz.Matrix(72 / 72, 72 / 72)   # 72 dpi thumbnail
        for i in range(n):
            page = doc[i]
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            out  = EXPLORE_DIR / f"{alias}_p{i:03d}.png"
            pix.save(str(out))
        doc.close()

    print(f"\nThumbnails en: {EXPLORE_DIR}")
    print("Revisa los thumbnails para confirmar los numeros de pagina en EXTRACTIONS.")


def do_extract():
    """Extrae las páginas definidas en EXTRACTIONS como PNG de alta resolución."""
    try:
        import fitz
    except ImportError:
        sys.exit("ERROR: PyMuPDF no instalado. Ejecuta: pip install pymupdf")

    GRAFICOS.mkdir(parents=True, exist_ok=True)
    print("\n=== EXTRACT: figuras finales ===")

    open_docs = {}

    for alias, page0, outname, dpi in EXTRACTIONS:
        pdf_path = PDF_SOURCES.get(alias)
        if pdf_path is None or not pdf_path.exists():
            print(f"  MISS  {alias}")
            continue

        if alias not in open_docs:
            open_docs[alias] = fitz.open(str(pdf_path))
        doc = open_docs[alias]

        if page0 >= doc.page_count:
            print(f"  WARN  {alias} p{page0}: fuera de rango (total={doc.page_count})")  # noqa
            continue

        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        page = doc[page0]
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        out  = GRAFICOS / outname
        pix.save(str(out))
        print(f"  OK  {outname}  ({dpi} dpi, p{page0})")

    for doc in open_docs.values():
        doc.close()

    print(f"\nFiguras en: {GRAFICOS}")


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae figuras para la presentación PPT")
    parser.add_argument("--copy",    action="store_true", help="Copiar PNGs propios del workspace")
    parser.add_argument("--explore", action="store_true", help="Generar thumbnails de todos las páginas de los PDFs")
    parser.add_argument("--extract", action="store_true", help="Extraer páginas específicas como PNG")
    args = parser.parse_args()

    if not (args.copy or args.explore or args.extract):
        parser.print_help()
        sys.exit(0)

    if args.copy:
        do_copy()
    if args.explore:
        do_explore()
    if args.extract:
        do_extract()
