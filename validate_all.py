"""Validacion rapida de todos (o un subconjunto) de los ejemplos.

Descubre automaticamente todos los scripts ``run_*.py`` en cada carpeta de
ejemplos, parchea ``solver_params.csv`` con pasos minimos, ejecuta cada
script y muestra un resumen comparativo.

Uso::

    # Todos los ejemplos con perfil quick (defecto)
    python validate_all.py

    # Una carpeta con perfil quick (defecto)
    python validate_all.py kim_2024_154kv_bedding

    # Una carpeta con perfil especifico
    python validate_all.py kim_2024_154kv_bedding research

    # Varias carpetas, cada una con su perfil
    python validate_all.py aras_2005_154kv kim_2024_154kv_bedding research

    # Script especifico con perfil
    python validate_all.py kim_2024_154kv_bedding/run_multilayer.py research

    # Global default diferente (aplica a carpetas sin perfil propio)
    python validate_all.py --profile research

    # Sin parchear el CSV (usa la config real)
    python validate_all.py --no-fast
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"

FAST_OVERRIDES = {
    "adam_steps":  "200",
    "lbfgs_steps": "0",
    "print_every": "100",
    "save_every":  "0",
}

# Extra CLI arguments required by specific scripts (beyond --profile <p>)
SCRIPT_EXTRA_ARGS: dict[str, list[str]] = {
    "run_multilayer.py": ["--case", "both"],
}

# Profile names recognised in positional args
_KNOWN_PROFILES = {"quick", "research"}


# ---------------------------------------------------------------------------
# Positional-arg parser
# ---------------------------------------------------------------------------

def _parse_folder_args(raw: list[str], default: str) -> list[tuple[str, str]]:
    """Parse interleaved ``FOLDER [PROFILE] FOLDER [PROFILE]...`` positional args.

    A token that matches a known profile name immediately after a folder token
    is consumed as that folder's profile.  Any other profile token is attached
    to the most recent folder.  Folders without an explicit profile token use
    *default*.

    Examples::

        ["kim_2024_154kv_bedding", "research"]         -> [("kim…", "research")]
        ["aras", "kim_2024_154kv_bedding", "research"] -> [("aras", "quick"), ("kim…", "research")]
        ["kim_2024_154kv_bedding/run_multilayer.py", "research"] -> [("kim…/run…", "research")]

    Returns:
        List of ``(folder_or_pinned_script, profile)`` pairs.
    """
    result: list[tuple[str, str]] = []
    i = 0
    while i < len(raw):
        token = raw[i]
        if token in _KNOWN_PROFILES:
            # Stray profile token — attach to the previous entry
            if result:
                folder, _ = result[-1]
                result[-1] = (folder, token)
        else:
            # folder (or folder/script.py) — peek for a following profile token
            if i + 1 < len(raw) and raw[i + 1] in _KNOWN_PROFILES:
                result.append((token, raw[i + 1]))
                i += 1  # consume the profile token
            else:
                result.append((token, default))
        i += 1
    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_jobs(
    folder_specs: list[tuple[str, str]] | None = None,
    default_profile: str = "quick",
) -> list[tuple[str, str, str]]:
    """Return ``(example_dir_name, script_name, profile)`` triples to run.

    Args:
        folder_specs: If given (non-empty), each entry is
                      ``(folder_or_pinned_script, profile)``.  A pinned entry
                      such as ``"folder/run_x.py"`` runs only that script.
                      ``None`` means *all* folders with *default_profile*.
        default_profile: Profile used when *folder_specs* is ``None``.
    """
    # Build dirname -> (pinned_script_or_None, profile)
    spec_map: dict[str, tuple[str | None, str]] = {}
    if folder_specs:
        for spec, prof in folder_specs:
            norm = spec.replace("\\", "/")
            if "/" in norm:
                parts = norm.split("/", 1)
                spec_map[parts[0]] = (parts[1], prof)
            else:
                spec_map[spec] = (None, prof)

    result: list[tuple[str, str, str]] = []
    for example_path in sorted(EXAMPLES_DIR.iterdir()):
        if not example_path.is_dir():
            continue
        dirname = example_path.name

        if folder_specs is not None:
            if dirname not in spec_map:
                continue
            pinned_script, prof = spec_map[dirname]
            if pinned_script is not None:
                result.append((dirname, pinned_script, prof))
                continue
            # No specific script pinned — run all scripts with this prof
        else:
            prof = default_profile

        for script in sorted(s.name for s in example_path.glob("run_*.py")):
            result.append((dirname, script, prof))

    return result


# ---------------------------------------------------------------------------
# CSV patching
# ---------------------------------------------------------------------------

def patch_csv(csv_path: Path) -> Path:
    """Backup CSV and overwrite with minimal training steps."""
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)

    rows: list[list[str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() in FAST_OVERRIDES:
                row[1] = FAST_OVERRIDES[row[0].strip()]
            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    return backup


def restore_csv(csv_path: Path, backup: Path) -> None:
    shutil.move(str(backup), str(csv_path))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_script(example_dir: str, script: str, profile: str) -> dict:
    """Run one script under *examples/<example_dir>/* with the given profile."""
    data_dir = EXAMPLES_DIR / example_dir / "data"
    csv_path = data_dir / "solver_params.csv"
    script_path = EXAMPLES_DIR / example_dir / script
    label = f"{example_dir}/{script}"

    if not script_path.exists():
        return {
            "label": label, "rc": -3, "elapsed": 0.0,
            "stdout": "", "stderr": f"Script not found: {script_path}",
        }

    backup = patch_csv(csv_path) if csv_path.exists() else None

    extra = SCRIPT_EXTRA_ARGS.get(script, [])
    cmd = [sys.executable, "-u", str(script_path), "--profile", profile] + extra

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        stdout, stderr, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        stdout, stderr, rc = "", "TIMEOUT (600s)", -1
    except Exception as exc:
        elapsed = time.time() - t0
        stdout, stderr, rc = "", str(exc), -2
    finally:
        if backup is not None:
            restore_csv(csv_path, backup)

    return {
        "label": label,
        "rc": rc,
        "elapsed": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

_RESULT_KEYWORDS = (
    "t_cond", "degc", "\u00b0c", "temperatura", "conductor", "t_max",
    "pinn", "fem", "iec", "t_cond ref", "dt total", "error", "peor",
    "case a", "case b", "summary", "worst", "resumen",
)


def extract_result_lines(output: str) -> list[str]:
    """Return the last result-bearing lines from script stdout."""
    lines = [
        line.strip()
        for line in output.splitlines()
        if any(kw in line.lower() for kw in _RESULT_KEYWORDS) and line.strip()
    ]
    return lines[-18:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Validar ejemplos PINN (200 Adam, 0 L-BFGS por defecto).\n"
            "Sin argumentos ejecuta todos los ejemplos disponibles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "folders",
        nargs="*",
        metavar="FOLDER_O_PERFIL",
        help=(
            "Carpetas a ejecutar, con perfil opcional por carpeta. "
            "Formato: CARPETA [PERFIL] CARPETA [PERFIL] ... "
            "Si no se indica perfil tras una carpeta se usa el defecto (quick). "
            "Ejemplo: kim_2024_154kv_bedding research aras_2005_154kv"
        ),
    )
    ap.add_argument(
        "--profile", "-p",
        default="quick",
        metavar="PROFILE",
        help="Perfil de solver a pasar a todos los scripts (default: quick).",
    )
    ap.add_argument(
        "--no-fast",
        action="store_true",
        help="No parchear solver_params.csv; usa la configuracion real.",
    )
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    default_profile: str = args.profile
    fast: bool = not args.no_fast

    if args.folders:
        folder_specs = _parse_folder_args(args.folders, default_profile)
    else:
        folder_specs = None

    jobs = discover_jobs(folder_specs, default_profile=default_profile)

    if not jobs:
        available = sorted(d.name for d in EXAMPLES_DIR.iterdir() if d.is_dir())
        print("No se encontraron scripts para los filtros especificados.")
        print(f"  Carpetas disponibles: {available}")
        sys.exit(1)

    SEP = "=" * 72
    fast_note = "adam_steps=200, lbfgs_steps=0" if fast else "SIN parchear CSV"
    profiles_used = "/".join(sorted({p for _, _, p in jobs}))
    print(SEP)
    print(
        f"  VALIDACION DE EJEMPLOS  |  perfil(es)={profiles_used}  "
        f"|  {len(jobs)} script(s)  |  {fast_note}"
    )
    print(SEP)

    results: list[dict] = []
    for idx, (example_dir, script, job_profile) in enumerate(jobs, 1):
        label = f"{example_dir}/{script}"
        extra_str = " ".join(SCRIPT_EXTRA_ARGS.get(script, []))
        extra_display = f"  {extra_str}" if extra_str else ""
        print(f"\n{'-' * 72}")
        print(f"  [{idx}/{len(jobs)}] {label}  --profile {job_profile}{extra_display}")
        print(f"{'-' * 72}", flush=True)

        if fast:
            info = run_script(example_dir, script, job_profile)
        else:
            script_path = EXAMPLES_DIR / example_dir / script
            extra = SCRIPT_EXTRA_ARGS.get(script, [])
            cmd = [sys.executable, "-u", str(script_path), "--profile", job_profile] + extra
            t0 = time.time()
            try:
                res = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT),
                )
                info = {
                    "label": label, "rc": res.returncode,
                    "elapsed": time.time() - t0,
                    "stdout": res.stdout, "stderr": res.stderr,
                }
            except Exception as exc:
                info = {
                    "label": label, "rc": -2, "elapsed": time.time() - t0,
                    "stdout": "", "stderr": str(exc),
                }

        info["profile"] = job_profile
        results.append(info)

        status = "OK" if info["rc"] == 0 else f"FAIL (rc={info['rc']})"
        print(f"  Estado: {status}  ({info['elapsed']:.1f}s)")

        if info["rc"] == 0:
            res_lines = extract_result_lines(info["stdout"])
            if res_lines:
                print("  Resultados:")
                for ln in res_lines:
                    print(f"    {ln}")
            else:
                print("  (sin lineas de resultado reconocibles en stdout)")
        else:
            print("  Error (ultimas lineas):")
            for ln in info["stderr"].strip().splitlines()[-20:]:
                print(f"    {ln}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("  RESUMEN COMPARATIVO")
    print(SEP)

    w  = max(len(r["label"]) for r in results) + 2
    pw = 10  # profile column width
    print(f"  {'Script':<{w}} {'Perfil':<{pw}} {'Estado':>8}  {'Tiempo':>7}")
    print(f"  {'-'*w} {'-'*pw} {'-'*8}  {'-'*7}")
    for r in results:
        sym = "OK    " if r["rc"] == 0 else "FAIL  "
        print(f"  {r['label']:<{w}} {r['profile']:<{pw}} {sym:>8}  {r['elapsed']:6.1f}s")

    ok_n  = sum(1 for r in results if r["rc"] == 0)
    bad_n = len(results) - ok_n
    total_t = sum(r["elapsed"] for r in results)
    print(f"\n  Total: {ok_n} OK, {bad_n} FAIL de {len(results)}  ({total_t:.1f}s)")

    if bad_n:
        print("\n  Scripts con error:")
        for r in results:
            if r["rc"] != 0:
                print(f"    * {r['label']}  rc={r['rc']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
