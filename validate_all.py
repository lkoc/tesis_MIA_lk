"""Validación rápida de todos los ejemplos con pasos mínimos.

Parcha temporalmente solver_params.csv de cada ejemplo para usar
200 Adam + 0 L-BFGS, ejecuta el script y restaura el CSV original.
"""
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXAMPLES = [
    ("xlpe_single_cable",       "run_example.py",      "quick"),
    ("aras_2005_154kv",         "run_example.py",      "quick"),
    ("aras_2005_154kv_flat",    "run_example.py",      "quick"),
    ("xlpe_trefoil",            "run_example.py",      "quick"),
    ("xlpe_three_trefoils",     "run_example.py",      "quick"),
    ("kim_2024_154kv_bedding",  "run_example.py",      "quick"),
    ("kim_2024_154kv_bedding",  "run_research_pac.py", "quick"),
]

FAST_OVERRIDES = {
    "adam_steps": "200",
    "lbfgs_steps": "0",
    "print_every": "100",
    "save_every": "0",
}


def patch_csv(csv_path: Path) -> Path:
    """Backup CSV and write patched version with minimal steps."""
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip() in FAST_OVERRIDES:
                row[1] = FAST_OVERRIDES[row[0].strip()]
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return backup


def restore_csv(csv_path: Path, backup: Path) -> None:
    shutil.move(str(backup), str(csv_path))


def run_example(example_dir: str, script: str, profile: str) -> dict:
    data_dir = ROOT / "examples" / example_dir / "data"
    csv_path = data_dir / "solver_params.csv"

    backup = patch_csv(csv_path)
    script_path = ROOT / "examples" / example_dir / script
    label = f"{example_dir}/{script}"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script_path), "--profile", profile],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per example
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        stdout = result.stdout
        stderr = result.stderr
        rc = result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        stdout = ""
        stderr = "TIMEOUT (300s)"
        rc = -1
    except Exception as e:
        elapsed = time.time() - t0
        stdout = ""
        stderr = str(e)
        rc = -2
    finally:
        restore_csv(csv_path, backup)

    return {
        "label": label,
        "rc": rc,
        "elapsed": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


def extract_temps(output: str) -> str:
    """Extract conductor temperature lines from output."""
    lines = output.splitlines()
    temp_lines = []
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in [
            "t_cond", "degc", "°c", "temperatura", "conductor",
            "cable", "t_max", "pinn", "fem", "iec",
        ]):
            temp_lines.append(line.strip())
    # Also grab IEC reference
    for line in lines:
        if "T_cond ref" in line or "dT TOTAL" in line:
            temp_lines.append(line.strip())
    return "\n    ".join(temp_lines[-12:]) if temp_lines else "(no temp info found)"


def main():
    print("=" * 70)
    print("  VALIDACION RAPIDA DE TODOS LOS EJEMPLOS (200 Adam, 0 LBFGS)")
    print("=" * 70)

    results = []
    for example_dir, script, profile in EXAMPLES:
        label = f"{example_dir}/{script}"
        print(f"\n{'-' * 70}")
        print(f"  Ejecutando: {label} --profile {profile}")
        print(f"{'-' * 70}")

        info = run_example(example_dir, script, profile)
        results.append(info)

        status = "OK" if info["rc"] == 0 else f"FAIL (rc={info['rc']})"
        print(f"  Estado: {status}  ({info['elapsed']:.1f}s)")

        if info["rc"] == 0:
            temps = extract_temps(info["stdout"])
            print(f"  Temperaturas:\n    {temps}")
        else:
            # Show last 20 lines of stderr
            err_lines = info["stderr"].strip().splitlines()[-20:]
            print(f"  Error:\n    " + "\n    ".join(err_lines))

    # Summary
    print(f"\n{'=' * 70}")
    print("  RESUMEN")
    print(f"{'=' * 70}")
    ok_count = sum(1 for r in results if r["rc"] == 0)
    fail_count = len(results) - ok_count
    for r in results:
        sym = "OK" if r["rc"] == 0 else "FAIL"
        print(f"  {sym} {r['label']:50s} {r['elapsed']:6.1f}s  rc={r['rc']}")
    print(f"\n  Total: {ok_count} OK, {fail_count} FAIL de {len(results)}")


if __name__ == "__main__":
    main()
