"""Phase 7: aggregate the PINN studies already on disk into one table.

Only reads results/phase*/*/summary.json; it never retrains and never
touches the source folders. Output goes to a new folder.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

COLS = ["study", "case", "level", "architecture", "seed", "status", "acceptance",
        "nrmse_pct", "error_rise_pct", "error_Tmax_abs_K", "max_error_K",
        "worst_region_rmse_rise_pct", "balance_global_pct",
        "interface_T_jump_max_K", "interface_flux_jump_rms_pct",
        "final_loss", "parameters", "elapsed_s", "failed_gates"]


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("phase*/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "rows" not in data:  # FEM studies have no PINN rows
            continue
        for r in data["rows"]:
            out = {k: r.get(k) for k in COLS if k not in ("study", "failed_gates")}
            out["study"] = path.parent.name
            out["failed_gates"] = ",".join(k for k, ok in (r.get("gates") or {}).items() if not ok)
            rows.append(out)
    return rows


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "all_runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    lines = ["# Fase 7 - Resumen agregado de los estudios PINN", "",
             "Generado por `pinn_v2.experiments.aggregate` a partir de los `summary.json` existentes.",
             "Los criterios de aceptacion son propuestos, no confirmados por el usuario.", "",
             "| estudio | caso | arq. | sem. | NRMSE % | dT % | Tmax K | balance % | veredicto | gates fallidos |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[k]) for k in (
            "study", "case", "architecture", "seed", "nrmse_pct", "error_rise_pct",
            "error_Tmax_abs_K", "balance_global_pct", "acceptance", "failed_gates")) + " |")
    n_ok = sum(1 for r in rows if r["acceptance"] == "accepted")
    lines += ["", f"Corridas: {len(rows)}; aceptadas: {n_ok}."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "results"
    rows = collect(root)
    write_outputs(rows, root / "phase7" / "aggregate")
    print(f"{len(rows)} runs -> {root / 'phase7' / 'aggregate'}")


if __name__ == "__main__":
    main()