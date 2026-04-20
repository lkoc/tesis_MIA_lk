"""Boundary-condition report: console + text file.

Provides a single public function :func:`write_bc_report` that summarises
the boundary conditions loaded from CSV (including any attached
:class:`~pinn_cables.physics.ground_temp.GroundTempProfile`) and writes
the report both to *stdout* and to ``<out_dir>/bc_report.txt``.

Typical usage in a run script::

    problem = load_problem(DATA_DIR)
    write_bc_report(problem, RESULTS_DIR, label="Kim 2024 — summer")
    # ... rest of main() ...
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinn_cables.io.readers import ProblemDefinition

# Column widths for the table
_W_EDGE = 8
_W_TYPE = 12
_W_VAL  = 14
_W_H    = 10
_W_PROF = 40


def _profile_summary(profile) -> str:
    """One-line description of a GroundTempProfile (or None)."""
    if profile is None:
        return "scalar (CSV value)"

    cls = type(profile).__name__

    if cls == "ConstantProfile":
        return f"ConstantProfile  T={profile.T_K - 273.15:.2f} °C"

    if cls == "PiecewiseLinearProfile":
        knots = profile.knots
        depths = [f"{z:.1f}" for z, _ in knots]
        temps  = [f"{T - 273.15:.1f}" for _, T in knots]
        return (
            f"PiecewiseLinear  z=[{', '.join(depths)}] m  "
            f"T=[{', '.join(temps)}] °C"
        )

    if cls == "CosineGroundProfile":
        return (
            f"CosineGround  Tg={profile.T_g - 273.15:.1f} °C  "
            f"As={profile.A_s:.1f} K  tp={profile.tp:.0f}d  "
            f"D={profile.damping_depth():.2f} m  "
            f"T_surf={profile.T_surface() - 273.15:.1f} °C"
        )

    # Generic fallback
    return repr(profile)


def _format_value(bc) -> str:
    """Format the scalar BC value as 'XXX.XX K (YY.YY °C)'."""
    v = bc.value
    if v == 0.0:
        return "T_amb  (fallback)"
    return f"{v:.2f} K  ({v - 273.15:.2f} °C)"


def _build_report(problem: "ProblemDefinition", label: str) -> str:
    """Build the full report string without printing."""
    dom = problem.domain
    scenario = problem.scenarios[0] if problem.scenarios else None

    lines: list[str] = []
    sep  = "=" * 72
    sep2 = "-" * 72

    lines.append(sep)
    lines.append("  CONDICIONES DE FRONTERA  —  " + label)
    lines.append(sep)

    # ----------------------------------------------------------------
    # Domain
    # ----------------------------------------------------------------
    lines.append(f"  Dominio : x ∈ [{dom.xmin:.1f}, {dom.xmax:.1f}] m"
                 f"   y ∈ [{dom.ymin:.1f}, {dom.ymax:.1f}] m")
    if scenario is not None:
        lines.append(f"  Escenario : {scenario.scenario_id}  ({scenario.mode})")
        lines.append(
            f"  T_amb     : {scenario.T_amb:.2f} K  "
            f"({scenario.T_amb - 273.15:.2f} °C)  "
            f"|  k_suelo = {scenario.k_soil:.3f} W/(m·K)"
        )

    # ----------------------------------------------------------------
    # Boundary table header
    # ----------------------------------------------------------------
    lines.append("")
    hdr = (
        f"  {'Frontera':<{_W_EDGE}}  "
        f"{'Tipo':<{_W_TYPE}}  "
        f"{'Valor CSV':<{_W_VAL}}  "
        f"{'h [W/m²K]':<{_W_H}}  "
        f"{'Perfil adjunto'}"
    )
    lines.append(hdr)
    lines.append("  " + sep2)

    order = ["top", "bottom", "left", "right"]
    bcs = problem.bcs
    shown: set[str] = set()

    for edge in order:
        if edge not in bcs:
            continue
        shown.add(edge)
        bc = bcs[edge]
        h_str = f"{bc.h:.3f}" if bc.bc_type == "robin" else "—"
        val_str = _format_value(bc) if bc.bc_type in ("dirichlet", "robin") else f"{bc.value:.4f} W/m²"
        prof_str = _profile_summary(bc.profile)
        lines.append(
            f"  {edge:<{_W_EDGE}}  "
            f"{bc.bc_type:<{_W_TYPE}}  "
            f"{val_str:<{_W_VAL}}  "
            f"{h_str:<{_W_H}}  "
            f"{prof_str}"
        )

    # Any extra edges not in the canonical order
    for edge, bc in bcs.items():
        if edge in shown:
            continue
        h_str = f"{bc.h:.3f}" if bc.bc_type == "robin" else "—"
        val_str = _format_value(bc) if bc.bc_type in ("dirichlet", "robin") else f"{bc.value:.4f} W/m²"
        prof_str = _profile_summary(bc.profile)
        lines.append(
            f"  {edge:<{_W_EDGE}}  "
            f"{bc.bc_type:<{_W_TYPE}}  "
            f"{val_str:<{_W_VAL}}  "
            f"{h_str:<{_W_H}}  "
            f"{prof_str}"
        )

    lines.append("  " + sep2)

    # ----------------------------------------------------------------
    # Per-profile detail block (only when profiles are attached)
    # ----------------------------------------------------------------
    profiled = [(e, bcs[e]) for e in bcs if bcs[e].profile is not None]
    if profiled:
        lines.append("")
        lines.append("  Detalle de perfiles activos:")
        for edge, bc in profiled:
            cls = type(bc.profile).__name__
            lines.append(f"    [{edge}]  {cls}")

            if cls == "PiecewiseLinearProfile":
                lines.append(f"    {'  z [m]':>10}   {'T [K]':>8}   {'T [°C]':>8}")
                for z, T in bc.profile.knots:
                    lines.append(f"    {z:>10.3f}   {T:>8.3f}   {T - 273.15:>8.3f}")

            elif cls == "CosineGroundProfile":
                p = bc.profile
                lines.append(
                    f"    Tg={p.T_g:.3f} K  As={p.A_s:.2f} K  "
                    f"tau={p.tau:.0f} d  alpha={p.alpha:.4f} m²/d  "
                    f"D={p.damping_depth():.3f} m"
                )
                lines.append(
                    f"    t0={p.t0:.0f} d  tp={p.tp:.0f} d  "
                    f"T_surface={p.T_surface() - 273.15:.2f} °C"
                )
                lines.append("    Perfil de profundidad:")
                lines.append(f"    {'  z [m]':>10}   {'T [K]':>8}   {'T [°C]':>8}")
                for z in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                    T = p.T_at_depth(z)
                    lines.append(f"    {z:>10.2f}   {T:>8.3f}   {T - 273.15:>8.3f}")

    # ----------------------------------------------------------------
    # Summary note
    # ----------------------------------------------------------------
    lines.append("")
    n_profiled = len(profiled)
    if n_profiled:
        lines.append(
            f"  NOTA: {n_profiled} frontera(s) usan perfil T(z) cargado de "
            "boundary_profiles.csv (sobreescribe valor CSV)."
        )
    else:
        lines.append(
            "  NOTA: todas las fronteras usan valores escalares del CSV."
        )
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def write_bc_report(
    problem: "ProblemDefinition",
    out_dir: str | Path,
    label: str = "caso simulado",
    filename: str = "bc_report.txt",
) -> str:
    """Print a boundary-condition report and save it to *out_dir/filename*.

    Writes identical content to *stdout* and to a text file so that both
    the interactive terminal and post-run analysis have the information.

    Args:
        problem:  Loaded :class:`~pinn_cables.io.readers.ProblemDefinition`.
        out_dir:  Results directory (created if absent).
        label:    One-line description shown in the report header.
        filename: Output file name inside *out_dir* (default ``bc_report.txt``).

    Returns:
        The full report string (same text as printed and saved).
    """
    report = _build_report(problem, label)

    # --- terminal ---
    import sys
    sys.stdout.buffer.write(report.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()

    # --- file ---
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(report, encoding="utf-8")

    return report
