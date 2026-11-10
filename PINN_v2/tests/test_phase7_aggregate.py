"""Phase 7 aggregator: reads existing summaries and writes a new folder."""
import json

from pinn_v2.experiments import aggregate


def test_collect_and_write(tmp_path):
    study = tmp_path / "phase9" / "s1"
    study.mkdir(parents=True)
    row = {"case": "c", "architecture": "M1", "seed": 0, "acceptance": "rejected",
           "nrmse_pct": 7.0, "gates": {"nrmse_pct": False, "balance_pct": True}}
    (study / "summary.json").write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    fem = tmp_path / "phase9" / "fem"
    fem.mkdir()
    (fem / "summary.json").write_text(json.dumps({"cases": {}}), encoding="utf-8")
    rows = aggregate.collect(tmp_path)
    assert len(rows) == 1 and rows[0]["failed_gates"] == "nrmse_pct"
    aggregate.write_outputs(rows, tmp_path / "out")
    assert "rejected" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")