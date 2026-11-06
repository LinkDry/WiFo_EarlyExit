import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
R = W / "results"


def rows(name):
    with (R / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    analysis = json.loads((R / "04_analysis.json").read_text(encoding="utf-8"))
    kinds = ("h1", "h0") if analysis["h0_trigger"] else ("h1",)
    val = rows("04_val_summary.csv")
    training = rows("04_training.csv")
    costs = rows("04_costs.csv")
    controls = rows("04_controls.csv")
    assert len(val) == 64 * len(kinds), len(val)
    assert len(training) >= 2 and len({(row["backbone"], row["head_kind"]) for row in training}) == 2 * len(kinds)
    assert len(costs) == 2 * len(kinds), len(costs)
    assert len(controls) == 4
    persample = np.load(R / "04_persample.npz")
    assert len(persample.files) == len(val)
    for row in val:
        key = f"full_{row['head_kind']}/{row['backbone']}/e6d4/{row['dataset']}/{row['task']}/0.5/val/seed100"
        values = persample[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
    for backbone in ("base", "small"):
        for task in ("temporal", "fre"):
            generic = analysis["generic"][backbone][task]
            assert math.isclose(generic["full_db"] - generic["full_h1_db"], generic["generic_gain_db"], rel_tol=1e-12)
    expected_trigger = any(analysis["generic"][backbone][task]["generic_gain_db"] > 0.05 for backbone in ("base", "small") for task in ("temporal", "fre"))
    assert analysis["h0_trigger"] == expected_trigger
    for row in controls:
        generic = float(row["generic_gain_db"])
        recovery = float(row["total_recovery_db"])
        assert math.isclose(recovery - generic, float(row["specific_recovery_db"]), rel_tol=1e-12)
        assert math.isclose(float(row["trunc_h1_db"]) - float(row["full_h1_db"]), float(row["depth_cost_db"]), rel_tol=1e-12)
    for backbone in ("base", "small"):
        for kind in kinds:
            state = torch.load(W / f"checkpoints/04_{backbone}_{kind}.pt", map_location="cpu", weights_only=False)
            assert state["backbone"] == backbone and state["head_kind"] == kind
            assert set(state["best_heads"]) == {"e6d4"}
            assert "blocks" not in state and "decoder_blocks" not in state
    for name in ("04_training.csv", "04_val_summary.csv", "04_persample.npz", "04_costs.csv", "04_controls.csv", "04_analysis.json", "04_report.md"):
        assert (R / name).is_file(), name
    print(f"PASS Task 04 validation: kinds={kinds} val={len(val)} costs={len(costs)} controls={len(controls)} persample={len(persample.files)}")


if __name__ == "__main__":
    main()
