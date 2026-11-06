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
    analysis = json.loads((R / "06_analysis.json").read_text(encoding="utf-8"))
    triggered = sum(record["interface_trigger"] for record in analysis.values())
    val = rows("06_val_summary.csv")
    training = rows("06_training.csv")
    costs = rows("06_costs.csv")
    controls = rows("06_controls.csv")
    assert len(val) == 96 + 32 * triggered, len(val)
    assert len(costs) == 3 + triggered, len(costs)
    assert len(controls) == 4
    assert len({(row["exit_key"], row["head_kind"]) for row in training}) == 3 + triggered
    persample = np.load(R / "06_persample.npz")
    assert len(persample.files) == len(val)
    for row in val:
        key = f"{row['mode']}/base/e{row['enc_depth']}d{row['dec_depth']}/{row['dataset']}/{row['task']}/0.5/val/seed100"
        values = persample[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
    for point, record in analysis.items():
        assert record["interface_trigger"] == (max(record["probe_gain_db"].values()) > 0.05)
    for row in controls:
        assert math.isclose(float(row["output_h1_db"]) - float(row["probe_db"]), float(row["probe_gain_db"]), rel_tol=1e-12)
        assert math.isclose(float(row["probe_db"]) - float(row["full_h1_db"]), float(row["residual_depth_cost_db"]), rel_tol=1e-12)
        triggered_here = analysis[f"e{row['enc_depth']}d{row['dec_depth']}"]["interface_trigger"]
        if triggered_here:
            assert math.isclose(float(row["output_h1_db"]) - float(row["interface_db"]), float(row["interface_gain_db"]), rel_tol=1e-12)
        else:
            assert row["interface_db"] == "" and row["interface_gain_db"] == ""
    for enc, dec in ((3, 4), (6, 2)):
        modes = ["probe"] + (["interface"] if analysis[f"e{enc}d{dec}"]["interface_trigger"] else [])
        if (enc, dec) == (6, 2):
            modes.append("output_h1")
        for mode in modes:
            state = torch.load(W / f"checkpoints/06_base_e{enc}d{dec}_{mode}.pt", map_location="cpu", weights_only=False)
            assert state["enc_depth"] == enc and state["dec_depth"] == dec and state["mode"] == mode
            assert "best" in state and "current" in state
            assert "model" not in state and "backbone" not in state
    print(f"PASS Task 06 validation: val={len(val)} costs={len(costs)} controls={len(controls)} persample={len(persample.files)} interface_triggered={triggered}")


if __name__ == "__main__":
    main()
