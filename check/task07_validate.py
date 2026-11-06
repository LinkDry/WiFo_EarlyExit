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
    val = rows("07_val_summary.csv")
    training = rows("07_training.csv")
    costs = rows("07_costs.csv")
    controls = rows("07_controls.csv")
    analysis = json.loads((R / "07_analysis.json").read_text(encoding="utf-8"))
    assert len(val) == 352, len(val)
    assert len(costs) == 11, len(costs)
    assert len(controls) == 16, len(controls)
    assert len(analysis) == 4
    assert len({(row["backbone"], row["exit_key"]) for row in training}) == 7
    persample = np.load(R / "07_persample.npz")
    assert len(persample.files) == len(val)
    for row in val:
        key = f"{row['mode']}/{row['backbone']}/e{row['enc_depth']}d{row['dec_depth']}/{row['dataset']}/{row['task']}/0.5/val/seed100"
        values = persample[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
    for row in controls:
        assert math.isclose(float(row["full_db"]) - float(row["full_h1_db"]), float(row["generic_gain_db"]), rel_tol=1e-12)
        assert math.isclose(float(row["untrained_db"]) - float(row["trunc_h1_db"]), float(row["total_recovery_db"]), rel_tol=1e-12)
        assert math.isclose(float(row["trunc_h1_db"]) - float(row["full_h1_db"]), float(row["depth_cost_db"]), rel_tol=1e-12)
        assert math.isclose(float(row["total_recovery_db"]) - float(row["generic_gain_db"]), float(row["specific_recovery_db"]), rel_tol=1e-12)
    for record in analysis:
        increments = [later["depth_cost_db"] - earlier["depth_cost_db"] for earlier, later in zip(record["records"], record["records"][1:])]
        assert record["narrower_more_degradation"] == all(value >= -0.01 for value in increments)
    for backbone, enc, dec in (("little", 6, 4), ("little", 3, 4), ("little", 6, 2), ("tiny", 6, 4), ("tiny", 3, 4), ("tiny", 6, 2), ("small", 6, 2)):
        state = torch.load(W / f"checkpoints/07_{backbone}_e{enc}d{dec}_h1.pt", map_location="cpu", weights_only=False)
        assert state["backbone"] == backbone and state["enc_depth"] == enc and state["dec_depth"] == dec
        assert set(state["best"]) == {"head"}
    assert (R / "07_width.png").stat().st_size > 10000
    print(f"PASS Task 07 validation: val={len(val)} costs={len(costs)} controls={len(controls)} persample={len(persample.files)} width_cases={len(analysis)}")


if __name__ == "__main__":
    main()
