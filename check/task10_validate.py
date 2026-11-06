import csv
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
    val = rows("10_val_summary.csv")
    training = rows("10_training.csv")
    costs = rows("10_costs.csv")
    curves = rows("10_curves.csv")
    selection = rows("10_selection.csv")
    assert len(val) == 768, len(val)
    assert len(costs) == 24, len(costs)
    assert len(curves) == 72, len(curves)
    assert len(selection) == 36, len(selection)
    assert len({(row["backbone"], row["exit_key"], row["train_seed"]) for row in training}) == 24
    arrays = np.load(R / "10_persample.npz")
    assert len(arrays.files) == 768
    for row in val:
        key = f"h1/{row['backbone']}/e{row['enc_depth']}d{row['dec_depth']}/train{row['train_seed']}/mask0.5/{row['dataset']}/{row['task']}/val/eval100"
        values = arrays[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
    for row in curves:
        assert math.isclose(float(row["trunc_h1_db"]) - float(row["full_h1_db"]), float(row["depth_cost_db"]), rel_tol=1e-12)
    for row in selection:
        if row["feasible"] == "False":
            assert row["selected_enc_depth"] == ""
    for backbone, enc, dec, seed in [(b, e, d, s) for b in ("base", "small") for e, d in ((5, 4), (4, 4), (6, 3)) for s in (100, 200, 300)] + [("small", e, d, s) for e, d in ((6, 4), (3, 4), (6, 2)) for s in (200, 300)]:
        state = torch.load(W / f"checkpoints/10_{backbone}_e{enc}d{dec}_h1_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert state["backbone"] == backbone and state["enc_depth"] == enc and state["dec_depth"] == dec and state["train_seed"] == seed
    assert (R / "10_intermediate.png").stat().st_size > 10000
    assert (R / "10_report.md").is_file()
    print(f"PASS Task 10 validation: val={len(val)} costs={len(costs)} curves={len(curves)} selection={len(selection)}")


if __name__ == "__main__":
    main()
