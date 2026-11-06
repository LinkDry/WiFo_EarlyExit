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
    val = rows("12_val_summary.csv")
    training = rows("12_training.csv")
    costs = rows("12_costs.csv")
    controls = rows("12_controls.csv")
    assert len(val) == 384, len(val)
    assert len(costs) == 12, len(costs)
    assert len(controls) == 12, len(controls)
    assert len({(row["mode"], row["exit_key"], row["train_seed"]) for row in training}) == 12
    arrays = np.load(R / "12_persample.npz")
    assert len(arrays.files) == 384
    for row in val:
        key = f"{row['mode']}/small/e{row['enc_depth']}d{row['dec_depth']}/train{row['train_seed']}/mask0.5/{row['dataset']}/{row['task']}/val/eval100"
        values = arrays[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
    for row in controls:
        assert math.isclose(float(row["trunc_h1_db"]) - float(row["probe_db"]), float(row["probe_gain_vs_h1_db"]), rel_tol=1e-12)
        assert math.isclose(float(row["probe_db"]) - float(row["full_update_db"]), float(row["residual_depth_cost_db"]), rel_tol=1e-12)
    for mode, enc, dec, seed in [(m, 6, 4, s) for m in ("full_enc_update", "full_dec_update") for s in (100, 200, 300)] + [("probe", e, d, s) for e, d in ((3, 4), (6, 2)) for s in (100, 200, 300)]:
        state = torch.load(W / f"checkpoints/12_small_{mode}_e{enc}d{dec}_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert state["backbone"] == "small" and state["mode"] == mode and state["train_seed"] == seed
    assert (R / "12_recovery.png").stat().st_size > 10000
    assert (R / "12_report.md").is_file()
    print(f"PASS Task 12 validation: val={len(val)} costs={len(costs)} controls={len(controls)}")


if __name__ == "__main__":
    main()
