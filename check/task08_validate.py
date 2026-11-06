import csv
import json
import math
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
R = W / "results"


def rows(name):
    with (R / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    val = rows("08_val_summary.csv")
    aggregate = rows("08_aggregate.csv")
    width = rows("08_width_summary.csv")
    probe = rows("08_probe_summary.csv")
    selection = rows("08_depth_selection.csv")
    config = rows("08_config_selection.csv")
    stability = rows("08_stability.csv")
    costs = rows("08_costs.csv")
    training = rows("08_training.csv")
    assert len(val) == 3264, len(val)
    assert len(aggregate) == 204, len(aggregate)
    assert len(width) == 48, len(width)
    assert len(probe) == 12, len(probe)
    assert len(selection) == 144, len(selection)
    assert len(config) == 2304, len(config)
    assert len(stability) == 576, len(stability)
    assert len(costs) == 26, len(costs)
    groups = {(row["backbone"], row["exit_key"], row["head_kind"], row["train_seed"], row["mask_ratio"]) for row in training}
    assert len(groups) == 78, len(groups)
    arrays = np.load(R / "08_persample.npz")
    assert len(arrays.files) == 3264
    for row in val:
        key = f"{row['mode']}/{row['backbone']}/e{row['enc_depth']}d{row['dec_depth']}/train{row['train_seed']}/mask{row['mask_ratio']}/{row['dataset']}/{row['task']}/val/eval100"
        values = arrays[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
    for row in aggregate:
        selected = [item for item in val if item["mode"] == row["method"] and item["backbone"] == row["backbone"] and int(item["enc_depth"]) == int(row["enc_depth"]) and int(item["dec_depth"]) == int(row["dec_depth"]) and int(item["train_seed"]) == int(row["train_seed"]) and float(item["mask_ratio"]) == float(row["mask_ratio"]) and item["task"] == row["task"]]
        assert len(selected) == 16
        assert math.isclose(float(np.mean([float(item["nmse_db"]) for item in selected])), float(row["nmse_db_mean"]), rel_tol=1e-12)
    for row in probe:
        assert math.isclose(float(row["output_h1_db"]) - float(row["probe_db"]), float(row["probe_gain_db"]), rel_tol=1e-12)
    for row in stability:
        assert math.isclose(float(row["p_e6d4"]) + float(row["p_e3d4"]) + float(row["p_e6d2"]) + float(row["p_no_feasible"]), 1.0, abs_tol=1e-12)
    analysis = json.loads((R / "08_analysis.json").read_text(encoding="utf-8"))
    assert len(analysis["width_trends"]) == 8
    for record in analysis["width_trends"]:
        assert record["monotonic_seed_count"] == sum(item["monotonic"] for item in record["per_seed"])
    assert (R / "08_width.png").stat().st_size > 10000
    assert (R / "08_depth_selection.png").stat().st_size > 10000
    assert (R / "08_report.md").is_file()
    print(f"PASS Task 08 validation: val={len(val)} aggregate={len(aggregate)} width={len(width)} probe={len(probe)} selection={len(selection)} config={len(config)} stability={len(stability)} models={len(groups)}")


if __name__ == "__main__":
    main()
