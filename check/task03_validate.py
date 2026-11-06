import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
RESULTS = W / "results"


def rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    required = [
        "03_training.csv", "03_val_summary.csv", "03_val_aggregate.csv",
        "03_persample.npz", "03_costs.csv", "03_pareto.csv",
        "03_h1_selection.json", "03_g1.png", "03_report.md",
    ]
    for name in required:
        assert (RESULTS / name).is_file(), f"missing {name}"
    training = rows("03_training.csv")
    val = rows("03_val_summary.csv")
    aggregate = rows("03_val_aggregate.csv")
    costs = rows("03_costs.csv")
    pareto = rows("03_pareto.csv")
    assert len(val) == 576, len(val)
    assert len(aggregate) == 36, len(aggregate)
    assert len(costs) == 8, len(costs)
    assert len(pareto) == 84, len(pareto)
    assert not {row["dataset"] for row in val} - {f"D{index}" for index in range(1, 17)}
    assert len({(row["method"], row["backbone"], row["enc_depth"], row["dec_depth"], row["dataset"], row["task"]) for row in val}) == 576
    persample = np.load(RESULTS / "03_persample.npz")
    assert len(persample.files) == 576
    val_lookup = {}
    for row in val:
        key = f"{row['method']}/{row['backbone']}/e{row['enc_depth']}d{row['dec_depth']}/{row['dataset']}/{row['task']}/0.5/val"
        assert key in persample
        values = persample[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
        val_lookup[(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["dataset"], row["task"])] = row
    for row in aggregate:
        enc, dec = int(row["enc_depth"]), int(row["dec_depth"])
        selected = [val_lookup[(row["method"], row["backbone"], enc, dec, f"D{index}", row["task"])] for index in range(1, 17)]
        linear_mean = float(np.mean([float(item["nmse_lin"]) for item in selected]))
        db_mean = float(np.mean([float(item["nmse_db"]) for item in selected]))
        assert math.isclose(linear_mean, float(row["nmse_lin_mean"]), rel_tol=1e-12)
        assert math.isclose(db_mean, float(row["nmse_db_mean"]), rel_tol=1e-12)
        official = [val_lookup[("official", row["backbone"], 6, 4, f"D{index}", row["task"])] for index in range(1, 17)]
        official_db = float(np.mean([float(item["nmse_db"]) for item in official]))
        assert math.isclose(db_mean - official_db, float(row["degradation_from_full_db"]), rel_tol=1e-12)
        if row["method"] in ("h0", "h1"):
            untrained = [val_lookup[("untrained", row["backbone"], enc, dec, f"D{index}", row["task"])] for index in range(1, 17)]
            untrained_db = float(np.mean([float(item["nmse_db"]) for item in untrained]))
            assert math.isclose(untrained_db - db_mean, float(row["recovery_db"]), rel_tol=1e-12)
    groups = {(row["backbone"], row["head_kind"], row["exit_key"]) for row in training}
    assert len(groups) == 8, groups
    assert all(1 <= int(row["epoch"]) <= 10 for row in training)
    selection = json.loads((RESULTS / "03_h1_selection.json").read_text(encoding="utf-8"))
    assert set(selection) == {"base", "small"}
    for backbone, record in selection.items():
        expected = min(record["candidates"], key=lambda row: (row["score_db_mean"], row["gflops"]))
        assert record["selected"] == expected
        assert any(row["method"] == "h1" and row["backbone"] == backbone and int(row["enc_depth"]) == expected["enc_depth"] and int(row["dec_depth"]) == expected["dec_depth"] for row in costs)
    for row in pareto:
        if row["required_db"] == "":
            assert row["gap_db"] == ""
        else:
            assert math.isclose(float(row["gap_db"]), float(row["nmse_db"]) - float(row["required_db"]), rel_tol=1e-12)
    for backbone in ("base", "small"):
        for kind in ("h0", "h1"):
            checkpoint = torch.load(W / f"checkpoints/03_{backbone}_{kind}.pt", map_location="cpu", weights_only=False)
            assert checkpoint["backbone"] == backbone and checkpoint["head_kind"] == kind
            assert "best_heads" in checkpoint and "heads" in checkpoint
            assert "blocks" not in checkpoint and "decoder_blocks" not in checkpoint
    assert (RESULTS / "03_g1.png").stat().st_size > 10000
    print(f"PASS Task 03 validation: training_groups={len(groups)} val={len(val)} aggregate={len(aggregate)} costs={len(costs)} pareto={len(pareto)} persample={len(persample.files)}")


if __name__ == "__main__":
    main()
