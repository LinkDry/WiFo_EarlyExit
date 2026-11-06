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
    val = rows("05_val_summary.csv")
    costs = rows("05_costs.csv")
    aggregate = rows("05_aggregate.csv")
    equal = rows("05_equal_cost.csv")
    analysis = json.loads((R / "05_analysis.json").read_text(encoding="utf-8"))
    assert len(val) == 640, len(val)
    assert len(costs) == 20, len(costs)
    assert len(aggregate) == 40, len(aggregate)
    assert len(equal) == 12, len(equal)
    assert len(analysis) == 8, len(analysis)
    assert sum(row["source"] == "task05" for row in val) == 448
    assert sum(row["source"].startswith("task03") for row in val) == 192
    persample = np.load(R / "05_persample.npz")
    assert len(persample.files) == 640
    lookup = {}
    for row in val:
        key = f"untrained/{row['backbone']}/e{row['enc_depth']}d{row['dec_depth']}/{row['dataset']}/{row['task']}/0.5/val/seed100"
        values = persample[key]
        assert len(values) == 2000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
        lookup[(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["dataset"], row["task"])] = row
    for row in aggregate:
        selected = [lookup[(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), f"D{index}", row["task"])] for index in range(1, 17)]
        full = [lookup[(row["backbone"], 6, 4, f"D{index}", row["task"])] for index in range(1, 17)]
        db = float(np.mean([float(item["nmse_db"]) for item in selected]))
        full_db = float(np.mean([float(item["nmse_db"]) for item in full]))
        assert math.isclose(db, float(row["nmse_db_mean"]), rel_tol=1e-12)
        assert math.isclose(db - full_db, float(row["degradation_db"]), rel_tol=1e-12)
    for row in analysis:
        largest = row["largest_jump"]
        if largest["increment_db"] <= 0.25:
            assert row["status"] == "no_clear_knee" and row["candidate"] is None
        elif not row["monotonic"]:
            assert row["status"] == "nonmonotonic" and row["candidate"] is None
        else:
            assert row["status"] == "candidate_knee" and row["candidate"] == largest["from"]
    assert (R / "05_depth_curves.png").stat().st_size > 10000
    assert (R / "05_report.md").is_file()
    print(f"PASS Task 05 validation: val={len(val)} costs={len(costs)} aggregate={len(aggregate)} equal_cost={len(equal)} knees={len(analysis)} persample={len(persample.files)}")


if __name__ == "__main__":
    main()
