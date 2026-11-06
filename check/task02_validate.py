import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
RESULTS = W / "results"


def rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    required = [
        "02_baselines.csv", "02_costs.csv", "02_pareto.csv", "02_persample.npz",
        "02_pareto.png", "02_report.md",
    ]
    for name in required:
        assert (RESULTS / name).is_file(), f"missing {name}"
    baselines = rows("02_baselines.csv")
    costs = rows("02_costs.csv")
    pareto = rows("02_pareto.csv")
    assert len(baselines) == 156, len(baselines)
    assert len(costs) == 68, len(costs)
    assert len(pareto) > 0
    baseline_keys = set()
    persample = np.load(RESULTS / "02_persample.npz")
    for row in baselines:
        key = f"{row['model']}/{row['dataset']}/{row['task']}/{float(row['mask_ratio']):g}/{row['split']}"
        assert key not in baseline_keys, key
        baseline_keys.add(key)
        assert key in persample, key
        expected_count = 1000 if row["split"] == "test" else 2000
        assert len(persample[key]) == expected_count, key
        linear = float(np.mean(persample[key]))
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12), key
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12), key
    assert len(persample.files) == 156
    expected = {
        ("D17", "temporal"): -5.267876176216529,
        ("D17", "fre"): -6.307384,
        ("D18", "temporal"): -3.780606,
        ("D18", "fre"): -5.522577,
    }
    for (dataset, task), reference in expected.items():
        row = next(row for row in baselines if row["model"] == "base" and row["dataset"] == dataset and row["split"] == "test" and row["task"] == task)
        assert abs(float(row["nmse_db"]) - reference) < 0.01, (dataset, task, row["nmse_db"], reference)
    mask_rows = [row for row in baselines if float(row["mask_ratio"]) != 0.5 or row["split"] == "val"]
    assert len(mask_rows) == 12
    assert {(row["dataset"], row["split"]) for row in mask_rows} == {("D1", "val"), ("D16", "val")}
    cost_keys = set()
    for row in costs:
        key = (row["model"], row["shape"], row["enc_depth"], row["dec_depth"])
        assert key not in cost_keys, key
        cost_keys.add(key)
        for field in ("params_exec_range", "gflops_range", "gpu_b1_ms_range", "gpu_thr_b64_range", "gpu_thr_b256_range", "cpu_b1_ms_t1_range", "cpu_b1_ms_t4_range"):
            assert row[field] == "" or float(row[field]) >= 0, (key, field)
    for shape in ("D17", "D1"):
        official = next(row for row in costs if row["model"] == "base_official" and row["shape"] == shape)
        truncated = next(row for row in costs if row["model"] == "base" and row["shape"] == shape and row["enc_depth"] == "6" and row["dec_depth"] == "4")
        assert official["params_exec"] == truncated["params_exec"]
        assert math.isclose(float(official["gflops"]), float(truncated["gflops"]), rel_tol=1e-12)
    for row in pareto:
        if row["required_db"] == "":
            assert row["gap_db"] == ""
        else:
            assert math.isclose(float(row["gap_db"]), float(row["nmse_db_untrained"]) - float(row["required_db"]), rel_tol=1e-12)
    manifest = json.loads((W / "cache/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 50
    assert all(record["validation"]["max_abs_diff"] == 0 for record in manifest["files"])
    assert (RESULTS / "02_pareto.png").stat().st_size > 10000
    print(f"PASS Task 02 validation: baselines={len(baselines)} costs={len(costs)} pareto={len(pareto)} persample={len(persample.files)} cache=50")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise
