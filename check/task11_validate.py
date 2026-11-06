import csv
import math
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
R = W / "results"


def rows(name):
    with (R / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    pareto = rows("11_pareto.csv")
    costs = rows("11_costs.csv")
    review = rows("11_selection_review.csv")
    assert len(pareto) == 96, len(pareto)
    assert len(costs) == 48, len(costs)
    assert len(review) == 72, len(review)
    arrays = np.load(R / "11_persample.npz")
    assert len(arrays.files) == 96
    for row in pareto:
        key = f"{row['name']}/{row['dataset']}/{row['task']}/0.5/test/eval100"
        values = arrays[key]
        assert len(values) == 1000 and np.isfinite(values).all()
        linear = float(values.mean())
        assert math.isclose(linear, float(row["nmse_lin"]), rel_tol=1e-12)
        assert math.isclose(10 * math.log10(linear), float(row["nmse_db"]), rel_tol=1e-12)
        cost = next(item for item in costs if item["name"] == row["name"] and item["shape"] == row["dataset"])
        assert math.isclose(float(cost["gflops"]), float(row["gflops"]), rel_tol=1e-12)
    for row in review:
        delta = float(row["selected_nmse_db"]) - float(row["full_h1_nmse_db"])
        assert math.isclose(delta, float(row["review_delta_db"]), rel_tol=1e-12)
        assert (delta <= float(row["epsilon_db"]) + 1e-12) == (row["constraint_satisfied"] == "True")
    assert (R / "11_pareto.png").stat().st_size > 10000
    assert (R / "11_report.md").is_file()
    print(f"PASS Task 11 validation: pareto={len(pareto)} costs={len(costs)} review={len(review)} persample={len(arrays.files)}")


if __name__ == "__main__":
    main()
