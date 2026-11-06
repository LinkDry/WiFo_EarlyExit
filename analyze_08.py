import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

import run_02
import run_03

W = Path(__file__).resolve().parent
R = W / "results"
WORK = R / "08_work"
BACKBONES = ("base", "small", "little", "tiny")
DEPTHS = ((6, 4), (3, 4), (6, 2))
SEEDS = (100, 200, 300)
MASKS = (0.5, 0.25)
EPSILONS = (0.25, 0.5, 1.0)
SAMPLE_SIZES = (16, 64, 256, 2000)


def evaluation_key(backbone, enc, dec, mode, train_seed, mask, dataset, task):
    return f"{mode}/{backbone}/e{enc}d{dec}/train{train_seed}/mask{mask}/{dataset}/{task}/val/eval100"


def aggregate(rows):
    output = []
    groups = {}
    for row in rows:
        key = (row["mode"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), int(row["train_seed"]), float(row["mask_ratio"]), row["task"])
        groups.setdefault(key, []).append(row)
    for (mode, backbone, enc, dec, seed, mask, task), selected in groups.items():
        assert len(selected) == 16, (mode, backbone, enc, dec, seed, mask, task, len(selected))
        output.append({
            "method": mode, "backbone": backbone, "enc_depth": enc, "dec_depth": dec,
            "train_seed": seed, "mask_ratio": mask, "task": task,
            "nmse_lin_mean": f"{np.mean([float(row['nmse_lin']) for row in selected]):.17g}",
            "nmse_db_mean": f"{np.mean([float(row['nmse_db']) for row in selected]):.17g}",
        })
    return output


def aggregate_lookup(aggregate_rows):
    return {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), int(row["train_seed"]), float(row["mask_ratio"]), row["task"]): row for row in aggregate_rows}


def width_summary(aggregate_rows, costs):
    lookup = aggregate_lookup(aggregate_rows)
    output = []
    for backbone in BACKBONES:
        for enc, dec in DEPTHS:
            for mask in MASKS:
                cost = next(row for row in costs if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and float(row["mask_ratio"]) == mask and row["mode"] == "h1")
                full_cost = next(row for row in costs if row["backbone"] == backbone and int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == 4 and float(row["mask_ratio"]) == mask and row["mode"] == "h1")
                for task in run_02.TASKS:
                    selected = [lookup[("h1", backbone, enc, dec, seed, mask, task)] for seed in SEEDS]
                    full = [lookup[("h1", backbone, 6, 4, seed, mask, task)] for seed in SEEDS]
                    depth_costs = [float(row["nmse_db_mean"]) - float(base["nmse_db_mean"]) for row, base in zip(selected, full)]
                    output.append({
                        "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "mask_ratio": mask, "task": task,
                        "seed_mean_db": f"{np.mean([float(row['nmse_db_mean']) for row in selected]):.17g}",
                        "seed_std_db": f"{np.std([float(row['nmse_db_mean']) for row in selected]):.17g}",
                        "seed_min_db": f"{min(float(row['nmse_db_mean']) for row in selected):.17g}",
                        "seed_max_db": f"{max(float(row['nmse_db_mean']) for row in selected):.17g}",
                        "depth_cost_mean_db": f"{np.mean(depth_costs):.17g}",
                        "depth_cost_std_db": f"{np.std(depth_costs):.17g}",
                        "cost_reduction_pct": f"{(float(full_cost['gflops']) - float(cost['gflops'])) / float(full_cost['gflops']) * 100:.17g}",
                    })
    return output


def width_analysis(aggregate_rows):
    lookup = aggregate_lookup(aggregate_rows)
    output = []
    for enc, dec in ((3, 4), (6, 2)):
        for mask in MASKS:
            for task in run_02.TASKS:
                per_seed = []
                for seed in SEEDS:
                    values = []
                    for backbone in BACKBONES:
                        trunc = lookup[("h1", backbone, enc, dec, seed, mask, task)]
                        full = lookup[("h1", backbone, 6, 4, seed, mask, task)]
                        values.append(float(trunc["nmse_db_mean"]) - float(full["nmse_db_mean"]))
                    increments = [later - earlier for earlier, later in zip(values, values[1:])]
                    per_seed.append({"train_seed": seed, "depth_costs": values, "monotonic": all(value >= -0.01 for value in increments)})
                output.append({
                    "enc_depth": enc, "dec_depth": dec, "mask_ratio": mask, "task": task,
                    "per_seed": per_seed,
                    "monotonic_seed_count": sum(row["monotonic"] for row in per_seed),
                    "monotonic_seed_fraction": sum(row["monotonic"] for row in per_seed) / len(per_seed),
                })
    return output


def probe_summary(aggregate_rows):
    lookup = aggregate_lookup(aggregate_rows)
    output = []
    for enc, dec in ((3, 4), (6, 2)):
        for seed in SEEDS:
            for task in run_02.TASKS:
                h1 = lookup[("h1", "base", enc, dec, seed, 0.5, task)]
                probe = lookup[("probe", "base", enc, dec, seed, 0.5, task)]
                full = lookup[("h1", "base", 6, 4, seed, 0.5, task)]
                output.append({
                    "backbone": "base", "enc_depth": enc, "dec_depth": dec, "train_seed": seed, "task": task,
                    "output_h1_db": h1["nmse_db_mean"], "probe_db": probe["nmse_db_mean"],
                    "probe_gain_db": f"{float(h1['nmse_db_mean']) - float(probe['nmse_db_mean']):.17g}",
                    "residual_depth_cost_db": f"{float(probe['nmse_db_mean']) - float(full['nmse_db_mean']):.17g}",
                })
    return output


def cost_lookup(costs):
    return {(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), float(row["mask_ratio"])): float(row["gflops"]) for row in costs if row["mode"] == "h1"}


def depth_selection(rows, aggregate_rows, costs):
    lookup = aggregate_lookup(aggregate_rows)
    cost = cost_lookup(costs)
    global_rows = []
    config_rows = []
    for backbone in BACKBONES:
        for seed in SEEDS:
            for mask in MASKS:
                for task in run_02.TASKS:
                    for epsilon in EPSILONS:
                        candidates = []
                        for enc, dec in DEPTHS:
                            row = lookup[("h1", backbone, enc, dec, seed, mask, task)]
                            full = lookup[("h1", backbone, 6, 4, seed, mask, task)]
                            degradation = float(row["nmse_db_mean"]) - float(full["nmse_db_mean"])
                            candidates.append((cost[(backbone, enc, dec, mask)], enc, dec, degradation))
                        feasible = [row for row in candidates if row[3] <= epsilon]
                        chosen = min(feasible) if feasible else None
                        global_rows.append({
                            "backbone": backbone, "train_seed": seed, "mask_ratio": mask, "task": task,
                            "epsilon_db": epsilon, "feasible": bool(chosen),
                            "selected_enc_depth": "" if chosen is None else chosen[1],
                            "selected_dec_depth": "" if chosen is None else chosen[2],
                            "selected_gflops": "" if chosen is None else f"{chosen[0]:.17g}",
                            "selected_degradation_db": "" if chosen is None else f"{chosen[3]:.17g}",
                        })
                        for dataset in run_02.DATASETS[:16]:
                            config_candidates = []
                            config_subset = [row for row in rows if row["backbone"] == backbone and int(row["train_seed"]) == seed and float(row["mask_ratio"]) == mask and row["mode"] == "h1" and row["dataset"] == dataset and row["task"] == task]
                            for enc, dec in DEPTHS:
                                row = next(row for row in config_subset if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec)
                                full = next(row for row in config_subset if int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == 4)
                                degradation = float(row["nmse_db"]) - float(full["nmse_db"])
                                config_candidates.append((cost[(backbone, enc, dec, mask)], enc, dec, degradation))
                            feasible = [row for row in config_candidates if row[3] <= epsilon]
                            chosen = min(feasible) if feasible else None
                            config_rows.append({
                                "backbone": backbone, "train_seed": seed, "mask_ratio": mask, "task": task,
                                "dataset": dataset, "epsilon_db": epsilon, "feasible": bool(chosen),
                                "selected_enc_depth": "" if chosen is None else chosen[1],
                                "selected_dec_depth": "" if chosen is None else chosen[2],
                                "selected_gflops": "" if chosen is None else f"{chosen[0]:.17g}",
                                "selected_degradation_db": "" if chosen is None else f"{chosen[3]:.17g}",
                            })
    return global_rows, config_rows


def stability_analysis(arrays, costs):
    cost = cost_lookup(costs)
    output = []
    rng = np.random.default_rng(20261031)
    for backbone in BACKBONES:
        for seed in SEEDS:
            for mask in MASKS:
                for task in run_02.TASKS:
                    grouped = {}
                    for enc, dec in DEPTHS:
                        grouped[(enc, dec)] = [arrays[evaluation_key(backbone, enc, dec, "h1", seed, mask, dataset, task)] for dataset in run_02.DATASETS[:16]]
                    for epsilon in EPSILONS:
                        for sample_size in SAMPLE_SIZES:
                            selections = []
                            selected_costs = []
                            for _ in range(200):
                                candidates = []
                                for enc, dec in DEPTHS:
                                    degradations = []
                                    for dataset_index in range(16):
                                        current = grouped[(enc, dec)][dataset_index]
                                        full = grouped[(6, 4)][dataset_index]
                                        indices = rng.integers(0, 2000, size=sample_size)
                                        current_nmse = 10 * math.log10(float(np.mean(current[indices])))
                                        full_nmse = 10 * math.log10(float(np.mean(full[indices])))
                                        degradations.append(current_nmse - full_nmse)
                                    candidates.append((cost[(backbone, enc, dec, mask)], enc, dec, float(np.mean(degradations))))
                                feasible = [row for row in candidates if row[3] <= epsilon]
                                chosen = min(feasible) if feasible else None
                                selections.append("none" if chosen is None else point_name(chosen[1], chosen[2]))
                                if chosen is not None:
                                    selected_costs.append(chosen[0])
                            counts = Counter(selections)
                            output.append({
                                "backbone": backbone, "train_seed": seed, "mask_ratio": mask, "task": task,
                                "epsilon_db": epsilon, "sample_size": sample_size,
                                "p_e6d4": counts["e6d4"] / 200, "p_e3d4": counts["e3d4"] / 200,
                                "p_e6d2": counts["e6d2"] / 200, "p_no_feasible": counts["none"] / 200,
                                "mean_selected_gflops": "" if not selected_costs else f"{np.mean(selected_costs):.17g}",
                            })
    return output


def point_name(enc, dec):
    return f"e{enc}d{dec}"


def make_width_plot(analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_index, mask in enumerate(MASKS):
        for col_index, task in enumerate(run_02.TASKS):
            axis = axes[row_index, col_index]
            for enc, dec in ((3, 4), (6, 2)):
                record = next(row for row in analysis if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and float(row["mask_ratio"]) == mask and row["task"] == task)
                values = np.array([row["depth_costs"] for row in record["per_seed"]])
                axis.errorbar(BACKBONES, values.mean(axis=0), yerr=values.std(axis=0), marker="o", capsize=4, label=f"e{enc}d{dec}")
            axis.set_title(f"mask={mask} / {task}")
            axis.set_ylabel("Depth cost under same H1 (dB)")
            axis.grid(alpha=0.2)
            axis.legend()
    fig.tight_layout()
    path = R / "08_width.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_selection_plot(global_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for row_index, backbone in enumerate(BACKBONES[:2]):
        for col_index, task in enumerate(run_02.TASKS):
            axis = axes[row_index, col_index]
            bottom = np.zeros(len(EPSILONS))
            for depth_name in ("e6d4", "e3d4", "e6d2", "none"):
                fractions = []
                for epsilon in EPSILONS:
                    selected = [row for row in global_rows if row["backbone"] == backbone and float(row["mask_ratio"]) == 0.5 and row["task"] == task and float(row["epsilon_db"]) == epsilon]
                    if depth_name == "none":
                        count = sum(not row["feasible"] for row in selected)
                    else:
                        enc, dec = int(depth_name[1]), int(depth_name[3])
                        count = sum(int(row["selected_enc_depth"]) == enc and int(row["selected_dec_depth"]) == dec for row in selected if row["feasible"])
                    fractions.append(count / len(selected))
                axis.bar([str(value) for value in EPSILONS], fractions, bottom=bottom, label=depth_name)
                bottom += np.array(fractions)
            axis.set_ylim(0, 1)
            axis.set_title(f"{backbone} / {task} / mask=0.5")
            axis.set_ylabel("Selection fraction across seeds")
            axis.set_xlabel("Epsilon (dB)")
            axis.legend(fontsize=8)
    fig.tight_layout()
    path = R / "08_depth_selection.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_report(width_rows, width_analysis_rows, probe_rows, global_rows, stability_rows):
    lines = [
        "# Task 08：精简多种子正式验证报告",
        "",
        "## 协议与完整性",
        "",
        "- 核心矩阵：4 骨干 × 3 深度 × 3 种子 × 2 掩码，共 72 个全量数据 H1。",
        "- 另有 Base 两个深度的 6 个有限解冻探针，沿用原 2000 样本子集。",
        "- 最终评价仅使用 D1–D16 val；训练种子为 100/200/300，评价噪声 seed=100。",
        "- 深度选择只在 `(6,4)`、`(3,4)`、`(6,2)` 中进行，ε=0.25/0.5/1.0 dB。",
        "",
        "## 宽度趋势按种子检查",
        "",
        "| 深度 | mask | 任务 | 单调种子数/3 | Base 均值 | Small 均值 | Little 均值 | Tiny 均值 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in width_analysis_rows:
        values = np.array([item["depth_costs"] for item in row["per_seed"]])
        lines.append(f"| e{row['enc_depth']}d{row['dec_depth']} | {row['mask_ratio']} | {row['task']} | {row['monotonic_seed_count']}/3 | " + " | ".join(f"{value:.4f}" for value in values.mean(axis=0)) + " |")
    lines.extend([
        "",
        "## 多种子精度与深度代价",
        "",
        "| 骨干 | 深度 | mask | 任务 | H1 种子均值±标准差 | 深度代价均值±标准差 | 成本下降 % |",
        "|---|---|---:|---|---:|---:|---:|",
    ])
    for row in width_rows:
        if int(row["enc_depth"]) == 6:
            continue
        lines.append(f"| {row['backbone']} | e{row['enc_depth']}d{row['dec_depth']} | {row['mask_ratio']} | {row['task']} | {float(row['seed_mean_db']):.4f}±{float(row['seed_std_db']):.4f} | {float(row['depth_cost_mean_db']):.4f}±{float(row['depth_cost_std_db']):.4f} | {float(row['cost_reduction_pct']):.3f} |")
    lines.extend([
        "",
        "## Base 有限解冻探针",
        "",
        "| 深度 | 种子 | 任务 | 输出 H1 | 探针 | 解冻增益 | 剩余深度代价 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in probe_rows:
        lines.append(f"| e{row['enc_depth']}d{row['dec_depth']} | {row['train_seed']} | {row['task']} | {float(row['output_h1_db']):.4f} | {float(row['probe_db']):.4f} | {float(row['probe_gain_db']):.4f} | {float(row['residual_depth_cost_db']):.4f} |")
    lines.extend([
        "",
        "## 全局深度选择分布",
        "",
        "每档 ε 包含 48 个骨干×种子×掩码×任务组合；完整路径始终可行，因此不存在无可行点。",
        "",
        "| ε | 完整 e6d4 | 编码器 e3d4 | 解码器 e6d2 |",
        "|---:|---:|---:|---:|",
    ])
    for epsilon in EPSILONS:
        selected = [row for row in global_rows if float(row["epsilon_db"]) == epsilon]
        counts = Counter(point_name(int(row["selected_enc_depth"]), int(row["selected_dec_depth"])) for row in selected if row["feasible"])
        lines.append(f"| {epsilon} | {counts['e6d4']} | {counts['e3d4']} | {counts['e6d2']} |")
    lines.extend([
        "",
        "## Bootstrap 深度选择稳定性",
        "",
        "每格汇总 144 个骨干×种子×掩码×任务组合。主导比例为最深出口被选择的概率最大值；同时列出三种深度的选择概率中位数，避免只凭中位数 1.000 误判为所有组合都稳定。",
        "",
        "| 样本量 | ε | 主导最小 | 主导中位 | 主导≥0.9 比例 | e6d4 中位 | e3d4 中位 | e6d2 中位 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for sample_size in SAMPLE_SIZES:
        for epsilon in EPSILONS:
            selected = [row for row in stability_rows if int(row["sample_size"]) == sample_size and float(row["epsilon_db"]) == epsilon]
            dominant = [max(float(row["p_e6d4"]), float(row["p_e3d4"]), float(row["p_e6d2"])) for row in selected]
            cells = [
                min(dominant), float(np.median(dominant)), float(np.mean(np.array(dominant) >= 0.9)),
                float(np.median([float(row["p_e6d4"]) for row in selected])),
                float(np.median([float(row["p_e3d4"]) for row in selected])),
                float(np.median([float(row["p_e6d2"]) for row in selected])),
            ]
            lines.append(f"| {sample_size} | {epsilon} | " + " | ".join(f"{value:.3f}" for value in cells) + " |")
    lines.extend(["", "## 边界", "", "- 本报告检验训练种子和验证样本扰动，不将大量样本次生置信区间替代模型训练不确定性。", "- 结果不代表严格未见配置泛化，也不证明动态路由或部署上优于官方小模型。", ""])
    path = R / "08_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = run_02.read_csv(WORK / "val_summary.csv")
    arrays = dict(np.load(WORK / "persample.npz").items())
    costs = run_02.read_csv(WORK / "costs.csv")
    assert len(rows) == 3264, len(rows)
    assert len(arrays) == 3264, len(arrays)
    assert len(costs) == 26, len(costs)
    aggregate_rows = aggregate(rows)
    width_rows = width_summary(aggregate_rows, costs)
    width_analysis_rows = width_analysis(aggregate_rows)
    probe_rows = probe_summary(aggregate_rows)
    global_rows, config_rows = depth_selection(rows, aggregate_rows, costs)
    stability_rows = stability_analysis(arrays, costs)
    history = []
    for path in sorted(WORK.glob("*_training.csv")):
        history.extend(run_02.read_csv(path))
    run_02.atomic_csv(R / "08_training.csv", history, tuple(history[0]))
    run_02.atomic_csv(R / "08_val_summary.csv", rows, tuple(rows[0]))
    run_02.atomic_npz(R / "08_persample.npz", arrays)
    run_02.atomic_csv(R / "08_costs.csv", costs, tuple(costs[0]))
    run_02.atomic_csv(R / "08_aggregate.csv", aggregate_rows, tuple(aggregate_rows[0]))
    run_02.atomic_csv(R / "08_width_summary.csv", width_rows, tuple(width_rows[0]))
    run_02.atomic_csv(R / "08_probe_summary.csv", probe_rows, tuple(probe_rows[0]))
    run_02.atomic_csv(R / "08_depth_selection.csv", global_rows, tuple(global_rows[0]))
    run_02.atomic_csv(R / "08_config_selection.csv", config_rows, tuple(config_rows[0]))
    run_02.atomic_csv(R / "08_stability.csv", stability_rows, tuple(stability_rows[0]))
    analysis = {
        "width_trends": width_analysis_rows,
        "protocol": {"core_models": 72, "probe_models": 6, "bootstrap_repeats": 200, "sample_sizes": SAMPLE_SIZES, "epsilons": EPSILONS},
    }
    run_02.atomic_json(R / "08_analysis.json", analysis)
    make_width_plot(width_analysis_rows)
    make_selection_plot(global_rows)
    make_report(width_rows, width_analysis_rows, probe_rows, global_rows, stability_rows)
    run_02.atomic_json(WORK / "analysis_complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "val_rows": len(rows), "aggregate_rows": len(aggregate_rows), "stability_rows": len(stability_rows)})
    print(f"PASS Task 08 analysis: val={len(rows)} aggregate={len(aggregate_rows)} stability={len(stability_rows)}")


if __name__ == "__main__":
    main()
