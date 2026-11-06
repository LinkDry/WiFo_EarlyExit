import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch

import run_02
import run_03
from exit_heads import evaluate_exit

W = Path(__file__).resolve().parent
RESULTS = W / "results"
WORK = RESULTS / "05_work"
LOGS = W / "logs"
VAL_CSV = RESULTS / "05_val_summary.csv"
PERSAMPLE_NPZ = RESULTS / "05_persample.npz"
COSTS_CSV = RESULTS / "05_costs.csv"
AGGREGATE_CSV = RESULTS / "05_aggregate.csv"
EQUAL_CSV = RESULTS / "05_equal_cost.csv"
ANALYSIS_JSON = RESULTS / "05_analysis.json"
PLOT_PNG = RESULTS / "05_depth_curves.png"
REPORT_MD = RESULTS / "05_report.md"
PATHS = (
    (6, 4), (5, 4), (4, 4), (3, 4), (2, 4), (1, 4),
    (6, 3), (6, 2), (6, 1), (6, 0),
)
REUSED = {(6, 4), (3, 4), (6, 2)}
VAL_FIELDS = (
    "method", "backbone", "enc_depth", "dec_depth", "dataset", "task",
    "mask_ratio", "split", "seed", "source", "nmse_lin", "nmse_db",
)


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_05")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_05.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def key(backbone, enc, dec, dataset, task):
    return f"untrained/{backbone}/e{enc}d{dec}/{dataset}/{task}/0.5/val/seed100"


def route(enc, dec):
    if enc == 6 and dec == 4:
        return "full"
    if dec == 4:
        return "encoder"
    return "decoder"


def run_new_evaluation(logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["dataset"], row["task"]) for row in rows}
    for backbone in ("base", "small"):
        pending_paths = [(enc, dec) for enc, dec in PATHS if (enc, dec) not in REUSED]
        model, _ = run_02.build_model(backbone, torch.device("cuda"))
        run_03.freeze_backbone(model)
        for enc, dec in pending_paths:
            for dataset in run_02.DATASETS[:16]:
                for task in run_02.TASKS:
                    if (backbone, enc, dec, dataset, task) in done:
                        continue
                    values = evaluate_exit(model, None, run_02.cached(dataset, "val"), torch.device("cuda"), task, enc, dec)
                    linear, db = run_02.summarize(values)
                    rows.append({
                        "method": "untrained", "backbone": backbone,
                        "enc_depth": enc, "dec_depth": dec, "dataset": dataset, "task": task,
                        "mask_ratio": 0.5, "split": "val", "seed": 100, "source": "task05",
                        "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
                    })
                    arrays[key(backbone, enc, dec, dataset, task)] = values
                    run_02.atomic_csv(work_csv, rows, VAL_FIELDS)
                    run_02.atomic_npz(work_npz, arrays)
            logger.info("evaluated %s e%dd%d", backbone, enc, dec)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows, arrays


def combine_results(new_rows, new_arrays):
    old_rows = run_02.read_csv(run_03.VAL_CSV)
    old_arrays = np.load(run_03.PERSAMPLE_NPZ)
    combined_rows = []
    combined_arrays = {}
    for backbone in ("base", "small"):
        for enc, dec in PATHS:
            for dataset in run_02.DATASETS[:16]:
                for task in run_02.TASKS:
                    output_key = key(backbone, enc, dec, dataset, task)
                    if (enc, dec) in REUSED:
                        method = "official" if (enc, dec) == (6, 4) else "untrained"
                        source = f"task03_{method}"
                        row = next(row for row in old_rows if row["method"] == method and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["dataset"] == dataset and row["task"] == task)
                        old_key = f"{method}/{backbone}/e{enc}d{dec}/{dataset}/{task}/0.5/val"
                        combined_arrays[output_key] = old_arrays[old_key]
                        combined_rows.append({
                            "method": "untrained", "backbone": backbone,
                            "enc_depth": enc, "dec_depth": dec, "dataset": dataset, "task": task,
                            "mask_ratio": 0.5, "split": "val", "seed": 100, "source": source,
                            "nmse_lin": row["nmse_lin"], "nmse_db": row["nmse_db"],
                        })
                    else:
                        row = next(row for row in new_rows if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["dataset"] == dataset and row["task"] == task)
                        combined_rows.append(row)
                        combined_arrays[output_key] = new_arrays[output_key]
    return combined_rows, combined_arrays


def run_costs(logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    done = {(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"])) for row in rows}
    noisy = run_02.noisy_input("D17")
    for backbone in ("base", "small"):
        for enc, dec in PATHS:
            if (backbone, enc, dec) in done:
                continue
            logger.info("cost %s e%dd%d", backbone, enc, dec)
            model, _ = run_02.build_model(backbone, torch.device("cuda"))
            run_03.freeze_backbone(model)
            gflops = run_03.cost_gflops(model, None, noisy[:1].cuda(), enc, dec)
            gpu_b1, gpu_b1_range = run_03.cost_timing(model, None, noisy, enc, dec, torch.device("cuda"), "gpu_b1")
            gpu_b64, gpu_b64_range = run_03.cost_timing(model, None, noisy, enc, dec, torch.device("cuda"), "gpu_b64")
            model = model.cpu()
            torch.cuda.empty_cache()
            cpu_b1, cpu_b1_range = run_03.cost_timing(model, None, noisy, enc, dec, torch.device("cpu"), "cpu_b1")
            rows.append({
                "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "shape": "D17",
                "params_exec": run_02.params_exec(model, enc, dec), "gflops": f"{gflops:.17g}",
                "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
                "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
                "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
            })
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            del model
            gc.collect()
    return rows


def aggregate_results(rows):
    output = []
    for backbone in ("base", "small"):
        full = [row for row in rows if row["backbone"] == backbone and int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == 4]
        for enc, dec in PATHS:
            for task in run_02.TASKS:
                selected = [row for row in rows if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
                full_task = [row for row in full if row["task"] == task]
                linear = float(np.mean([float(row["nmse_lin"]) for row in selected]))
                db = float(np.mean([float(row["nmse_db"]) for row in selected]))
                full_linear = float(np.mean([float(row["nmse_lin"]) for row in full_task]))
                full_db = float(np.mean([float(row["nmse_db"]) for row in full_task]))
                output.append({
                    "backbone": backbone, "route": route(enc, dec), "enc_depth": enc, "dec_depth": dec,
                    "task": task, "nmse_lin_mean": f"{linear:.17g}", "nmse_db_mean": f"{db:.17g}",
                    "degradation_db": f"{db - full_db:.17g}", "full_lin_mean": f"{full_linear:.17g}",
                    "degradation_lin": f"{linear - full_linear:.17g}",
                })
    return output


def analyze_knees(aggregate):
    output = []
    for backbone in ("base", "small"):
        for task in run_02.TASKS:
            for route_name in ("encoder", "decoder"):
                selected = [row for row in aggregate if row["backbone"] == backbone and row["task"] == task and row["route"] in (route_name, "full")]
                if route_name == "encoder":
                    selected.sort(key=lambda row: int(row["enc_depth"]), reverse=True)
                else:
                    selected.sort(key=lambda row: int(row["dec_depth"]), reverse=True)
                increments = []
                for deeper, shallower in zip(selected, selected[1:]):
                    increments.append({
                        "from": f"e{deeper['enc_depth']}d{deeper['dec_depth']}",
                        "to": f"e{shallower['enc_depth']}d{shallower['dec_depth']}",
                        "increment_db": float(shallower["degradation_db"]) - float(deeper["degradation_db"]),
                    })
                largest = max(increments, key=lambda row: row["increment_db"])
                monotonic = all(row["increment_db"] >= -0.01 for row in increments)
                if largest["increment_db"] <= 0.25:
                    status = "no_clear_knee"
                elif not monotonic:
                    status = "nonmonotonic"
                else:
                    status = "candidate_knee"
                output.append({
                    "backbone": backbone, "task": task, "route": route_name,
                    "monotonic": monotonic, "largest_jump": largest,
                    "candidate": largest["from"] if status == "candidate_knee" else None,
                    "status": status, "increments": increments,
                })
    return output


def equal_cost_pairs(aggregate, costs):
    output = []
    for backbone in ("base", "small"):
        for task in run_02.TASKS:
            decoder_rows = [row for row in aggregate if row["backbone"] == backbone and row["route"] == "decoder" and row["task"] == task and int(row["dec_depth"]) >= 1]
            encoder_rows = [row for row in aggregate if row["backbone"] == backbone and row["route"] == "encoder" and row["task"] == task]
            for decoder in decoder_rows:
                dec_cost = next(float(row["gflops"]) for row in costs if row["backbone"] == backbone and int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == int(decoder["dec_depth"]))
                candidates = []
                for encoder in encoder_rows:
                    enc_cost = next(float(row["gflops"]) for row in costs if row["backbone"] == backbone and int(row["enc_depth"]) == int(encoder["enc_depth"]) and int(row["dec_depth"]) == 4)
                    candidates.append((abs(enc_cost - dec_cost), encoder, enc_cost))
                _, encoder, enc_cost = min(candidates, key=lambda item: item[0])
                output.append({
                    "backbone": backbone, "task": task,
                    "encoder_enc_depth": encoder["enc_depth"], "encoder_dec_depth": 4,
                    "decoder_enc_depth": 6, "decoder_dec_depth": decoder["dec_depth"],
                    "encoder_gflops": f"{enc_cost:.17g}", "decoder_gflops": f"{dec_cost:.17g}",
                    "cost_ratio": f"{enc_cost / dec_cost:.17g}",
                    "encoder_degradation_db": encoder["degradation_db"],
                    "decoder_degradation_db": decoder["degradation_db"],
                    "encoder_minus_decoder_db": f"{float(encoder['degradation_db']) - float(decoder['degradation_db']):.17g}",
                })
    return output


def make_plot(aggregate, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_index, backbone in enumerate(("base", "small")):
        for col_index, task in enumerate(run_02.TASKS):
            axis = axes[row_index, col_index]
            for route_name, color in (("encoder", "tab:blue"), ("decoder", "tab:orange")):
                selected = [row for row in aggregate if row["backbone"] == backbone and row["task"] == task and row["route"] in (route_name, "full")]
                selected.sort(key=lambda row: next(float(cost["gflops"]) for cost in costs if cost["backbone"] == backbone and int(cost["enc_depth"]) == int(row["enc_depth"]) and int(cost["dec_depth"]) == int(row["dec_depth"])))
                xs = [next(float(cost["gflops"]) for cost in costs if cost["backbone"] == backbone and int(cost["enc_depth"]) == int(row["enc_depth"]) and int(cost["dec_depth"]) == int(row["dec_depth"])) for row in selected]
                ys = [float(row["degradation_db"]) for row in selected]
                axis.plot(xs, ys, marker="o", color=color, label=route_name)
                for row, x, y in zip(selected, xs, ys):
                    axis.annotate(f"e{row['enc_depth']}d{row['dec_depth']}", (x, y), fontsize=8)
            axis.set_xscale("log")
            axis.set_xlabel("GFLOPs (log scale)")
            axis.set_ylabel("Degradation from full model (dB)")
            axis.set_title(f"{backbone} / {task}")
            axis.grid(alpha=0.2)
            axis.legend()
    fig.tight_layout()
    temp = PLOT_PNG.with_suffix(f".{time.time_ns()}.tmp.png")
    fig.savefig(temp, dpi=180)
    plt.close(fig)
    temp.replace(PLOT_PNG)


def make_report(aggregate, costs, equal_costs, analysis):
    lines = [
        "# Task 05：Base、Small 零训练深度曲线报告",
        "",
        "## 协议与完整性",
        "",
        "- 每个骨干包含 10 个唯一深度点，D1–D16 val 全 2000 样本，temporal/fre，mask=0.5，seed 100。",
        "- 共 640 行配置级结果、640 组逐样本结果和 20 条实际成本路径；未训练，D17、D18 未用于深度选择。",
        "- 相对退化 = 当前点 dB NMSE − 同骨干完整 `(6,4)` dB NMSE；线性聚合保留在 `05_aggregate.csv`。",
        "",
        "## 相邻深度跃迁与候选拐点",
        "",
        "最大跃迁 ≤0.25 dB 记为无明显拐点；存在 <−0.01 dB 增量时标记非单调。候选点只是描述性标记，不自动替换后续探针点。",
        "",
        "| 骨干 | 任务 | 路线 | 状态 | 最大跃迁 | 候选点 |",
        "|---|---|---|---|---:|---|",
    ]
    for row in analysis:
        candidate = row["candidate"] or "—"
        lines.append(f"| {row['backbone']} | {row['task']} | {row['route']} | {row['status']} | {row['largest_jump']['increment_db']:.6f} | {candidate} |")
    lines.extend([
        "",
        "## 相近 GFLOPs 的编码器/解码器配对",
        "",
        "负的 encoder-minus-decoder 表示相同近似成本下编码器侧退化更小。",
        "",
        "| 骨干 | 任务 | 编码器点 | 解码器点 | enc GF | dec GF | 成本比 | enc 退化 | dec 退化 | 差值 dB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in equal_costs:
        lines.append(f"| {row['backbone']} | {row['task']} | e{row['encoder_enc_depth']}d4 | e6d{row['decoder_dec_depth']} | {float(row['encoder_gflops']):.4f} | {float(row['decoder_gflops']):.4f} | {float(row['cost_ratio']):.3f} | {float(row['encoder_degradation_db']):.4f} | {float(row['decoder_degradation_db']):.4f} | {float(row['encoder_minus_decoder_db']):.4f} |")
    lines.extend([
        "",
        "## 聚合曲线",
        "",
        "| 骨干 | 路线 | enc | dec | 任务 | 线性均值 | dB 均值 | 相对完整退化 dB |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ])
    for row in aggregate:
        lines.append(f"| {row['backbone']} | {row['route']} | {row['enc_depth']} | {row['dec_depth']} | {row['task']} | {float(row['nmse_lin_mean']):.6f} | {float(row['nmse_db_mean']):.6f} | {float(row['degradation_db']):.6f} |")
    lines.extend([
        "",
        "## 实际成本（D17 形状）",
        "",
        "| 骨干 | enc | dec | params | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in costs:
        lines.append(f"| {row['backbone']} | {row['enc_depth']} | {row['dec_depth']} | {row['params_exec']} | {float(row['gflops']):.4f} | {float(row['gpu_b1_ms']):.3f} | {float(row['gpu_thr_b64']):.3f} | {float(row['cpu_b1_ms_t1']):.3f} |")
    lines.extend(["", "## 边界", "", "- 本报告为零训练验证集曲线，只用于选择后续诊断点，不证明编码器冗余或解码器不可压缩。", "- 等成本配对是离散点上的最近 GFLOPs 配对，不是连续插值，也不代表相同深度比例。", ""])
    temp = REPORT_MD.with_suffix(f".{time.time_ns()}.tmp.md")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(REPORT_MD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    final_paths = (VAL_CSV, PERSAMPLE_NPZ, COSTS_CSV, AGGREGATE_CSV, EQUAL_CSV, ANALYSIS_JSON, PLOT_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume:
        raise FileExistsError("Task 05 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    new_rows, new_arrays = run_new_evaluation(logger)
    rows, arrays = combine_results(new_rows, new_arrays)
    costs = run_costs(logger)
    aggregate = aggregate_results(rows)
    equal_costs = equal_cost_pairs(aggregate, costs)
    analysis = analyze_knees(aggregate)
    run_02.atomic_csv(VAL_CSV, rows, VAL_FIELDS)
    run_02.atomic_npz(PERSAMPLE_NPZ, arrays)
    run_02.atomic_csv(COSTS_CSV, costs, tuple(costs[0]))
    run_02.atomic_csv(AGGREGATE_CSV, aggregate, tuple(aggregate[0]))
    run_02.atomic_csv(EQUAL_CSV, equal_costs, tuple(equal_costs[0]))
    run_02.atomic_json(ANALYSIS_JSON, analysis)
    make_plot(aggregate, costs)
    make_report(aggregate, costs, equal_costs, analysis)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "val_rows": len(rows), "cost_rows": len(costs), "aggregate_rows": len(aggregate)})
    logger.info("PASS Task 05: depth curves complete")


if __name__ == "__main__":
    main()
