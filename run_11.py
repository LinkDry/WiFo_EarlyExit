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
import run_04
import run_06
import run_07
import run_08
import run_10
from exit_heads import evaluate_exit

W = Path(__file__).resolve().parent
R = W / "results"
WORK = R / "11_work"
LOGS = W / "logs"
PARETO_CSV = R / "11_pareto.csv"
PERSAMPLE_NPZ = R / "11_persample.npz"
COSTS_CSV = R / "11_costs.csv"
SELECTION_CSV = R / "11_selection_review.csv"
ANALYSIS_JSON = R / "11_analysis.json"
PLOT_PNG = R / "11_pareto.png"
REPORT_MD = R / "11_report.md"


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_11")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_11.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def candidates():
    items = [(f"official_{b}", b, 6, 4, "official") for b in ("base", "small", "little", "tiny")]
    for b in ("base", "small"):
        items.append((f"{b}_full_h1", b, 6, 4, "h1"))
        for enc, dec in ((5, 4), (4, 4), (3, 4), (6, 3), (6, 2)):
            items.append((f"{b}_h1_e{enc}d{dec}", b, enc, dec, "h1"))
    for enc, dec in ((3, 4), (6, 2)):
        items.append((f"base_probe_e{enc}d{dec}", "base", enc, dec, "probe"))
    for b in ("little", "tiny"):
        for enc, dec in ((6, 4), (3, 4), (6, 2)):
            items.append((f"{b}_h1_e{enc}d{dec}", b, enc, dec, "h1"))
    return items


def load_candidate(item):
    name, backbone, enc, dec, kind = item
    if kind == "official":
        model, _ = run_02.build_model(backbone, torch.device("cuda"))
        run_03.freeze_backbone(model)
        return model, None, None
    if kind == "probe":
        model, block, head = run_08.load_model(backbone, enc, dec, 100, 0.5, "probe", torch.device("cuda"))
        return model, block, head
    if (enc, dec) == (6, 4) and backbone in ("base", "small"):
        model, head = run_04.load_head(backbone, "h1", torch.device("cuda"))
    elif backbone == "base" and (enc, dec) == (3, 4):
        model, head = run_03.load_best_head("base", "h1", "e3d4", torch.device("cuda"))
    elif backbone == "base" and (enc, dec) == (6, 2):
        model, block, head, adapter = run_06.load_point(6, 2, "output_h1", torch.device("cuda"))
    elif backbone == "small" and (enc, dec) == (3, 4):
        model, head = run_03.load_best_head("small", "h1", "e3d4", torch.device("cuda"))
    elif backbone == "small" and (enc, dec) == (6, 2):
        model, head = run_07.load_h1("small", 6, 2, torch.device("cuda"))
    elif backbone in ("little", "tiny"):
        model, head = run_07.load_h1(backbone, enc, dec, torch.device("cuda"))
    else:
        model, head = run_10.load_model(backbone, enc, dec, 100, torch.device("cuda"))
    return model, None, head


def evaluation_key(name, dataset, task):
    return f"{name}/{dataset}/{task}/0.5/test/eval100"


def evaluate_candidate(item, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    name, backbone, enc, dec, kind = item
    done = {(row["name"], row["dataset"], row["task"]) for row in rows}
    if all((name, dataset, task) in done for dataset in ("D17", "D18") for task in run_02.TASKS):
        return
    model, block, head = load_candidate(item)
    for dataset in ("D17", "D18"):
        for task in run_02.TASKS:
            if (name, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "test"), torch.device("cuda"), task, enc, dec)
            linear, db = run_02.summarize(values)
            rows.append({"name": name, "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "kind": kind, "dataset": dataset, "task": task, "mask_ratio": 0.5, "eval_seed": 100, "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}"})
            arrays[evaluation_key(name, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s NMSE=%.6f", evaluation_key(name, dataset, task), db)
    del model, block, head
    gc.collect()
    torch.cuda.empty_cache()


def measure_cost(item, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    name, backbone, enc, dec, kind = item
    done = {(row["name"], row["shape"]) for row in rows}
    noisy_by_shape = {}
    for shape in ("D17", "D18"):
        if (name, shape) in done:
            continue
        if shape not in noisy_by_shape:
            noisy_by_shape[shape] = run_02.noisy_input(shape)
        noisy = noisy_by_shape[shape]
        model, block, head = load_candidate(item)
        gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), enc, dec)
        gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b1")
        gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b64")
        extra = 0 if head is None else sum(parameter.numel() for parameter in head.adapter.parameters())
        params = run_02.params_exec(model, enc, dec) + extra
        model = model.cpu()
        if head is not None:
            head = head.cpu()
        if block is not None:
            block = block.cpu()
        torch.cuda.empty_cache()
        cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cpu"), "cpu_b1")
        rows.append({"name": name, "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "kind": kind, "shape": shape, "params_exec": params, "gflops": f"{gflops:.17g}", "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}", "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}", "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}"})
        run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
        logger.info("PASS cost %s/%s GFLOPs=%.6f", name, shape, gflops)
        del model, block, head
        gc.collect()


def make_pareto(rows, costs):
    cost_lookup = {(row["name"], row["shape"]): row for row in costs}
    output = []
    for row in rows:
        cost = cost_lookup[(row["name"], row["dataset"])]
        item = dict(row)
        for field in ("params_exec", "gflops", "gpu_b1_ms", "gpu_thr_b64", "cpu_b1_ms_t1"):
            item[field] = cost[field]
        output.append(item)
    return output


def non_dominated(rows, dataset, task, metric):
    selected = [row for row in rows if row["dataset"] == dataset and row["task"] == task]
    result = []
    for row in selected:
        cost = float(row[metric])
        nmse = float(row["nmse_db"])
        dominated = any(other["name"] != row["name"] and float(other[metric]) <= cost and float(other["nmse_db"]) <= nmse and (float(other[metric]) < cost or float(other["nmse_db"]) < nmse) for other in selected)
        if not dominated:
            result.append(row["name"])
    return result


def make_selection_review(rows):
    selected = run_02.read_csv(R / "10_selection.csv")
    lookup = {(row["name"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["dataset"], row["task"]): row for row in rows}
    output = []
    for choice in selected:
        backbone = choice["backbone"]
        task = choice["task"]
        seed = int(choice["train_seed"])
        enc, dec = int(choice["selected_enc_depth"]), int(choice["selected_dec_depth"])
        for dataset in ("D17", "D18"):
            chosen_name = f"{backbone}_full_h1" if (enc, dec) == (6, 4) else f"{backbone}_h1_e{enc}d{dec}"
            chosen = lookup[(chosen_name, backbone, enc, dec, dataset, task)]
            full = lookup[(f"{backbone}_full_h1", backbone, 6, 4, dataset, task)]
            delta = float(chosen["nmse_db"]) - float(full["nmse_db"])
            epsilon = float(choice["epsilon_db"])
            output.append({
                "backbone": backbone, "train_seed": seed, "task": task, "dataset": dataset,
                "epsilon_db": f"{epsilon:.17g}", "selected_enc_depth": enc, "selected_dec_depth": dec,
                "selected_nmse_db": chosen["nmse_db"], "full_h1_nmse_db": full["nmse_db"],
                "review_delta_db": f"{delta:.17g}", "constraint_satisfied": delta <= epsilon + 1e-12,
            })
    return output


def make_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for row_index, dataset in enumerate(("D17", "D18")):
        for col_index, task in enumerate(run_02.TASKS):
            axis = axes[row_index, col_index]
            selected = [row for row in rows if row["dataset"] == dataset and row["task"] == task]
            for kind, marker in (("official", "o"), ("h1", "s"), ("probe", "*")):
                group = [row for row in selected if row["kind"] == kind]
                axis.scatter([float(row["gflops"]) for row in group], [float(row["nmse_db"]) for row in group], marker=marker, s=65 if kind == "probe" else 35, label=kind, alpha=0.75)
                for row in group:
                    axis.annotate(row["name"], (float(row["gflops"]), float(row["nmse_db"])), fontsize=7, alpha=0.7)
            axis.set_xscale("log")
            axis.set_xlabel("GFLOPs (log scale)")
            axis.set_ylabel("NMSE (dB)")
            axis.set_title(f"{dataset} / {task}")
            axis.grid(alpha=0.2)
            axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=180)
    plt.close(fig)


def make_report(rows, review, analysis):
    lines = [
        "# Task 11：公平部署 Pareto 与选择复核报告",
        "",
        "## 协议与边界",
        "",
        "- 24 个候选包含官方模型、恢复截断和 Base 有限解冻探针，统一 seed 100、mask=0.5、评价 seed 100。",
        "- 96 组逐样本 NMSE 和 48 条 D17/D18 形状成本。绝对 NMSE 与成本同图，不使用相对各自完整模型的退化替代部署比较。",
        "- D17、D18 已用于前期诊断，因此本报告只作留出配置复核，不称为完全独立测试。",
        "",
        "## 非支配点",
        "",
        "| 数据 | 任务 | GFLOPs 非支配点 | GPU b1 非支配点 | CPU t1 非支配点 |",
        "|---|---|---|---|---|",
    ]
    for dataset in ("D17", "D18"):
        for task in run_02.TASKS:
            lines.append(f"| {dataset} | {task} | {', '.join(analysis[f'{dataset}/{task}/gflops'])} | {', '.join(analysis[f'{dataset}/{task}/gpu_b1_ms'])} | {', '.join(analysis[f'{dataset}/{task}/cpu_b1_ms_t1'])} |")
    lines.extend([
        "",
        "## 验证集选择在留出配置上的复核",
        "",
        "| 数据 | ε | 组合数 | 约束满足数 | 满足率 | 平均超限 dB | 最大超限 dB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in ("D17", "D18"):
        for epsilon in (0.25, 0.5, 1.0):
            selected = [row for row in review if row["dataset"] == dataset and float(row["epsilon_db"]) == epsilon]
            satisfied = sum(row["constraint_satisfied"] for row in selected)
            excess = [max(0.0, float(row["review_delta_db"]) - epsilon) for row in selected]
            lines.append(f"| {dataset} | {epsilon} | {len(selected)} | {satisfied} | {satisfied / len(selected):.3f} | {np.mean(excess):.4f} | {max(excess):.4f} |")
    lines.extend([
        "",
        "## 候选绝对 NMSE 与成本",
        "",
        "| 数据 | 任务 | 候选 | NMSE dB | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['task']} | {row['name']} | {float(row['nmse_db']):.4f} | {float(row['gflops']):.4f} | {float(row['gpu_b1_ms']):.3f} | {float(row['gpu_thr_b64']):.3f} | {float(row['cpu_b1_ms_t1']):.3f} |")
    lines.extend(["", "## 边界", "", "- 非支配点只针对当前候选和当前配置，不能推断所有模型或新配置。", "- 选择复核不参与重新选择；真正独立部署验证仍需新增未参与开发的数据。", ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    final_paths = (PARETO_CSV, PERSAMPLE_NPZ, COSTS_CSV, SELECTION_CSV, ANALYSIS_JSON, PLOT_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume:
        raise FileExistsError("Task 11 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    for item in candidates():
        evaluate_candidate(item, logger)
        measure_cost(item, logger)
    rows = run_02.read_csv(WORK / "val_summary.csv")
    costs = run_02.read_csv(WORK / "costs.csv")
    pareto = make_pareto(rows, costs)
    review = make_selection_review(pareto)
    analysis = {}
    for dataset in ("D17", "D18"):
        for task in run_02.TASKS:
            for metric in ("gflops", "gpu_b1_ms", "cpu_b1_ms_t1"):
                analysis[f"{dataset}/{task}/{metric}"] = non_dominated(pareto, dataset, task, metric)
    run_02.atomic_csv(PARETO_CSV, pareto, tuple(pareto[0]))
    run_02.atomic_npz(PERSAMPLE_NPZ, dict(np.load(WORK / "persample.npz").items()))
    run_02.atomic_csv(COSTS_CSV, costs, tuple(costs[0]))
    run_02.atomic_csv(SELECTION_CSV, review, tuple(review[0]))
    run_02.atomic_json(ANALYSIS_JSON, analysis)
    make_plot(pareto)
    make_report(pareto, review, analysis)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "candidates": len(candidates()), "pareto_rows": len(pareto), "review_rows": len(review)})
    logger.info("PASS Task 11: deployment review complete")


if __name__ == "__main__":
    main()
