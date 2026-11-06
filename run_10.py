import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch

import run_02
import run_03
from exit_heads import ExitHead, add_noise, evaluate_exit, exit_hidden_from_noisy, exit_loss_and_outputs

W = Path(__file__).resolve().parent
R = W / "results"
WORK = R / "10_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
TRAINING_CSV = R / "10_training.csv"
VAL_CSV = R / "10_val_summary.csv"
PERSAMPLE_NPZ = R / "10_persample.npz"
COSTS_CSV = R / "10_costs.csv"
CURVES_CSV = R / "10_curves.csv"
SELECTION_CSV = R / "10_selection.csv"
ANALYSIS_JSON = R / "10_analysis.json"
PLOT_PNG = R / "10_intermediate.png"
REPORT_MD = R / "10_report.md"
SEEDS = (100, 200, 300)
DEPTHS = ((6, 4), (5, 4), (4, 4), (3, 4), (6, 3), (6, 2))
NEW_DEPTHS = ((5, 4), (4, 4), (6, 3))
NEW_MODELS = tuple((backbone, enc, dec, seed) for backbone in ("base", "small") for enc, dec in NEW_DEPTHS for seed in SEEDS)
NEW_MODELS += tuple(("small", enc, dec, seed) for enc, dec in ((6, 4), (3, 4), (6, 2)) for seed in (200, 300))


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_10.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def point_name(enc, dec):
    return f"e{enc}d{dec}"


def checkpoint_path(backbone, enc, dec, seed):
    return CHECKPOINTS / f"10_{backbone}_{point_name(enc, dec)}_h1_seed{seed}.pt"


def prepare(backbone, seed, device):
    model, _ = run_02.build_model(backbone, device)
    run_03.freeze_backbone(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    head = ExitHead(model, "h1").to(device)
    return model, head


def load_model(backbone, enc, dec, seed, device):
    model, head = prepare(backbone, seed, device)
    state = torch.load(checkpoint_path(backbone, enc, dec, seed), map_location="cpu", weights_only=False)
    head.load_state_dict(state["best"]["head"])
    model.eval()
    head.eval()
    return model, head


def save_checkpoint(path, epoch, backbone, enc, dec, seed, head, best, best_score, bad_epochs, optimizer, scheduler, history):
    state = {
        "epoch": epoch, "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "train_seed": seed,
        "current": {"head": head.state_dict()}, "best": best, "best_score": best_score,
        "bad_epochs": bad_epochs, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
    }
    temp = path.with_suffix(f".{time.time_ns()}.tmp.pt")
    torch.save(state, temp)
    temp.replace(path)


def train_h1(backbone, enc, dec, seed, resume, logger):
    path = checkpoint_path(backbone, enc, dec, seed)
    history_path = WORK / f"{backbone}_{point_name(enc, dec)}_h1_seed{seed}_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, head = prepare(backbone, seed, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best = None
    best_score = math.inf
    bad_epochs = 0
    start_epoch = 0
    history = []
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(state["current"]["head"])
        best = state["best"]
        best_score = state["best_score"]
        bad_epochs = state["bad_epochs"]
        start_epoch = state["epoch"]
        history = state["history"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if bad_epochs >= 3:
            return history
    total_start = time.time()
    for epoch in range(start_epoch + 1, 11):
        jobs = run_03.make_batch_jobs(epoch, run_03.split_indices())
        losses = []
        for batch_index, (dataset, task, batch_indices) in enumerate(jobs):
            imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
            torch.manual_seed(seed * 100000 + epoch * 10000 + batch_index)
            torch.cuda.manual_seed(seed * 100000 + epoch * 10000 + batch_index)
            imgs_n = add_noise(imgs)
            with torch.no_grad():
                hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc, dec)
            loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {backbone}/{point_name(enc, dec)}/seed{seed}/epoch{epoch}/batch{batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 100 == 0:
                logger.info("%s/%s/seed%d epoch=%d batch=%d loss=%.6g", backbone, point_name(enc, dec), seed, epoch, batch_index + 1, np.mean(losses[-100:]))
        scheduler.step()
        config_linear, config_db = [], []
        indices = run_03.split_indices()
        for dataset in run_02.DATASETS[:16]:
            data = run_02.cached(dataset, "val")[indices[f"{dataset}/val_early"]]
            for task in run_02.TASKS:
                values = evaluate_exit(model, head, data, device, task, enc, dec)
                linear, db = run_02.summarize(values)
                config_linear.append(linear)
                config_db.append(db)
        val_linear = float(np.mean(config_linear))
        val_db = float(np.mean(config_db))
        if val_db < best_score:
            best_score = val_db
            best = {"head": {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
            "mode": "h1", "backbone": backbone, "head_kind": "h1", "epoch": epoch,
            "exit_key": point_name(enc, dec), "train_seed": seed, "mask_ratio": 0.5,
            "train_loss": f"{np.mean(losses):.17g}", "lr": f"{optimizer.param_groups[0]['lr']:.17g}",
            "val_lin_mean": f"{val_linear:.17g}", "val_db_mean": f"{val_db:.17g}",
            "global_val_db_mean": f"{val_db:.17g}", "bad_epochs": bad_epochs,
            "elapsed_sec": f"{time.time() - total_start:.3f}",
        })
        save_checkpoint(path, epoch, backbone, enc, dec, seed, head, best, best_score, bad_epochs, optimizer, scheduler, history)
        run_02.atomic_csv(history_path, history, tuple(history[0]))
        logger.info("%s/%s/seed%d epoch=%d loss=%.6g val_db=%.6f bad=%d", backbone, point_name(enc, dec), seed, epoch, np.mean(losses), val_db, bad_epochs)
        if bad_epochs >= 3:
            break
    del model, head, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def evaluation_key(backbone, enc, dec, seed, dataset, task):
    return f"h1/{backbone}/e{enc}d{dec}/train{seed}/mask0.5/{dataset}/{task}/val/eval100"


def evaluate_one(backbone, enc, dec, seed, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), int(row["train_seed"]), row["dataset"], row["task"]) for row in rows}
    if all((backbone, enc, dec, seed, dataset, task) in done for dataset in run_02.DATASETS[:16] for task in run_02.TASKS):
        return
    model, head = load_model(backbone, enc, dec, seed, torch.device("cuda"))
    for dataset in run_02.DATASETS[:16]:
        for task in run_02.TASKS:
            if (backbone, enc, dec, seed, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), torch.device("cuda"), task, enc, dec)
            linear, db = run_02.summarize(values)
            rows.append({
                "method": "h1", "backbone": backbone, "mode": "h1",
                "enc_depth": enc, "dec_depth": dec, "train_seed": seed, "mask_ratio": 0.5,
                "dataset": dataset, "task": task, "split": "val", "eval_seed": 100,
                "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(backbone, enc, dec, seed, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s NMSE=%.6f", evaluation_key(backbone, enc, dec, seed, dataset, task), db)
    del model, head
    gc.collect()
    torch.cuda.empty_cache()


def measure_cost(backbone, enc, dec, seed, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    if any(row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and int(row["train_seed"]) == seed for row in rows):
        return
    noisy = run_02.noisy_input("D17")
    model, head = load_model(backbone, enc, dec, seed, torch.device("cuda"))
    gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), enc, dec)
    gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b1")
    gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b64")
    model = model.cpu()
    head = head.cpu()
    torch.cuda.empty_cache()
    cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cpu"), "cpu_b1")
    extra = sum(parameter.numel() for parameter in head.adapter.parameters())
    rows.append({
        "method": "h1", "backbone": backbone, "mode": "h1", "enc_depth": enc, "dec_depth": dec,
        "train_seed": seed, "params_exec": run_02.params_exec(model, enc, dec) + extra,
        "trainable_params": sum(parameter.numel() for parameter in head.parameters()),
        "gflops": f"{gflops:.17g}", "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
        "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
        "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
    })
    run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
    logger.info("PASS cost %s/%s/seed%d GFLOPs=%.6f", backbone, point_name(enc, dec), seed, gflops)
    del model, head
    gc.collect()


def aggregate_new(rows, backbone, enc, dec, seed, task):
    selected = [row for row in rows if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and int(row["train_seed"]) == seed and row["task"] == task]
    assert len(selected) == 16, (backbone, enc, dec, seed, task, len(selected))
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def aggregate03(rows, method, backbone, enc, dec, task):
    selected = [row for row in rows if row["method"] == method and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
    assert len(selected) == 16
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def full_h1_db(new_rows, old04_rows, old09_rows, backbone, seed, task):
    if backbone == "base":
        selected = [row for row in old09_rows if row["mode"] == "full_h1" and int(row["train_seed"]) == seed and row["task"] == task]
        if seed == 100:
            selected = [row for row in old04_rows if row["head_kind"] == "h1" and row["backbone"] == "base" and row["task"] == task]
    else:
        selected = [row for row in old04_rows if row["head_kind"] == "h1" and row["backbone"] == "small" and row["task"] == task] if seed == 100 else [row for row in new_rows if row["backbone"] == "small" and row["mode"] == "h1" and int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == 4 and int(row["train_seed"]) == seed and row["task"] == task]
    assert len(selected) == 16, (backbone, seed, task, len(selected))
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def trunc_h1_db(new_rows, old03_rows, old06_rows, old09_rows, backbone, enc, dec, seed, task):
    if backbone == "base" and (enc, dec) == (3, 4) and seed == 100:
        return aggregate03(old03_rows, "h1", backbone, enc, dec, task)
    if backbone == "base" and (enc, dec) == (6, 2) and seed == 100:
        selected = [row for row in old06_rows if row["mode"] == "output_h1" and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    if backbone == "base" and seed > 100 and (enc, dec) in ((3, 4), (6, 2)):
        selected = [row for row in old09_rows if row["mode"] == "trunc_h1" and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and int(row["train_seed"]) == seed and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    if backbone == "small" and (enc, dec) == (6, 2) and seed == 100:
        selected = [row for row in run_02.read_csv(R / "07_val_summary.csv") if row["backbone"] == "small" and row["mode"] == "h1" and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    if backbone == "small" and seed == 100 and (enc, dec) in ((3, 4), (6, 2)):
        return aggregate03(old03_rows, "h1", backbone, enc, dec, task)
    return aggregate_new(new_rows, backbone, enc, dec, seed, task)


def cost_for_depth(new_costs, backbone, enc, dec):
    if backbone == "base" and (enc, dec) == (6, 4):
        row = next(row for row in run_02.read_csv(R / "04_costs.csv") if row["backbone"] == "base" and row["head_kind"] == "h1")
    elif backbone == "small" and (enc, dec) == (6, 4):
        row = next(row for row in run_02.read_csv(R / "04_costs.csv") if row["backbone"] == "small" and row["head_kind"] == "h1")
    elif backbone == "base" and (enc, dec) == (6, 2):
        row = next(row for row in run_02.read_csv(R / "06_costs.csv") if row["mode"] == "output_h1")
    elif backbone == "small" and (enc, dec) == (6, 2):
        row = next(row for row in run_02.read_csv(R / "07_costs.csv") if row["mode"] == "h1")
    elif (enc, dec) in ((3, 4), (6, 2)):
        row = next(row for row in run_02.read_csv(R / "03_costs.csv") if row["method"] == "h1" and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec)
    else:
        row = next(row for row in new_costs if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec)
    return float(row["gflops"])


def make_curves(new_rows, old03_rows, old04_rows, old06_rows, old09_rows, new_costs):
    output = []
    for backbone in ("base", "small"):
        full_cost = cost_for_depth(new_costs, backbone, 6, 4)
        for enc, dec in DEPTHS:
            cost = cost_for_depth(new_costs, backbone, enc, dec)
            for seed in SEEDS:
                full = full_h1_db(new_rows, old04_rows, old09_rows, backbone, seed, "temporal")
                for task in run_02.TASKS:
                    full = full_h1_db(new_rows, old04_rows, old09_rows, backbone, seed, task)
                    trunc = full if (enc, dec) == (6, 4) else trunc_h1_db(new_rows, old03_rows, old06_rows, old09_rows, backbone, enc, dec, seed, task)
                    output.append({
                        "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "train_seed": seed, "task": task,
                        "full_h1_db": f"{full:.17g}", "trunc_h1_db": f"{trunc:.17g}",
                        "depth_cost_db": f"{trunc - full:.17g}", "gflops": f"{cost:.17g}",
                        "cost_reduction_pct": f"{(full_cost - cost) / full_cost * 100:.17g}",
                    })
    return output


def make_selection(curves):
    output = []
    for backbone in ("base", "small"):
        for seed in SEEDS:
            for task in run_02.TASKS:
                for epsilon in (0.25, 0.5, 1.0):
                    selected = [row for row in curves if row["backbone"] == backbone and int(row["train_seed"]) == seed and row["task"] == task]
                    full = next(row for row in selected if int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == 4)
                    candidates = [(float(row["gflops"]), int(row["enc_depth"]), int(row["dec_depth"]), float(row["depth_cost_db"])) for row in selected]
                    feasible = [row for row in candidates if row[3] <= epsilon]
                    chosen = min(feasible) if feasible else None
                    output.append({
                        "backbone": backbone, "train_seed": seed, "task": task, "epsilon_db": epsilon,
                        "feasible": bool(chosen), "selected_enc_depth": "" if chosen is None else chosen[1],
                        "selected_dec_depth": "" if chosen is None else chosen[2],
                        "selected_gflops": "" if chosen is None else f"{chosen[0]:.17g}",
                        "selected_degradation_db": "" if chosen is None else f"{chosen[3]:.17g}",
                    })
    return output


def analyze_results(curves, selection):
    summary = []
    for backbone in ("base", "small"):
        for task in run_02.TASKS:
            for enc, dec in DEPTHS:
                selected = [row for row in curves if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
                summary.append({
                    "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "task": task,
                    "depth_cost_mean_db": float(np.mean([float(row["depth_cost_db"]) for row in selected])),
                    "depth_cost_std_db": float(np.std([float(row["depth_cost_db"]) for row in selected])),
                })
    selections = []
    for epsilon in (0.25, 0.5, 1.0):
        selected = [row for row in selection if float(row["epsilon_db"]) == epsilon]
        counts = {}
        for enc, dec in DEPTHS:
            counts[point_name(enc, dec)] = sum(int(row["selected_enc_depth"]) == enc and int(row["selected_dec_depth"]) == dec for row in selected if row["feasible"])
        selections.append({"epsilon_db": epsilon, "counts": counts})
    return {"curve_summary": summary, "selection_summary": selections}


def make_plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row_index, backbone in enumerate(("base", "small")):
        for col_index, task in enumerate(run_02.TASKS):
            axis = axes[row_index, col_index]
            for route_name, depths, color in (("encoder", ((6, 4), (5, 4), (4, 4), (3, 4)), "tab:blue"), ("decoder", ((6, 4), (6, 3), (6, 2)), "tab:orange")):
                selected = [row for row in curves if row["backbone"] == backbone and row["task"] == task and (int(row["enc_depth"]), int(row["dec_depth"])) in depths]
                grouped = []
                for enc, dec in depths:
                    rows = [row for row in selected if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec]
                    grouped.append((float(rows[0]["gflops"]), float(np.mean([float(row["depth_cost_db"]) for row in rows])), float(np.std([float(row["depth_cost_db"]) for row in rows]))))
                grouped.sort()
                axis.errorbar([row[0] for row in grouped], [row[1] for row in grouped], yerr=[row[2] for row in grouped], marker="o", capsize=3, color=color, label=route_name)
                for (enc, dec), (x, y, _) in zip(depths, grouped):
                    axis.annotate(point_name(enc, dec), (x, y), fontsize=8)
            axis.set_xscale("log")
            axis.set_xlabel("GFLOPs (log scale)")
            axis.set_ylabel("Depth cost vs full H1 (dB)")
            axis.set_title(f"{backbone} / {task}")
            axis.grid(alpha=0.2)
            axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=180)
    plt.close(fig)


def make_report(curves, selection, analysis):
    lines = [
        "# Task 10：Base、Small 中间深度恢复曲线报告",
        "",
        "## 协议与完整性",
        "",
        "- 新增 24 个 H1，覆盖 `(5,4)`、`(4,4)`、`(6,3)` 及 Small 缺失种子；mask=0.5，三种子，统一每配置 2000 样本。",
        "- 恢复后候选为 `(6,4),(5,4),(4,4),(3,4),(6,3),(6,2)`，完整 H1 为参考。",
        "",
        "## 恢复后的深度—成本曲线",
        "",
        "| 骨干 | 深度 | 任务 | 深度代价均值 dB | 种子标准差 | 成本下降 % |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in analysis["curve_summary"]:
        curve_row = next(item for item in curves if item["backbone"] == row["backbone"] and int(item["enc_depth"]) == int(row["enc_depth"]) and int(item["dec_depth"]) == int(row["dec_depth"]) and item["task"] == row["task"])
        cost = float(curve_row["cost_reduction_pct"])
        lines.append(f"| {row['backbone']} | e{row['enc_depth']}d{row['dec_depth']} | {row['task']} | {row['depth_cost_mean_db']:.4f} | {row['depth_cost_std_db']:.4f} | {cost:.3f} |")
    lines.extend([
        "",
        "## 扩展候选下的深度选择",
        "",
        "每档 ε 包含 12 个骨干×种子×任务组合。",
        "",
        "| ε | e6d4 | e5d4 | e4d4 | e3d4 | e6d3 | e6d2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in analysis["selection_summary"]:
        counts = row["counts"]
        lines.append(f"| {row['epsilon_db']} | {counts['e6d4']} | {counts['e5d4']} | {counts['e4d4']} | {counts['e3d4']} | {counts['e6d3']} | {counts['e6d2']} |")
    lines.extend(["", "## 边界", "", "- 本报告使用恢复后曲线选择候选，不用零训练拐点替代恢复结论。", "- 中间点是否优于 `(3,4)` 必须按实际精度与成本判断，不预设其一定成为最佳部署点。", ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    final_paths = (TRAINING_CSV, VAL_CSV, PERSAMPLE_NPZ, COSTS_CSV, CURVES_CSV, SELECTION_CSV, ANALYSIS_JSON, PLOT_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume:
        raise FileExistsError("Task 10 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    for backbone, enc, dec, seed in NEW_MODELS:
        train_h1(backbone, enc, dec, seed, args.resume, logger)
        evaluate_one(backbone, enc, dec, seed, logger)
        measure_cost(backbone, enc, dec, seed, logger)
    history = []
    for backbone, enc, dec, seed in NEW_MODELS:
        history.extend(run_02.read_csv(WORK / f"{backbone}_{point_name(enc, dec)}_h1_seed{seed}_training.csv"))
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    curves = make_curves(new_rows, run_02.read_csv(run_03.VAL_CSV), run_02.read_csv(R / "04_val_summary.csv"), run_02.read_csv(R / "06_val_summary.csv"), run_02.read_csv(R / "09_val_summary.csv"), run_02.read_csv(WORK / "costs.csv"))
    selection = make_selection(curves)
    analysis = analyze_results(curves, selection)
    run_02.atomic_csv(TRAINING_CSV, history, tuple(history[0]))
    run_02.atomic_csv(VAL_CSV, new_rows, tuple(new_rows[0]))
    run_02.atomic_npz(PERSAMPLE_NPZ, dict(np.load(WORK / "persample.npz").items()))
    run_02.atomic_csv(COSTS_CSV, run_02.read_csv(WORK / "costs.csv"), tuple(run_02.read_csv(WORK / "costs.csv")[0]))
    run_02.atomic_csv(CURVES_CSV, curves, tuple(curves[0]))
    run_02.atomic_csv(SELECTION_CSV, selection, tuple(selection[0]))
    run_02.atomic_json(ANALYSIS_JSON, analysis)
    make_plot(curves)
    make_report(curves, selection, analysis)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "new_models": len(NEW_MODELS), "val_rows": len(new_rows)})
    logger.info("PASS Task 10: intermediate depth recovery complete")


if __name__ == "__main__":
    main()
