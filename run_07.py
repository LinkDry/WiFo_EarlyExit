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
RESULTS = W / "results"
WORK = RESULTS / "07_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
TRAINING_CSV = RESULTS / "07_training.csv"
VAL_CSV = RESULTS / "07_val_summary.csv"
PERSAMPLE_NPZ = RESULTS / "07_persample.npz"
COSTS_CSV = RESULTS / "07_costs.csv"
CONTROLS_CSV = RESULTS / "07_controls.csv"
WIDTH_CSV = RESULTS / "07_width_summary.csv"
ANALYSIS_JSON = RESULTS / "07_analysis.json"
PLOT_PNG = RESULTS / "07_width.png"
REPORT_MD = RESULTS / "07_report.md"
NEW_H1 = (
    ("little", 6, 4), ("little", 3, 4), ("little", 6, 2),
    ("tiny", 6, 4), ("tiny", 3, 4), ("tiny", 6, 2),
    ("small", 6, 2),
)
NEW_UNTRAINED = (("little", 3, 4), ("little", 6, 2), ("tiny", 3, 4), ("tiny", 6, 2))
BACKBONES = ("base", "small", "little", "tiny")
POINTS = ((3, 4), (6, 2))


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_07")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_07.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def point_name(enc, dec):
    return f"e{enc}d{dec}"


def checkpoint_path(backbone, enc, dec):
    return CHECKPOINTS / f"07_{backbone}_{point_name(enc, dec)}_h1.pt"


def load_h1(backbone, enc, dec, device):
    model, _ = run_02.build_model(backbone, device)
    run_03.freeze_backbone(model)
    head = ExitHead(model, "h1").to(device).eval()
    state = torch.load(checkpoint_path(backbone, enc, dec), map_location="cpu", weights_only=False)
    head.load_state_dict(state["best"]["head"])
    return model, head


def save_checkpoint(path, epoch, backbone, enc, dec, head, best, best_score, bad_epochs, optimizer, scheduler, history):
    state = {
        "epoch": epoch, "backbone": backbone, "enc_depth": enc, "dec_depth": dec,
        "current": {"head": head.state_dict()}, "best": best, "best_score": best_score,
        "bad_epochs": bad_epochs, "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "history": history,
    }
    temp = path.with_suffix(f".{time.time_ns()}.tmp.pt")
    torch.save(state, temp)
    temp.replace(path)


def train_h1(backbone, enc, dec, indices, resume, logger):
    path = checkpoint_path(backbone, enc, dec)
    history_path = WORK / f"{backbone}_{point_name(enc, dec)}_h1_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, _ = run_02.build_model(backbone, device)
    run_03.freeze_backbone(model)
    head = ExitHead(model, "h1").to(device)
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
        jobs = run_03.make_batch_jobs(epoch, indices)
        losses = []
        for batch_index, (dataset, task, batch_indices) in enumerate(jobs):
            imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
            torch.manual_seed(100000 + epoch * 10000 + batch_index)
            torch.cuda.manual_seed(100000 + epoch * 10000 + batch_index)
            imgs_n = add_noise(imgs)
            with torch.no_grad():
                hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc, dec)
            loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {backbone}/{point_name(enc, dec)}/epoch{epoch}/batch{batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 100 == 0:
                logger.info("%s/%s epoch=%d batch=%d loss=%.6g", backbone, point_name(enc, dec), epoch, batch_index + 1, np.mean(losses[-100:]))
        scheduler.step()
        config_linear, config_db = [], []
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
            "backbone": backbone, "head_kind": "h1", "epoch": epoch, "exit_key": point_name(enc, dec),
            "train_loss": f"{np.mean(losses):.17g}", "lr": f"{optimizer.param_groups[0]['lr']:.17g}",
            "val_lin_mean": f"{val_linear:.17g}", "val_db_mean": f"{val_db:.17g}",
            "global_val_db_mean": f"{val_db:.17g}", "bad_epochs": bad_epochs,
            "elapsed_sec": f"{time.time() - total_start:.3f}",
        })
        save_checkpoint(path, epoch, backbone, enc, dec, head, best, best_score, bad_epochs, optimizer, scheduler, history)
        run_02.atomic_csv(history_path, history, run_03.TRAINING_FIELDS)
        logger.info("%s/%s epoch=%d loss=%.6g val_db=%.6f bad=%d", backbone, point_name(enc, dec), epoch, np.mean(losses), val_db, bad_epochs)
        if bad_epochs >= 3:
            break
    del model, head, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def smoke(indices, logger):
    results = {}
    for backbone in ("little", "tiny"):
        device = torch.device("cuda")
        model, _ = run_02.build_model(backbone, device)
        run_03.freeze_backbone(model)
        head = ExitHead(model, "h1").to(device)
        data = run_02.cached("D1", "val")[:32]
        original = evaluate_exit(model, None, data, device, "temporal", 6, 4)
        initial = evaluate_exit(model, head, data, device, "temporal", 6, 4)
        assert np.array_equal(original, initial)
        jobs = run_03.make_batch_jobs(1, indices, smoke=True)
        dataset, task, batch_indices = jobs[0]
        imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
        torch.manual_seed(101001)
        torch.cuda.manual_seed(101001)
        imgs_n = add_noise(imgs)
        with torch.no_grad():
            hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, 6, 4)
        loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None for parameter in model.parameters())
        optimizer.step()
        results[backbone] = {"initial_max_diff": float(np.max(np.abs(original - initial))), "loss": float(loss.detach().cpu())}
        logger.info("PASS smoke %s loss=%.6g", backbone, loss)
        del model, head, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    run_02.atomic_json(WORK / "smoke.json", results)


def evaluation_key(backbone, enc, dec, mode, dataset, task):
    return f"{mode}/{backbone}/e{enc}d{dec}/{dataset}/{task}/0.5/val/seed100"


def evaluate_point(backbone, enc, dec, mode, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["mode"], row["dataset"], row["task"]) for row in rows}
    if all((backbone, enc, dec, mode, dataset, task) in done for dataset in run_02.DATASETS[:16] for task in run_02.TASKS):
        return
    if mode == "h1":
        model, head = load_h1(backbone, enc, dec, torch.device("cuda"))
    else:
        model, _ = run_02.build_model(backbone, torch.device("cuda"))
        run_03.freeze_backbone(model)
        head = None
    for dataset in run_02.DATASETS[:16]:
        for task in run_02.TASKS:
            if (backbone, enc, dec, mode, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), torch.device("cuda"), task, enc, dec)
            linear, db = run_02.summarize(values)
            rows.append({
                "method": mode, "backbone": backbone, "mode": mode,
                "enc_depth": enc, "dec_depth": dec, "dataset": dataset, "task": task,
                "mask_ratio": 0.5, "split": "val", "seed": 100,
                "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(backbone, enc, dec, mode, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s NMSE=%.6f", evaluation_key(backbone, enc, dec, mode, dataset, task), db)
    del model, head
    gc.collect()
    torch.cuda.empty_cache()


def measure_cost(backbone, enc, dec, mode, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    if any(row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["mode"] == mode for row in rows):
        return
    noisy = run_02.noisy_input("D17")
    if mode == "h1":
        model, head = load_h1(backbone, enc, dec, torch.device("cuda"))
        extra = sum(parameter.numel() for parameter in head.adapter.parameters())
        trainable = sum(parameter.numel() for parameter in head.parameters())
    else:
        model, _ = run_02.build_model(backbone, torch.device("cuda"))
        run_03.freeze_backbone(model)
        head = None
        extra = 0
        trainable = 0
    gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), enc, dec)
    gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b1")
    gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b64")
    model = model.cpu()
    if head is not None:
        head = head.cpu()
    torch.cuda.empty_cache()
    cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cpu"), "cpu_b1")
    rows.append({
        "method": mode, "backbone": backbone, "mode": mode, "enc_depth": enc, "dec_depth": dec,
        "params_exec": run_02.params_exec(model, enc, dec) + extra, "trainable_params": trainable,
        "gflops": f"{gflops:.17g}", "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
        "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
        "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
    })
    run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
    logger.info("PASS cost %s/%s/%s GFLOPs=%.6f", backbone, point_name(enc, dec), mode, gflops)
    del model, head
    gc.collect()


def aggregate_new(rows, backbone, enc, dec, mode, task):
    selected = [row for row in rows if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["mode"] == mode and row["task"] == task]
    assert len(selected) == 16, (backbone, enc, dec, mode, task, len(selected))
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def aggregate03(rows, method, backbone, enc, dec, task):
    selected = [row for row in rows if row["method"] == method and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
    assert len(selected) == 16, (method, backbone, enc, dec, task, len(selected))
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def untrained_db(new_rows, old03_rows, backbone, enc, dec, task):
    if backbone in ("base", "small"):
        return aggregate03(old03_rows, "untrained", backbone, enc, dec, task)
    return aggregate_new(new_rows, backbone, enc, dec, "untrained", task)


def full_h1_db(new_rows, old04_rows, backbone, task):
    if backbone in ("base", "small"):
        selected = [row for row in old04_rows if row["backbone"] == backbone and row["head_kind"] == "h1" and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    return aggregate_new(new_rows, backbone, 6, 4, "h1", task)


def trunc_h1_db(new_rows, old03_rows, old06_rows, backbone, enc, dec, task):
    if (enc, dec) == (3, 4) and backbone in ("base", "small"):
        return aggregate03(old03_rows, "h1", backbone, enc, dec, task)
    if (enc, dec) == (6, 2) and backbone == "base":
        selected = [row for row in old06_rows if row["mode"] == "output_h1" and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    return aggregate_new(new_rows, backbone, enc, dec, "h1", task)


def existing_cost(cost07, cost06, cost04, cost03, backbone, enc, dec):
    if (enc, dec) == (6, 4) and backbone in ("base", "small"):
        row = next(row for row in cost04 if row["backbone"] == backbone and row["head_kind"] == "h1")
    elif (enc, dec) == (3, 4) and backbone in ("base", "small"):
        row = next(row for row in cost03 if row["method"] == "h1" and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec)
    elif (enc, dec) == (6, 2) and backbone == "base":
        row = next(row for row in cost06 if row["mode"] == "output_h1")
    else:
        row = next(row for row in cost07 if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["mode"] == "h1")
    return row


def make_controls(new_rows, old03_rows, old04_rows, old06_rows, cost07):
    cost06 = run_02.read_csv(RESULTS / "06_costs.csv")
    cost04 = run_02.read_csv(RESULTS / "04_costs.csv")
    cost03 = run_02.read_csv(RESULTS / "03_costs.csv")
    output = []
    for backbone in BACKBONES:
        for enc, dec in POINTS:
            full_cost = existing_cost(cost07, cost06, cost04, cost03, backbone, 6, 4)
            trunc_cost = existing_cost(cost07, cost06, cost04, cost03, backbone, enc, dec)
            reduction = (float(full_cost["gflops"]) - float(trunc_cost["gflops"])) / float(full_cost["gflops"]) * 100
            for task in run_02.TASKS:
                full = aggregate03(old03_rows, "official", backbone, 6, 4, task)
                full_h1 = full_h1_db(new_rows, old04_rows, backbone, task)
                untrained = untrained_db(new_rows, old03_rows, backbone, enc, dec, task)
                trunc_h1 = trunc_h1_db(new_rows, old03_rows, old06_rows, backbone, enc, dec, task)
                output.append({
                    "backbone": backbone, "enc_depth": enc, "dec_depth": dec, "task": task,
                    "full_db": f"{full:.17g}", "full_h1_db": f"{full_h1:.17g}",
                    "generic_gain_db": f"{full - full_h1:.17g}", "untrained_db": f"{untrained:.17g}",
                    "trunc_h1_db": f"{trunc_h1:.17g}", "total_recovery_db": f"{untrained - trunc_h1:.17g}",
                    "depth_cost_db": f"{trunc_h1 - full_h1:.17g}",
                    "specific_recovery_db": f"{(untrained - trunc_h1) - (full - full_h1):.17g}",
                    "full_h1_gflops": full_cost["gflops"], "trunc_h1_gflops": trunc_cost["gflops"],
                    "cost_reduction_pct": f"{reduction:.17g}",
                })
    return output


def analyze_width(controls):
    output = []
    for enc, dec in POINTS:
        for task in run_02.TASKS:
            records = []
            for backbone in BACKBONES:
                row = next(row for row in controls if row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task)
                records.append({"backbone": backbone, "depth_cost_db": float(row["depth_cost_db"]), "cost_reduction_pct": float(row["cost_reduction_pct"])})
            increments = [later["depth_cost_db"] - earlier["depth_cost_db"] for earlier, later in zip(records, records[1:])]
            output.append({
                "enc_depth": enc, "dec_depth": dec, "task": task, "records": records,
                "increments_db": increments,
                "narrower_more_degradation": all(value >= -0.01 for value in increments),
            })
    return output


def make_plot(analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, task in zip(axes, run_02.TASKS):
        for enc, dec in POINTS:
            record = next(row for row in analysis if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task)
            axis.plot(BACKBONES, [row["depth_cost_db"] for row in record["records"]], marker="o", label=f"e{enc}d{dec}")
        axis.set_title(task)
        axis.set_ylabel("Depth cost under same H1 (dB)")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    temp = PLOT_PNG.with_suffix(f".{time.time_ns()}.tmp.png")
    fig.savefig(temp, dpi=180)
    plt.close(fig)
    temp.replace(PLOT_PNG)


def make_report(history, controls, costs, analysis):
    lines = [
        "# Task 07：Little、Tiny 代表位置与四宽度检验报告",
        "",
        "## 协议",
        "",
        "- 新增 7 个 H1：Little/Tiny 各完整、`(3,4)`、`(6,2)`，另补 Small `(6,2)`。",
        "- 所有方法均与各自完整模型和完整 H1 对齐，D1–D16 val 全 2000 样本，单种子，未使用 D17、D18 选择。",
        "- 宽度顺序固定 Base → Small → Little → Tiny；相同 H1 深度代价非递减才记为符合“较窄退化更大”。",
        "",
        "## 宽度趋势",
        "",
        "| 深度 | 任务 | Base | Small | Little | Tiny | 符合单调趋势 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for enc, dec in POINTS:
        for task in run_02.TASKS:
            record = next(row for row in analysis if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task)
            values = [row["depth_cost_db"] for row in record["records"]]
            lines.append(f"| e{enc}d{dec} | {task} | " + " | ".join(f"{value:.6f}" for value in values) + f" | {record['narrower_more_degradation']} |")
    lines.extend([
        "",
        "## 四宽度对照",
        "",
        "| 骨干 | 深度 | 任务 | 完整 H1 | 截断 H1 | 总恢复 | 通用收益 | 深度代价 | 成本下降 % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in controls:
        lines.append(f"| {row['backbone']} | e{row['enc_depth']}d{row['dec_depth']} | {row['task']} | {float(row['full_h1_db']):.6f} | {float(row['trunc_h1_db']):.6f} | {float(row['total_recovery_db']):.6f} | {float(row['generic_gain_db']):.6f} | {float(row['depth_cost_db']):.6f} | {float(row['cost_reduction_pct']):.3f} |")
    lines.extend([
        "",
        "## 新增训练历史",
        "",
        "| 骨干 | 深度 | 最佳 epoch | 最佳 val 子集 dB |",
        "|---|---|---:|---:|",
    ])
    seen = set()
    for row in history:
        key = (row["backbone"], row["exit_key"])
        if key in seen:
            continue
        seen.add(key)
        selected = [item for item in history if item["backbone"] == key[0] and item["exit_key"] == key[1]]
        best = min(selected, key=lambda item: float(item["val_db_mean"]))
        lines.append(f"| {key[0]} | {key[1]} | {best['epoch']} | {float(best['val_db_mean']):.6f} |")
    lines.extend([
        "",
        "## 新增实际成本（D17 形状）",
        "",
        "| 骨干 | 深度 | 方法 | 执行参数 | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in costs:
        lines.append(f"| {row['backbone']} | e{row['enc_depth']}d{row['dec_depth']} | {row['mode']} | {row['params_exec']} | {float(row['gflops']):.6f} | {float(row['gpu_b1_ms']):.3f} | {float(row['gpu_thr_b64']):.3f} | {float(row['cpu_b1_ms_t1']):.3f} |")
    lines.extend(["", "## 边界", "", "- 单种子、官方 checkpoint 和代表位置只能用于检验趋势，不证明模型宽度决定深度冗余。", "- 单调趋势按深度、任务分别判断；若不成立，应报告 checkpoint 和任务依赖，而不是选择性汇总。", ""])
    temp = REPORT_MD.with_suffix(f".{time.time_ns()}.tmp.md")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(REPORT_MD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    final_paths = (TRAINING_CSV, VAL_CSV, PERSAMPLE_NPZ, COSTS_CSV, CONTROLS_CSV, WIDTH_CSV, ANALYSIS_JSON, PLOT_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume and not args.smoke:
        raise FileExistsError("Task 07 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    indices = run_03.split_indices()
    if args.smoke:
        smoke(indices, logger)
        return
    if not (WORK / "smoke.json").exists() or not args.resume:
        smoke(indices, logger)
    for backbone, enc, dec in NEW_UNTRAINED:
        evaluate_point(backbone, enc, dec, "untrained", logger)
    for backbone, enc, dec in NEW_H1:
        train_h1(backbone, enc, dec, indices, args.resume, logger)
        evaluate_point(backbone, enc, dec, "h1", logger)
    for backbone, enc, dec in NEW_UNTRAINED:
        measure_cost(backbone, enc, dec, "untrained", logger)
    for backbone, enc, dec in NEW_H1:
        measure_cost(backbone, enc, dec, "h1", logger)
    history = []
    for backbone, enc, dec in NEW_H1:
        history.extend(run_02.read_csv(WORK / f"{backbone}_{point_name(enc, dec)}_h1_training.csv"))
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    old03_rows = run_02.read_csv(run_03.VAL_CSV)
    old04_rows = run_02.read_csv(RESULTS / "04_val_summary.csv")
    old06_rows = run_02.read_csv(RESULTS / "06_val_summary.csv")
    cost07 = run_02.read_csv(WORK / "costs.csv")
    controls = make_controls(new_rows, old03_rows, old04_rows, old06_rows, cost07)
    analysis = analyze_width(controls)
    run_02.atomic_csv(TRAINING_CSV, history, run_03.TRAINING_FIELDS)
    run_02.atomic_csv(VAL_CSV, new_rows, tuple(new_rows[0]))
    run_02.atomic_npz(PERSAMPLE_NPZ, dict(np.load(WORK / "persample.npz").items()))
    run_02.atomic_csv(COSTS_CSV, cost07, tuple(cost07[0]))
    run_02.atomic_csv(CONTROLS_CSV, controls, tuple(controls[0]))
    run_02.atomic_csv(WIDTH_CSV, controls, tuple(controls[0]))
    run_02.atomic_json(ANALYSIS_JSON, analysis)
    make_plot(analysis)
    make_report(history, controls, cost07, analysis)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "val_rows": len(new_rows), "cost_rows": len(cost07), "control_rows": len(controls)})
    logger.info("PASS Task 07: width checks complete")


if __name__ == "__main__":
    main()
