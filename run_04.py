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
WORK = RESULTS / "04_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
TRAINING_CSV = RESULTS / "04_training.csv"
VAL_CSV = RESULTS / "04_val_summary.csv"
PERSAMPLE_NPZ = RESULTS / "04_persample.npz"
COSTS_CSV = RESULTS / "04_costs.csv"
CONTROLS_CSV = RESULTS / "04_controls.csv"
ANALYSIS_JSON = RESULTS / "04_analysis.json"
REPORT_MD = RESULTS / "04_report.md"
SPECS = (("full", 6, 4),)
CONTROL_FIELDS = (
    "backbone", "task", "full_db", "full_h1_db", "generic_gain_db",
    "untrained_db", "trunc_h1_db", "total_recovery_db", "depth_cost_db",
    "specific_recovery_db", "full_lin_mean", "full_h1_lin_mean",
    "untrained_lin_mean", "trunc_h1_lin_mean", "generic_gain_lin",
    "total_recovery_lin", "depth_cost_lin", "full_h0_db", "h0_to_h1_db",
)


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_04")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_04.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def checkpoint_path(backbone, kind):
    return CHECKPOINTS / f"04_{backbone}_{kind}.pt"


def load_head(backbone, kind, device):
    state = torch.load(checkpoint_path(backbone, kind), map_location="cpu", weights_only=False)
    model, _ = run_02.build_model(backbone, device)
    run_03.freeze_backbone(model)
    head = ExitHead(model, kind).to(device).eval()
    head.load_state_dict(state["best_heads"]["e6d4"])
    return model, head


def train(backbone, kind, indices, resume, logger):
    path = checkpoint_path(backbone, kind)
    history_path = WORK / f"{backbone}_{kind}_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, _ = run_02.build_model(backbone, device)
    run_03.freeze_backbone(model)
    heads = torch.nn.ModuleDict({"e6d4": ExitHead(model, kind).to(device)})
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_heads = {}
    best_scores = {"e6d4": math.inf}
    best_global = math.inf
    bad_epochs = 0
    start_epoch = 0
    history = []
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        heads["e6d4"].load_state_dict(state["heads"]["e6d4"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_heads = state["best_heads"]
        best_scores = state["best_scores"]
        best_global = state["best_global"]
        bad_epochs = state["bad_epochs"]
        start_epoch = state["epoch"]
        history = state["history"]
        logger.info("Resuming %s/%s from epoch %d", backbone, kind, start_epoch)
        if bad_epochs >= 3:
            return history
    total_start = time.time()
    for epoch in range(start_epoch + 1, 11):
        jobs = run_03.make_batch_jobs(epoch, indices)
        losses = []
        for batch_index, (dataset, task, batch_indices) in enumerate(jobs):
            data = run_02.cached(dataset, "train")
            imgs = torch.from_numpy(np.array(data[batch_indices], copy=True)).to(device)
            torch.manual_seed(100000 + epoch * 10000 + batch_index)
            torch.cuda.manual_seed(100000 + epoch * 10000 + batch_index)
            imgs_n = add_noise(imgs)
            with torch.no_grad():
                hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, 6, 4)
            loss, _, _ = exit_loss_and_outputs(model, heads["e6d4"], hidden, imgs_n, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {backbone}/{kind}/epoch{epoch}/batch{batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads["e6d4"].parameters(), 0.05)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 100 == 0:
                logger.info("%s/%s epoch=%d batch=%d/%d loss=%.6g", backbone, kind, epoch, batch_index + 1, len(jobs), np.mean(losses[-100:]))
        scheduler.step()
        config_linear, config_db = [], []
        for dataset in run_02.DATASETS[:16]:
            data = run_02.cached(dataset, "val")[indices[f"{dataset}/val_early"]]
            for task in run_02.TASKS:
                values = evaluate_exit(model, heads["e6d4"], data, device, task, 6, 4)
                linear, db = run_02.summarize(values)
                config_linear.append(linear)
                config_db.append(db)
        val_linear = float(np.mean(config_linear))
        val_db = float(np.mean(config_db))
        if val_db < best_scores["e6d4"]:
            best_scores["e6d4"] = val_db
            best_heads["e6d4"] = {name: tensor.detach().cpu().clone() for name, tensor in heads["e6d4"].state_dict().items()}
        if val_db < best_global - 1e-6:
            best_global = val_db
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
            "backbone": backbone, "head_kind": kind, "epoch": epoch, "exit_key": "e6d4",
            "train_loss": f"{np.mean(losses):.17g}", "lr": f"{optimizer.param_groups[0]['lr']:.17g}",
            "val_lin_mean": f"{val_linear:.17g}", "val_db_mean": f"{val_db:.17g}",
            "global_val_db_mean": f"{val_db:.17g}", "bad_epochs": bad_epochs,
            "elapsed_sec": f"{time.time() - total_start:.3f}",
        })
        run_03.save_training_checkpoint(path, epoch, heads, optimizer, scheduler, best_heads, best_scores, best_global, bad_epochs, history, backbone, kind, SPECS)
        run_02.atomic_csv(history_path, history, run_03.TRAINING_FIELDS)
        logger.info("%s/%s epoch=%d loss=%.6g val_db=%.6f bad=%d", backbone, kind, epoch, np.mean(losses), val_db, bad_epochs)
        if bad_epochs >= 3:
            break
    del model, heads, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def smoke(indices, logger):
    results = {}
    for backbone in ("base", "small"):
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
        before = head.pred.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert not torch.equal(before, head.pred.weight)
        assert all(parameter.grad is None for parameter in model.parameters())
        results[backbone] = {"initial_max_diff": float(np.max(np.abs(original - initial))), "loss": float(loss.detach().cpu()), "head_updated": True}
        logger.info("PASS full-depth smoke %s loss=%.6g", backbone, loss)
        del model, head, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    run_02.atomic_json(WORK / "smoke.json", results)


def evaluation_key(backbone, kind, dataset, task):
    return f"full_{kind}/{backbone}/e6d4/{dataset}/{task}/0.5/val/seed100"


def evaluate(backbone, kind, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(row["backbone"], row["head_kind"], row["dataset"], row["task"]) for row in rows}
    if all((backbone, kind, dataset, task) in done for dataset in run_02.DATASETS[:16] for task in run_02.TASKS):
        return
    model, head = load_head(backbone, kind, torch.device("cuda"))
    for dataset in run_02.DATASETS[:16]:
        for task in run_02.TASKS:
            if (backbone, kind, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), torch.device("cuda"), task, 6, 4)
            linear, db = run_02.summarize(values)
            rows.append({
                "method": f"full_{kind}", "backbone": backbone, "head_kind": kind,
                "enc_depth": 6, "dec_depth": 4, "dataset": dataset, "task": task,
                "mask_ratio": 0.5, "split": "val", "seed": 100,
                "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(backbone, kind, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s %s/%s NMSE=%.6f", evaluation_key(backbone, kind, dataset, task), dataset, task, db)
    del model, head
    gc.collect()
    torch.cuda.empty_cache()


def aggregate(rows, backbone, kind, task):
    selected = [row for row in rows if row["backbone"] == backbone and row["head_kind"] == kind and row["task"] == task]
    assert len(selected) == 16, (backbone, kind, task, len(selected))
    return (
        float(np.mean([float(row["nmse_lin"]) for row in selected])),
        float(np.mean([float(row["nmse_db"]) for row in selected])),
    )


def aggregate03(rows, method, backbone, enc, dec, task):
    selected = [row for row in rows if row["method"] == method and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
    assert len(selected) == 16, (method, backbone, enc, dec, task, len(selected))
    return (
        float(np.mean([float(row["nmse_lin"]) for row in selected])),
        float(np.mean([float(row["nmse_db"]) for row in selected])),
    )


def analyze_trigger(new_rows, old_rows):
    generic = {}
    trigger = False
    for backbone in ("base", "small"):
        generic[backbone] = {}
        for task in run_02.TASKS:
            _, full_db = aggregate03(old_rows, "official", backbone, 6, 4, task)
            _, h1_db = aggregate(new_rows, backbone, "h1", task)
            gain = full_db - h1_db
            generic[backbone][task] = {"full_db": full_db, "full_h1_db": h1_db, "generic_gain_db": gain}
            trigger = trigger or gain > 0.05
    analysis = {
        "h0_trigger": trigger,
        "rule": "train full H0 for both backbones if any full-H1 generic gain > 0.05 dB",
        "generic": generic,
    }
    run_02.atomic_json(ANALYSIS_JSON, analysis)
    return analysis


def measure_cost(backbone, kind, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    if any(row["backbone"] == backbone and row["head_kind"] == kind for row in rows):
        return
    noisy = run_02.noisy_input("D17")
    model, head = load_head(backbone, kind, torch.device("cuda"))
    gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), 6, 4)
    gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, 6, 4, torch.device("cuda"), "gpu_b1")
    gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, 6, 4, torch.device("cuda"), "gpu_b64")
    adapter_params = sum(parameter.numel() for name, parameter in head.named_parameters() if name.startswith("adapter."))
    params = run_02.params_exec(model, 6, 4) + adapter_params
    model = model.cpu()
    head = head.cpu()
    torch.cuda.empty_cache()
    cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, 6, 4, torch.device("cpu"), "cpu_b1")
    rows.append({
        "method": f"full_{kind}", "backbone": backbone, "head_kind": kind,
        "enc_depth": 6, "dec_depth": 4, "params_exec": params,
        "gflops": f"{gflops:.17g}", "gpu_b1_ms": f"{gpu_b1:.17g}",
        "gpu_b1_ms_range": f"{gpu_b1_range:.17g}", "gpu_thr_b64": f"{gpu_b64:.17g}",
        "gpu_thr_b64_range": f"{gpu_b64_range:.17g}", "cpu_b1_ms_t1": f"{cpu_b1:.17g}",
        "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
    })
    run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
    logger.info("PASS cost %s/%s GFLOPs=%.6f", backbone, kind, gflops)
    del model, head
    gc.collect()


def make_controls(new_rows, old_rows, include_h0):
    output = []
    for backbone in ("base", "small"):
        for task in run_02.TASKS:
            full_lin, full_db = aggregate03(old_rows, "official", backbone, 6, 4, task)
            h1_lin, h1_db = aggregate(new_rows, backbone, "h1", task)
            untrained_lin, untrained_db = aggregate03(old_rows, "untrained", backbone, 3, 4, task)
            trunc_lin, trunc_db = aggregate03(old_rows, "h1", backbone, 3, 4, task)
            row = {
                "backbone": backbone, "task": task,
                "full_db": f"{full_db:.17g}", "full_h1_db": f"{h1_db:.17g}",
                "generic_gain_db": f"{full_db - h1_db:.17g}",
                "untrained_db": f"{untrained_db:.17g}", "trunc_h1_db": f"{trunc_db:.17g}",
                "total_recovery_db": f"{untrained_db - trunc_db:.17g}",
                "depth_cost_db": f"{trunc_db - h1_db:.17g}",
                "specific_recovery_db": f"{(untrained_db - trunc_db) - (full_db - h1_db):.17g}",
                "full_lin_mean": f"{full_lin:.17g}", "full_h1_lin_mean": f"{h1_lin:.17g}",
                "untrained_lin_mean": f"{untrained_lin:.17g}", "trunc_h1_lin_mean": f"{trunc_lin:.17g}",
                "generic_gain_lin": f"{full_lin - h1_lin:.17g}",
                "total_recovery_lin": f"{untrained_lin - trunc_lin:.17g}",
                "depth_cost_lin": f"{trunc_lin - h1_lin:.17g}",
                "full_h0_db": "", "h0_to_h1_db": "",
            }
            if include_h0:
                _, h0_db = aggregate(new_rows, backbone, "h0", task)
                row["full_h0_db"] = f"{h0_db:.17g}"
                row["h0_to_h1_db"] = f"{h0_db - h1_db:.17g}"
            output.append(row)
    return output


def make_report(history, new_rows, controls, costs, analysis):
    lines = [
        "# Task 04：完整模型输出适配对照报告",
        "",
        "## 协议",
        "",
        "- Base、Small 的完整 `(6,4)+H1` 使用与 Task 03 相同的训练索引、早停索引、batch、噪声、优化器和训练轮数。",
        "- 骨干冻结，H1 位于完整解码器之后。最终评价仅使用 D1–D16 val，未使用 D17、D18 训练或选择。",
        "- 通用适配收益 = 原始完整模型 dB − 完整 H1 dB；总恢复量 = 未训练截断 dB − 截断 H1 dB；深度代价 = 截断 H1 dB − 完整 H1 dB。",
        "- 近似截断特异恢复 = 总恢复量 − 通用适配收益。它只是当前单种子对照下的差分指标，不是独立的统计显著性证明。",
        "",
        "## 完整 H0 触发",
        "",
        f"- 规则：任一完整 H1 通用适配收益 > 0.05 dB 则补两个骨干的完整 H0。实际触发：**{analysis['h0_trigger']}**。",
        "",
        "## 通用适配收益",
        "",
        "| 骨干 | 任务 | 原始完整 dB | 完整 H1 dB | 通用适配收益 dB |",
        "|---|---|---:|---:|---:|",
    ]
    for backbone in ("base", "small"):
        for task in run_02.TASKS:
            row = analysis["generic"][backbone][task]
            lines.append(f"| {backbone} | {task} | {row['full_db']:.6f} | {row['full_h1_db']:.6f} | {row['generic_gain_db']:.6f} |")
    lines.extend([
        "",
        "## 与 `(3,4)` 截断 H1 的对照",
        "",
        "| 骨干 | 任务 | 通用适配收益 | 总恢复量 | 相同 H1 深度代价 | 近似特异恢复 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in controls:
        lines.append(f"| {row['backbone']} | {row['task']} | {float(row['generic_gain_db']):.6f} | {float(row['total_recovery_db']):.6f} | {float(row['depth_cost_db']):.6f} | {float(row['specific_recovery_db']):.6f} |")
    lines.extend([
        "",
        "对应的跨配置线性 NMSE 均值保存在 `04_controls.csv`，dB 均值与线性均值不互相替代。",
        "",
        "## 新增训练历史",
        "",
        "| 骨干 | 头 | 最佳 epoch | 最佳 val 子集 dB |",
        "|---|---|---:|---:|",
    ])
    for backbone in ("base", "small"):
        kinds = ("h1", "h0") if analysis["h0_trigger"] else ("h1",)
        for kind in kinds:
            selected = [row for row in history if row["backbone"] == backbone and row["head_kind"] == kind]
            best = min(selected, key=lambda row: float(row["val_db_mean"]))
            lines.append(f"| {backbone} | {kind} | {best['epoch']} | {float(best['val_db_mean']):.6f} |")
    if analysis["h0_trigger"]:
        lines.extend(["", "## 完整 H0 与 H1", "", "| 骨干 | 任务 | 完整 H0 dB | 完整 H1 dB | H0→H1 改善 dB |", "|---|---|---:|---:|---:|"])
        for row in controls:
            lines.append(f"| {row['backbone']} | {row['task']} | {float(row['full_h0_db']):.6f} | {float(row['full_h1_db']):.6f} | {float(row['h0_to_h1_db']):.6f} |")
    lines.extend([
        "",
        "## 实际成本（D17 形状）",
        "",
        "| 骨干 | 头 | params | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in costs:
        lines.append(f"| {row['backbone']} | {row['head_kind']} | {row['params_exec']} | {float(row['gflops']):.6f} | {float(row['gpu_b1_ms']):.3f} | {float(row['gpu_thr_b64']):.3f} | {float(row['cpu_b1_ms_t1']):.3f} |")
    lines.extend(["", "## 边界", "", "- 本报告只区分通用适配收益、截断总恢复和相同 H1 下的深度代价，不据此直接宣称编码器冗余或解码器不可压缩。", "- 单种子试训不包含训练种子显著性分析；D17 仅用于成本输入，不作泛化评价。", ""])
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
    final_paths = (TRAINING_CSV, VAL_CSV, PERSAMPLE_NPZ, COSTS_CSV, CONTROLS_CSV, ANALYSIS_JSON, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume and not args.smoke:
        raise FileExistsError("Task 04 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    indices = run_03.split_indices()
    if args.smoke:
        smoke(indices, logger)
        return
    if not (WORK / "smoke.json").exists() or not args.resume:
        smoke(indices, logger)
    old_rows = run_02.read_csv(run_03.VAL_CSV)
    for backbone in ("base", "small"):
        train(backbone, "h1", indices, args.resume, logger)
        evaluate(backbone, "h1", logger)
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    if ANALYSIS_JSON.exists() and args.resume:
        analysis = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    else:
        analysis = analyze_trigger(new_rows, old_rows)
    if analysis["h0_trigger"]:
        for backbone in ("base", "small"):
            train(backbone, "h0", indices, args.resume, logger)
            evaluate(backbone, "h0", logger)
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    kinds = ("h1", "h0") if analysis["h0_trigger"] else ("h1",)
    for backbone in ("base", "small"):
        for kind in kinds:
            measure_cost(backbone, kind, logger)
    costs = run_02.read_csv(WORK / "costs.csv")
    history = []
    for backbone in ("base", "small"):
        for kind in kinds:
            history.extend(run_02.read_csv(WORK / f"{backbone}_{kind}_training.csv"))
    controls = make_controls(new_rows, old_rows, analysis["h0_trigger"])
    run_02.atomic_csv(TRAINING_CSV, history, run_03.TRAINING_FIELDS)
    run_02.atomic_csv(VAL_CSV, new_rows, tuple(new_rows[0]))
    arrays = dict(np.load(WORK / "persample.npz").items())
    run_02.atomic_npz(PERSAMPLE_NPZ, arrays)
    run_02.atomic_csv(COSTS_CSV, costs, tuple(costs[0]))
    run_02.atomic_csv(CONTROLS_CSV, controls, CONTROL_FIELDS)
    make_report(history, new_rows, controls, costs, analysis)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "h0_trigger": analysis["h0_trigger"], "val_rows": len(new_rows), "cost_rows": len(costs)})
    logger.info("PASS Task 04: full-depth controls complete")


if __name__ == "__main__":
    main()
