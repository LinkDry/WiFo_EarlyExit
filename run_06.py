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
from torch import nn

import run_02
import run_03
from exit_heads import ExitHead, add_noise, evaluate_exit, exit_hidden_from_noisy, exit_loss_and_outputs

W = Path(__file__).resolve().parent
RESULTS = W / "results"
WORK = RESULTS / "06_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
TRAINING_CSV = RESULTS / "06_training.csv"
VAL_CSV = RESULTS / "06_val_summary.csv"
PERSAMPLE_NPZ = RESULTS / "06_persample.npz"
COSTS_CSV = RESULTS / "06_costs.csv"
CONTROLS_CSV = RESULTS / "06_controls.csv"
ANALYSIS_JSON = RESULTS / "06_analysis.json"
REPORT_MD = RESULTS / "06_report.md"
POINTS = ((3, 4), (6, 2))


class InterfaceAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(channels, channels // 4), nn.GELU(), nn.Linear(channels // 4, channels))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, hidden):
        return hidden + self.layers(hidden)


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_06")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_06.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def point_name(enc, dec):
    return f"e{enc}d{dec}"


def checkpoint_path(enc, dec, mode):
    return CHECKPOINTS / f"06_base_{point_name(enc, dec)}_{mode}.pt"


def retained_block(model, enc, dec):
    return model.blocks[enc - 1] if dec == 4 else model.decoder_blocks[dec - 1]


def prepare(enc, dec, mode, device):
    model, _ = run_02.build_model("base", device)
    run_03.freeze_backbone(model)
    if mode == "probe":
        block = retained_block(model, enc, dec)
        for parameter in block.parameters():
            parameter.requires_grad_(True)
        head = ExitHead(model, "h1").to(device)
        adapter = None
        trainable = list(block.parameters()) + list(head.parameters())
    elif mode == "output_h1":
        block = None
        head = ExitHead(model, "h1").to(device)
        adapter = None
        trainable = list(head.parameters())
    elif mode == "interface":
        block = None
        head = None
        adapter = InterfaceAdapter(model.decoder_embed.out_features).to(device)
        trainable = list(adapter.parameters())
    else:
        raise ValueError(mode)
    return model, block, head, adapter, trainable


def load_point(enc, dec, mode, device):
    model, block, head, adapter, _ = prepare(enc, dec, mode, device)
    state = torch.load(checkpoint_path(enc, dec, mode), map_location="cpu", weights_only=False)
    if mode in ("probe", "output_h1"):
        head.load_state_dict(state["best"]["head"])
        if mode == "probe":
            block.load_state_dict(state["best"]["block"])
    else:
        adapter.load_state_dict(state["best"]["adapter"])
    model.eval()
    if head is not None:
        head.eval()
    if adapter is not None:
        adapter.eval()
    return model, block, head, adapter


def save_checkpoint(path, epoch, enc, dec, mode, block, head, adapter, best, best_score, bad_epochs, optimizer, scheduler, history):
    state = {
        "epoch": epoch, "enc_depth": enc, "dec_depth": dec, "mode": mode,
        "current": {
            "head": None if head is None else head.state_dict(),
            "block": None if block is None else block.state_dict(),
            "adapter": None if adapter is None else adapter.state_dict(),
        },
        "best": best, "best_score": best_score, "bad_epochs": bad_epochs,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
    }
    temp = path.with_suffix(f".{time.time_ns()}.tmp.pt")
    torch.save(state, temp)
    temp.replace(path)


def train_point(enc, dec, mode, indices, resume, logger):
    path = checkpoint_path(enc, dec, mode)
    history_path = WORK / f"{point_name(enc, dec)}_{mode}_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, block, head, adapter, trainable = prepare(enc, dec, mode, device)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best = None
    best_score = math.inf
    bad_epochs = 0
    start_epoch = 0
    history = []
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        current = state["current"]
        if mode in ("probe", "output_h1"):
            head.load_state_dict(current["head"])
            if mode == "probe":
                block.load_state_dict(current["block"])
        else:
            adapter.load_state_dict(current["adapter"])
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
            hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc, dec, adapter)
            loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {point_name(enc, dec)}/{mode}/epoch{epoch}/batch{batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if head is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            if block is not None:
                torch.nn.utils.clip_grad_norm_(block.parameters(), 0.05)
            if adapter is not None:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.05)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 100 == 0:
                logger.info("%s/%s epoch=%d batch=%d loss=%.6g", point_name(enc, dec), mode, epoch, batch_index + 1, np.mean(losses[-100:]))
        scheduler.step()
        config_linear, config_db = [], []
        for dataset in run_02.DATASETS[:16]:
            data = run_02.cached(dataset, "val")[indices[f"{dataset}/val_early"]]
            for task in run_02.TASKS:
                values = evaluate_exit(model, head, data, device, task, enc, dec, interface_adapter=adapter)
                linear, db = run_02.summarize(values)
                config_linear.append(linear)
                config_db.append(db)
        val_linear = float(np.mean(config_linear))
        val_db = float(np.mean(config_db))
        if val_db < best_score:
            best_score = val_db
            best = {
                "head": None if head is None else {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()},
                "block": None if block is None else {name: tensor.detach().cpu().clone() for name, tensor in block.state_dict().items()},
                "adapter": None if adapter is None else {name: tensor.detach().cpu().clone() for name, tensor in adapter.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
            "backbone": "base", "head_kind": mode, "epoch": epoch, "exit_key": point_name(enc, dec),
            "train_loss": f"{np.mean(losses):.17g}", "lr": f"{optimizer.param_groups[0]['lr']:.17g}",
            "val_lin_mean": f"{val_linear:.17g}", "val_db_mean": f"{val_db:.17g}",
            "global_val_db_mean": f"{val_db:.17g}", "bad_epochs": bad_epochs,
            "elapsed_sec": f"{time.time() - total_start:.3f}",
        })
        save_checkpoint(path, epoch, enc, dec, mode, block, head, adapter, best, best_score, bad_epochs, optimizer, scheduler, history)
        run_02.atomic_csv(history_path, history, run_03.TRAINING_FIELDS)
        logger.info("%s/%s epoch=%d loss=%.6g val_db=%.6f bad=%d", point_name(enc, dec), mode, epoch, np.mean(losses), val_db, bad_epochs)
        if bad_epochs >= 3:
            break
    del model, block, head, adapter, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def smoke(indices, logger):
    results = {}
    device = torch.device("cuda")
    for mode in ("probe", "interface"):
        model, block, head, adapter, trainable = prepare(3, 4, mode, device)
        data = run_02.cached("D1", "val")[:32]
        original = evaluate_exit(model, None, data, device, "temporal", 3, 4)
        initial = evaluate_exit(model, head, data, device, "temporal", 3, 4, interface_adapter=adapter)
        assert np.array_equal(original, initial), mode
        jobs = run_03.make_batch_jobs(1, indices, smoke=True)
        dataset, task, batch_indices = jobs[0]
        imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
        torch.manual_seed(101001)
        torch.cuda.manual_seed(101001)
        imgs_n = add_noise(imgs)
        hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, 3, 4, adapter)
        loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        if mode == "probe":
            assert any(parameter.grad is not None for parameter in block.parameters())
            assert any(parameter.grad is not None for parameter in head.parameters())
        else:
            assert any(parameter.grad is not None for parameter in adapter.parameters())
        frozen = model.decoder_blocks[3].parameters() if mode == "interface" else model.blocks[0].parameters()
        assert all(parameter.grad is None for parameter in frozen)
        optimizer.step()
        results[mode] = {"initial_max_diff": float(np.max(np.abs(original - initial))), "loss": float(loss.detach().cpu()), "backward": True}
        logger.info("PASS smoke %s loss=%.6g", mode, loss)
        del model, block, head, adapter, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    run_02.atomic_json(WORK / "smoke.json", results)


def evaluation_key(enc, dec, mode, dataset, task):
    return f"{mode}/base/e{enc}d{dec}/{dataset}/{task}/0.5/val/seed100"


def evaluate_point(enc, dec, mode, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(int(row["enc_depth"]), int(row["dec_depth"]), row["mode"], row["dataset"], row["task"]) for row in rows}
    if all((enc, dec, mode, dataset, task) in done for dataset in run_02.DATASETS[:16] for task in run_02.TASKS):
        return
    model, block, head, adapter = load_point(enc, dec, mode, torch.device("cuda"))
    for dataset in run_02.DATASETS[:16]:
        for task in run_02.TASKS:
            if (enc, dec, mode, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), torch.device("cuda"), task, enc, dec, interface_adapter=adapter)
            linear, db = run_02.summarize(values)
            rows.append({
                "method": mode, "backbone": "base", "mode": mode,
                "enc_depth": enc, "dec_depth": dec, "dataset": dataset, "task": task,
                "mask_ratio": 0.5, "split": "val", "seed": 100,
                "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(enc, dec, mode, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s NMSE=%.6f", evaluation_key(enc, dec, mode, dataset, task), db)
    del model, block, head, adapter
    gc.collect()
    torch.cuda.empty_cache()


def measure_cost(enc, dec, mode, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    if any(int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["mode"] == mode for row in rows):
        return
    noisy = run_02.noisy_input("D17")
    model, block, head, adapter = load_point(enc, dec, mode, torch.device("cuda"))
    gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), enc, dec, adapter)
    gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b1", adapter)
    gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b64", adapter)
    model = model.cpu()
    if head is not None:
        head = head.cpu()
    if adapter is not None:
        adapter = adapter.cpu()
    if block is not None:
        block = block.cpu()
    torch.cuda.empty_cache()
    cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cpu"), "cpu_b1", adapter)
    if mode == "probe":
        extra = sum(parameter.numel() for parameter in head.adapter.parameters())
        trainable = sum(parameter.numel() for parameter in head.parameters()) + sum(parameter.numel() for parameter in block.parameters())
    elif mode == "output_h1":
        extra = sum(parameter.numel() for parameter in head.adapter.parameters())
        trainable = sum(parameter.numel() for parameter in head.parameters())
    else:
        extra = sum(parameter.numel() for parameter in adapter.parameters())
        trainable = extra
    rows.append({
        "method": mode, "backbone": "base", "mode": mode, "enc_depth": enc, "dec_depth": dec,
        "params_exec": run_02.params_exec(model, enc, dec) + extra,
        "trainable_params": trainable, "gflops": f"{gflops:.17g}",
        "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
        "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
        "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
    })
    run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
    logger.info("PASS cost %s/%s GFLOPs=%.6f", point_name(enc, dec), mode, gflops)
    del model, block, head, adapter
    gc.collect()


def aggregate(rows, enc, dec, mode, task):
    selected = [row for row in rows if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["mode"] == mode and row["task"] == task]
    assert len(selected) == 16, (enc, dec, mode, task, len(selected))
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def aggregate03(rows, method, enc, dec, task):
    selected = [row for row in rows if row["method"] == method and row["backbone"] == "base" and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
    assert len(selected) == 16
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def output_h1_db(new_rows, old_rows, enc, dec, task):
    selected = [row for row in old_rows if row["method"] == "h1" and row["backbone"] == "base" and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
    if len(selected) == 16:
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    return aggregate(new_rows, enc, dec, "output_h1", task)


def analyze_trigger(new_rows, old_rows):
    output = {}
    for enc, dec in POINTS:
        gains = {}
        for task in run_02.TASKS:
            output_h1 = output_h1_db(new_rows, old_rows, enc, dec, task)
            probe = aggregate(new_rows, enc, dec, "probe", task)
            gains[task] = output_h1 - probe
        output[point_name(enc, dec)] = {
            "probe_gain_db": gains,
            "interface_trigger": max(gains.values()) > 0.05,
        }
    run_02.atomic_json(ANALYSIS_JSON, output)
    return output


def make_controls(new_rows, old_rows, analysis):
    full04 = json.loads((RESULTS / "04_analysis.json").read_text(encoding="utf-8"))
    output = []
    for enc, dec in POINTS:
        for task in run_02.TASKS:
            untrained = aggregate03(old_rows, "untrained", enc, dec, task)
            output_h1 = output_h1_db(new_rows, old_rows, enc, dec, task)
            probe = aggregate(new_rows, enc, dec, "probe", task)
            full_h1 = full04["generic"]["base"][task]["full_h1_db"]
            triggered = analysis[point_name(enc, dec)]["interface_trigger"]
            row = {
                "backbone": "base", "enc_depth": enc, "dec_depth": dec, "task": task,
                "untrained_db": f"{untrained:.17g}", "output_h1_db": f"{output_h1:.17g}",
                "probe_db": f"{probe:.17g}", "probe_gain_db": f"{output_h1 - probe:.17g}",
                "full_h1_db": f"{full_h1:.17g}", "residual_depth_cost_db": f"{probe - full_h1:.17g}",
                "interface_db": "", "interface_gain_db": "",
            }
            if triggered:
                interface = aggregate(new_rows, enc, dec, "interface", task)
                row["interface_db"] = f"{interface:.17g}"
                row["interface_gain_db"] = f"{output_h1 - interface:.17g}"
            output.append(row)
    return output


def make_report(history, controls, costs, analysis):
    lines = [
        "# Task 06：Base 恢复能力探针报告",
        "",
        "## 协议",
        "",
        "- `(3,4)` 仅解冻最后一个保留编码器块与输出端 H1；`(6,2)` 仅解冻最后一个保留解码器块与输出端 H1。",
        "- 两个探针都从官方 Base 和零初始化 H1 开始，训练数据、索引、优化器和 10 epoch 上限与 Task 03 相同，不从既有 H1 续训。",
        "- 解冻增益 = 输出端 H1 dB − 有限解冻探针 dB。若任一任务增益 >0.05 dB，则为该点训练中间接口适配器。",
        "",
        "## 接口分支触发",
        "",
        "| 深度 | temporal 解冻增益 | fre 解冻增益 | 触发接口 |",
        "|---|---:|---:|---|",
    ]
    for enc, dec in POINTS:
        row = analysis[point_name(enc, dec)]
        lines.append(f"| e{enc}d{dec} | {row['probe_gain_db']['temporal']:.6f} | {row['probe_gain_db']['fre']:.6f} | {row['interface_trigger']} |")
    lines.extend([
        "",
        "## 恢复对照",
        "",
        "剩余深度代价 = 有限解冻探针 dB − 完整 H1 dB；接口改善 = 输出端 H1 dB − 中间接口适配器 dB。",
        "",
        "| 深度 | 任务 | 未训练 | 输出 H1 | 有限解冻 | 解冻增益 | 完整 H1 | 剩余深度代价 | 接口适配 | 接口改善 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in controls:
        interface = "—" if row["interface_db"] == "" else f"{float(row['interface_db']):.6f}"
        interface_gain = "—" if row["interface_gain_db"] == "" else f"{float(row['interface_gain_db']):.6f}"
        lines.append(f"| e{row['enc_depth']}d{row['dec_depth']} | {row['task']} | {float(row['untrained_db']):.6f} | {float(row['output_h1_db']):.6f} | {float(row['probe_db']):.6f} | {float(row['probe_gain_db']):.6f} | {float(row['full_h1_db']):.6f} | {float(row['residual_depth_cost_db']):.6f} | {interface} | {interface_gain} |")
    lines.extend([
        "",
        "## 训练历史",
        "",
        "| 深度 | 方法 | 最佳 epoch | 最佳 val 子集 dB |",
        "|---|---|---:|---:|",
    ])
    seen = set()
    for row in history:
        key = (row["exit_key"], row["head_kind"])
        if key in seen:
            continue
        seen.add(key)
        selected = [item for item in history if item["exit_key"] == key[0] and item["head_kind"] == key[1]]
        best = min(selected, key=lambda item: float(item["val_db_mean"]))
        lines.append(f"| {key[0]} | {key[1]} | {best['epoch']} | {float(best['val_db_mean']):.6f} |")
    lines.extend([
        "",
        "## 实际成本与可训练参数（D17 形状）",
        "",
        "| 深度 | 方法 | 执行参数 | 可训练参数 | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in costs:
        lines.append(f"| e{row['enc_depth']}d{row['dec_depth']} | {row['mode']} | {row['params_exec']} | {row['trainable_params']} | {float(row['gflops']):.6f} | {float(row['gpu_b1_ms']):.3f} | {float(row['gpu_thr_b64']):.3f} | {float(row['cpu_b1_ms_t1']):.3f} |")
    lines.extend(["", "## 边界", "", "- 解冻增益只说明当前预算下冻结限制的影响，不等同于证明接口问题，也不证明任何深度绝对不可恢复。", "- 接口分支只在预设规则触发时执行；单种子结果不作统计显著性或宽度规律结论。", ""])
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
        raise FileExistsError("Task 06 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    indices = run_03.split_indices()
    if args.smoke:
        smoke(indices, logger)
        return
    if not (WORK / "smoke.json").exists() or not args.resume:
        smoke(indices, logger)
    old_rows = run_02.read_csv(run_03.VAL_CSV)
    for enc, dec in POINTS:
        train_point(enc, dec, "probe", indices, args.resume, logger)
        evaluate_point(enc, dec, "probe", logger)
    train_point(6, 2, "output_h1", indices, args.resume, logger)
    evaluate_point(6, 2, "output_h1", logger)
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    if ANALYSIS_JSON.exists() and args.resume:
        analysis = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    else:
        analysis = analyze_trigger(new_rows, old_rows)
    for enc, dec in POINTS:
        if analysis[point_name(enc, dec)]["interface_trigger"]:
            train_point(enc, dec, "interface", indices, args.resume, logger)
            evaluate_point(enc, dec, "interface", logger)
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    for enc, dec in POINTS:
        measure_cost(enc, dec, "probe", logger)
        if analysis[point_name(enc, dec)]["interface_trigger"]:
            measure_cost(enc, dec, "interface", logger)
    measure_cost(6, 2, "output_h1", logger)
    history = []
    for enc, dec in POINTS:
        modes = ["probe"] + (["interface"] if analysis[point_name(enc, dec)]["interface_trigger"] else [])
        if (enc, dec) == (6, 2):
            modes.append("output_h1")
        for mode in modes:
            history.extend(run_02.read_csv(WORK / f"{point_name(enc, dec)}_{mode}_training.csv"))
    costs = run_02.read_csv(WORK / "costs.csv")
    controls = make_controls(new_rows, old_rows, analysis)
    run_02.atomic_csv(TRAINING_CSV, history, run_03.TRAINING_FIELDS)
    run_02.atomic_csv(VAL_CSV, new_rows, tuple(new_rows[0]))
    run_02.atomic_npz(PERSAMPLE_NPZ, dict(np.load(WORK / "persample.npz").items()))
    run_02.atomic_csv(COSTS_CSV, costs, tuple(costs[0]))
    run_02.atomic_csv(CONTROLS_CSV, controls, tuple(controls[0]))
    make_report(history, controls, costs, analysis)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "val_rows": len(new_rows), "cost_rows": len(costs), "analysis": analysis})
    logger.info("PASS Task 06: recovery probes complete")


if __name__ == "__main__":
    main()
