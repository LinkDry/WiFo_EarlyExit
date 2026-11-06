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
WORK = R / "09_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
TRAINING_CSV = R / "09_training.csv"
VAL_CSV = R / "09_val_summary.csv"
PERSAMPLE_NPZ = R / "09_persample.npz"
COSTS_CSV = R / "09_costs.csv"
CONTROLS_CSV = R / "09_controls.csv"
ANALYSIS_JSON = R / "09_analysis.json"
PLOT_PNG = R / "09_recovery.png"
REPORT_MD = R / "09_report.md"
SEEDS = (100, 200, 300)
NEW_MODELS = (
    ("full_h1", 6, 4, 200), ("full_h1", 6, 4, 300),
    ("full_enc_update", 6, 4, 100), ("full_enc_update", 6, 4, 200), ("full_enc_update", 6, 4, 300),
    ("full_dec_update", 6, 4, 100), ("full_dec_update", 6, 4, 200), ("full_dec_update", 6, 4, 300),
    ("trunc_h1", 3, 4, 200), ("trunc_h1", 3, 4, 300),
    ("trunc_h1", 6, 2, 200), ("trunc_h1", 6, 2, 300),
)


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_09")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_09.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def point_name(enc, dec):
    return f"e{enc}d{dec}"


def checkpoint_path(mode, enc, dec, seed):
    return CHECKPOINTS / f"09_{mode}_{point_name(enc, dec)}_seed{seed}.pt"


def prepare(mode, seed, device):
    model, _ = run_02.build_model("base", device)
    run_03.freeze_backbone(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    head = ExitHead(model, "h1").to(device)
    block = None
    if mode == "full_enc_update":
        block = model.blocks[5]
    elif mode == "full_dec_update":
        block = model.decoder_blocks[3]
    if block is not None:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
        trainable = list(head.parameters()) + list(block.parameters())
    else:
        trainable = list(head.parameters())
    return model, block, head, trainable


def load_model(mode, enc, dec, seed, device):
    model, block, head, _ = prepare(mode, seed, device)
    state = torch.load(checkpoint_path(mode, enc, dec, seed), map_location="cpu", weights_only=False)
    head.load_state_dict(state["best"]["head"])
    if block is not None:
        block.load_state_dict(state["best"]["block"])
    model.eval()
    head.eval()
    return model, block, head


def save_checkpoint(path, epoch, mode, enc, dec, seed, block, head, best, best_score, bad_epochs, optimizer, scheduler, history):
    state = {
        "epoch": epoch, "mode": mode, "enc_depth": enc, "dec_depth": dec, "train_seed": seed,
        "current": {"head": head.state_dict(), "block": None if block is None else block.state_dict()},
        "best": best, "best_score": best_score, "bad_epochs": bad_epochs,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
    }
    temp = path.with_suffix(f".{time.time_ns()}.tmp.pt")
    torch.save(state, temp)
    temp.replace(path)


def train_one(mode, enc, dec, seed, resume, logger):
    path = checkpoint_path(mode, enc, dec, seed)
    history_path = WORK / f"{mode}_{point_name(enc, dec)}_seed{seed}_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, block, head, trainable = prepare(mode, seed, device)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best = None
    best_score = math.inf
    bad_epochs = 0
    start_epoch = 0
    history = []
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(state["current"]["head"])
        if block is not None:
            block.load_state_dict(state["current"]["block"])
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
            if block is None:
                with torch.no_grad():
                    hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc, dec)
            else:
                hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc, dec)
            loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {mode}/{point_name(enc, dec)}/seed{seed}/epoch{epoch}/batch{batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            if block is not None:
                torch.nn.utils.clip_grad_norm_(block.parameters(), 0.05)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 100 == 0:
                logger.info("%s/%s/seed%d epoch=%d batch=%d loss=%.6g", mode, point_name(enc, dec), seed, epoch, batch_index + 1, np.mean(losses[-100:]))
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
            best = {
                "head": {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()},
                "block": None if block is None else {name: tensor.detach().cpu().clone() for name, tensor in block.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({
            "mode": mode, "backbone": "base", "head_kind": mode, "epoch": epoch,
            "exit_key": point_name(enc, dec), "train_seed": seed, "mask_ratio": 0.5,
            "train_loss": f"{np.mean(losses):.17g}", "lr": f"{optimizer.param_groups[0]['lr']:.17g}",
            "val_lin_mean": f"{val_linear:.17g}", "val_db_mean": f"{val_db:.17g}",
            "global_val_db_mean": f"{val_db:.17g}", "bad_epochs": bad_epochs,
            "elapsed_sec": f"{time.time() - total_start:.3f}",
        })
        save_checkpoint(path, epoch, mode, enc, dec, seed, block, head, best, best_score, bad_epochs, optimizer, scheduler, history)
        run_02.atomic_csv(history_path, history, tuple(history[0]))
        logger.info("%s/%s/seed%d epoch=%d loss=%.6g val_db=%.6f bad=%d", mode, point_name(enc, dec), seed, epoch, np.mean(losses), val_db, bad_epochs)
        if bad_epochs >= 3:
            break
    del model, block, head, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def evaluation_key(mode, enc, dec, seed, dataset, task):
    return f"{mode}/base/e{enc}d{dec}/train{seed}/mask0.5/{dataset}/{task}/val/eval100"


def evaluate_one(mode, enc, dec, seed, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(row["mode"], int(row["enc_depth"]), int(row["dec_depth"]), int(row["train_seed"]), row["dataset"], row["task"]) for row in rows}
    if all((mode, enc, dec, seed, dataset, task) in done for dataset in run_02.DATASETS[:16] for task in run_02.TASKS):
        return
    model, block, head = load_model(mode, enc, dec, seed, torch.device("cuda"))
    for dataset in run_02.DATASETS[:16]:
        for task in run_02.TASKS:
            if (mode, enc, dec, seed, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), torch.device("cuda"), task, enc, dec)
            linear, db = run_02.summarize(values)
            rows.append({
                "method": mode, "backbone": "base", "mode": mode,
                "enc_depth": enc, "dec_depth": dec, "train_seed": seed, "mask_ratio": 0.5,
                "dataset": dataset, "task": task, "split": "val", "eval_seed": 100,
                "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(mode, enc, dec, seed, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s NMSE=%.6f", evaluation_key(mode, enc, dec, seed, dataset, task), db)
    del model, block, head
    gc.collect()
    torch.cuda.empty_cache()


def measure_cost(mode, enc, dec, seed, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    if any(row["mode"] == mode and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and int(row["train_seed"]) == seed for row in rows):
        return
    noisy = run_02.noisy_input("D17")
    model, block, head = load_model(mode, enc, dec, seed, torch.device("cuda"))
    gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), enc, dec)
    gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b1")
    gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b64")
    model = model.cpu()
    head = head.cpu()
    if block is not None:
        block = block.cpu()
    torch.cuda.empty_cache()
    cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cpu"), "cpu_b1")
    extra = sum(parameter.numel() for parameter in head.adapter.parameters())
    trainable = sum(parameter.numel() for parameter in head.parameters())
    if block is not None:
        trainable += sum(parameter.numel() for parameter in block.parameters())
    rows.append({
        "method": mode, "backbone": "base", "mode": mode, "enc_depth": enc, "dec_depth": dec,
        "train_seed": seed, "params_exec": run_02.params_exec(model, enc, dec) + extra,
        "trainable_params": trainable, "gflops": f"{gflops:.17g}",
        "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
        "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
        "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
    })
    run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
    logger.info("PASS cost %s/%s/seed%d GFLOPs=%.6f", mode, point_name(enc, dec), seed, gflops)
    del model, block, head
    gc.collect()


def aggregate(rows, mode, enc, dec, seed, task):
    selected = [row for row in rows if row["mode"] == mode and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and int(row["train_seed"]) == seed and row["task"] == task]
    assert len(selected) == 16, (mode, enc, dec, seed, task, len(selected))
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def aggregate03(rows, method, enc, dec, task):
    selected = [row for row in rows if row["method"] == method and row["backbone"] == "base" and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
    assert len(selected) == 16
    return float(np.mean([float(row["nmse_db"]) for row in selected]))


def full_h1_db(new_rows, old04_rows, seed, task):
    if seed == 100:
        selected = [row for row in old04_rows if row["head_kind"] == "h1" and row["backbone"] == "base" and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    return aggregate(new_rows, "full_h1", 6, 4, seed, task)


def trunc_h1_db(new_rows, old03_rows, old06_rows, enc, dec, seed, task):
    if seed == 100 and (enc, dec) == (3, 4):
        return aggregate03(old03_rows, "h1", enc, dec, task)
    if seed == 100 and (enc, dec) == (6, 2):
        selected = [row for row in old06_rows if row["mode"] == "output_h1" and row["task"] == task]
        assert len(selected) == 16
        return float(np.mean([float(row["nmse_db"]) for row in selected]))
    return aggregate(new_rows, "trunc_h1", enc, dec, seed, task)


def make_controls(new_rows, old03_rows, old04_rows, old06_rows, old08_rows):
    output = []
    for enc, dec in ((3, 4), (6, 2)):
        update_mode = "full_enc_update" if dec == 4 else "full_dec_update"
        for seed in SEEDS:
            for task in run_02.TASKS:
                full_original = aggregate03(old03_rows, "official", 6, 4, task)
                full_h1 = full_h1_db(new_rows, old04_rows, seed, task)
                full_update = aggregate(new_rows, update_mode, 6, 4, seed, task)
                untrained = aggregate03(old03_rows, "untrained", enc, dec, task)
                trunc_h1 = trunc_h1_db(new_rows, old03_rows, old06_rows, enc, dec, seed, task)
                probe = aggregate(old08_rows, "probe", enc, dec, seed, task)
                output.append({
                    "backbone": "base", "enc_depth": enc, "dec_depth": dec, "train_seed": seed, "task": task,
                    "full_original_db": f"{full_original:.17g}", "full_h1_db": f"{full_h1:.17g}",
                    "full_update_db": f"{full_update:.17g}", "untrained_db": f"{untrained:.17g}",
                    "trunc_h1_db": f"{trunc_h1:.17g}", "probe_db": f"{probe:.17g}",
                    "full_update_gain_vs_h1_db": f"{full_h1 - full_update:.17g}",
                    "probe_gain_vs_h1_db": f"{trunc_h1 - probe:.17g}",
                    "residual_depth_cost_db": f"{probe - full_update:.17g}",
                    "depth_cost_vs_full_h1_db": f"{probe - full_h1:.17g}",
                })
    return output


def analyze_controls(controls):
    output = []
    for enc, dec in ((3, 4), (6, 2)):
        for task in run_02.TASKS:
            selected = [row for row in controls if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
            output.append({
                "enc_depth": enc, "dec_depth": dec, "task": task,
                "full_update_gain_mean_db": float(np.mean([float(row["full_update_gain_vs_h1_db"]) for row in selected])),
                "probe_gain_mean_db": float(np.mean([float(row["probe_gain_vs_h1_db"]) for row in selected])),
                "probe_gain_std_db": float(np.std([float(row["probe_gain_vs_h1_db"]) for row in selected])),
                "residual_mean_db": float(np.mean([float(row["residual_depth_cost_db"]) for row in selected])),
                "residual_std_db": float(np.std([float(row["residual_depth_cost_db"]) for row in selected])),
                "residual_min_db": min(float(row["residual_depth_cost_db"]) for row in selected),
                "residual_max_db": max(float(row["residual_depth_cost_db"]) for row in selected),
            })
    return output


def make_plot(controls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, task in zip(axes, run_02.TASKS):
        for enc, dec, label in ((3, 4, "Encoder (3,4)"), (6, 2, "Decoder (6,2)")):
            selected = [row for row in controls if int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and row["task"] == task]
            seeds = [int(row["train_seed"]) for row in selected]
            values = [float(row["residual_depth_cost_db"]) for row in selected]
            axis.plot(seeds, values, marker="o", label=label)
        axis.set_xticks(SEEDS)
        axis.set_title(task)
        axis.set_ylabel("Residual depth cost vs matched full update (dB)")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=180)
    plt.close(fig)


def make_report(controls, analysis, costs):
    lines = [
        "# Task 09：Base 同预算有限解冻公平对照报告",
        "",
        "## 协议",
        "",
        "- 所有新旧对照均使用每配置 2000 个 train 样本、相同早停与最终 val、mask=0.5、三个种子和相同评价噪声。",
        "- 编码器侧参考为完整 E6+H1，解码器侧参考为完整 D4+H1；同时保留完整 H1 参考。",
        "- 新训练仅 12 个缺失对照，其他结果在协议一致时复用。",
        "",
        "## 公平恢复对照",
        "",
        "| 深度 | 种子 | 任务 | 完整 H1 | 完整有限解冻 | 截断 H1 | 截断有限解冻 | 完整额外恢复 | 截断额外恢复 | 剩余深度代价 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controls:
        lines.append(f"| e{row['enc_depth']}d{row['dec_depth']} | {row['train_seed']} | {row['task']} | {float(row['full_h1_db']):.4f} | {float(row['full_update_db']):.4f} | {float(row['trunc_h1_db']):.4f} | {float(row['probe_db']):.4f} | {float(row['full_update_gain_vs_h1_db']):.4f} | {float(row['probe_gain_vs_h1_db']):.4f} | {float(row['residual_depth_cost_db']):.4f} |")
    lines.extend([
        "",
        "## 多种子汇总",
        "",
        "| 深度 | 任务 | 完整额外恢复均值 | 截断额外恢复均值±标准差 | 剩余代价均值±标准差 | 剩余代价范围 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in analysis:
        lines.append(f"| e{row['enc_depth']}d{row['dec_depth']} | {row['task']} | {row['full_update_gain_mean_db']:.4f} | {row['probe_gain_mean_db']:.4f}±{row['probe_gain_std_db']:.4f} | {row['residual_mean_db']:.4f}±{row['residual_std_db']:.4f} | {row['residual_min_db']:.4f}–{row['residual_max_db']:.4f} |")
    lines.extend([
        "",
        "## 新增成本（D17 形状）",
        "",
        "| 模式 | 深度 | 种子 | 执行参数 | 可训练参数 | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in costs:
        lines.append(f"| {row['mode']} | e{row['enc_depth']}d{row['dec_depth']} | {row['train_seed']} | {row['params_exec']} | {row['trainable_params']} | {float(row['gflops']):.6f} | {float(row['gpu_b1_ms']):.3f} | {float(row['gpu_thr_b64']):.3f} | {float(row['cpu_b1_ms_t1']):.3f} |")
    lines.extend(["", "## 边界", "", "- 完整有限解冻也获得恢复权限后，原先的编码器剩余差距不能原样沿用，应以本报告的新对照为准。", "- 有限解冻是机制诊断，不自动等同于轻量部署方法；训练参数量、数据预算和实际成本需分别解释。", ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    final_paths = (TRAINING_CSV, VAL_CSV, PERSAMPLE_NPZ, COSTS_CSV, CONTROLS_CSV, ANALYSIS_JSON, PLOT_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume:
        raise FileExistsError("Task 09 outputs exist; use --resume only to continue")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    for mode, enc, dec, seed in NEW_MODELS:
        train_one(mode, enc, dec, seed, args.resume, logger)
        evaluate_one(mode, enc, dec, seed, logger)
        measure_cost(mode, enc, dec, seed, logger)
    history = []
    for mode, enc, dec, seed in NEW_MODELS:
        history.extend(run_02.read_csv(WORK / f"{mode}_{point_name(enc, dec)}_seed{seed}_training.csv"))
    new_rows = run_02.read_csv(WORK / "val_summary.csv")
    controls = make_controls(new_rows, run_02.read_csv(run_03.VAL_CSV), run_02.read_csv(R / "04_val_summary.csv"), run_02.read_csv(R / "06_val_summary.csv"), run_02.read_csv(R / "08_work/val_summary.csv"))
    analysis = analyze_controls(controls)
    run_02.atomic_csv(TRAINING_CSV, history, tuple(history[0]))
    run_02.atomic_csv(VAL_CSV, new_rows, tuple(new_rows[0]))
    run_02.atomic_npz(PERSAMPLE_NPZ, dict(np.load(WORK / "persample.npz").items()))
    run_02.atomic_csv(COSTS_CSV, run_02.read_csv(WORK / "costs.csv"), tuple(run_02.read_csv(WORK / "costs.csv")[0]))
    run_02.atomic_csv(CONTROLS_CSV, controls, tuple(controls[0]))
    run_02.atomic_json(ANALYSIS_JSON, analysis)
    make_plot(controls)
    make_report(controls, analysis, run_02.read_csv(WORK / "costs.csv"))
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "new_models": len(NEW_MODELS), "val_rows": len(new_rows)})
    logger.info("PASS Task 09: fair recovery controls complete")


if __name__ == "__main__":
    main()
