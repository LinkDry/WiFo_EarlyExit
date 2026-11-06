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
from run_06 import InterfaceAdapter

W = Path(__file__).resolve().parent
RESULTS = W / "results"
WORK = RESULTS / "08_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
BACKBONES = ("base", "small", "little", "tiny")
DEPTHS = ((6, 4), (3, 4), (6, 2))
SEEDS = (100, 200, 300)
MASKS = (0.5, 0.25)
CORE = tuple((backbone, enc, dec, seed, mask) for backbone in BACKBONES for enc, dec in DEPTHS for seed in SEEDS for mask in MASKS)
PROBES = tuple((enc, dec, seed) for enc, dec in ((3, 4), (6, 2)) for seed in SEEDS)


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_08")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_08.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def point_name(enc, dec):
    return f"e{enc}d{dec}"


def mask_name(mask):
    return str(mask).replace(".", "p")


def checkpoint_path(backbone, enc, dec, seed, mask, mode="h1"):
    return CHECKPOINTS / f"08_{backbone}_{point_name(enc, dec)}_{mode}_seed{seed}_mask{mask_name(mask)}.pt"


def retained_block(model, enc, dec):
    return model.blocks[enc - 1] if dec == 4 else model.decoder_blocks[dec - 1]


def prepare(backbone, enc, dec, seed, mode, device):
    model, _ = run_02.build_model(backbone, device)
    run_03.freeze_backbone(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    head = ExitHead(model, "h1").to(device)
    block = None
    if mode == "probe":
        block = retained_block(model, enc, dec)
        for parameter in block.parameters():
            parameter.requires_grad_(True)
        trainable = list(head.parameters()) + list(block.parameters())
    else:
        trainable = list(head.parameters())
    return model, block, head, trainable


def load_model(backbone, enc, dec, seed, mask, mode, device):
    model, block, head, _ = prepare(backbone, enc, dec, seed, mode, device)
    state = torch.load(checkpoint_path(backbone, enc, dec, seed, mask, mode), map_location="cpu", weights_only=False)
    head.load_state_dict(state["best"]["head"])
    if mode == "probe":
        block.load_state_dict(state["best"]["block"])
    model.eval()
    head.eval()
    return model, block, head


def make_jobs(train_seed, epoch, mask, subset):
    jobs = []
    split = run_03.split_indices() if subset else None
    for dataset in run_02.DATASETS[:16]:
        selected = split[f"{dataset}/train"] if subset else np.arange(9000)
        rng = np.random.default_rng(train_seed + epoch)
        order = rng.permutation(len(selected))
        batches = [selected[order[start:start + 64]] for start in range(0, len(selected), 64)]
        tasks = ["temporal"] * (len(batches) // 2) + ["fre"] * (len(batches) - len(batches) // 2)
        rng.shuffle(tasks)
        for batch_indices, task in zip(batches, tasks):
            jobs.append((dataset, task, batch_indices))
    rng = np.random.default_rng(train_seed * 10 + epoch)
    rng.shuffle(jobs)
    return jobs


def save_checkpoint(path, epoch, backbone, enc, dec, seed, mask, mode, block, head, best, best_score, bad_epochs, optimizer, scheduler, history):
    state = {
        "epoch": epoch, "backbone": backbone, "enc_depth": enc, "dec_depth": dec,
        "train_seed": seed, "mask_ratio": mask, "mode": mode,
        "current": {"head": head.state_dict(), "block": None if block is None else block.state_dict()},
        "best": best, "best_score": best_score, "bad_epochs": bad_epochs,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
    }
    temp = path.with_suffix(f".{time.time_ns()}.tmp.pt")
    torch.save(state, temp)
    temp.replace(path)


def train_one(backbone, enc, dec, seed, mask, mode, resume, logger):
    path = checkpoint_path(backbone, enc, dec, seed, mask, mode)
    history_path = WORK / f"{backbone}_{point_name(enc, dec)}_{mode}_seed{seed}_mask{mask_name(mask)}_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, block, head, trainable = prepare(backbone, enc, dec, seed, mode, device)
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
        if mode == "probe":
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
    subset = mode == "probe"
    for epoch in range(start_epoch + 1, 11):
        jobs = make_jobs(seed, epoch, mask, subset)
        losses = []
        for batch_index, (dataset, task, batch_indices) in enumerate(jobs):
            imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
            torch.manual_seed(seed * 100000 + epoch * 10000 + batch_index)
            torch.cuda.manual_seed(seed * 100000 + epoch * 10000 + batch_index)
            imgs_n = add_noise(imgs)
            if mode == "probe":
                hidden, mask_tensor = exit_hidden_from_noisy(model, imgs_n, mask, task, enc, dec)
            else:
                with torch.no_grad():
                    hidden, mask_tensor = exit_hidden_from_noisy(model, imgs_n, mask, task, enc, dec)
            loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask_tensor)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {backbone}/{point_name(enc, dec)}/{mode}/seed{seed}/mask{mask}/epoch{epoch}/batch{batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            if block is not None:
                torch.nn.utils.clip_grad_norm_(block.parameters(), 0.05)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 500 == 0:
                logger.info("%s/%s/%s/seed%d/mask%s epoch=%d batch=%d/%d loss=%.6g", backbone, point_name(enc, dec), mode, seed, mask, epoch, batch_index + 1, len(jobs), np.mean(losses[-500:]))
        scheduler.step()
        config_linear, config_db = [], []
        split = run_03.split_indices()
        for dataset in run_02.DATASETS[:16]:
            data = run_02.cached(dataset, "val")[split[f"{dataset}/val_early"]]
            for task in run_02.TASKS:
                values = evaluate_exit(model, head, data, device, task, enc, dec, mask_ratio=mask)
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
            "backbone": backbone, "head_kind": mode, "epoch": epoch, "exit_key": point_name(enc, dec),
            "train_seed": seed, "mask_ratio": mask, "train_loss": f"{np.mean(losses):.17g}",
            "lr": f"{optimizer.param_groups[0]['lr']:.17g}", "val_lin_mean": f"{val_linear:.17g}",
            "val_db_mean": f"{val_db:.17g}", "global_val_db_mean": f"{val_db:.17g}",
            "bad_epochs": bad_epochs, "elapsed_sec": f"{time.time() - total_start:.3f}",
        })
        save_checkpoint(path, epoch, backbone, enc, dec, seed, mask, mode, block, head, best, best_score, bad_epochs, optimizer, scheduler, history)
        run_02.atomic_csv(history_path, history, tuple(history[0]))
        logger.info("%s/%s/%s/seed%d/mask%s epoch=%d loss=%.6g val_db=%.6f bad=%d", backbone, point_name(enc, dec), mode, seed, mask, epoch, np.mean(losses), val_db, bad_epochs)
        if bad_epochs >= 3:
            break
    del model, block, head, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def evaluation_key(backbone, enc, dec, mode, train_seed, mask, dataset, task):
    return f"{mode}/{backbone}/e{enc}d{dec}/train{train_seed}/mask{mask}/{dataset}/{task}/val/eval100"


def evaluate_one(backbone, enc, dec, seed, mask, mode, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv)
    arrays = dict(np.load(work_npz).items()) if work_npz.exists() else {}
    done = {(row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["mode"], int(row["train_seed"]), float(row["mask_ratio"]), row["dataset"], row["task"]) for row in rows}
    if all((backbone, enc, dec, mode, seed, mask, dataset, task) in done for dataset in run_02.DATASETS[:16] for task in run_02.TASKS):
        return
    if mode == "untrained":
        model, _ = run_02.build_model(backbone, torch.device("cuda"))
        run_03.freeze_backbone(model)
        block, head = None, None
    else:
        model, block, head = load_model(backbone, enc, dec, seed, mask, mode, torch.device("cuda"))
    for dataset in run_02.DATASETS[:16]:
        for task in run_02.TASKS:
            if (backbone, enc, dec, mode, seed, mask, dataset, task) in done:
                continue
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), torch.device("cuda"), task, enc, dec, mask_ratio=mask)
            linear, db = run_02.summarize(values)
            rows.append({
                "method": mode, "backbone": backbone, "mode": mode,
                "enc_depth": enc, "dec_depth": dec, "train_seed": seed, "mask_ratio": mask,
                "dataset": dataset, "task": task, "split": "val", "eval_seed": 100,
                "nmse_lin": f"{linear:.17g}", "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(backbone, enc, dec, mode, seed, mask, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s NMSE=%.6f", evaluation_key(backbone, enc, dec, mode, seed, mask, dataset, task), db)
    del model, block, head
    gc.collect()
    torch.cuda.empty_cache()


def measure_cost(backbone, enc, dec, mask, mode, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv)
    if any(row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec and float(row["mask_ratio"]) == mask and row["mode"] == mode for row in rows):
        return
    noisy = run_02.noisy_input("D17")
    model, block, head = load_model(backbone, enc, dec, 100, mask, mode, torch.device("cuda"))
    gflops = run_03.cost_gflops(model, head, noisy[:1].cuda(), enc, dec, mask_ratio=mask)
    gpu_b1, gpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b1", mask_ratio=mask)
    gpu_b64, gpu_b64_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cuda"), "gpu_b64", mask_ratio=mask)
    model = model.cpu()
    head = head.cpu()
    if block is not None:
        block = block.cpu()
    torch.cuda.empty_cache()
    cpu_b1, cpu_b1_range = run_03.cost_timing(model, head, noisy, enc, dec, torch.device("cpu"), "cpu_b1", mask_ratio=mask)
    extra = sum(parameter.numel() for parameter in head.adapter.parameters())
    trainable = sum(parameter.numel() for parameter in head.parameters())
    if block is not None:
        trainable += sum(parameter.numel() for parameter in block.parameters())
    rows.append({
        "method": mode, "backbone": backbone, "mode": mode, "enc_depth": enc, "dec_depth": dec,
        "mask_ratio": mask, "params_exec": run_02.params_exec(model, enc, dec) + extra,
        "trainable_params": trainable, "gflops": f"{gflops:.17g}",
        "gpu_b1_ms": f"{gpu_b1:.17g}", "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
        "gpu_thr_b64": f"{gpu_b64:.17g}", "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
        "cpu_b1_ms_t1": f"{cpu_b1:.17g}", "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
    })
    run_02.atomic_csv(work_csv, rows, tuple(rows[0]))
    logger.info("PASS cost %s/%s/%s/mask%s GFLOPs=%.6f", backbone, point_name(enc, dec), mode, mask, gflops)
    del model, block, head
    gc.collect()


def smoke(logger):
    results = {}
    for backbone in BACKBONES:
        device = torch.device("cuda")
        model, block, head, trainable = prepare(backbone, 6, 4, 999, "h1", device)
        data = run_02.cached("D1", "val")[:32]
        original = evaluate_exit(model, None, data, device, "temporal", 6, 4, mask_ratio=0.25)
        initial = evaluate_exit(model, head, data, device, "temporal", 6, 4, mask_ratio=0.25)
        assert np.array_equal(original, initial)
        jobs = make_jobs(999, 1, 0.25, True)
        dataset, task, batch_indices = jobs[0]
        imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
        torch.manual_seed(999000)
        torch.cuda.manual_seed(999000)
        imgs_n = add_noise(imgs)
        with torch.no_grad():
            hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.25, task, 6, 4)
        loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None for parameter in model.parameters())
        optimizer.step()
        results[backbone] = {"initial_max_diff": float(np.max(np.abs(original - initial))), "loss": float(loss.detach().cpu())}
        logger.info("PASS smoke %s loss=%.6g", backbone, loss)
        del model, block, head, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    model, block, head, trainable = prepare("base", 6, 2, 999, "probe", device)
    jobs = make_jobs(999, 1, 0.5, True)
    dataset, task, batch_indices = jobs[0]
    imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
    imgs_n = add_noise(imgs)
    hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, 6, 2)
    loss, _, _ = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in block.parameters())
    optimizer.step()
    results["base_probe"] = {"loss": float(loss.detach().cpu()), "block_backward": True}
    logger.info("PASS smoke base probe loss=%.6g", loss)
    del model, block, head, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    run_02.atomic_json(WORK / "smoke.json", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    if args.smoke:
        smoke(logger)
        return
    if not (WORK / "smoke.json").exists() or not args.resume:
        smoke(logger)
    backbones = (args.backbone,) if args.backbone else BACKBONES
    for backbone in backbones:
        for enc, dec in DEPTHS:
            for seed in SEEDS:
                for mask in MASKS:
                    train_one(backbone, enc, dec, seed, mask, "h1", args.resume, logger)
                    evaluate_one(backbone, enc, dec, seed, mask, "h1", logger)
        if backbone == "base":
            for enc, dec in ((3, 4), (6, 2)):
                for seed in SEEDS:
                    train_one(backbone, enc, dec, seed, 0.5, "probe", args.resume, logger)
                    evaluate_one(backbone, enc, dec, seed, 0.5, "probe", logger)
        for enc, dec in DEPTHS:
            for mask in MASKS:
                evaluate_one(backbone, enc, dec, 0, mask, "untrained", logger)
                measure_cost(backbone, enc, dec, mask, "h1", logger)
                if backbone == "base" and (enc, dec) != (6, 4) and mask == 0.5:
                    measure_cost(backbone, enc, dec, mask, "probe", logger)
    logger.info("PASS Task 08 training stage: %s", ",".join(backbones))


if __name__ == "__main__":
    main()
