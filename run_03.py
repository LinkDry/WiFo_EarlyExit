import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import run_02
from exit_heads import ExitHead, add_noise, evaluate_exit, exit_hidden_from_noisy, exit_loss_and_outputs

W = Path(__file__).resolve().parent
RESULTS = W / "results"
WORK = RESULTS / "03_work"
CHECKPOINTS = W / "checkpoints"
LOGS = W / "logs"
TRAINING_CSV = RESULTS / "03_training.csv"
VAL_CSV = RESULTS / "03_val_summary.csv"
AGGREGATE_CSV = RESULTS / "03_val_aggregate.csv"
PERSAMPLE_NPZ = RESULTS / "03_persample.npz"
COSTS_CSV = RESULTS / "03_costs.csv"
PARETO_CSV = RESULTS / "03_pareto.csv"
SELECTION_JSON = RESULTS / "03_h1_selection.json"
G1_PNG = RESULTS / "03_g1.png"
REPORT_MD = RESULTS / "03_report.md"
SPECS = (("decoder", 6, 2), ("encoder", 3, 4), ("lowcost", 2, 1))
TRAINING_FIELDS = (
    "backbone", "head_kind", "epoch", "exit_key", "train_loss", "lr",
    "val_lin_mean", "val_db_mean", "global_val_db_mean", "bad_epochs", "elapsed_sec",
)
VAL_FIELDS = (
    "method", "backbone", "enc_depth", "dec_depth", "dataset", "task",
    "mask_ratio", "split", "nmse_lin", "nmse_db",
)
AGGREGATE_FIELDS = (
    "method", "backbone", "enc_depth", "dec_depth", "task", "nmse_lin_mean",
    "nmse_db_mean", "recovery_db", "degradation_from_full_db",
)
COST_FIELDS = (
    "method", "backbone", "enc_depth", "dec_depth", "params_exec", "gflops",
    "gpu_b1_ms", "gpu_b1_ms_range", "gpu_thr_b64", "gpu_thr_b64_range",
    "cpu_b1_ms_t1", "cpu_b1_ms_t1_range",
)
PARETO_FIELDS = (
    "method", "backbone", "enc_depth", "dec_depth", "task", "cost_metric",
    "cost", "nmse_db", "required_db", "gap_db",
)


def exit_key(enc_depth, dec_depth):
    return f"e{enc_depth}d{dec_depth}"


def make_logger():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_03")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_03.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def split_indices():
    path = WORK / "split_indices.npz"
    if path.exists():
        return dict(np.load(path).items())
    arrays = {}
    for dataset in run_02.DATASETS[:16]:
        rng = np.random.default_rng(100)
        arrays[f"{dataset}/train"] = np.sort(rng.permutation(9000)[:2000])
        arrays[f"{dataset}/val_early"] = np.sort(rng.permutation(2000)[:256])
    run_02.atomic_npz(path, arrays)
    return arrays


def make_batch_jobs(epoch, indices, batch_size=64, smoke=False):
    jobs = []
    datasets = ("D1",) if smoke else run_02.DATASETS[:16]
    for dataset in datasets:
        selected = indices[f"{dataset}/train"]
        if smoke:
            selected = selected[:64]
        rng = np.random.default_rng(100 + epoch)
        order = rng.permutation(len(selected))
        batches = [selected[order[start:start + batch_size]] for start in range(0, len(selected), batch_size)]
        tasks = ["temporal"] * (len(batches) // 2) + ["fre"] * (len(batches) - len(batches) // 2)
        rng.shuffle(tasks)
        for batch_indices, task in zip(batches, tasks):
            jobs.append((dataset, task, batch_indices))
        if smoke:
            jobs = jobs[:1]
    rng = np.random.default_rng(1000 + epoch)
    rng.shuffle(jobs)
    return jobs


def freeze_backbone(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_best_head(backbone, kind, key, device):
    checkpoint = torch.load(CHECKPOINTS / f"03_{backbone}_{kind}.pt", map_location="cpu", weights_only=False)
    model, _ = run_02.build_model(backbone, device)
    freeze_backbone(model)
    head = ExitHead(model, kind).to(device).eval()
    head.load_state_dict(checkpoint["best_heads"][key])
    return model, head


def training_checkpoint(backbone, kind):
    return CHECKPOINTS / f"03_{backbone}_{kind}.pt"


def save_training_checkpoint(path, epoch, heads, optimizer, scheduler, best_heads, best_scores, best_global, bad_epochs, history, backbone, kind, specs):
    state = {
        "epoch": epoch,
        "backbone": backbone,
        "head_kind": kind,
        "specs": specs,
        "heads": {key: head.state_dict() for key, head in heads.items()},
        "best_heads": best_heads,
        "best_scores": best_scores,
        "best_global": best_global,
        "bad_epochs": bad_epochs,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
    }
    temp = path.with_suffix(f".{os.getpid()}.tmp.pt")
    torch.save(state, temp)
    os.replace(temp, path)


def train_backbone(backbone, kind, specs, indices, resume, logger):
    path = training_checkpoint(backbone, kind)
    history_path = WORK / f"{backbone}_{kind}_training.csv"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    device = torch.device("cuda")
    model, _ = run_02.build_model(backbone, device)
    freeze_backbone(model)
    heads = torch.nn.ModuleDict({exit_key(enc, dec): ExitHead(model, kind).to(device) for _, enc, dec in specs})
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_heads = {}
    best_scores = {exit_key(enc, dec): math.inf for _, enc, dec in specs}
    best_global = math.inf
    bad_epochs = 0
    start_epoch = 0
    history = []
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, head in heads.items():
            head.load_state_dict(state["heads"][key])
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
            del model, heads, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
            return history
    total_start = time.time()
    for epoch in range(start_epoch + 1, 11):
        epoch_start = time.time()
        jobs = make_batch_jobs(epoch, indices)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (dataset, task, batch_indices) in enumerate(jobs):
            data = run_02.cached(dataset, "train")
            array = np.array(data[batch_indices], copy=True)
            imgs = torch.from_numpy(array).to(device)
            torch.manual_seed(100000 + epoch * 10000 + batch_index)
            torch.cuda.manual_seed(100000 + epoch * 10000 + batch_index)
            imgs_n = add_noise(imgs)
            batch_losses = []
            optimizer.zero_grad(set_to_none=True)
            for _, enc_depth, dec_depth in specs:
                key = exit_key(enc_depth, dec_depth)
                with torch.no_grad():
                    hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc_depth, dec_depth)
                loss, _, _ = exit_loss_and_outputs(model, heads[key], hidden, imgs_n, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {backbone}/{kind}/{key}/epoch{epoch}/batch{batch_index}")
                loss.backward()
                batch_losses.append(float(loss.detach().cpu()))
            for head in heads.values():
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            optimizer.step()
            losses.extend(batch_losses)
            if (batch_index + 1) % 50 == 0:
                logger.info("%s/%s epoch=%d batch=%d/%d loss=%.6g", backbone, kind, epoch, batch_index + 1, len(jobs), np.mean(batch_losses))
        scheduler.step()
        epoch_rows = []
        global_scores = []
        for _, enc_depth, dec_depth in specs:
            key = exit_key(enc_depth, dec_depth)
            config_linear = []
            config_db = []
            for dataset in run_02.DATASETS[:16]:
                early_indices = indices[f"{dataset}/val_early"]
                data = run_02.cached(dataset, "val")[early_indices]
                for task in run_02.TASKS:
                    values = evaluate_exit(model, heads[key], data, device, task, enc_depth, dec_depth)
                    linear, db = run_02.summarize(values)
                    config_linear.append(linear)
                    config_db.append(db)
            val_linear = float(np.mean(config_linear))
            val_db = float(np.mean(config_db))
            global_scores.append(val_db)
            if val_db < best_scores[key]:
                best_scores[key] = val_db
                best_heads[key] = {name: tensor.detach().cpu().clone() for name, tensor in heads[key].state_dict().items()}
            epoch_rows.append((key, val_linear, val_db))
        global_db = float(np.mean(global_scores))
        if global_db < best_global - 1e-6:
            best_global = global_db
            bad_epochs = 0
        else:
            bad_epochs += 1
        elapsed = time.time() - total_start
        for key, val_linear, val_db in epoch_rows:
            history.append({
                "backbone": backbone,
                "head_kind": kind,
                "epoch": epoch,
                "exit_key": key,
                "train_loss": f"{np.mean(losses):.17g}",
                "lr": f"{optimizer.param_groups[0]['lr']:.17g}",
                "val_lin_mean": f"{val_linear:.17g}",
                "val_db_mean": f"{val_db:.17g}",
                "global_val_db_mean": f"{global_db:.17g}",
                "bad_epochs": bad_epochs,
                "elapsed_sec": f"{elapsed:.3f}",
            })
        save_training_checkpoint(path, epoch, heads, optimizer, scheduler, best_heads, best_scores, best_global, bad_epochs, history, backbone, kind, specs)
        run_02.atomic_csv(history_path, history, TRAINING_FIELDS)
        logger.info("%s/%s epoch=%d train_loss=%.6g val_db=%.6f bad=%d elapsed=%.1fs", backbone, kind, epoch, np.mean(losses), global_db, bad_epochs, time.time() - epoch_start)
        if bad_epochs >= 3:
            logger.info("Early stopping %s/%s at epoch %d", backbone, kind, epoch)
            break
    if len(best_heads) != len(specs):
        raise RuntimeError(f"Missing best heads for {backbone}/{kind}")
    del model, heads, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return history


def smoke_test(indices, logger):
    results = {}
    for backbone in ("base", "small"):
        device = torch.device("cuda")
        model, _ = run_02.build_model(backbone, device)
        freeze_backbone(model)
        for kind in ("h0", "h1"):
            heads = torch.nn.ModuleDict({exit_key(enc, dec): ExitHead(model, kind).to(device) for _, enc, dec in SPECS})
            optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3)
            before = heads["e6d2"].pred.weight.detach().clone()
            jobs = make_batch_jobs(1, indices, smoke=True)
            dataset, task, batch_indices = jobs[0]
            imgs = torch.from_numpy(np.array(run_02.cached(dataset, "train")[batch_indices], copy=True)).to(device)
            torch.manual_seed(101001)
            torch.cuda.manual_seed(101001)
            imgs_n = add_noise(imgs)
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for _, enc_depth, dec_depth in SPECS:
                key = exit_key(enc_depth, dec_depth)
                with torch.no_grad():
                    hidden, mask = exit_hidden_from_noisy(model, imgs_n, 0.5, task, enc_depth, dec_depth)
                loss, _, _ = exit_loss_and_outputs(model, heads[key], hidden, imgs_n, mask)
                loss.backward()
                losses.append(float(loss.detach().cpu()))
            for head in heads.values():
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.05)
            optimizer.step()
            assert all(parameter.grad is None for parameter in model.parameters())
            assert np.isfinite(losses).all()
            assert not torch.equal(before, heads["e6d2"].pred.weight)
            checkpoint = CHECKPOINTS / f"03_smoke_{backbone}_{kind}.pt"
            torch.save({"heads": {key: head.state_dict() for key, head in heads.items()}}, checkpoint)
            results[f"{backbone}/{kind}"] = {"losses": losses, "backbone_gradients": 0, "head_updated": True}
            logger.info("PASS smoke %s/%s losses=%s", backbone, kind, losses)
            del heads, optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
    run_02.atomic_json(WORK / "smoke.json", results)
    return results


def evaluation_methods(selection=None):
    methods = []
    for backbone in run_02.MODEL_SIZES:
        methods.append({"method": "official", "backbone": backbone, "enc": 6, "dec": 4, "head": None})
    for kind in ("untrained", "h0"):
        for backbone in ("base", "small"):
            for _, enc, dec in SPECS:
                methods.append({"method": kind, "backbone": backbone, "enc": enc, "dec": dec, "head": None if kind == "untrained" else "h0"})
    if selection is not None:
        for backbone in ("base", "small"):
            chosen = selection[backbone]["selected"]
            methods.append({"method": "h1", "backbone": backbone, "enc": chosen["enc_depth"], "dec": chosen["dec_depth"], "head": "h1"})
    return methods


def evaluation_key(method, dataset, task):
    return f"{method['method']}/{method['backbone']}/e{method['enc']}d{method['dec']}/{dataset}/{task}/0.5/val"


def run_evaluation(methods, resume, logger):
    work_csv = WORK / "val_summary.csv"
    work_npz = WORK / "persample.npz"
    rows = run_02.read_csv(work_csv) if resume else []
    arrays = dict(np.load(work_npz).items()) if resume and work_npz.exists() else {}
    done = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["dataset"], row["task"]) for row in rows}
    device = torch.device("cuda")
    for method_index, method in enumerate(methods, 1):
        pending = [(dataset, task) for dataset in run_02.DATASETS[:16] for task in run_02.TASKS if (method["method"], method["backbone"], method["enc"], method["dec"], dataset, task) not in done]
        if not pending:
            continue
        logger.info("[%d/%d] evaluate %s/%s enc=%d dec=%d", method_index, len(methods), method["method"], method["backbone"], method["enc"], method["dec"])
        if method["head"] is None:
            model, _ = run_02.build_model(method["backbone"], device)
            freeze_backbone(model)
            head = None
        else:
            model, head = load_best_head(method["backbone"], method["head"], exit_key(method["enc"], method["dec"]), device)
        for dataset, task in pending:
            values = evaluate_exit(model, head, run_02.cached(dataset, "val"), device, task, method["enc"], method["dec"])
            linear, db = run_02.summarize(values)
            rows.append({
                "method": method["method"],
                "backbone": method["backbone"],
                "enc_depth": method["enc"],
                "dec_depth": method["dec"],
                "dataset": dataset,
                "task": task,
                "mask_ratio": "0.5",
                "split": "val",
                "nmse_lin": f"{linear:.17g}",
                "nmse_db": f"{db:.17g}",
            })
            arrays[evaluation_key(method, dataset, task)] = values
            run_02.atomic_csv(work_csv, rows, VAL_FIELDS)
            run_02.atomic_npz(work_npz, arrays)
            logger.info("%s %s/%s NMSE=%.6f dB", evaluation_key(method, dataset, task), dataset, task, db)
        del model, head
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def aggregate_rows(rows):
    output = []
    methods = {}
    for row in rows:
        key = (row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]))
        methods.setdefault(key, []).append(row)
    lookup = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["dataset"], row["task"]): row for row in rows}
    for (method, backbone, enc, dec), method_rows in methods.items():
        for task in run_02.TASKS:
            selected = [row for row in method_rows if row["task"] == task]
            linear_mean = float(np.mean([float(row["nmse_lin"]) for row in selected]))
            db_mean = float(np.mean([float(row["nmse_db"]) for row in selected]))
            official_db = float(np.mean([float(lookup[("official", backbone, 6, 4, dataset, task)]["nmse_db"]) for dataset in run_02.DATASETS[:16]]))
            degradation = db_mean - official_db
            if method in ("h0", "h1"):
                untrained_db = float(np.mean([float(lookup[("untrained", backbone, enc, dec, dataset, task)]["nmse_db"]) for dataset in run_02.DATASETS[:16]]))
                recovery = untrained_db - db_mean
            else:
                recovery = None
            output.append({
                "method": method,
                "backbone": backbone,
                "enc_depth": enc,
                "dec_depth": dec,
                "task": task,
                "nmse_lin_mean": f"{linear_mean:.17g}",
                "nmse_db_mean": f"{db_mean:.17g}",
                "recovery_db": "" if recovery is None else f"{recovery:.17g}",
                "degradation_from_full_db": f"{degradation:.17g}",
            })
    return output


def forward_cost(model, head, noisy, task, enc_depth, dec_depth, interface_adapter=None, mask_ratio=0.5):
    hidden, mask = exit_hidden_from_noisy(model, noisy, mask_ratio, task, enc_depth, dec_depth, interface_adapter)
    return exit_loss_and_outputs(model, head, hidden, noisy, mask)


def cost_gflops(model, head, noisy, enc_depth, dec_depth, interface_adapter=None, mask_ratio=0.5):
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        forward_cost(model, head, noisy, "temporal", enc_depth, dec_depth, interface_adapter, mask_ratio)
    return float(counter.get_total_flops() / 1e9)


def timing_summary(values):
    return float(np.median(values)), float(np.max(values) - np.min(values))


def cost_timing(model, head, noisy, enc_depth, dec_depth, device, mode, interface_adapter=None, mask_ratio=0.5):
    if mode == "gpu_b1":
        batch_size, warmup, iterations, rounds = 1, 20, 100, 3
    elif mode == "gpu_b64":
        batch_size, warmup, iterations, rounds = 64, 10, 30, 3
    else:
        batch_size, warmup, iterations, rounds = 1, 5, 30, 3
    batch = noisy[:batch_size].to(device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    round_medians = []
    with torch.inference_mode():
        for _ in range(rounds):
            for _ in range(warmup):
                forward_cost(model, head, batch, "temporal", enc_depth, dec_depth, interface_adapter, mask_ratio)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = []
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                forward_cost(model, head, batch, "temporal", enc_depth, dec_depth, interface_adapter, mask_ratio)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
            median = float(np.median(elapsed))
            round_medians.append(64 / median if mode == "gpu_b64" else median * 1000)
    return timing_summary(round_medians)


def run_costs(methods, resume, logger):
    work_csv = WORK / "costs.csv"
    rows = run_02.read_csv(work_csv) if resume else []
    done = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"])) for row in rows}
    noisy = run_02.noisy_input("D17")
    for method in methods:
        key = (method["method"], method["backbone"], method["enc"], method["dec"])
        if key in done:
            continue
        logger.info("cost %s", key)
        model, head = load_best_head(method["backbone"], method["head"], exit_key(method["enc"], method["dec"]), torch.device("cuda"))
        gflops = cost_gflops(model, head, noisy[:1].cuda(), method["enc"], method["dec"])
        gpu_b1, gpu_b1_range = cost_timing(model, head, noisy, method["enc"], method["dec"], torch.device("cuda"), "gpu_b1")
        gpu_b64, gpu_b64_range = cost_timing(model, head, noisy, method["enc"], method["dec"], torch.device("cuda"), "gpu_b64")
        adapter_params = sum(parameter.numel() for name, parameter in head.named_parameters() if name.startswith("adapter."))
        params = run_02.params_exec(model, method["enc"], method["dec"]) + adapter_params
        model = model.cpu()
        head = head.cpu()
        torch.cuda.empty_cache()
        cpu_b1, cpu_b1_range = cost_timing(model, head, noisy, method["enc"], method["dec"], torch.device("cpu"), "cpu_b1")
        rows.append({
            "method": method["method"],
            "backbone": method["backbone"],
            "enc_depth": method["enc"],
            "dec_depth": method["dec"],
            "params_exec": params,
            "gflops": f"{gflops:.17g}",
            "gpu_b1_ms": f"{gpu_b1:.17g}",
            "gpu_b1_ms_range": f"{gpu_b1_range:.17g}",
            "gpu_thr_b64": f"{gpu_b64:.17g}",
            "gpu_thr_b64_range": f"{gpu_b64_range:.17g}",
            "cpu_b1_ms_t1": f"{cpu_b1:.17g}",
            "cpu_b1_ms_t1_range": f"{cpu_b1_range:.17g}",
        })
        run_02.atomic_csv(work_csv, rows, COST_FIELDS)
        del model, head
        gc.collect()
    return rows


def select_h1(aggregate, costs, logger):
    selection = {}
    for backbone in ("base", "small"):
        candidates = []
        for _, enc, dec in SPECS:
            rows = [row for row in aggregate if row["method"] == "h0" and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec]
            score = float(np.mean([float(row["nmse_db_mean"]) for row in rows]))
            cost = next(float(row["gflops"]) for row in costs if row["method"] == "h0" and row["backbone"] == backbone and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec)
            candidates.append({"enc_depth": enc, "dec_depth": dec, "score_db_mean": score, "gflops": cost})
        chosen = min(candidates, key=lambda row: (row["score_db_mean"], row["gflops"]))
        selection[backbone] = {"rule": "minimum mean of temporal/fre cross-dataset dB NMSE; ties broken by lower GFLOPs", "candidates": candidates, "selected": chosen}
        logger.info("H1 selection %s: %s", backbone, chosen)
    run_02.atomic_json(SELECTION_JSON, selection)
    return selection


def build_pareto(aggregate, costs):
    aggregate_lookup = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["task"]): float(row["nmse_db_mean"]) for row in aggregate}
    task02_costs = run_02.read_csv(run_02.COSTS_CSV)
    official_costs = [row for row in task02_costs if row["shape"] == "D17" and row["model"] in ("base_official", "small", "little", "tiny")]
    trained_costs = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"])): row for row in costs}
    output = []
    for method, backbone, enc, dec in [(method["method"], method["backbone"], method["enc"], method["dec"]) for method in evaluation_methods()]:
        if method == "official":
            continue
        if method == "untrained":
            cost_row = trained_costs[("h0", backbone, enc, dec)]
        else:
            cost_row = trained_costs[(method, backbone, enc, dec)]
        for task in run_02.TASKS:
            official = []
            for row in official_costs:
                model = "base" if row["model"] == "base_official" else row["model"]
                nmse = aggregate_lookup[("official", model, 6, 4, task)]
                official.append((row, nmse))
            for metric in ("gflops", "1/gpu_thr_b64", "cpu_b1_ms_t1"):
                if metric == "1/gpu_thr_b64":
                    cost = 1.0 / float(cost_row["gpu_thr_b64"])
                    frontier = [(1.0 / float(row["gpu_thr_b64"]), nmse) for row, nmse in official]
                else:
                    cost = float(cost_row[metric])
                    frontier = [(float(row[metric]), nmse) for row, nmse in official]
                eligible = [nmse for point_cost, nmse in frontier if point_cost <= cost + 1e-15]
                required = min(eligible) if eligible else None
                nmse_db = aggregate_lookup[(method, backbone, enc, dec, task)]
                output.append({
                    "method": method,
                    "backbone": backbone,
                    "enc_depth": enc,
                    "dec_depth": dec,
                    "task": task,
                    "cost_metric": metric,
                    "cost": f"{cost:.17g}",
                    "nmse_db": f"{nmse_db:.17g}",
                    "required_db": "" if required is None else f"{required:.17g}",
                    "gap_db": "" if required is None else f"{nmse_db - required:.17g}",
                })
    for selection in json.loads(SELECTION_JSON.read_text(encoding="utf-8")).values():
        chosen = selection["selected"]
        backbone = next(key for key, value in json.loads(SELECTION_JSON.read_text(encoding="utf-8")).items() if value == selection)
        enc, dec = chosen["enc_depth"], chosen["dec_depth"]
        cost_row = trained_costs[("h1", backbone, enc, dec)]
        for task in run_02.TASKS:
            for metric in ("gflops", "1/gpu_thr_b64", "cpu_b1_ms_t1"):
                official = []
                for row in official_costs:
                    model = "base" if row["model"] == "base_official" else row["model"]
                    nmse = aggregate_lookup[("official", model, 6, 4, task)]
                    if metric == "1/gpu_thr_b64":
                        point_cost = 1.0 / float(row["gpu_thr_b64"])
                    else:
                        point_cost = float(row[metric])
                    official.append((point_cost, nmse))
                cost = 1.0 / float(cost_row["gpu_thr_b64"]) if metric == "1/gpu_thr_b64" else float(cost_row[metric])
                eligible = [nmse for point_cost, nmse in official if point_cost <= cost + 1e-15]
                required = min(eligible) if eligible else None
                nmse_db = aggregate_lookup[("h1", backbone, enc, dec, task)]
                output.append({
                    "method": "h1", "backbone": backbone, "enc_depth": enc, "dec_depth": dec,
                    "task": task, "cost_metric": metric, "cost": f"{cost:.17g}",
                    "nmse_db": f"{nmse_db:.17g}", "required_db": "" if required is None else f"{required:.17g}",
                    "gap_db": "" if required is None else f"{nmse_db - required:.17g}",
                })
    run_02.atomic_csv(PARETO_CSV, output, PARETO_FIELDS)
    return output


def make_g1_plot(aggregate, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]), row["task"]): float(row["nmse_db_mean"]) for row in aggregate}
    cost_lookup = {(row["method"], row["backbone"], int(row["enc_depth"]), int(row["dec_depth"])): float(row["gflops"]) for row in costs}
    task02_costs = run_02.read_csv(run_02.COSTS_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, task in zip(axes, run_02.TASKS):
        for row in task02_costs:
            if row["shape"] != "D17" or row["model"] not in ("base_official", "small", "little", "tiny"):
                continue
            backbone = "base" if row["model"] == "base_official" else row["model"]
            axis.scatter(float(row["gflops"]), lookup[("official", backbone, 6, 4, task)], color="black", s=48)
            axis.annotate(backbone, (float(row["gflops"]), lookup[("official", backbone, 6, 4, task)]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        for method in ("untrained", "h0", "h1"):
            for row in aggregate:
                if row["method"] != method or row["task"] != task:
                    continue
                backbone, enc, dec = row["backbone"], int(row["enc_depth"]), int(row["dec_depth"])
                cost = cost_lookup[("h0", backbone, enc, dec)] if method == "untrained" else cost_lookup[(method, backbone, enc, dec)]
                marker = "o" if method == "untrained" else ("s" if method == "h0" else "*")
                color = "tab:blue" if backbone == "base" else "tab:orange"
                face = "none" if method == "untrained" else color
                axis.scatter(cost, float(row["nmse_db_mean"]), marker=marker, s=90 if method == "h1" else 45, facecolors=face, edgecolors=color, label=f"{backbone}/{method}" if (backbone, method, task) not in getattr(axis, "_labels", set()) else None)
                axis.annotate(f"({enc},{dec})", (cost, float(row["nmse_db_mean"])), fontsize=8)
                if not hasattr(axis, "_labels"):
                    axis._labels = set()
                axis._labels.add((backbone, method, task))
        axis.set_xscale("log")
        axis.set_xlabel("GFLOPs (log scale)")
        axis.set_ylabel("D1–D16 val mean NMSE (dB)")
        axis.set_title(task)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    temp = G1_PNG.with_suffix(f".{os.getpid()}.tmp.png")
    fig.savefig(temp, dpi=180)
    plt.close(fig)
    os.replace(temp, G1_PNG)


def make_report(history, aggregate, costs, pareto):
    selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    lines = [
        "# Task 03：Base + Small 轻量出口试训报告",
        "",
        "## 协议与完成情况",
        "",
        "- 出口：`(6,2)` 解码器侧、`(3,4)` 编码器侧、`(2,1)` 低成本；Base 与 Small 相同。",
        "- 训练数据为 D1–D16 train 各固定 2000 个样本，mask=0.5，temporal/fre batch 数各半，单种子 100，batch 64，最多 10 epoch。",
        "- 早停使用各配置 val 的固定 256 个样本；最终评价使用全部 2000 个 val 样本。D17、D18 未用于训练、早停、出口或掩码选择。",
        "- 骨干冻结，仅出口头训练；H0 共 6 个，H1 共 2 个。逐样本结果、训练历史和 checkpoint 已保存。",
        "",
        "## H1 选择",
        "",
        "选择规则为 H0 的 temporal/fre 跨配置平均 dB NMSE 均值最低，并列时取 GFLOPs 更低者。",
        "",
        "| 骨干 | 入选 enc | 入选 dec | H0 两任务均值 dB | GFLOPs |",
        "|---|---:|---:|---:|---:|",
    ]
    for backbone in ("base", "small"):
        chosen = selection[backbone]["selected"]
        lines.append(f"| {backbone} | {chosen['enc_depth']} | {chosen['dec_depth']} | {chosen['score_db_mean']:.6f} | {chosen['gflops']:.6f} |")
    lines.extend([
        "",
        "## 训练历史摘要",
        "",
        "下表为各头在固定验证子集上的最佳 epoch；耗时为训练脚本记录的累计时间。",
        "",
        "| 骨干 | 头 | 出口 | 最佳 epoch | 最佳 val 子集 dB | 最后记录耗时秒 |",
        "|---|---|---|---:|---:|---:|",
    ])
    for backbone in ("base", "small"):
        for kind in ("h0", "h1"):
            for _, enc, dec in SPECS:
                key = exit_key(enc, dec)
                rows = [row for row in history if row["backbone"] == backbone and row["head_kind"] == kind and row["exit_key"] == key]
                if not rows:
                    continue
                best = min(rows, key=lambda row: float(row["val_db_mean"]))
                lines.append(f"| {backbone} | {kind} | {key} | {best['epoch']} | {float(best['val_db_mean']):.6f} | {float(rows[-1]['elapsed_sec']):.1f} |")
    lines.extend([
        "",
        "## 全验证集聚合结果",
        "",
        "跨配置同时计算线性平均和各配置 dB 平均。恢复量为未训练 dB − 训练后 dB，正值表示改善；完整退化量为当前方法 dB − 同骨干完整模型 dB。",
        "",
        "| 方法 | 骨干 | enc | dec | 任务 | 线性平均 | dB 平均 | 恢复 dB | 完整退化 dB |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ])
    for row in aggregate:
        recovery = "—" if row["recovery_db"] == "" else f"{float(row['recovery_db']):.6f}"
        lines.append(f"| {row['method']} | {row['backbone']} | {row['enc_depth']} | {row['dec_depth']} | {row['task']} | {float(row['nmse_lin_mean']):.6f} | {float(row['nmse_db_mean']):.6f} | {recovery} | {float(row['degradation_from_full_db']):.6f} |")
    lines.extend([
        "",
        "## 出口实际成本（D17 形状）",
        "",
        "时延为三轮中位数，括号内为三轮极差；成本包含出口头。D17 仅作成本输入，不参与模型选择。",
        "",
        "| 方法 | 骨干 | enc | dec | params | GFLOPs | GPU b1 ms | GPU thr b64 | CPU t1 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in costs:
        lines.append(f"| {row['method']} | {row['backbone']} | {row['enc_depth']} | {row['dec_depth']} | {row['params_exec']} | {float(row['gflops']):.6f} | {float(row['gpu_b1_ms']):.3f} ({float(row['gpu_b1_ms_range']):.3f}) | {float(row['gpu_thr_b64']):.3f} ({float(row['gpu_thr_b64_range']):.3f}) | {float(row['cpu_b1_ms_t1']):.3f} ({float(row['cpu_b1_ms_t1_range']):.3f}) |")
    lines.extend(["", "## H0/H1 相对官方前沿的差距", "", "仅列训练后的出口；完整逐点结果见 `03_pareto.csv`。", "", "| 方法 | 骨干 | 出口 | 任务 | 成本口径 | NMSE dB | required dB | gap dB |", "|---|---|---|---|---|---:|---:|---:|"])
    for row in pareto:
        if row["method"] not in ("h0", "h1"):
            continue
        required = "无官方参照" if row["required_db"] == "" else f"{float(row['required_db']):.6f}"
        gap = "—" if row["gap_db"] == "" else f"{float(row['gap_db']):.6f}"
        lines.append(f"| {row['method']} | {row['backbone']} | e{row['enc_depth']}d{row['dec_depth']} | {row['task']} | {row['cost_metric']} | {float(row['nmse_db']):.6f} | {required} | {gap} |")
    lines.extend(["", "## 问题与边界", "", "- 执行中曾出现空官方前沿值格式化、checkpoint 嵌套恢复、续跑完成键缺少深度、逐样本键缺少深度和恢复量参照未按深度对齐的问题；均已修正，续跑深度场景已通过回归检查。最终 576 组逐样本结果已补评并通过独立校验，旧冲突文件保留为 `03_work/persample_collision_backup.npz`。", "- 本报告为单种子、小规模试训，只提供 G1 材料，不作最终骨干或论文主张选择。", "- 未进行 D17、D18 泛化评价，未进行多种子显著性检验，也未训练完整骨干或蒸馏学生。", ""])
    temp = REPORT_MD.with_suffix(f".{os.getpid()}.tmp.md")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, REPORT_MD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    logger = make_logger()
    WORK.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    final_paths = (TRAINING_CSV, VAL_CSV, AGGREGATE_CSV, PERSAMPLE_NPZ, COSTS_CSV, PARETO_CSV, SELECTION_JSON, G1_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume and not args.smoke and not args.self_check_only:
        raise FileExistsError("Task 03 outputs exist; use --resume only to continue this task")
    check = run_02.selfcheck(logger)
    run_02.atomic_json(WORK / "selfcheck.json", check)
    if args.self_check_only:
        return
    indices = split_indices()
    if args.smoke:
        smoke_test(indices, logger)
        return
    if not (WORK / "smoke.json").exists() or not args.resume:
        smoke_test(indices, logger)
    for backbone in ("base", "small"):
        train_backbone(backbone, "h0", SPECS, indices, args.resume, logger)
    methods = evaluation_methods()
    rows = run_evaluation(methods, args.resume, logger)
    aggregate = aggregate_rows(rows)
    h0_methods = [method for method in methods if method["method"] == "h0"]
    costs = run_costs(h0_methods, args.resume, logger)
    if SELECTION_JSON.exists() and args.resume:
        selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    else:
        selection = select_h1(aggregate, costs, logger)
    for backbone in ("base", "small"):
        chosen = selection[backbone]["selected"]
        specs = (("selected", chosen["enc_depth"], chosen["dec_depth"]),)
        train_backbone(backbone, "h1", specs, indices, args.resume, logger)
    h1_methods = evaluation_methods(selection)[-2:]
    rows = run_evaluation(h1_methods, True, logger)
    h1_costs = run_costs(h1_methods, True, logger)
    costs = run_02.read_csv(WORK / "costs.csv")
    aggregate = aggregate_rows(rows)
    expected_methods = evaluation_methods(selection)
    if len(rows) != len(expected_methods) * 32:
        raise RuntimeError(f"Validation incomplete: {len(rows)}/{len(expected_methods) * 32}")
    history = []
    for backbone in ("base", "small"):
        for kind in ("h0", "h1"):
            history.extend(run_02.read_csv(WORK / f"{backbone}_{kind}_training.csv"))
    run_02.atomic_csv(TRAINING_CSV, history, TRAINING_FIELDS)
    run_02.atomic_csv(VAL_CSV, rows, VAL_FIELDS)
    run_02.atomic_csv(AGGREGATE_CSV, aggregate, AGGREGATE_FIELDS)
    arrays = dict(np.load(WORK / "persample.npz").items())
    run_02.atomic_npz(PERSAMPLE_NPZ, arrays)
    run_02.atomic_csv(COSTS_CSV, costs, COST_FIELDS)
    pareto = build_pareto(aggregate, costs)
    make_g1_plot(aggregate, costs)
    make_report(history, aggregate, costs, pareto)
    run_02.atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "val_rows": len(rows), "cost_rows": len(costs), "pareto_rows": len(pareto)})
    logger.info("PASS Task 03: all outputs completed")


if __name__ == "__main__":
    main()
