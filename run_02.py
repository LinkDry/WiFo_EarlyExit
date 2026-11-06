import argparse
import csv
import gc
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

sys.dont_write_bytecode = True

import hdf5storage
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

W = Path(__file__).resolve().parent
SRC = Path(os.environ.get("WIFO_SRC", str(W / "third_party" / "WiFo" / "src")))
WEIGHTS = Path(os.environ.get("WIFO_WEIGHTS", str(W / "assets" / "WiFo")))
R = Path(os.environ.get("WIFO_DATA", str(W / "data" / "RF-only_channel_dataset_for_WiFo")))
CACHE = W / "cache"
RESULTS = W / "results"
WORK = RESULTS / "02_work"
LOGS = W / "logs"
BASELINES_CSV = RESULTS / "02_baselines.csv"
PERSAMPLE_NPZ = RESULTS / "02_persample.npz"
COSTS_CSV = RESULTS / "02_costs.csv"
PARETO_CSV = RESULTS / "02_pareto.csv"
PARETO_PNG = RESULTS / "02_pareto.png"
REPORT_MD = RESULTS / "02_report.md"
TASK01_CSV = RESULTS / "01_trunc_grid.csv"
MODEL_SIZES = ("base", "small", "little", "tiny")
DATASETS = tuple(f"D{index}" for index in range(1, 19))
TASKS = ("temporal", "fre")
MASK_RATIOS = (0.25, 0.5, 0.75)
BASELINE_FIELDS = ("model", "dataset", "split", "task", "mask_ratio", "nmse_lin", "nmse_db")
COST_FIELDS = (
    "model", "enc_depth", "dec_depth", "shape", "params_exec", "params_exec_range",
    "gflops", "gflops_range", "gpu_b1_ms", "gpu_b1_ms_range",
    "gpu_thr_b64", "gpu_thr_b64_range", "gpu_thr_b256", "gpu_thr_b256_range",
    "cpu_b1_ms_t1", "cpu_b1_ms_t1_range", "cpu_b1_ms_t4", "cpu_b1_ms_t4_range",
)
PARETO_FIELDS = (
    "dataset", "task", "enc_depth", "dec_depth", "cost_metric", "cost",
    "nmse_db_untrained", "required_db", "gap_db",
)

sys.path.insert(0, str(SRC))
from model import WiFo_model
from wifo_trunc import forward_trunc, forward_trunc_from_noisy


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("run_02")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "run_02.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def atomic_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_every_run():
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)


def build_model(size, device):
    args = Namespace(size=size, t_patch_size=4, patch_size=4, pos_emb="SinCos_3D", no_qkv_bias=0)
    model = WiFo_model(args)
    state_dict = torch.load(WEIGHTS / f"wifo_{size}.pkl", map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    assert missing == [], f"{size}: missing keys {missing}"
    assert len(unexpected) == 4 and all(key.startswith("decoder_pred_dual.") for key in unexpected), (
        f"{size}: unexpected keys {unexpected}"
    )
    return model.to(device).eval(), unexpected


def as_imgs_list(batch):
    return list(batch.unsqueeze(1))


def evaluate_persample(model, data, device, task, mask_ratio, batch_size=64):
    values = []
    seed_every_run()
    with torch.inference_mode():
        for start in range(0, data.shape[0], batch_size):
            array = np.array(data[start:start + batch_size], copy=True)
            batch = torch.from_numpy(array).to(device)
            _, _, pred, target, mask = forward_trunc(
                model, as_imgs_list(batch), mask_ratio, task, len(model.blocks), len(model.decoder_blocks)
            )
            expanded_mask = mask.reshape(pred.shape[0], -1).unsqueeze(-1)
            numerator = (torch.abs(pred - target) ** 2 * expanded_mask).sum(dim=(1, 2))
            denominator = (torch.abs(target) ** 2 * expanded_mask).sum(dim=(1, 2))
            values.append((numerator / denominator).float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def summarize(values):
    nmse_lin = float(np.mean(values))
    return nmse_lin, float(10 * math.log10(nmse_lin))


def load_source(path):
    raw = hdf5storage.loadmat(str(path))
    keys = [key for key in raw if not key.startswith("__")]
    assert len(keys) == 1, f"{path}: variables {keys}"
    array = np.asarray(raw[keys[0]])
    converted = np.stack((array.real, array.imag), axis=1).astype(np.float32, copy=False)
    return converted


def cached(dataset, split):
    path = CACHE / f"{dataset}_{split}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache {path}; run python prep_cache.py first")
    return np.load(path, mmap_mode="r")


def selfcheck(logger):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    disk_free = shutil.disk_usage(W.anchor).free
    assert gpu_free / 2**30 > 10, f"GPU free memory {gpu_free / 2**30:.2f} GiB <= 10"
    assert disk_free / 2**30 > 40, f"Disk free {disk_free / 2**30:.2f} GiB <= 40"
    model, unexpected = build_model("base", device)
    shapes = {}
    checks = (
        ("D1_train", R / "dataset4train/D1/X_train.mat", 9000),
        ("D1_val", R / "dataset4train/D1/X_val.mat", 2000),
        ("D17_test", R / "dataset/D17/X_test.mat", 1000),
    )
    data17 = None
    for name, path, expected in checks:
        array = load_source(path)
        assert array.shape[0] == expected, f"{name}: {array.shape}"
        shapes[name] = list(array.shape)
        if name == "D17_test":
            data17 = array
        else:
            del array
            gc.collect()
    batch = torch.from_numpy(data17[:128]).to(device)
    imgs = as_imgs_list(batch)
    with torch.inference_mode():
        seed_every_run()
        official = model(imgs, 0.5, "temporal")
        seed_every_run()
        truncated = forward_trunc(model, imgs, 0.5, "temporal", 6, 4)
    max_diff = float((official[2] - truncated[2]).abs().max().item())
    assert max_diff < 1e-5, f"forward difference {max_diff}"
    values = evaluate_persample(model, data17, device, "temporal", 0.5)
    nmse_lin, nmse_db = summarize(values)
    assert abs(nmse_db - (-5.267876176216529)) < 0.01, f"D17 baseline {nmse_db}"
    result = {
        "status": "PASS",
        "weight_missing_keys": 0,
        "weight_unexpected_keys": unexpected,
        "data_shapes": shapes,
        "forward_max_abs_diff": max_diff,
        "d17_temporal_nmse_lin": nmse_lin,
        "d17_temporal_nmse_db": nmse_db,
        "gpu_free_gib": gpu_free / 2**30,
        "gpu_total_gib": gpu_total / 2**30,
        "disk_free_gib": disk_free / 2**30,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
        },
    }
    atomic_json(WORK / "selfcheck.json", result)
    logger.info("PASS selfcheck: %s", json.dumps({k: v for k, v in result.items() if k != "environment"}))
    del model, data17
    gc.collect()
    torch.cuda.empty_cache()
    return result


def baseline_jobs():
    jobs = []
    for size in MODEL_SIZES:
        for dataset in DATASETS:
            for task in TASKS:
                jobs.append((size, dataset, "test", task, 0.5))
    for dataset in ("D1", "D16"):
        for task in TASKS:
            for ratio in MASK_RATIOS:
                jobs.append(("base", dataset, "val", task, ratio))
    return jobs


def baseline_key(job):
    size, dataset, split, task, ratio = job
    return f"{size}/{dataset}/{task}/{ratio:g}/{split}"


def run_baselines(logger, resume):
    if BASELINES_CSV.exists() and PERSAMPLE_NPZ.exists() and resume:
        logger.info("Skipping completed baselines")
        return read_csv(BASELINES_CSV)
    work_csv = WORK / "baselines.csv"
    work_npz = WORK / "persample.npz"
    rows = read_csv(work_csv) if resume else []
    arrays = dict(np.load(work_npz).items()) if resume and work_npz.exists() else {}
    done = {baseline_key((row["model"], row["dataset"], row["split"], row["task"], float(row["mask_ratio"]))) for row in rows}
    jobs = baseline_jobs()
    device = torch.device("cuda")
    active_size = None
    model = None
    try:
        for index, job in enumerate(jobs, 1):
            size, dataset, split, task, ratio = job
            key = baseline_key(job)
            if key in done:
                continue
            if size != active_size:
                if model is not None:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                model, _ = build_model(size, device)
                active_size = size
            logger.info("[%d/%d] baseline %s", index, len(jobs), key)
            values = evaluate_persample(model, cached(dataset, split), device, task, ratio)
            nmse_lin, nmse_db = summarize(values)
            row = {
                "model": size,
                "dataset": dataset,
                "split": split,
                "task": task,
                "mask_ratio": f"{ratio:g}",
                "nmse_lin": f"{nmse_lin:.17g}",
                "nmse_db": f"{nmse_db:.17g}",
            }
            rows.append(row)
            arrays[key] = values
            atomic_csv(work_csv, rows, BASELINE_FIELDS)
            atomic_npz(work_npz, arrays)
            logger.info("NMSE lin=%.9g dB=%.6f", nmse_lin, nmse_db)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    if len(rows) != len(jobs):
        raise RuntimeError(f"Baseline incomplete: {len(rows)}/{len(jobs)}; rerun with --resume")
    ordered = sorted(rows, key=lambda row: jobs.index((row["model"], row["dataset"], row["split"], row["task"], float(row["mask_ratio"]))))
    atomic_csv(BASELINES_CSV, ordered, BASELINE_FIELDS)
    atomic_npz(PERSAMPLE_NPZ, arrays)
    logger.info("PASS baselines: rows=%d persample=%d", len(ordered), len(arrays))
    return ordered


def params_exec(model, enc_depth, dec_depth):
    total = 0
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "decoder_pred_dual":
            continue
        if parts[0] == "blocks" and int(parts[1]) >= enc_depth:
            continue
        if parts[0] == "decoder_blocks" and int(parts[1]) >= dec_depth:
            continue
        total += parameter.numel()
    return int(total)


def noisy_input(dataset):
    data = cached(dataset, "test")[:256]
    imgs = torch.from_numpy(np.array(data, copy=True))
    seed_every_run()
    noise_power = torch.mean(imgs**2) * 10 ** (-20 / 10)
    noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
    return imgs + noise


def measure_gflops(model, sample, enc_depth, dec_depth):
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        forward_trunc_from_noisy(model, sample, 0.5, "temporal", enc_depth, dec_depth)
    return float(counter.get_total_flops() / 1e9)


def summarize_timings(values):
    return float(np.median(values)), float(np.max(values) - np.min(values))


def gpu_b1(model, noisy, enc_depth, dec_depth):
    batch = noisy[:1].cuda()
    rounds = []
    with torch.inference_mode():
        for _ in range(3):
            for _ in range(20):
                forward_trunc_from_noisy(model, batch, 0.5, "temporal", enc_depth, dec_depth)
            torch.cuda.synchronize()
            elapsed = []
            for _ in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                forward_trunc_from_noisy(model, batch, 0.5, "temporal", enc_depth, dec_depth)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter() - start) * 1000)
            rounds.append(float(np.median(elapsed)))
    return summarize_timings(rounds)


def gpu_throughput(model, noisy, batch_size, enc_depth, dec_depth):
    rounds = []
    try:
        batch = noisy[:batch_size].cuda()
        with torch.inference_mode():
            for _ in range(3):
                for _ in range(10):
                    forward_trunc_from_noisy(model, batch, 0.5, "temporal", enc_depth, dec_depth)
                torch.cuda.synchronize()
                elapsed = []
                for _ in range(30):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    forward_trunc_from_noisy(model, batch, 0.5, "temporal", enc_depth, dec_depth)
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - start)
                rounds.append(batch_size / float(np.median(elapsed)))
        del batch
        return summarize_timings(rounds)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        return None
    finally:
        torch.cuda.empty_cache()


def cpu_b1(model, noisy, enc_depth, dec_depth, threads):
    torch.set_num_threads(threads)
    batch = noisy[:1].cpu()
    rounds = []
    with torch.inference_mode():
        for _ in range(3):
            for _ in range(5):
                forward_trunc_from_noisy(model, batch, 0.5, "temporal", enc_depth, dec_depth)
            elapsed = []
            for _ in range(30):
                start = time.perf_counter()
                forward_trunc_from_noisy(model, batch, 0.5, "temporal", enc_depth, dec_depth)
                elapsed.append((time.perf_counter() - start) * 1000)
            rounds.append(float(np.median(elapsed)))
    return summarize_timings(rounds)


def set_metric(row, name, result):
    if result is None:
        row[name] = ""
        row[f"{name}_range"] = ""
    else:
        median, value_range = result
        row[name] = f"{median:.17g}"
        row[f"{name}_range"] = f"{value_range:.17g}"


def cost_paths():
    paths = [("base_official", 6, 4)] + [(size, 6, 4) for size in MODEL_SIZES[1:]]
    paths += [("base", enc, dec) for enc in range(1, 7) for dec in range(5)]
    return paths


def gpu_utilization():
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def run_costs(logger, resume):
    if COSTS_CSV.exists() and resume:
        logger.info("Skipping completed costs")
        return read_csv(COSTS_CSV)
    work_csv = WORK / "costs.csv"
    rows = read_csv(work_csv) if resume else []
    done = {(row["model"], row["shape"], int(row["enc_depth"]), int(row["dec_depth"])) for row in rows}
    paths = cost_paths()
    expected = [(model, shape, enc, dec) for model, enc, dec in paths for shape in ("D17", "D1")]
    utilization = gpu_utilization()
    atomic_json(WORK / "costs_meta.json", {"gpu_before_costs": utilization, "started_at": time.strftime("%F %T")})
    logger.info("GPU before costs: %s", utilization)
    base_check_done = resume and (WORK / "flops_task_check.json").exists()
    for index, (model_name, shape, enc_depth, dec_depth) in enumerate(expected, 1):
        key = (model_name, shape, enc_depth, dec_depth)
        if key in done:
            continue
        size = "base" if model_name in ("base", "base_official") else model_name
        logger.info("[%d/%d] costs %s %s enc=%d dec=%d", index, len(expected), model_name, shape, enc_depth, dec_depth)
        noisy = noisy_input(shape)
        model, _ = build_model(size, torch.device("cuda"))
        flops_values = [measure_gflops(model, noisy[:1].cuda(), enc_depth, dec_depth) for _ in range(3)]
        if not base_check_done and model_name == "base_official" and shape == "D17":
            with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                forward_trunc_from_noisy(model, noisy[:1].cuda(), 0.5, "fre", enc_depth, dec_depth)
            fre_gflops = float(counter.get_total_flops() / 1e9)
            assert abs(fre_gflops - flops_values[0]) < 1e-9, (fre_gflops, flops_values[0])
            atomic_json(WORK / "flops_task_check.json", {"temporal": flops_values[0], "fre": fre_gflops, "max_abs_diff": abs(fre_gflops - flops_values[0])})
            base_check_done = True
        row = {
            "model": model_name,
            "enc_depth": enc_depth,
            "dec_depth": dec_depth,
            "shape": shape,
            "params_exec": str(params_exec(model, enc_depth, dec_depth)),
            "params_exec_range": "0",
        }
        set_metric(row, "gflops", summarize_timings(flops_values))
        set_metric(row, "gpu_b1_ms", gpu_b1(model, noisy, enc_depth, dec_depth))
        set_metric(row, "gpu_thr_b64", gpu_throughput(model, noisy, 64, enc_depth, dec_depth))
        set_metric(row, "gpu_thr_b256", gpu_throughput(model, noisy, 256, enc_depth, dec_depth))
        model = model.cpu()
        torch.cuda.empty_cache()
        set_metric(row, "cpu_b1_ms_t1", cpu_b1(model, noisy, enc_depth, dec_depth, 1))
        set_metric(row, "cpu_b1_ms_t4", cpu_b1(model, noisy, enc_depth, dec_depth, 4))
        rows.append(row)
        atomic_csv(work_csv, rows, COST_FIELDS)
        logger.info("cost row: %s", row)
        del model, noisy
        gc.collect()
    if len(rows) != len(expected):
        raise RuntimeError(f"Costs incomplete: {len(rows)}/{len(expected)}; rerun with --resume")
    order = {key: index for index, key in enumerate(expected)}
    ordered = sorted(rows, key=lambda row: order[(row["model"], row["shape"], int(row["enc_depth"]), int(row["dec_depth"]))])
    atomic_csv(COSTS_CSV, ordered, COST_FIELDS)
    logger.info("PASS costs: rows=%d", len(ordered))
    return ordered


def official_frontier(cost_rows, baseline_rows, dataset, task, metric):
    baseline = {
        (row["model"], row["dataset"], row["task"]): float(row["nmse_db"])
        for row in baseline_rows
        if row["split"] == "test" and float(row["mask_ratio"]) == 0.5
    }
    points = []
    for row in cost_rows:
        if row["shape"] != "D17" or row["model"] not in ("base_official", "small", "little", "tiny"):
            continue
        model = "base" if row["model"] == "base_official" else row["model"]
        if metric == "1/gpu_thr_b64":
            if not row["gpu_thr_b64"]:
                continue
            cost = 1.0 / float(row["gpu_thr_b64"])
        else:
            if not row[metric]:
                continue
            cost = float(row[metric])
        points.append((cost, baseline[(model, dataset, task)], model))
    return sorted(points)


def required_value(points, cost):
    eligible = [nmse for point_cost, nmse, _ in points if point_cost <= cost + 1e-15]
    return min(eligible) if eligible else None


def run_pareto(logger, baseline_rows, cost_rows):
    task01 = {
        (row["dataset"], row["task"], int(row["enc_depth"]), int(row["dec_depth"])): float(row["nmse_db"])
        for row in read_csv(TASK01_CSV)
    }
    rows = []
    for dataset in ("D17", "D18"):
        for task in TASKS:
            for metric in ("gflops", "1/gpu_thr_b64", "cpu_b1_ms_t1"):
                frontier = official_frontier(cost_rows, baseline_rows, dataset, task, metric)
                for cost_row in cost_rows:
                    if cost_row["shape"] != "D17" or cost_row["model"] != "base":
                        continue
                    if metric == "1/gpu_thr_b64":
                        if not cost_row["gpu_thr_b64"]:
                            continue
                        cost = 1.0 / float(cost_row["gpu_thr_b64"])
                    else:
                        if not cost_row[metric]:
                            continue
                        cost = float(cost_row[metric])
                    required = required_value(frontier, cost)
                    nmse = task01[(dataset, task, int(cost_row["enc_depth"]), int(cost_row["dec_depth"]))]
                    rows.append({
                        "dataset": dataset,
                        "task": task,
                        "enc_depth": cost_row["enc_depth"],
                        "dec_depth": cost_row["dec_depth"],
                        "cost_metric": metric,
                        "cost": f"{cost:.17g}",
                        "nmse_db_untrained": f"{nmse:.17g}",
                        "required_db": "" if required is None else f"{required:.17g}",
                        "gap_db": "" if required is None else f"{nmse - required:.17g}",
                    })
    atomic_csv(PARETO_CSV, rows, PARETO_FIELDS)
    make_pareto_plot(baseline_rows, cost_rows, task01)
    logger.info("PASS pareto: rows=%d", len(rows))
    return rows


def make_pareto_plot(baseline_rows, cost_rows, task01):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for row_index, dataset in enumerate(("D17", "D18")):
        for col_index, task in enumerate(TASKS):
            ax = axes[row_index, col_index]
            frontier = official_frontier(cost_rows, baseline_rows, dataset, task, "gflops")
            xs, ys = [], []
            best = math.inf
            for cost, nmse, _ in frontier:
                if nmse < best:
                    xs.append(cost)
                    ys.append(nmse)
                    best = nmse
            ax.step(xs, ys, where="post", color="black", linewidth=1.5, label="Official frontier")
            ax.scatter([point[0] for point in frontier], [point[1] for point in frontier], color="black", s=38)
            for cost, nmse, model in frontier:
                ax.annotate(model, (cost, nmse), xytext=(4, 4), textcoords="offset points", fontsize=8)
            base_costs = [row for row in cost_rows if row["shape"] == "D17" and row["model"] == "base"]
            ax.scatter(
                [float(row["gflops"]) for row in base_costs],
                [task01[(dataset, task, int(row["enc_depth"]), int(row["dec_depth"]))] for row in base_costs],
                facecolors="none",
                edgecolors="tab:blue",
                s=30,
                label="Base untrained truncations",
            )
            for cost_row in base_costs:
                enc, dec = int(cost_row["enc_depth"]), int(cost_row["dec_depth"])
                ax.annotate(
                    f"({enc},{dec})",
                    (float(cost_row["gflops"]), task01[(dataset, task, enc, dec)]),
                    fontsize=8,
                    alpha=0.75,
                )
            ax.set_xscale("log")
            ax.set_xlabel("GFLOPs (log scale)", fontsize=9)
            ax.set_ylabel("NMSE (dB)", fontsize=9)
            ax.set_title(f"{dataset} / {task}", fontsize=11)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    temp = PARETO_PNG.with_suffix(f".{os.getpid()}.tmp.png")
    fig.savefig(temp, dpi=180)
    plt.close(fig)
    os.replace(temp, PARETO_PNG)


def fmt(value, digits=4):
    if value is None or value == "":
        return "OOM"
    return f"{float(value):.{digits}f}"


def make_report(selfcheck, baseline_rows, cost_rows, pareto_rows):
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    baseline = {
        (row["model"], row["dataset"], row["split"], row["task"], float(row["mask_ratio"])): row
        for row in baseline_rows
    }
    lines = [
        "# Task 02：数据缓存、官方模型基线与成本全表",
        "",
        "## 交接自检",
        "",
        f"- 结果：**{selfcheck['status']}**。WiFo-Base 为 0 个 missing key、4 个预期 `decoder_pred_dual.*` unexpected key。",
        f"- D1 train/val 与 D17 test 形状：`{selfcheck['data_shapes']}`。",
        f"- 完整截断前向与官方前向最大差值：{selfcheck['forward_max_abs_diff']:.12g}。",
        f"- D17 temporal 完整模型 NMSE：{selfcheck['d17_temporal_nmse_lin']:.9g} / {selfcheck['d17_temporal_nmse_db']:.6f} dB。",
        f"- 自检时 GPU 空闲 {selfcheck['gpu_free_gib']:.2f} GiB，E 盘空闲 {selfcheck['disk_free_gib']:.2f} GiB。",
        f"- 环境：{selfcheck['environment']}",
        "",
        "## 数据缓存",
        "",
        f"- 共 {len(manifest['files'])} 个 `.npy` 文件，总计 {manifest['total_bytes'] / 2**30:.3f} GiB。",
        "- 每个文件使用固定随机种子抽取 5 个样本，与源 MAT 逐值比较，最大差值均为 0；重复运行只跳过通过校验的文件。",
        "- 数组统一为 `[N,2,T,A,K]` float32；源路径、形状、样本数、MD5 前 12 位和校验索引见 `cache/manifest.json`。",
        "",
        "| 内容 | 文件数 | 样本数合计 |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        selected = [record for record in manifest["files"] if record["split"] == split]
        lines.append(f"| {split} | {len(selected)} | {sum(record['samples'] for record in selected)} |")
    lines.extend([
        "",
        "## 与 WiFo 论文 Table 4 对照",
        "",
        "下表为线性 NMSE；D1,D5,D9,D15 列先对四个配置的线性 NMSE 取平均。括号内为论文值。",
        "",
        "| 模型 | 任务 | D1,D5,D9,D15 | D17 | D18 |",
        "|---|---|---:|---:|---:|",
    ])
    paper = {
        "tiny": {"temporal": (0.315, 0.444, 0.506), "fre": (0.343, 0.440, 0.399)},
        "little": {"temporal": (0.271, 0.371, 0.446), "fre": (0.260, 0.284, 0.301)},
        "small": {"temporal": (0.245, 0.326, 0.421), "fre": (0.239, 0.245, 0.299)},
        "base": {"temporal": (0.237, 0.305, 0.420), "fre": (0.232, 0.229, 0.280)},
    }
    for model in ("tiny", "little", "small", "base"):
        for task in TASKS:
            group = float(np.mean([float(baseline[(model, dataset, "test", task, 0.5)]["nmse_lin"]) for dataset in ("D1", "D5", "D9", "D15")]))
            values = (group,) + tuple(float(baseline[(model, dataset, "test", task, 0.5)]["nmse_lin"]) for dataset in ("D17", "D18"))
            expected = paper[model][task]
            cells = [f"{value:.6f} ({reference:.3f})" for value, reference in zip(values, expected)]
            lines.append(f"| {model} | {task} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 掩码比例摸底（仅 D1、D16 val）",
        "",
        "这些验证集结果可用于后续选档；D17、D18 未用于掩码选档。",
        "",
        "| 数据集 | 任务 | mask | 保留 patch 数 | NMSE 线性 | NMSE dB |",
        "|---|---|---:|---:|---:|---:|",
    ])
    shapes = {"D1": (24, 4, 128), "D16": (16, 32, 32)}
    for dataset in ("D1", "D16"):
        for task in TASKS:
            for ratio in MASK_RATIOS:
                shape = shapes[dataset]
                patches = int((shape[0] // 4 if task == "temporal" else shape[2] // 4) * (1 - ratio))
                row = baseline[("base", dataset, "val", task, ratio)]
                lines.append(f"| {dataset} | {task} | {ratio:g} | {patches} | {float(row['nmse_lin']):.6f} | {float(row['nmse_db']):.6f} |")
    meta = json.loads((WORK / "costs_meta.json").read_text(encoding="utf-8"))
    lines.extend([
        "",
        "## 成本全表（D17 形状）",
        "",
        f"- 成本测量前 `nvidia-smi`：`{meta['gpu_before_costs']}`。时延为三轮中位数，括号内为三轮极差；吞吐单位为样本/秒。",
        "- `base_official` 是官方完整模型参照，`base` 是 Base 截断网格；二者在 (6,4) 为重复执行检查点。",
        "- temporal 与 fre 的 FLOPs 已在 Base 完整模型、D17 形状上验证一致；FLOPs 不含加噪，时延使用已加噪输入和真实截断路径。",
        "",
        "| 模型 | enc | dec | params_exec | GFLOPs | GPU b1 ms | GPU thr b64 | GPU thr b256 | CPU t1 ms | CPU t4 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cost_rows:
        if row["shape"] != "D17":
            continue
        lines.append(
            f"| {row['model']} | {row['enc_depth']} | {row['dec_depth']} | {int(row['params_exec'])} | {fmt(row['gflops'])} ({fmt(row['gflops_range'])}) | "
            f"{fmt(row['gpu_b1_ms'], 2)} ({fmt(row['gpu_b1_ms_range'], 2)}) | {fmt(row['gpu_thr_b64'], 1)} ({fmt(row['gpu_thr_b64_range'], 1)}) | "
            f"{fmt(row['gpu_thr_b256'], 1)} ({fmt(row['gpu_thr_b256_range'], 1)}) | {fmt(row['cpu_b1_ms_t1'], 2)} ({fmt(row['cpu_b1_ms_t1_range'], 2)}) | "
            f"{fmt(row['cpu_b1_ms_t4'], 2)} ({fmt(row['cpu_b1_ms_t4_range'], 2)}) |"
        )
    lines.extend(["", "## 每种成本口径下 gap_db 最小的 5 个截断点", "", "正值表示未训练截断被官方前沿压制；required_db 为空表示没有成本更低的官方参照。", ""])
    for metric in ("gflops", "1/gpu_thr_b64", "cpu_b1_ms_t1"):
        for dataset in ("D17", "D18"):
            for task in TASKS:
                selected = [row for row in pareto_rows if row["cost_metric"] == metric and row["dataset"] == dataset and row["task"] == task and row["gap_db"] != ""]
                selected.sort(key=lambda row: float(row["gap_db"]))
                lines.extend([f"### {metric} / {dataset} / {task}", "", "| enc | dec | cost | NMSE 未训练 dB | required dB | gap dB |", "|---:|---:|---:|---:|---:|---:|"])
                for row in selected[:5]:
                    lines.append(f"| {row['enc_depth']} | {row['dec_depth']} | {float(row['cost']):.6g} | {float(row['nmse_db_untrained']):.6f} | {float(row['required_db']):.6f} | {float(row['gap_db']):.6f} |")
                lines.append("")
    oom = [row for row in cost_rows if row["shape"] == "D17" and not row["gpu_thr_b256"]]
    lines.extend([
        "## 遇到的问题",
        "",
        f"- D17 形状下 batch=256 吞吐 OOM 的路径数：{len(oom)}；OOM 单元格在 CSV 中留空。",
        "- PyTorch 提示未安装 Triton；本任务使用常规 PyTorch 算子，FLOPs 仍由 FlopCounterMode 统计。",
        "- 本报告只列基线、成本与差距数字，不作主骨干或论文主张选择。",
        "",
    ])
    temp = REPORT_MD.with_suffix(f".{os.getpid()}.tmp.md")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, REPORT_MD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logger = setup_logging()
    RESULTS.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    final_paths = (BASELINES_CSV, PERSAMPLE_NPZ, COSTS_CSV, PARETO_CSV, PARETO_PNG, REPORT_MD)
    if any(path.exists() for path in final_paths) and not args.resume and not args.self_check_only:
        raise FileExistsError("Task 02 outputs already exist; use --resume only to continue this task, not to overwrite them")
    check = selfcheck(logger)
    if args.self_check_only:
        return
    if not (CACHE / "manifest.json").is_file():
        raise FileNotFoundError("Run python prep_cache.py before the full Task 02 pipeline")
    baseline_rows = run_baselines(logger, args.resume)
    cost_rows = run_costs(logger, args.resume)
    pareto_rows = run_pareto(logger, baseline_rows, cost_rows)
    make_report(check, baseline_rows, cost_rows, pareto_rows)
    atomic_json(WORK / "complete.json", {"status": "PASS", "completed_at": time.strftime("%F %T"), "baseline_rows": len(baseline_rows), "cost_rows": len(cost_rows), "pareto_rows": len(pareto_rows)})
    logger.info("PASS Task 02: all outputs completed")


if __name__ == "__main__":
    main()
