import csv
import math
import os
import platform
import sys
import time
from argparse import Namespace
from pathlib import Path

sys.dont_write_bytecode = True

import hdf5storage
import numpy as np
import scipy
import torch
from torch.utils.flop_counter import FlopCounterMode

W = Path(__file__).resolve().parent
SRC = Path(os.environ.get("WIFO_SRC", str(W / "third_party" / "WiFo" / "src")))
WEIGHTS = Path(os.environ.get("WIFO_WEIGHTS", str(W / "assets" / "WiFo" / "wifo_base.pkl")))
R = Path(os.environ.get("WIFO_DATA", str(W / "data" / "RF-only_channel_dataset_for_WiFo")))
DATA_ROOT = R / "dataset"
RESULTS = W / "results"
CSV_PATH = RESULTS / "01_trunc_grid.csv"
REPORT_PATH = RESULTS / "01_report.md"
DATASETS = ("D1", "D16", "D17", "D18")
TASKS = ("temporal", "fre")
EXPECTED_SHAPES = {
    "D1": (1000, 2, 24, 4, 128),
    "D16": (1000, 2, 16, 32, 32),
    "D17": (1000, 2, 16, 32, 32),
    "D18": (1000, 2, 24, 16, 64),
}

sys.path.insert(0, str(SRC))
from model import WiFo_model
from wifo_trunc import forward_trunc, forward_trunc_from_noisy


def seed_every_run():
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)


def build_model(device):
    args = Namespace(
        size="base",
        t_patch_size=4,
        patch_size=4,
        pos_emb="SinCos_3D",
        no_qkv_bias=0,
    )
    model = WiFo_model(args)
    state_dict = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    assert missing == [], f"Unexpected missing keys: {missing}"
    assert len(unexpected) == 4 and all(
        key.startswith("decoder_pred_dual.") for key in unexpected
    ), f"Unexpected keys do not match the expected four decoder_pred_dual keys: {unexpected}"
    print(f"Weight load check: missing={missing}, unexpected={unexpected}", flush=True)
    return model.to(device).eval(), unexpected


def load_datasets():
    datasets = {}
    for name in DATASETS:
        path = DATA_ROOT / name / "X_test.mat"
        raw = hdf5storage.loadmat(str(path))["X_val"]
        complex_tensor = torch.tensor(np.array(raw, dtype=complex)).unsqueeze(1)
        tensor = torch.cat((complex_tensor.real, complex_tensor.imag), dim=1).float()
        datasets[name] = tensor
        shape = tuple(tensor.shape)
        match = "matches expected" if shape == EXPECTED_SHAPES[name] else "DIFFERS from expected"
        print(f"{name} final shape: {shape} ({match})", flush=True)
    return datasets


def as_imgs_list(batch):
    return list(batch.unsqueeze(1))


def correctness_check(model, data, device):
    batch = data[:128].to(device)
    imgs_list = as_imgs_list(batch)
    with torch.no_grad():
        seed_every_run()
        official = model(imgs_list, mask_ratio=0.5, mask_strategy="temporal")
        seed_every_run()
        truncated = forward_trunc(model, imgs_list, 0.5, "temporal", 6, 4)
    max_diff = (official[2] - truncated[2]).abs().max().item()
    print(f"Correctness pred max absolute difference: {max_diff:.12g}", flush=True)
    if not max_diff < 1e-5:
        RESULTS.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            "# WiFo-Base 深度截断诊断（失败）\n\n"
            f"正确性检查未通过：pred 最大绝对差为 `{max_diff:.12g}`，要求 `< 1e-5`。\n"
            "按任务要求已停止，未执行网格评测。\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Correctness check failed: {max_diff:.12g} >= 1e-5")
    return max_diff


def evaluate_nmse(model, data, device, task, enc_depth, dec_depth):
    error_nmse = 0.0
    num = 0
    seed_every_run()
    with torch.no_grad():
        for start in range(0, data.shape[0], 64):
            batch = data[start : start + 64].to(device)
            _, _, pred, target, mask = forward_trunc(
                model, as_imgs_list(batch), 0.5, task, enc_depth, dec_depth
            )
            dim1 = pred.shape[0]
            pred_mask = pred.squeeze(dim=2)
            target_mask = target.squeeze(dim=2)
            y_pred = (
                pred_mask[mask == 1]
                .reshape(-1, 1)
                .reshape(dim1, -1)
                .detach()
                .cpu()
                .numpy()
            )
            y_target = (
                target_mask[mask == 1]
                .reshape(-1, 1)
                .reshape(dim1, -1)
                .detach()
                .cpu()
                .numpy()
            )
            error_nmse += np.sum(
                np.mean(np.abs(y_target - y_pred) ** 2, axis=1)
                / np.mean(np.abs(y_target) ** 2, axis=1)
            )
            num += y_pred.shape[0]
    nmse_lin = float(error_nmse / num)
    return nmse_lin, float(10 * math.log10(nmse_lin))


def measure_gflops(model, sample, task, enc_depth, dec_depth):
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            forward_trunc_from_noisy(
                model, sample, 0.5, task, enc_depth, dec_depth
            )
    return counter.get_total_flops() / 1e9


def measure_latency_ms(model, batch, task, enc_depth, dec_depth):
    imgs_list = as_imgs_list(batch)
    seed_every_run()
    with torch.no_grad():
        for _ in range(20):
            forward_trunc(model, imgs_list, 0.5, task, enc_depth, dec_depth)
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            forward_trunc(model, imgs_list, 0.5, task, enc_depth, dec_depth)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
    return float(np.median(elapsed))


def write_csv_header():
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "task",
                "enc_depth",
                "dec_depth",
                "nmse_lin",
                "nmse_db",
                "gflops",
                "lat_b1_ms",
                "lat_b64_ms",
            ),
        )
        writer.writeheader()


def append_row(row):
    with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writerow(row)


def generate_report(rows, datasets, max_diff, unexpected_keys):
    by_key = {(row["dataset"], row["task"]): row for row in rows if row["enc_depth"] == 6 and row["dec_depth"] == 4}
    lines = [
        "# WiFo-Base 深度截断诊断报告",
        "",
        "## 设置与环境",
        "",
        "- 预训练 WiFo-Base：6 层编码器、4 层解码器；不训练，直接截断。",
        "- 任务：`temporal`、`fre`；`mask_ratio=0.5`；输入加 20 dB 噪声，target 同样来自加噪输入。",
        "- NMSE：仅在 `mask==1` 位置计算，每样本先求 NMSE，再对 1000 个样本求平均；dB 为 `10*log10(nmse_lin)`。",
        "- 每次评测前设置 CPU/CUDA 随机种子为 100；batch=64，shuffle=False。",
        f"- OS：{platform.platform()}；Python：{platform.python_version()}。",
        f"- PyTorch：{torch.__version__}；CUDA runtime：{torch.version.cuda}；GPU：{torch.cuda.get_device_name(0)}。",
        f"- NumPy：{np.__version__}；SciPy：{scipy.__version__}；hdf5storage：{hdf5storage.__version__}。",
        "- FLOPs 使用 `torch.utils.flop_counter.FlopCounterMode` 统计单样本前向，不含加噪；时延为 warm-up 20 次后 100 次同步计时的中位数。",
        "- `causal_masking`（temporal）与 `fre_masking`（fre）均使用 `torch.arange` 构造固定顺序，不含随机性；前向随机性来自 `torch.randn_like` 加噪。",
        f"- 权重加载断言通过：0 个 missing key，4 个 unexpected key（`{', '.join(unexpected_keys)}`）。",
        "",
        "## 数据形状与正确性检查",
        "",
        "| 数据集 | 最终形状 | 是否符合预期 |",
        "|---|---|---|",
    ]
    for name in DATASETS:
        shape = tuple(datasets[name].shape)
        lines.append(f"| {name} | {list(shape)} | {'是' if shape == EXPECTED_SHAPES[name] else '否'} |")
    lines.extend(
        [
            "",
            f"D17 前 128 个样本、同一随机种子下，`forward_trunc(6,4)` 与官方 `model.forward` 的 pred 最大绝对差为 **{max_diff:.12g}**，小于 `1e-5`，检查通过。",
            "",
            "## 完整模型（6,4）NMSE",
            "",
            "| 数据集 | temporal 线性 | temporal dB | fre 线性 | fre dB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in DATASETS:
        temporal = by_key[(name, "temporal")]
        fre = by_key[(name, "fre")]
        lines.append(
            f"| {name} | {temporal['nmse_lin']:.9g} | {temporal['nmse_db']:.6f} | "
            f"{fre['nmse_lin']:.9g} | {fre['nmse_db']:.6f} |"
        )

    lines.extend(["", "## 全部 NMSE(dB) 矩阵", "", "每行是 enc_depth=1..6，每列是 dec_depth=0..4。", ""])
    lookup = {(row["dataset"], row["task"], row["enc_depth"], row["dec_depth"]): row for row in rows}
    for name in DATASETS:
        for task in TASKS:
            lines.extend(
                [
                    f"### {name} / {task}",
                    "",
                    "| enc\\dec | 0 | 1 | 2 | 3 | 4 |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for enc in range(1, 7):
                values = [lookup[(name, task, enc, dec)]["nmse_db"] for dec in range(5)]
                lines.append(f"| {enc} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
            lines.append("")

    lines.extend(
        [
            "## 给定退化上限下的最低 FLOPs 配置",
            "",
            "退化量为当前配置 NMSE(dB) 减去同一数据集、同一任务完整模型 NMSE(dB)。",
            "",
            "| 数据集 | 任务 | 退化上限 | enc | dec | GFLOPs | NMSE(dB) | 实际退化(dB) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DATASETS:
        for task in TASKS:
            base = by_key[(name, task)]["nmse_db"]
            candidates = [row for row in rows if row["dataset"] == name and row["task"] == task]
            for threshold in (0.5, 1.0, 3.0):
                eligible = [row for row in candidates if row["nmse_db"] - base <= threshold + 1e-12]
                best = min(eligible, key=lambda row: (row["gflops"], row["enc_depth"], row["dec_depth"]))
                degradation = best["nmse_db"] - base
                lines.append(
                    f"| {name} | {task} | ≤{threshold:g} dB | {best['enc_depth']} | {best['dec_depth']} | "
                    f"{best['gflops']:.6f} | {best['nmse_db']:.6f} | {degradation:.6f} |"
                )

    enc_deltas = {}
    for enc in range(1, 6):
        enc_deltas[enc] = float(
            np.mean(
                [lookup[(name, task, enc, 4)]["nmse_db"] - by_key[(name, task)]["nmse_db"] for name in DATASETS for task in TASKS]
            )
        )
    dec_deltas = {}
    for dec in range(4):
        dec_deltas[dec] = float(
            np.mean(
                [lookup[(name, task, 6, dec)]["nmse_db"] - by_key[(name, task)]["nmse_db"] for name in DATASETS for task in TASKS]
            )
        )
    lines.extend(
        [
            "",
            "## 编码器与解码器截断的数值观察",
            "",
            "下表为四个数据集×两个任务共 8 组结果相对各自完整模型的平均 NMSE(dB) 退化。",
            "",
            "| 截断设置 | 深度 | 平均退化(dB) |",
            "|---|---:|---:|",
        ]
    )
    for enc, delta in enc_deltas.items():
        lines.append(f"| 编码器（dec=4） | enc={enc} | {delta:.6f} |")
    for dec, delta in dec_deltas.items():
        lines.append(f"| 解码器（enc=6） | dec={dec} | {delta:.6f} |")
    one_block_more = "编码器" if enc_deltas[5] > dec_deltas[3] else "解码器"
    lines.extend(
        [
            "",
            f"仅去掉一个 block 时，enc=5/dec=4 的平均退化为 {enc_deltas[5]:.6f} dB，enc=6/dec=3 的平均退化为 {dec_deltas[3]:.6f} dB；按此数值，{one_block_more}截断的退化更大。",
            "",
            "## 时延说明与遇到的问题",
            "",
            "- 仅 D1、D17 测量 batch=1 和 batch=64 的 GPU 时延；D16、D18 对应 CSV 单元格留空。",
            "- 未遇到阻断问题；四个数据集均成功读取，正确性检查与全部 240 组评测均完成。",
            "- PyTorch 在导入 FLOPs 工具时提示未安装 Triton；本前向使用常规 PyTorch 算子，统计仍由 `FlopCounterMode` 完成。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this task")
    device = torch.device("cuda")
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(
        f"Environment: Python {platform.python_version()}, torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    model, unexpected_keys = build_model(device)
    datasets = load_datasets()
    max_diff = correctness_check(model, datasets["D17"], device)

    write_csv_header()
    rows = []
    total = len(DATASETS) * len(TASKS) * 6 * 5
    completed = 0
    for name in DATASETS:
        data = datasets[name]
        sample = data[:1].to(device)
        batch64 = data[:64].to(device)
        for task in TASKS:
            for enc_depth in range(1, 7):
                for dec_depth in range(5):
                    nmse_lin, nmse_db = evaluate_nmse(
                        model, data, device, task, enc_depth, dec_depth
                    )
                    gflops = measure_gflops(model, sample, task, enc_depth, dec_depth)
                    lat_b1 = ""
                    lat_b64 = ""
                    if name in ("D1", "D17"):
                        lat_b1 = measure_latency_ms(
                            model, sample, task, enc_depth, dec_depth
                        )
                        lat_b64 = measure_latency_ms(
                            model, batch64, task, enc_depth, dec_depth
                        )
                    row = {
                        "dataset": name,
                        "task": task,
                        "enc_depth": enc_depth,
                        "dec_depth": dec_depth,
                        "nmse_lin": nmse_lin,
                        "nmse_db": nmse_db,
                        "gflops": gflops,
                        "lat_b1_ms": lat_b1,
                        "lat_b64_ms": lat_b64,
                    }
                    append_row(row)
                    rows.append(row)
                    completed += 1
                    print(
                        f"[{completed}/{total}] {name} {task} enc={enc_depth} dec={dec_depth} "
                        f"NMSE={nmse_db:.6f} dB GFLOPs={gflops:.6f} "
                        f"lat_b1={lat_b1 if lat_b1 != '' else '-'} lat_b64={lat_b64 if lat_b64 != '' else '-'}",
                        flush=True,
                    )
    generate_report(rows, datasets, max_diff, unexpected_keys)
    print(f"Completed {len(rows)} rows. CSV: {CSV_PATH}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
