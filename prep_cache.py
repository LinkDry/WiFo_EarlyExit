import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import hdf5storage
import numpy as np

W = Path(__file__).resolve().parent
R = Path(os.environ.get("WIFO_DATA", str(W / "data" / "RF-only_channel_dataset_for_WiFo")))
CACHE = W / "cache"
LOGS = W / "logs"
MANIFEST = CACHE / "manifest.json"
EXPECTED_COUNTS = {"train": 9000, "val": 2000, "test": 1000}


def setup_logging():
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("prep_cache")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS / "prep_cache.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(dataset, split):
    if dataset in ("D17", "D18"):
        assert split == "test"
        return R / "dataset" / dataset / "X_test.mat"
    filename = {"train": "X_train.mat", "val": "X_val.mat", "test": "X_test.mat"}[split]
    return R / "dataset4train" / dataset / filename


def load_source(path, expected_count):
    raw = hdf5storage.loadmat(str(path))
    keys = [key for key in raw if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one data variable in {path}, got {keys}")
    array = np.asarray(raw[keys[0]])
    if array.shape[0] != expected_count or array.ndim != 4 or not np.iscomplexobj(array):
        raise ValueError(f"Unexpected source array {path}: shape={array.shape}, dtype={array.dtype}")
    converted = np.stack((array.real, array.imag), axis=1).astype(np.float32, copy=False)
    return converted


def sample_indices(count):
    return np.random.default_rng(20261029).choice(count, size=5, replace=False)


def validate_array(converted, cached, dataset, split):
    if cached.dtype != np.float32 or cached.shape != converted.shape:
        raise ValueError(
            f"Cache mismatch for {dataset}/{split}: cached {cached.shape}/{cached.dtype}, "
            f"source {converted.shape}/{converted.dtype}"
        )
    indices = sample_indices(converted.shape[0])
    max_diff = float(np.max(np.abs(cached[indices] - converted[indices])))
    if max_diff != 0:
        raise ValueError(f"Cache sample mismatch for {dataset}/{split}: max_diff={max_diff}")
    return max_diff, sorted(int(index) for index in indices)


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def main():
    logger = setup_logging()
    started = time.time()
    free_gib = shutil.disk_usage(W.anchor).free / 2**30
    if free_gib < 40:
        raise RuntimeError(f"Insufficient disk space: {free_gib:.2f} GiB free, need > 40 GiB")
    CACHE.mkdir(exist_ok=True)
    jobs = [(f"D{index}", split) for index in range(1, 17) for split in ("train", "val", "test")]
    jobs += [("D17", "test"), ("D18", "test")]
    records = []
    logger.info("Starting %d cache jobs; disk free %.2f GiB", len(jobs), free_gib)
    for index, (dataset, split) in enumerate(jobs, 1):
        source = source_path(dataset, split)
        target = CACHE / f"{dataset}_{split}.npy"
        logger.info("[%d/%d] Reading %s", index, len(jobs), source)
        source_md5 = md5(source)
        converted = load_source(source, EXPECTED_COUNTS[split])
        indices = sample_indices(converted.shape[0])
        if target.exists():
            cached = np.load(target, mmap_mode="r")
            max_diff, checked = validate_array(converted, cached, dataset, split)
            action = "skipped_valid"
            del cached
        else:
            temp = CACHE / f"{dataset}_{split}.{os.getpid()}.tmp.npy"
            with temp.open("wb") as handle:
                np.save(handle, converted, allow_pickle=False)
            cached = np.load(temp, mmap_mode="r")
            max_diff, checked = validate_array(converted, cached, dataset, split)
            del cached
            os.replace(temp, target)
            action = "created"
            logger.info("Created %s", target)
        record = {
            "dataset": dataset,
            "split": split,
            "source": str(source),
            "source_md5_12": source_md5[:12],
            "cache": str(target.relative_to(W)),
            "shape": list(converted.shape),
            "samples": int(converted.shape[0]),
            "dtype": "float32",
            "bytes": int(target.stat().st_size),
            "validation": {
                "status": action,
                "sample_indices": checked,
                "max_abs_diff": max_diff,
            },
        }
        records.append(record)
        logger.info(
            "%s/%s %s shape=%s samples=%d max_diff=%g",
            dataset,
            split,
            action,
            tuple(converted.shape),
            converted.shape[0],
            max_diff,
        )
        del converted
        gc.collect()
    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(R),
        "array_layout": "[N,2,T,A,K] float32; axis 1 stores real and imaginary parts",
        "validation": "five deterministic random samples per file; exact equality required",
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    atomic_json(MANIFEST, manifest)
    logger.info(
        "PASS cache: files=%d total=%.3f GiB elapsed=%.1f min manifest=%s",
        len(records),
        manifest["total_bytes"] / 2**30,
        (time.time() - started) / 60,
        MANIFEST,
    )


if __name__ == "__main__":
    main()
