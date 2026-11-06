# WiFo Static Depth Configuration and Lightweight Recovery

This lightweight reproduction package provides the experiment scripts, validation scripts, aggregated results, and plotting utilities for **Static Depth Configuration and Lightweight Recovery for WiFo-Based Channel Prediction**.

It is intentionally small. It does **not** include raw datasets, official WiFo checkpoints, trained exit checkpoints, virtual environments, paper source files, or generated paper figures. Large files are downloaded or generated only when needed.

## Quick reproduction: regenerate the current figure and report

The aggregated CSV/JSON files in `results/` are included. You do not need the raw datasets or model checkpoints for this step.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
pip install -r requirements.txt
python make_depth_cost.py
python make_full_report.py
```

Outputs:

- `figures/cl_fig2_depth_cost.pdf`, `.svg`, and `.png`
- `WiFo_实验数据与证据汇总报告.md`

The figure uses `results/10_curves.csv`, `results/10_selection.csv`, and the D17 cost records in `results/11_pareto.csv`. The report aggregates the included result files.

## Full experiment reproduction

Full reruns require a CUDA-capable environment, the official WiFo code and weights, and the original datasets. Training and evaluation can take substantial time. Keep at least 80 GB of free disk space for the original data and generated caches.

### 1. Official dependencies

- WiFo code: <https://github.com/PKU-PCNI/WiFo>
- Official Base, Small, Little, and Tiny weights: <https://huggingface.co/liuboxun/WiFo>
- D1–D18 test data: <https://huggingface.co/datasets/liuboxun/WiFo-dataset>
- D1–D16 pretraining data: <https://disk.pku.edu.cn/link/AA003E48DD5EF343C18ACD92ACF3BB8E3E>

The WiFo code version checked during this work was:

```text
a0889e124aeb9dcc423fc37e6796f77c7c4f7f24
```

The dataset README specifies Apache-2.0. Third-party resources are linked rather than redistributed; consult their respective licenses.

### 2. Configure paths

The scripts use these environment variables:

- `WIFO_SRC`: directory containing the official `model.py` and `mask_strategy.py`
- `WIFO_WEIGHTS`: directory containing `wifo_base.pkl`, `wifo_small.pkl`, `wifo_little.pkl`, and `wifo_tiny.pkl`
- `WIFO_DATA`: the `RF-only_channel_dataset_for_WiFo` directory

Example for Bash:

```bash
export WIFO_SRC=/path/to/WiFo/src
export WIFO_WEIGHTS=/path/to/WiFo
export WIFO_DATA=/path/to/RF-only_channel_dataset_for_WiFo
```

Example for Windows PowerShell:

```powershell
$env:WIFO_SRC="C:\path\to\WiFo\src"
$env:WIFO_WEIGHTS="C:\path\to\WiFo"
$env:WIFO_DATA="C:\path\to\RF-only_channel_dataset_for_WiFo"
```

If these variables are not set, the scripts look under `third_party/WiFo`, `assets/WiFo`, and `data/RF-only_channel_dataset_for_WiFo` in this directory.

### 3. Prepare caches and run tasks

```bash
python prep_cache.py
python run_01.py
python run_02.py
```

Continue with `run_03.py` through `run_12.py` in numerical order. The scripts generate caches, trained exit checkpoints, logs, per-sample results, and final aggregated outputs. These generated artifacts are excluded from the lightweight package.

D1–D16 are used for training and validation. D17 and D18 are used for review. Official WiFo weights are downloaded separately; trained exit checkpoints are produced by the training tasks rather than bundled here.

## Environment used during this work

- Python 3.11.9
- PyTorch 2.14.0+cu130
- NumPy 2.4.6
- SciPy 1.17.1
- Matplotlib 3.10.8
- hdf5storage 0.2.2

Install an appropriate PyTorch build for your CUDA version rather than assuming that the CUDA 13.0 build is required.

## Validation

The `check/` directory contains task-specific validation scripts. After the relevant results have been generated, run the corresponding `taskXX_validate.py` script to check file integrity, aggregation, and evaluation invariants.

The selection summary may include the full model as a valid reference outcome. Paper tables can display truncated configurations only; this display choice does not alter the underlying results.
