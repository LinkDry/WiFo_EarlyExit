import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
R = W / "results"
OUT = W / "WiFo_实验数据与证据汇总报告.md"


def rows(name):
    with (R / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value):
    if value is None or value == "":
        return "—"
    try:
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        if value.is_integer():
            return str(int(value))
        if abs(value) >= 10000:
            return f"{value:.0f}"
        if abs(value) >= 100:
            return f"{value:.3f}"
        return f"{value:.6f}"
    except (TypeError, ValueError):
        return str(value).replace("|", "\\|")


def table(data, fields, labels=None, limit=None):
    if limit is not None:
        data = data[:limit]
    labels = labels or fields
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in data:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def json_rows(name):
    data = json.loads((R / name).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def aggregate(data, fields, value):
    grouped = {}
    for row in data:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(float(row[value]))
    return [{**dict(zip(fields, key)), value: float(np.mean(values)), f"{value}_std": float(np.std(values)), "n": len(values)} for key, values in grouped.items()]


def main():
    r01 = rows("01_trunc_grid.csv")
    b02 = rows("02_baselines.csv")
    c02 = rows("02_costs.csv")
    a03 = rows("03_val_aggregate.csv")
    c03 = rows("03_costs.csv")
    x04 = rows("04_controls.csv")
    a05 = rows("05_aggregate.csv")
    e05 = rows("05_equal_cost.csv")
    x06 = rows("06_controls.csv")
    c06 = rows("06_costs.csv")
    x07 = rows("07_controls.csv")
    w08 = rows("08_width_summary.csv")
    p08 = rows("08_probe_summary.csv")
    s08 = rows("08_stability.csv")
    x09 = rows("09_controls.csv")
    c09 = rows("09_costs.csv")
    v10 = rows("10_curves.csv")
    s10 = rows("10_selection.csv")
    p11 = rows("11_pareto.csv")
    r11 = rows("11_selection_review.csv")
    x12 = rows("12_controls.csv")

    out = []
    add = out.append
    add("# WiFo 深度截断与可恢复性：实验数据与证据汇总报告")
    add("")
    add("生成日期：2026-11-05")
    add("")
    add("本报告汇总 Task 01–12 的全部主要聚合结果、成本和证据边界。逐样本 NMSE、逐 epoch 训练历史及 checkpoint 不直接展开为数千行文本，而是通过末尾的完整文件索引定位。不同任务的数据预算、参考模型和聚合口径不混用；同一指标在多篇任务报告中出现时，以更新、更公平的对照为准。")
    add("")
    add("## 1. 研究问题与证据链")
    add("")
    add("研究问题不是单纯“WiFo 能否删层”，而是：**编码器与解码器截断的损伤是否不同？轻量适配和有限解冻能恢复多少？这些规律能否跨种子、宽度、任务和掩码复现，并转化为实际深度选择？**")
    add("")
    add(table([
        {"阶段":"原始截断","任务":"Task 01–02","回答":"不训练时损伤多少、成本多少"},
        {"阶段":"出口试训","任务":"Task 03","回答":"H0/H1 在 Base/Small 上能补回多少"},
        {"阶段":"完整适配对照","任务":"Task 04","回答":"完整模型是否也有通用适配收益"},
        {"阶段":"深度曲线","任务":"Task 05","回答":"相近成本下两侧损伤是否不同"},
        {"阶段":"机制探针","任务":"Task 06","回答":"冻结、输出适配和接口适配分别影响多少"},
        {"阶段":"宽度检验","任务":"Task 07","回答":"Base/Small 现象能否延伸到 Little/Tiny"},
        {"阶段":"多种子验证","任务":"Task 08","回答":"趋势是否依赖一次训练、掩码和样本量"},
        {"阶段":"公平恢复","任务":"Task 09","回答":"完整模型同等解冻后差异是否仍存在"},
        {"阶段":"中间深度","任务":"Task 10","回答":"更温和的截断是否更适合严格预算"},
        {"阶段":"部署复核","任务":"Task 11","回答":"绝对 NMSE—实际成本下是否有非支配点"},
        {"阶段":"跨骨干复核","任务":"Task 12","回答":"Small 是否复现 Base 的非对称恢复"},
    ], ["阶段","任务","回答"]))
    add("")
    add("## 2. 当前最强结论")
    add("")
    add("1. **编码器与解码器的截断敏感性不同。** 零训练相近成本比较、输出适配和有限解冻均支持这一判断。")
    add("2. **恢复评价必须有完整模型对应适配对照。** 完整 H1 本身也有通用收益，不能把截断 H1 的全部改善都算作恢复。")
    add("3. **公平恢复后编码器侧仍更易恢复。** Base 的 `(3,4)` 剩余代价约 0.188/0.283 dB，`(6,2)` 约 0.893/1.387 dB；Small 对应为 0.250/0.488 与 0.801/1.394 dB。")
    add("4. **中间深度比单一激进截断更有用。** `(5,4)` 在严格和中等预算下占主导，` (4,4)` 在 1 dB 下成为主要折中。")
    add("5. **宽度规律具有位置和任务边界。** `(3,4)` 的趋势较稳定，`(6,2)` 在 fre 和部分掩码下出现反例。")
    add("6. **部署优势不能夸大。** Base 探针可进入部分非支配集合，但官方小模型仍然强；D17、D18 的复核不等于完全独立外部测试。")
    add("")

    add("## 3. 公平恢复核心总表")
    add("")
    for name, data, label in (("Task 09 · Base", x09, "Base"), ("Task 12 · Small", x12, "Small")):
        add(f"### {label}")
        add("")
        add(table(data, ["enc_depth","dec_depth","train_seed","task","full_h1_db","full_update_db","trunc_h1_db","probe_db","probe_gain_vs_h1_db","residual_depth_cost_db"], ["enc","dec","seed","任务","完整 H1","完整有限解冻","截断 H1","截断有限解冻","解冻增益","剩余代价"]))
        add("")

    add("## 4. 中间深度与选择结果")
    add("")
    add("### 4.1 Task 10 恢复后曲线（按种子汇总）")
    add("")
    add(table(aggregate(v10,["backbone","enc_depth","dec_depth","task"],"depth_cost_db"),["backbone","enc_depth","dec_depth","task","depth_cost_db","depth_cost_db_std","n"],["骨干","enc","dec","任务","深度代价均值","种子标准差","种子数"]))
    add("")
    add("### 4.2 扩展候选选择分布")
    add("")
    selection_summary = []
    for epsilon in (0.25, 0.5, 1.0):
        selected = [row for row in s10 if float(row["epsilon_db"]) == epsilon]
        counts = Counter(f"e{row['selected_enc_depth']}d{row['selected_dec_depth']}" for row in selected if row["feasible"] == "True")
        selection_summary.append({"epsilon":epsilon, **{point:counts[point] for point in ("e6d4","e5d4","e4d4","e3d4","e6d3","e6d2")}})
    add(table(selection_summary,["epsilon","e6d4","e5d4","e4d4","e3d4","e6d3","e6d2"]))
    add("")

    add("## 5. 多种子宽度趋势")
    add("")
    trend_rows = []
    for depth in ((3,4),(6,2)):
        for mask in (0.5,0.25):
            for task in ("temporal","fre"):
                record = {"enc":depth[0],"dec":depth[1],"mask":mask,"task":task}
                values = []
                for backbone in ("base","small","little","tiny"):
                    row = next(row for row in w08 if row["backbone"] == backbone and int(row["enc_depth"]) == depth[0] and int(row["dec_depth"]) == depth[1] and float(row["mask_ratio"]) == mask and row["task"] == task)
                    record[backbone] = float(row["depth_cost_mean_db"])
                    values.append(float(row["depth_cost_mean_db"]))
                record["increments"] = ",".join(f"{b-a:.3f}" for a,b in zip(values,values[1:]))
                trend_rows.append(record)
    add(table(trend_rows,["enc","dec","mask","task","base","small","little","tiny","increments"],["enc","dec","mask","任务","Base","Small","Little","Tiny","相邻宽度增量"]))
    add("")

    add("## 6. Base 多种子解冻探针")
    add("")
    add(table(p08,["enc_depth","dec_depth","train_seed","task","output_h1_db","probe_db","probe_gain_db","residual_depth_cost_db"],["enc","dec","种子","任务","输出 H1","探针","解冻增益","剩余代价"]))
    add("")

    add("## 7. 部署 Pareto 与留出配置复核")
    add("")
    add("Task 11 使用 24 个候选、96 组逐样本结果和 48 条形状相关成本。下表列出全部候选的绝对 NMSE 与实际成本；非支配点及逐样本原文件见完整索引。")
    add("")
    add(table(p11,["name","dataset","task","nmse_db","gflops","gpu_b1_ms","gpu_thr_b64","cpu_b1_ms_t1"],["候选","配置","任务","NMSE dB","GFLOPs","GPU b1 ms","GPU thr b64","CPU t1 ms"]))
    add("")
    add("### 验证集选择的复核约束")
    add("")
    review_summary = []
    for dataset in ("D17","D18"):
        for epsilon in (0.25,0.5,1.0):
            selected = [row for row in r11 if row["dataset"] == dataset and float(row["epsilon_db"]) == epsilon]
            excess = [max(0.0,float(row["review_delta_db"])-epsilon) for row in selected]
            review_summary.append({"dataset":dataset,"epsilon":epsilon,"n":len(selected),"satisfied":sum(row["constraint_satisfied"]=="True" for row in selected),"mean_excess":float(np.mean(excess)),"max_excess":max(excess)})
    add(table(review_summary,["dataset","epsilon","n","satisfied","mean_excess","max_excess"],["配置","ε","组合数","满足数","平均超限 dB","最大超限 dB"]))
    add("")

    add("## 8. 各任务详细数据")
    add("")
    add("### 8.1 Task 01：零训练完整模型与截断诊断")
    add("")
    full01 = [row for row in r01 if int(row["enc_depth"]) == 6 and int(row["dec_depth"]) == 4]
    add(table(full01,["dataset","task","nmse_lin","nmse_db","gflops","lat_b1_ms","lat_b64_ms"]))
    add("")
    add("Task 01 共 240 个配置级截断结果，完整网格保存在 `01_trunc_grid.csv`。平均退化：编码器去 1/3 层约 0.17/0.82 dB；解码器去 1/2 层约 2.55/4.35 dB；dec=0 不可用。此任务是历史诊断，不用于后续验证集选点。")
    add("")

    add("### 8.2 Task 02：官方模型基线与成本")
    add("")
    paper_rows = []
    for model in ("tiny","little","small","base"):
        for task in ("temporal","fre"):
            selected = [row for row in b02 if row["model"] == model and row["task"] == task and row["split"] == "test"]
            group = np.mean([float(row["nmse_lin"]) for row in selected if row["dataset"] in ("D1","D5","D9","D15")])
            paper_rows.append({"model":model,"task":task,"D1,5,9,15":group,"D17":next(float(row["nmse_lin"]) for row in selected if row["dataset"] == "D17"),"D18":next(float(row["nmse_lin"]) for row in selected if row["dataset"] == "D18")})
    add(table(paper_rows,["model","task","D1,5,9,15","D17","D18"]))
    add("")
    add("官方完整模型在 D17 形状上的成本：")
    add("")
    official_costs = [row for row in c02 if row["shape"] == "D17" and row["model"] in ("base_official","small","little","tiny")]
    add(table(official_costs,["model","params_exec","gflops","gpu_b1_ms","gpu_thr_b64","gpu_thr_b256","cpu_b1_ms_t1","cpu_b1_ms_t4"]))
    add("")
    mask02 = [row for row in b02 if row["split"] == "val"]
    add("掩码比例摸底仅使用 D1、D16 val：")
    add("")
    add(table(mask02,["dataset","task","mask_ratio","nmse_lin","nmse_db"]))
    add("")

    add("### 8.3 Task 03：Base、Small 出口试训聚合")
    add("")
    add(table(a03,["method","backbone","enc_depth","dec_depth","task","nmse_lin_mean","nmse_db_mean","recovery_db","degradation_from_full_db"]))
    add("")
    add("出口成本：")
    add("")
    add(table(c03,["method","backbone","enc_depth","dec_depth","params_exec","gflops","gpu_b1_ms","gpu_thr_b64","cpu_b1_ms_t1"]))
    add("")

    add("### 8.4 Task 04：完整模型适配对照")
    add("")
    add(table(x04,["backbone","task","full_db","full_h1_db","generic_gain_db","untrained_db","trunc_h1_db","total_recovery_db","depth_cost_db","specific_recovery_db","full_h0_db","h0_to_h1_db"]))
    add("")

    add("### 8.5 Task 05：零训练深度曲线")
    add("")
    add(table(a05,["backbone","route","enc_depth","dec_depth","task","nmse_lin_mean","nmse_db_mean","degradation_db"]))
    add("")
    add("相近 GFLOPs 配对：")
    add("")
    add(table(e05,["backbone","task","encoder_enc_depth","decoder_dec_depth","encoder_gflops","decoder_gflops","cost_ratio","encoder_degradation_db","decoder_degradation_db","encoder_minus_decoder_db"]))
    add("")

    add("### 8.6 Task 06：输出适配、有限解冻与接口适配")
    add("")
    add(table(x06,["enc_depth","dec_depth","task","untrained_db","output_h1_db","probe_db","probe_gain_db","full_h1_db","residual_depth_cost_db","interface_db","interface_gain_db"]))
    add("")
    add("对应成本与可训练参数：")
    add("")
    add(table(c06,["mode","enc_depth","dec_depth","params_exec","trainable_params","gflops","gpu_b1_ms","gpu_thr_b64","cpu_b1_ms_t1"]))
    add("")

    add("### 8.7 Task 07：四宽度代表位置对照")
    add("")
    add(table(x07,["backbone","enc_depth","dec_depth","task","full_h1_db","trunc_h1_db","total_recovery_db","generic_gain_db","depth_cost_db","specific_recovery_db","cost_reduction_pct"]))
    add("")

    add("### 8.8 Task 08：多种子深度代价与成本")
    add("")
    add(table(w08,["backbone","enc_depth","dec_depth","mask_ratio","task","seed_mean_db","seed_std_db","depth_cost_mean_db","depth_cost_std_db","cost_reduction_pct"]))
    add("")
    add("Bootstrap 稳定性按样本量与 ε 汇总（144 个组合的统计）：")
    add("")
    stability_summary = []
    for n in (16,64,256,2000):
        for epsilon in (0.25,0.5,1.0):
            selected = [row for row in s08 if int(row["sample_size"]) == n and float(row["epsilon_db"]) == epsilon]
            dominant = [max(float(row["p_e6d4"]),float(row["p_e3d4"]),float(row["p_e6d2"])) for row in selected]
            stability_summary.append({"n":n,"epsilon":epsilon,"dominant_min":min(dominant),"dominant_median":float(np.median(dominant)),"dominant_ge_0.9":float(np.mean(np.array(dominant)>=0.9)),"p_full":float(np.median([float(row["p_e6d4"]) for row in selected])),"p_encoder":float(np.median([float(row["p_e3d4"]) for row in selected])),"p_decoder":float(np.median([float(row["p_e6d2"]) for row in selected]))})
    add(table(stability_summary,["n","epsilon","dominant_min","dominant_median","dominant_ge_0.9","p_full","p_encoder","p_decoder"],["样本量","ε","主导最小","主导中位","主导≥0.9 比例","完整中位","编码器中位","解码器中位"]))
    add("")

    add("### 8.9 Task 09–12 的文件")
    add("")
    add("Task 09 和 Task 12 的全部逐种子对照已列于第 3 节；Task 10 的逐种子曲线与选择见 `10_curves.csv`、`10_selection.csv`，本报告第 4 节给出按种子聚合；Task 11 的全部 96 个候选结果见第 7 节。")
    add("")

    add("## 9. 完整数据与复现文件索引")
    add("")
    add(f"实验根目录：`{W.as_posix()}/`")
    add("")
    inventory = []
    for path in sorted(R.iterdir()):
        if not path.is_file():
            continue
        entry = {"file":path.name,"size_kib":path.stat().st_size/1024,"type":path.suffix}
        if path.suffix == ".csv":
            entry["entries"] = len(rows(path.name))
        elif path.suffix == ".npz":
            entry["entries"] = len(np.load(path).files)
        else:
            entry["entries"] = "—"
        inventory.append(entry)
    add(table(inventory,["file","type","entries","size_kib"],["文件","类型","行/键数","KiB"]))
    add("")
    add("主要复现入口：")
    add("")
    add("- 数据与基线：`prep_cache.py`、`run_01.py`、`run_02.py`")
    add("- 出口试训与深度曲线：`run_03.py` 至 `run_07.py`")
    add("- 多种子与聚合：`run_08.py`、`analyze_08.py`")
    add("- 公平对照、中间深度与部署复核：`run_09.py` 至 `run_12.py`")
    add("- 独立校验：`check/task02_validate.py` 至 `check/task12_validate.py`（Task 01 以原报告和已有复核为准）")
    add("- 日志：`logs/run_01.log`（如有）、`logs/run_02.log` 至 `logs/run_12.log`")
    add("- checkpoint：`checkpoints/03_*.pt` 至 `checkpoints/12_*.pt`；探针只保存 H1 与允许更新的单个块，不保存完整骨干。")
    add("")

    add("## 10. 结论使用边界")
    add("")
    add("- 所有结论均受当前 WiFo 官方权重、前缀保留式截断、适配结构和训练协议约束。")
    add("- 零训练损伤、H1 恢复、有限解冻恢复是不同证据，不能合并成单一“早退有效性”数字。")
    add("- 完整模型有限解冻参考优先于仅完整 H1 参考；不同参考的结果不能直接混排。")
    add("- 训练种子波动、配置差异和验证样本 bootstrap 是不同不确定性来源。")
    add("- GPU 吞吐、GPU 时延和 CPU 时延分别报告；GFLOPs 下降不能直接写成实测时延下降。")
    add("- D17、D18 已参与前期诊断，只能作为复核配置。真正独立外部测试仍需新增未参与开发的数据。")
    add("- 论文应聚焦非对称深度截断与公平恢复证据，不宣称全面优于 Small、动态路由有效或所有模型都遵循统一宽度规律。")
    add("")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}: chars={len(chr(10).join(out))}, lines={len(out)}")


if __name__ == "__main__":
    main()
