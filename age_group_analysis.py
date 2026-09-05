# -*- coding: utf-8 -*-
"""
年龄分组优化分析脚本

功能：
1. 将患者按年龄分组（青中年 / 中年 / 老年 / 高龄）
2. 分析不同年龄组的疾病分布、BMI 风险、血压风险、血糖/胆固醇水平、吸烟结构
3. 补充统计检验（卡方检验、Kruskal-Wallis）
4. 结果输出到 results/优化补充/

输入数据：
D:/Programming/Data/Pycharm_data/数据探索/data/change.csv

输出目录：
D:/Programming/Data/Pycharm_data/数据探索/results/优化补充
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(r"D:\Programming\Data\Pycharm_data\数据探索")
DATA_PATH = PROJECT_DIR / "data" / "change.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "优化补充"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. 图表中文显示配置
# =========================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

# =========================================================
# 3. 安全读取 CSV
# =========================================================

def read_csv_safely(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("无法识别文件编码", b"", 0, 1, "请检查 CSV 文件编码")


df = read_csv_safely(DATA_PATH)

print("=" * 80)
print("数据加载完成")
print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")
print(f"字段列表：{df.columns.tolist()}")

# =========================================================
# 4. 数据预处理
# =========================================================

numeric_columns = [
    "age", "height_cm", "weight_kg",
    "bp_systolic", "bp_diastolic",
    "cholesterol_mg_dl", "blood_sugar_mg_dl"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# BMI
if "bmi" not in df.columns or df["bmi"].isna().all():
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
else:
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)

# =========================================================
# 5. 衍生字段构建
# =========================================================

# ---------- 5.1 年龄分组 ----------
def classify_age_group(age_value):
    if pd.isna(age_value):
        return np.nan
    if age_value < 40:
        return "青中年 (<40)"
    elif age_value < 60:
        return "中年 (40–59)"
    elif age_value < 75:
        return "老年 (60–74)"
    else:
        return "高龄 (≥75)"


df["age_group"] = df["age"].apply(classify_age_group)

age_group_order = ["青中年 (<40)", "中年 (40–59)", "老年 (60–74)", "高龄 (≥75)"]

print("\n年龄分组分布：")
print(df["age_group"].value_counts().reindex(age_group_order))

# ---------- 5.2 BMI 等级 ----------
def classify_bmi(bmi_value):
    if pd.isna(bmi_value):
        return np.nan
    if bmi_value < 18.5:
        return "偏瘦"
    elif bmi_value < 24:
        return "正常"
    elif bmi_value < 28:
        return "超重"
    else:
        return "肥胖"


df["bmi_level"] = df["bmi"].apply(classify_bmi)
bmi_order = ["偏瘦", "正常", "超重", "肥胖"]

# ---------- 5.3 血压风险评分 / 等级 ----------
def get_bp_risk_score(systolic, diastolic):
    if pd.isna(systolic) or pd.isna(diastolic):
        return np.nan
    if systolic >= 160 or diastolic >= 100:
        return 3
    elif systolic >= 140 or diastolic >= 90:
        return 2
    elif systolic >= 120 or diastolic >= 80:
        return 1
    else:
        return 0


def get_bp_risk_level(score):
    mapping = {0: "正常", 1: "偏高", 2: "高血压风险", 3: "高血压高风险"}
    return mapping.get(score, np.nan)


df["bp_risk_score"] = df.apply(
    lambda row: get_bp_risk_score(row["bp_systolic"], row["bp_diastolic"]),
    axis=1
)
df["bp_risk_level"] = df["bp_risk_score"].apply(get_bp_risk_level)
bp_level_order = ["正常", "偏高", "高血压风险", "高血压高风险"]

# ---------- 5.4 血糖等级 ----------
def classify_glucose(glucose_value):
    if pd.isna(glucose_value):
        return np.nan
    if glucose_value < 100:
        return "正常"
    elif glucose_value < 126:
        return "偏高"
    else:
        return "高血糖风险"


df["glucose_level"] = df["blood_sugar_mg_dl"].apply(classify_glucose)
glucose_order = ["正常", "偏高", "高血糖风险"]

# ---------- 5.5 胆固醇等级 ----------
def classify_cholesterol(cholesterol_value):
    if pd.isna(cholesterol_value):
        return np.nan
    if cholesterol_value < 200:
        return "理想"
    elif cholesterol_value < 240:
        return "边缘偏高"
    else:
        return "偏高"


df["cholesterol_level"] = df["cholesterol_mg_dl"].apply(classify_cholesterol)
cholesterol_order = ["理想", "边缘偏高", "偏高"]

# =========================================================
# 6. 通用绘图 / 输出函数
# =========================================================

def save_current_figure(file_name: str):
    save_path = RESULTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  已保存图表：{save_path}")


def draw_percent_stacked_bar(
    data: pd.DataFrame,
    group_col: str,
    stack_col: str,
    title: str,
    file_name: str,
    legend_title: str,
    group_order: list = None,
    category_order: list = None,
):
    """绘制百分比堆叠柱状图，同时保存交叉表 CSV。"""
    clean = data[[group_col, stack_col]].dropna()
    count_table = pd.crosstab(clean[group_col], clean[stack_col])

    if group_order is not None:
        count_table = count_table.reindex(
            [g for g in group_order if g in count_table.index]
        )
    if category_order is not None:
        count_table = count_table[
            [c for c in category_order if c in count_table.columns]
        ]

    # 保存交叉表
    csv_name = file_name.replace(".png", ".csv")
    count_table.to_csv(RESULTS_DIR / csv_name, encoding="utf-8-sig")
    print(f"  已保存交叉表：{RESULTS_DIR / csv_name}")

    # 百分比
    pct_table = count_table.div(count_table.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    x = np.arange(len(pct_table.index))
    bottom = np.zeros(len(pct_table.index))

    for col in pct_table.columns:
        values = pct_table[col].values
        plt.bar(x, values, bottom=bottom, label=str(col))
        bottom += values

    plt.title(title)
    plt.xlabel("年龄组")
    plt.ylabel("占比（%）")
    plt.xticks(x, pct_table.index.astype(str), rotation=15, ha="right")
    plt.ylim(0, 105)
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_current_figure(file_name)


def draw_grouped_boxplot(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    file_name: str,
    group_order: list = None,
):
    """绘制分组箱线图。"""
    clean = data[[group_col, value_col]].dropna()
    groups_to_plot = group_order if group_order else clean[group_col].unique()

    data_list = []
    labels = []
    for g in groups_to_plot:
        vals = clean.loc[clean[group_col] == g, value_col].dropna()
        if len(vals) > 0:
            data_list.append(vals)
            labels.append(str(g))

    if not data_list:
        print(f"  跳过 {file_name}：无有效数据")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list, patch_artist=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=15, ha="right")
    plt.title(title)
    plt.xlabel("年龄组")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    save_current_figure(file_name)


# =========================================================
# 7. 描述性分析
# =========================================================

print("\n" + "=" * 80)
print("描述性分析")
print("=" * 80)

# 7.1 不同年龄组的疾病分布
print("\n--- 7.1 年龄组 × 疾病分布 ---")
draw_percent_stacked_bar(
    data=df,
    group_col="age_group",
    stack_col="disease",
    title="不同年龄组的疾病类型分布",
    file_name="age_group_disease_distribution.png",
    legend_title="疾病类型",
    group_order=age_group_order,
)

# 7.2 不同年龄组的 BMI 风险
print("\n--- 7.2 年龄组 × BMI 风险 ---")
draw_percent_stacked_bar(
    data=df,
    group_col="age_group",
    stack_col="bmi_level",
    title="不同年龄组的 BMI 等级分布",
    file_name="age_group_bmi_distribution.png",
    legend_title="BMI 等级",
    group_order=age_group_order,
    category_order=bmi_order,
)

# 7.3 不同年龄组的血压风险
print("\n--- 7.3 年龄组 × 血压风险 ---")
draw_percent_stacked_bar(
    data=df,
    group_col="age_group",
    stack_col="bp_risk_level",
    title="不同年龄组的血压风险等级分布",
    file_name="age_group_bp_distribution.png",
    legend_title="血压风险等级",
    group_order=age_group_order,
    category_order=bp_level_order,
)

# 7.4 不同年龄组的血糖和胆固醇水平
print("\n--- 7.4 年龄组 × 血糖 / 胆固醇水平 ---")
draw_grouped_boxplot(
    data=df,
    group_col="age_group",
    value_col="blood_sugar_mg_dl",
    title="不同年龄组的血糖水平分布",
    ylabel="血糖（mg/dL）",
    file_name="age_group_glucose_boxplot.png",
    group_order=age_group_order,
)
draw_grouped_boxplot(
    data=df,
    group_col="age_group",
    value_col="cholesterol_mg_dl",
    title="不同年龄组的胆固醇水平分布",
    ylabel="胆固醇（mg/dL）",
    file_name="age_group_cholesterol_boxplot.png",
    group_order=age_group_order,
)

# 汇总统计表
glu_cho_stats = df.groupby("age_group").agg(
    血糖_均值=("blood_sugar_mg_dl", "mean"),
    血糖_中位数=("blood_sugar_mg_dl", "median"),
    血糖_标准差=("blood_sugar_mg_dl", "std"),
    胆固醇_均值=("cholesterol_mg_dl", "mean"),
    胆固醇_中位数=("cholesterol_mg_dl", "median"),
    胆固醇_标准差=("cholesterol_mg_dl", "std"),
    样本数=("blood_sugar_mg_dl", "count"),
).reindex(age_group_order)
glu_cho_stats.to_csv(RESULTS_DIR / "age_group_glucose_cholesterol_stats.csv", encoding="utf-8-sig")
print(f"  已保存血糖/胆固醇统计表：{RESULTS_DIR / 'age_group_glucose_cholesterol_stats.csv'}")

# 7.5 不同年龄组的吸烟结构
print("\n--- 7.5 年龄组 × 吸烟结构 ---")
draw_percent_stacked_bar(
    data=df,
    group_col="age_group",
    stack_col="smoking_history",
    title="不同年龄组的吸烟史结构",
    file_name="age_group_smoking_distribution.png",
    legend_title="吸烟史",
    group_order=age_group_order,
)

# =========================================================
# 8. 统计检验
# =========================================================

print("\n" + "=" * 80)
print("统计检验")
print("=" * 80)

test_results = []


def format_p_value(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (不显著)"


# ---------- 8.1 年龄组 × 疾病 卡方检验 ----------
print("\n--- 8.1 年龄组 vs 疾病类型 卡方检验 ---")
ct_age_disease = pd.crosstab(df["age_group"], df["disease"])
chi2, p_dof, dof, expected = stats.chi2_contingency(ct_age_disease)
print(f"  卡方统计量 = {chi2:.2f}, 自由度 = {dof}, p = {format_p_value(p_dof)}")
# 如果期望频数过小，使用 fisher_exact 的替代不现实（多维），提示 Cramér's V
n = ct_age_disease.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct_age_disease.shape) - 1))) if min(ct_age_disease.shape) > 1 else np.nan
print(f"  Cramér's V (效应量) = {cramers_v:.4f}")
test_results.append({
    "检验名称": "年龄组 × 疾病类型 卡方检验",
    "检验统计量": f"χ² = {chi2:.2f}",
    "自由度": dof,
    "p值": p_dof,
    "显著性": "***" if p_dof < 0.001 else ("**" if p_dof < 0.01 else ("*" if p_dof < 0.05 else "不显著")),
    "效应量": f"Cramér's V = {cramers_v:.4f}" if not np.isnan(cramers_v) else "N/A",
})

# ---------- 8.2 年龄组 × BMI 等级 卡方检验 ----------
print("\n--- 8.2 年龄组 vs BMI 等级 卡方检验 ---")
ct_age_bmi = pd.crosstab(df["age_group"], df["bmi_level"])
chi2, p_bmi, dof, expected = stats.chi2_contingency(ct_age_bmi)
print(f"  卡方统计量 = {chi2:.2f}, 自由度 = {dof}, p = {format_p_value(p_bmi)}")
cramers_v_bmi = np.sqrt(chi2 / (n * (min(ct_age_bmi.shape) - 1)))
print(f"  Cramér's V (效应量) = {cramers_v_bmi:.4f}")
test_results.append({
    "检验名称": "年龄组 × BMI 等级 卡方检验",
    "检验统计量": f"χ² = {chi2:.2f}",
    "自由度": dof,
    "p值": p_bmi,
    "显著性": "***" if p_bmi < 0.001 else ("**" if p_bmi < 0.01 else ("*" if p_bmi < 0.05 else "不显著")),
    "效应量": f"Cramér's V = {cramers_v_bmi:.4f}",
})

# ---------- 8.3 年龄组 × 血压风险等级 卡方检验 ----------
print("\n--- 8.3 年龄组 vs 血压风险等级 卡方检验 ---")
ct_age_bp = pd.crosstab(df["age_group"], df["bp_risk_level"])
chi2, p_bp, dof, expected = stats.chi2_contingency(ct_age_bp)
print(f"  卡方统计量 = {chi2:.2f}, 自由度 = {dof}, p = {format_p_value(p_bp)}")
cramers_v_bp = np.sqrt(chi2 / (n * (min(ct_age_bp.shape) - 1)))
print(f"  Cramér's V (效应量) = {cramers_v_bp:.4f}")
test_results.append({
    "检验名称": "年龄组 × 血压风险等级 卡方检验",
    "检验统计量": f"χ² = {chi2:.2f}",
    "自由度": dof,
    "p值": p_bp,
    "显著性": "***" if p_bp < 0.001 else ("**" if p_bp < 0.01 else ("*" if p_bp < 0.05 else "不显著")),
    "效应量": f"Cramér's V = {cramers_v_bp:.4f}",
})

# ---------- 8.4 年龄组 × 血糖水平 Kruskal-Wallis ----------
print("\n--- 8.4 年龄组 vs 血糖水平 Kruskal-Wallis 检验 ---")
glucose_groups = [
    df.loc[df["age_group"] == g, "blood_sugar_mg_dl"].dropna().values
    for g in age_group_order
]
glucose_groups = [g for g in glucose_groups if len(g) > 0]
if len(glucose_groups) >= 2:
    h_stat, p_glu = stats.kruskal(*glucose_groups)
    print(f"  H 统计量 = {h_stat:.2f}, p = {format_p_value(p_glu)}")
    test_results.append({
        "检验名称": "年龄组 × 血糖水平 Kruskal-Wallis 检验",
        "检验统计量": f"H = {h_stat:.2f}",
        "自由度": len(glucose_groups) - 1,
        "p值": p_glu,
        "显著性": "***" if p_glu < 0.001 else ("**" if p_glu < 0.01 else ("*" if p_glu < 0.05 else "不显著")),
        "效应量": "N/A",
    })
    # 事后两两比较（Dunn 检验简化版：逐对 Mann-Whitney U + Bonferroni 校正）
    n_comparisons = len(glucose_groups) * (len(glucose_groups) - 1) / 2
    for i in range(len(glucose_groups)):
        for j in range(i + 1, len(glucose_groups)):
            u_stat, p_pair = stats.mannwhitneyu(glucose_groups[i], glucose_groups[j], alternative="two-sided")
            p_corrected = min(p_pair * n_comparisons, 1.0)
            sig = "显著" if p_corrected < 0.05 else "不显著"
            print(f"    {age_group_order[i]} vs {age_group_order[j]}: U={u_stat:.1f}, p(校正)={p_corrected:.4f} ({sig})")

# ---------- 8.5 年龄组 × 胆固醇水平 Kruskal-Wallis ----------
print("\n--- 8.5 年龄组 vs 胆固醇水平 Kruskal-Wallis 检验 ---")
chol_groups = [
    df.loc[df["age_group"] == g, "cholesterol_mg_dl"].dropna().values
    for g in age_group_order
]
chol_groups = [g for g in chol_groups if len(g) > 0]
if len(chol_groups) >= 2:
    h_stat_cho, p_cho = stats.kruskal(*chol_groups)
    print(f"  H 统计量 = {h_stat_cho:.2f}, p = {format_p_value(p_cho)}")
    test_results.append({
        "检验名称": "年龄组 × 胆固醇水平 Kruskal-Wallis 检验",
        "检验统计量": f"H = {h_stat_cho:.2f}",
        "自由度": len(chol_groups) - 1,
        "p值": p_cho,
        "显著性": "***" if p_cho < 0.001 else ("**" if p_cho < 0.01 else ("*" if p_cho < 0.05 else "不显著")),
        "效应量": "N/A",
    })
    n_comparisons = len(chol_groups) * (len(chol_groups) - 1) / 2
    for i in range(len(chol_groups)):
        for j in range(i + 1, len(chol_groups)):
            u_stat, p_pair = stats.mannwhitneyu(chol_groups[i], chol_groups[j], alternative="two-sided")
            p_corrected = min(p_pair * n_comparisons, 1.0)
            sig = "显著" if p_corrected < 0.05 else "不显著"
            print(f"    {age_group_order[i]} vs {age_group_order[j]}: U={u_stat:.1f}, p(校正)={p_corrected:.4f} ({sig})")

# ---------- 8.6 年龄组 × 吸烟史 卡方检验 ----------
print("\n--- 8.6 年龄组 vs 吸烟史 卡方检验 ---")
ct_age_smoke = pd.crosstab(df["age_group"], df["smoking_history"])
chi2, p_sm, dof, expected = stats.chi2_contingency(ct_age_smoke)
print(f"  卡方统计量 = {chi2:.2f}, 自由度 = {dof}, p = {format_p_value(p_sm)}")
cramers_v_sm = np.sqrt(chi2 / (n * (min(ct_age_smoke.shape) - 1)))
print(f"  Cramér's V (效应量) = {cramers_v_sm:.4f}")
test_results.append({
    "检验名称": "年龄组 × 吸烟史 卡方检验",
    "检验统计量": f"χ² = {chi2:.2f}",
    "自由度": dof,
    "p值": p_sm,
    "显著性": "***" if p_sm < 0.001 else ("**" if p_sm < 0.01 else ("*" if p_sm < 0.05 else "不显著")),
    "效应量": f"Cramér's V = {cramers_v_sm:.4f}",
})

# 保存统计检验汇总表
tests_df = pd.DataFrame(test_results)
tests_df.to_csv(RESULTS_DIR / "statistical_tests_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n  已保存统计检验汇总：{RESULTS_DIR / 'statistical_tests_summary.csv'}")

# =========================================================
# 9. 描述性统计总表
# =========================================================

print("\n" + "=" * 80)
print("描述性统计总表")
print("=" * 80)

# 吸烟比例
smoke_pct = df.groupby("age_group")["smoking_history"].apply(
    lambda s: (s.isin(["Current", "Former"]).sum() / s.notna().sum() * 100)
)
# 女性比例
if "gender" in df.columns:
    female_pct = df.groupby("age_group")["gender"].apply(
        lambda s: (s == "Female").sum() / s.notna().sum() * 100
    )
else:
    female_pct = pd.Series(np.nan, index=age_group_order)

summary = df.groupby("age_group").agg(
    人数=("age", "count"),
    年龄均值=("age", "mean"),
    BMI均值=("bmi", "mean"),
    收缩压均值=("bp_systolic", "mean"),
    舒张压均值=("bp_diastolic", "mean"),
    血糖均值=("blood_sugar_mg_dl", "mean"),
    胆固醇均值=("cholesterol_mg_dl", "mean"),
).reindex(age_group_order)

summary["人数占比(%)"] = (summary["人数"] / len(df) * 100).round(1)
summary["吸烟比例(%)"] = smoke_pct.reindex(age_group_order).round(1)
summary["女性占比(%)"] = female_pct.reindex(age_group_order).round(1)

# 调整列顺序
summary = summary[[
    "人数", "人数占比(%)",
    "年龄均值", "BMI均值",
    "收缩压均值", "舒张压均值",
    "血糖均值", "胆固醇均值",
    "吸烟比例(%)", "女性占比(%)",
]]

print(summary.round(2))

summary.to_csv(RESULTS_DIR / "age_group_summary_stats.csv", encoding="utf-8-sig")
print(f"\n已保存描述性统计总表：{RESULTS_DIR / 'age_group_summary_stats.csv'}")

# =========================================================
# 10. 完成
# =========================================================

print("\n" + "=" * 80)
print("年龄分组优化分析完成")
print(f"所有结果已保存到：{RESULTS_DIR}")
print("=" * 80)
