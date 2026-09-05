# -*- coding: utf-8 -*-
"""
任务 13：疾病类型健康画像

绘制图表：
1. 疾病 × 年龄分布图
2. 疾病 × BMI 分布图
3. 疾病 × 血压风险分布图
4. 疾病 × 血糖分布图
5. 疾病 × 胆固醇分布图
6. 疾病 × 吸烟史结构图
7. 疾病 × 性别结构图
8. 疾病类型健康指标均值热力图

输入数据：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data\\chronic_patients.csv

输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\疾病类型健康画像
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(r"D:\Programming\Data\Pycharm_data\数据探索")

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "疾病类型健康画像"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE_NAME = "chronic_patients.csv"
DATA_PATH = DATA_DIR / DATA_FILE_NAME

if not DATA_PATH.exists():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"在数据目录中没有找到 CSV 文件：{DATA_DIR}")

    DATA_PATH = csv_files[0]
    print(f"未找到指定文件 {DATA_FILE_NAME}，已自动使用：{DATA_PATH.name}")


# =========================================================
# 2. 图表中文显示配置
# =========================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


# =========================================================
# 3. 读取数据
# =========================================================

def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    尝试使用常见编码读取 CSV 文件。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "无法识别文件编码",
        b"",
        0,
        1,
        "请检查 CSV 文件编码"
    )


df = read_csv_safely(DATA_PATH)

print("=" * 80)
print("数据读取成功")
print("=" * 80)
print(f"数据文件：{DATA_PATH}")
print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")
print("\n字段名称：")
print(df.columns.tolist())


# =========================================================
# 4. 检查必要字段
# =========================================================

required_columns = [
    "disease",
    "age",
    "height_cm",
    "weight_kg",
    "bp_systolic",
    "bp_diastolic",
    "blood_sugar_mg_dl",
    "cholesterol_mg_dl",
    "smoking_history"
]

missing_required_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_required_columns:
    raise ValueError(f"数据中缺少必要字段：{missing_required_columns}")


# =========================================================
# 5. 数值字段转换
# =========================================================

numeric_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "bp_systolic",
    "bp_diastolic",
    "blood_sugar_mg_dl",
    "cholesterol_mg_dl"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================================================
# 6. 构建 BMI 字段
# =========================================================

df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)


# =========================================================
# 7. 构建血压风险字段 bp_risk_score / bp_risk_level
# =========================================================

"""
血压风险评分规则（统一规则）：

0：正常         SBP < 120 且 DBP < 80
1：偏高         SBP 120–139 或 DBP 80–89
2：高血压风险    SBP 140–159 或 DBP 90–99
3：高血压高风险   SBP ≥ 160 或 DBP ≥ 100
"""

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
    mapping = {
        0: "正常",
        1: "偏高",
        2: "高血压风险",
        3: "高血压高风险"
    }
    return mapping.get(score, np.nan)


df["bp_risk_score"] = df.apply(
    lambda row: get_bp_risk_score(row["bp_systolic"], row["bp_diastolic"]),
    axis=1
)

df["bp_risk_level"] = df["bp_risk_score"].apply(get_bp_risk_level)


# =========================================================
# 8. 基础绘图数据准备
# =========================================================

plot_df = df.dropna(subset=["disease"]).copy()

# 疾病类型按出现频次排序，便于图表阅读
disease_order = plot_df["disease"].value_counts().index.tolist()

print("\n疾病类型分布：")
print(plot_df["disease"].value_counts())


# =========================================================
# 9. 通用绘图函数
# =========================================================

def save_current_figure(file_name: str):
    """
    保存当前图表。
    """
    save_path = RESULTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"已保存：{save_path}")


def draw_grouped_boxplot(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    file_name: str
):
    """
    绘制分组箱线图。
    兼容旧版本 Matplotlib，不在 boxplot 中使用 labels 参数。
    """
    clean_data = data[[group_col, value_col]].dropna()

    groups = []
    labels = []

    for disease in disease_order:
        values = clean_data.loc[
            clean_data[group_col] == disease,
            value_col
        ].dropna()

        if len(values) > 0:
            groups.append(values)
            labels.append(str(disease))

    if len(groups) == 0:
        print(f"跳过图表：{file_name}，无有效数据")
        return

    plt.figure(figsize=(12, 6))

    plt.boxplot(
        groups,
        showfliers=True
    )

    plt.title(title)
    plt.xlabel("疾病类型")
    plt.ylabel(ylabel)

    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=labels,
        rotation=45,
        ha="right"
    )

    save_current_figure(file_name)


def draw_percent_stacked_bar(
    data: pd.DataFrame,
    group_col: str,
    stack_col: str,
    title: str,
    file_name: str,
    legend_title: str,
    category_order=None
):
    """
    绘制百分比堆叠柱状图。
    """
    clean_data = data[[group_col, stack_col]].dropna()

    if clean_data.empty:
        print(f"跳过图表：{file_name}，无有效数据")
        return

    count_table = pd.crosstab(
        clean_data[group_col],
        clean_data[stack_col]
    )

    count_table = count_table.reindex(
        [disease for disease in disease_order if disease in count_table.index]
    )

    if category_order is not None:
        existing_categories = [
            c for c in category_order if c in count_table.columns
        ]
        remaining_categories = [
            c for c in count_table.columns if c not in existing_categories
        ]
        count_table = count_table[
            existing_categories + remaining_categories
        ]

    percent_table = count_table.div(
        count_table.sum(axis=1),
        axis=0
    ) * 100

    plt.figure(figsize=(12, 6))

    x = np.arange(len(percent_table.index))
    bottom = np.zeros(len(percent_table.index))

    for col in percent_table.columns:
        values = percent_table[col].values

        plt.bar(
            x,
            values,
            bottom=bottom,
            label=str(col)
        )

        bottom += values

    plt.title(title)
    plt.xlabel("疾病类型")
    plt.ylabel("占比（%）")

    plt.xticks(
        x,
        percent_table.index.astype(str),
        rotation=45,
        ha="right"
    )

    plt.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    save_current_figure(file_name)


def draw_mean_heatmap(
    data: pd.DataFrame,
    group_col: str,
    metric_cols: list,
    metric_labels: list,
    title: str,
    file_name: str
):
    """
    绘制疾病类型健康指标均值热力图。
    """
    clean_data = data[[group_col] + metric_cols].copy()

    for col in metric_cols:
        clean_data[col] = pd.to_numeric(
            clean_data[col],
            errors="coerce"
        )

    mean_df = clean_data.groupby(group_col)[metric_cols].mean()

    mean_df = mean_df.reindex(
        [disease for disease in disease_order if disease in mean_df.index]
    )

    # 保存均值统计表，字段名使用中文
    mean_df_chinese = mean_df.copy()
    mean_df_chinese.columns = metric_labels

    mean_table_path = RESULTS_DIR / "08_疾病类型健康指标均值统计表.csv"
    mean_df_chinese.to_csv(
        mean_table_path,
        encoding="utf-8-sig"
    )
    print(f"已保存：{mean_table_path}")

    # 标准化后用于热力图颜色展示，避免不同量纲影响颜色对比
    standardized_df = (
        mean_df - mean_df.mean()
    ) / mean_df.std(ddof=0)

    standardized_df = standardized_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    plt.figure(figsize=(11, max(5, len(standardized_df) * 0.45)))

    image = plt.imshow(
        standardized_df.values,
        aspect="auto"
    )

    plt.colorbar(image, label="标准化均值")

    plt.title(title)
    plt.xlabel("健康指标")
    plt.ylabel("疾病类型")

    plt.xticks(
        np.arange(len(metric_labels)),
        metric_labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        np.arange(len(standardized_df.index)),
        standardized_df.index.astype(str)
    )

    # 在热力图中标注原始均值
    for i in range(mean_df.shape[0]):
        for j in range(mean_df.shape[1]):
            value = mean_df.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

    save_current_figure(file_name)


# =========================================================
# 10. 绘制图 1：疾病 × 年龄分布图
# =========================================================

draw_grouped_boxplot(
    data=plot_df,
    group_col="disease",
    value_col="age",
    title="疾病类型与年龄分布画像",
    ylabel="年龄",
    file_name="01_疾病类型与年龄分布画像.png"
)


# =========================================================
# 11. 绘制图 2：疾病 × BMI 分布图
# =========================================================

draw_grouped_boxplot(
    data=plot_df,
    group_col="disease",
    value_col="bmi",
    title="疾病类型与BMI分布画像",
    ylabel="BMI",
    file_name="02_疾病类型与BMI分布画像.png"
)


# =========================================================
# 12. 绘制图 3：疾病 × 血压风险分布图
# =========================================================

bp_risk_order = [
    "正常",
    "偏高",
    "高血压风险",
    "高血压高风险"
]

draw_percent_stacked_bar(
    data=plot_df,
    group_col="disease",
    stack_col="bp_risk_level",
    title="疾病类型与血压风险分布画像",
    file_name="03_疾病类型与血压风险分布画像.png",
    legend_title="血压风险等级",
    category_order=bp_risk_order
)


# =========================================================
# 13. 绘制图 4：疾病 × 血糖分布图
# =========================================================

draw_grouped_boxplot(
    data=plot_df,
    group_col="disease",
    value_col="blood_sugar_mg_dl",
    title="疾病类型与血糖分布画像",
    ylabel="血糖（mg/dL）",
    file_name="04_疾病类型与血糖分布画像.png"
)


# =========================================================
# 14. 绘制图 5：疾病 × 胆固醇分布图
# =========================================================

draw_grouped_boxplot(
    data=plot_df,
    group_col="disease",
    value_col="cholesterol_mg_dl",
    title="疾病类型与胆固醇分布画像",
    ylabel="胆固醇（mg/dL）",
    file_name="05_疾病类型与胆固醇分布画像.png"
)


# =========================================================
# 15. 绘制图 6：疾病 × 吸烟史结构图
# =========================================================

draw_percent_stacked_bar(
    data=plot_df,
    group_col="disease",
    stack_col="smoking_history",
    title="疾病类型与吸烟史结构画像",
    file_name="06_疾病类型与吸烟史结构画像.png",
    legend_title="吸烟史"
)


# =========================================================
# 16. 绘制图 7：疾病 × 性别结构图
# =========================================================

if "gender" in plot_df.columns:
    draw_percent_stacked_bar(
        data=plot_df,
        group_col="disease",
        stack_col="gender",
        title="疾病类型与性别结构画像",
        file_name="07_疾病类型与性别结构画像.png",
        legend_title="性别"
    )
else:
    print("未发现 gender 字段，跳过疾病类型与性别结构画像。")


# =========================================================
# 17. 绘制图 8：疾病类型健康指标均值热力图
# =========================================================

core_health_metrics = [
    "age",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
    "blood_sugar_mg_dl",
    "cholesterol_mg_dl"
]

core_health_metric_labels = [
    "年龄",
    "BMI",
    "收缩压",
    "舒张压",
    "血糖",
    "胆固醇"
]

draw_mean_heatmap(
    data=plot_df,
    group_col="disease",
    metric_cols=core_health_metrics,
    metric_labels=core_health_metric_labels,
    title="疾病类型健康指标均值热力画像",
    file_name="08_疾病类型健康指标均值热力画像.png"
)


# =========================================================
# 18. 保存处理后的数据
# =========================================================

processed_data_path = RESULTS_DIR / "疾病类型健康画像处理后数据.csv"

df.to_csv(
    processed_data_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"已保存：{processed_data_path}")


# =========================================================
# 19. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("疾病类型健康画像绘制完成")
print("=" * 80)
print(f"输入数据文件：{DATA_PATH}")
print(f"结果输出目录：{RESULTS_DIR}")
print("已生成结果文件：")
print("1. 01_疾病类型与年龄分布画像.png")
print("2. 02_疾病类型与BMI分布画像.png")
print("3. 03_疾病类型与血压风险分布画像.png")
print("4. 04_疾病类型与血糖分布画像.png")
print("5. 05_疾病类型与胆固醇分布画像.png")
print("6. 06_疾病类型与吸烟史结构画像.png")
print("7. 07_疾病类型与性别结构画像.png")
print("8. 08_疾病类型健康指标均值统计表.csv")
print("9. 08_疾病类型健康指标均值热力画像.png")
print("10. 疾病类型健康画像处理后数据.csv")
print("=" * 80)
