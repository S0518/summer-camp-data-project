# -*- coding: utf-8 -*-
"""
任务一：数据探索分析——指标关系分析

绘制图表：
1. 数值变量相关性热力图
2. 胆固醇与血压风险关系图（箱线图）
3. 血糖与血压风险关系图（箱线图）
4. 年龄与血压风险关系图（散点图）

输入数据：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data\\change.csv

输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\指标关系分析
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
RESULTS_DIR = PROJECT_DIR / "results" / "指标关系分析"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "change.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"未找到清洗后的数据文件：{DATA_PATH}")


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
print("清洗后数据读取成功")
print("=" * 80)
print(f"数据文件：{DATA_PATH}")
print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")
print("\n字段名称：")
print(df.columns.tolist())
print("\n前 5 行数据：")
print(df.head())


# =========================================================
# 4. 检查必要字段
# =========================================================

required_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "bp_systolic",
    "bp_diastolic",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl",
    "smoking_history"
]

missing_required_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_required_columns:
    raise ValueError(f"数据中缺少必要字段：{missing_required_columns}")


# =========================================================
# 5. 数据预处理与衍生字段构建
# =========================================================

numeric_base_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "bp_systolic",
    "bp_diastolic",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl"
]

for col in numeric_base_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["smoking_history"] = df["smoking_history"].astype(str).str.strip()


# ---------- 5.1 构建 BMI ----------
if "bmi" not in df.columns:
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
else:
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)


# ---------- 5.2 构建血压风险评分 ----------
def get_bp_risk_score(systolic, diastolic):
    """
    血压风险评分：
    0：正常，收缩压 < 120 且 舒张压 < 80
    1：偏高，120 <= 收缩压 < 140 或 80 <= 舒张压 < 90
    2：高血压风险，140 <= 收缩压 < 160 或 90 <= 舒张压 < 100
    3：高血压高风险，收缩压 >= 160 或 舒张压 >= 100
    """
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


if "bp_risk_score" not in df.columns:
    df["bp_risk_score"] = df.apply(
        lambda row: get_bp_risk_score(
            row["bp_systolic"],
            row["bp_diastolic"]
        ),
        axis=1
    )
else:
    df["bp_risk_score"] = pd.to_numeric(df["bp_risk_score"], errors="coerce")


# ---------- 5.3 构建吸烟风险评分 ----------
def get_smoking_score(smoking_history):
    """
    Never=0, Former=1, Current=2
    """
    value = str(smoking_history).strip().lower()

    if value == "never":
        return 0
    elif value == "former":
        return 1
    elif value == "current":
        return 2
    else:
        return np.nan


if "smoking_score" not in df.columns:
    df["smoking_score"] = df["smoking_history"].apply(get_smoking_score)
else:
    df["smoking_score"] = pd.to_numeric(df["smoking_score"], errors="coerce")


# =========================================================
# 6. 字段配置
# =========================================================

corr_columns = [
    "age",
    "bmi",
    "bp_risk_score",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl",
    "smoking_score"
]

chinese_name_map = {
    "age": "年龄",
    "bmi": "BMI",
    "bp_risk_score": "血压风险评分",
    "cholesterol_mg_dl": "胆固醇",
    "blood_sugar_mg_dl": "血糖",
    "smoking_score": "吸烟风险评分"
}

bp_score_label_map = {
    0: "0-正常",
    1: "1-偏高",
    2: "2-高血压风险",
    3: "3-高血压高风险"
}

bp_score_order = [0, 1, 2, 3]
bp_score_labels = [bp_score_label_map[x] for x in bp_score_order]


# =========================================================
# 7. 绘制相关性热力图
# =========================================================

def plot_corr_heatmap(corr_df: pd.DataFrame, output_path: Path):
    """
    绘制数值变量相关性热力图。
    """
    labels = [chinese_name_map[col] for col in corr_df.columns]

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        corr_df.values,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10
            )

    ax.set_title("数值变量相关性热力图")
    fig.colorbar(im, ax=ax, shrink=0.9, label="相关系数")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# 8. 绘制按血压风险评分分组的箱线图
# =========================================================

def plot_bp_group_boxplot(
    data: pd.DataFrame,
    value_col: str,
    value_name: str,
    y_label: str,
    title: str,
    output_path: Path
):
    """
    绘制不同血压风险评分下某一连续变量的箱线图。

    说明：
    不使用 plt.boxplot(labels=...)，
    而是使用 plt.xticks(...) 设置标签，
    避免 matplotlib 版本兼容问题。
    """
    plot_df = data[["bp_risk_score", value_col]].dropna().copy()

    data_list = []
    labels = []

    for score in bp_score_order:
        group_values = plot_df.loc[
            plot_df["bp_risk_score"] == score,
            value_col
        ].dropna()

        if len(group_values) > 0:
            data_list.append(group_values)
            labels.append(bp_score_label_map[score])

    if len(data_list) == 0:
        print(f"{title} 没有有效数据，跳过绘图。")
        return

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        data_list,
        patch_artist=True,
        boxprops={
            "linewidth": 1.2
        },
        medianprops={
            "linewidth": 1.5
        },
        whiskerprops={
            "linewidth": 1.2
        },
        capprops={
            "linewidth": 1.2
        },
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "none",
            "markeredgewidth": 0.8
        }
    )

    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=labels,
        rotation=15,
        ha="right"
    )

    plt.title(title)
    plt.xlabel("血压风险评分")
    plt.ylabel(y_label)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# 9. 绘制年龄与血压风险散点图
# =========================================================

def plot_age_bp_scatter(data: pd.DataFrame, output_path: Path):
    """
    绘制年龄与血压风险评分散点图。
    """
    plot_df = data[["age", "bp_risk_score"]].dropna().copy()

    plt.figure(figsize=(7, 5))

    plt.scatter(
        plot_df["age"],
        plot_df["bp_risk_score"],
        s=18,
        alpha=0.45,
        edgecolors="black",
        linewidths=0.3,
        label="样本点"
    )

    # 添加线性趋势线
    if len(plot_df) >= 2 and plot_df["age"].nunique() > 1:
        x = plot_df["age"].to_numpy(dtype=float)
        y = plot_df["bp_risk_score"].to_numpy(dtype=float)

        coef = np.polyfit(x, y, deg=1)
        poly = np.poly1d(coef)

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = poly(x_line)

        plt.plot(
            x_line,
            y_line,
            linewidth=2,
            label="线性趋势线"
        )

    plt.title("年龄与血压风险关系图")
    plt.xlabel("年龄（岁）")
    plt.ylabel("血压风险评分")
    plt.yticks([0, 1, 2, 3], ["0", "1", "2", "3"])
    plt.grid(linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# 10. 构建统计表函数
# =========================================================

def build_pair_stat(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_name: str,
    y_name: str
) -> pd.DataFrame:
    """
    生成两个变量的基础统计表和皮尔逊相关系数。
    """
    plot_df = data[[x_col, y_col]].dropna().copy()

    if len(plot_df) > 1:
        corr_value = plot_df[x_col].corr(plot_df[y_col])
    else:
        corr_value = np.nan

    stat_df = pd.DataFrame({
        "项目": [
            "配对样本数",
            f"{x_name}均值",
            f"{y_name}均值",
            f"{x_name}与{y_name}皮尔逊相关系数"
        ],
        "结果": [
            len(plot_df),
            plot_df[x_col].mean(),
            plot_df[y_col].mean(),
            corr_value
        ]
    })

    return stat_df


def build_bp_score_group_table(
    data: pd.DataFrame,
    value_col: str,
    value_name: str
) -> pd.DataFrame:
    """
    按血压风险评分分组，统计某个连续变量的分布特征。
    """
    group_df = (
        data[["bp_risk_score", value_col]]
        .dropna()
        .groupby("bp_risk_score")[value_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    group_df["血压风险评分标签"] = group_df["bp_risk_score"].map(bp_score_label_map)
    group_df.insert(0, "分析指标", value_name)

    return group_df


# =========================================================
# 11. 数值变量相关性热力图
# =========================================================

corr_df = df[corr_columns].corr(method="pearson")

corr_df.to_csv(
    RESULTS_DIR / "01_数值变量相关性矩阵.csv",
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("数值变量相关性矩阵")
print("=" * 80)
print(corr_df)

plot_corr_heatmap(
    corr_df=corr_df,
    output_path=RESULTS_DIR / "01_数值变量相关性热力图.png"
)


# =========================================================
# 12. 胆固醇与血压风险关系图：箱线图
# =========================================================

chol_bp_stat = build_pair_stat(
    data=df,
    x_col="cholesterol_mg_dl",
    y_col="bp_risk_score",
    x_name="胆固醇",
    y_name="血压风险评分"
)

chol_bp_stat.to_csv(
    RESULTS_DIR / "02_胆固醇与血压风险统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

chol_bp_group_stat = build_bp_score_group_table(
    data=df,
    value_col="cholesterol_mg_dl",
    value_name="胆固醇"
)

chol_bp_group_stat.to_csv(
    RESULTS_DIR / "02_胆固醇按血压风险分组统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("胆固醇与血压风险统计")
print("=" * 80)
print(chol_bp_stat)

print("\n胆固醇按血压风险分组统计：")
print(chol_bp_group_stat)

plot_bp_group_boxplot(
    data=df,
    value_col="cholesterol_mg_dl",
    value_name="胆固醇",
    y_label="胆固醇（mg/dL）",
    title="胆固醇与血压风险关系图",
    output_path=RESULTS_DIR / "02_胆固醇与血压风险关系图.png"
)


# =========================================================
# 13. 血糖与血压风险关系图：箱线图
# =========================================================

glucose_bp_stat = build_pair_stat(
    data=df,
    x_col="blood_sugar_mg_dl",
    y_col="bp_risk_score",
    x_name="血糖",
    y_name="血压风险评分"
)

glucose_bp_stat.to_csv(
    RESULTS_DIR / "03_血糖与血压风险统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

glucose_bp_group_stat = build_bp_score_group_table(
    data=df,
    value_col="blood_sugar_mg_dl",
    value_name="血糖"
)

glucose_bp_group_stat.to_csv(
    RESULTS_DIR / "03_血糖按血压风险分组统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("血糖与血压风险统计")
print("=" * 80)
print(glucose_bp_stat)

print("\n血糖按血压风险分组统计：")
print(glucose_bp_group_stat)

plot_bp_group_boxplot(
    data=df,
    value_col="blood_sugar_mg_dl",
    value_name="血糖",
    y_label="血糖（mg/dL）",
    title="血糖与血压风险关系图",
    output_path=RESULTS_DIR / "03_血糖与血压风险关系图.png"
)


# =========================================================
# 14. 年龄与血压风险关系图：散点图
# =========================================================

age_bp_stat = build_pair_stat(
    data=df,
    x_col="age",
    y_col="bp_risk_score",
    x_name="年龄",
    y_name="血压风险评分"
)

age_bp_stat.to_csv(
    RESULTS_DIR / "04_年龄与血压风险统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

age_bp_group_stat = build_bp_score_group_table(
    data=df,
    value_col="age",
    value_name="年龄"
)

age_bp_group_stat.to_csv(
    RESULTS_DIR / "04_年龄按血压风险分组统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("年龄与血压风险统计")
print("=" * 80)
print(age_bp_stat)

print("\n年龄按血压风险分组统计：")
print(age_bp_group_stat)

plot_age_bp_scatter(
    data=df,
    output_path=RESULTS_DIR / "04_年龄与血压风险关系图.png"
)


# =========================================================
# 15. 汇总表
# =========================================================

summary_df = pd.DataFrame({
    "项目": [
        "总样本数",
        "相关性热力图使用字段数",
        "胆固醇与血压风险配对样本数",
        "血糖与血压风险配对样本数",
        "年龄与血压风险配对样本数"
    ],
    "结果": [
        len(df),
        len(corr_columns),
        len(df[["cholesterol_mg_dl", "bp_risk_score"]].dropna()),
        len(df[["blood_sugar_mg_dl", "bp_risk_score"]].dropna()),
        len(df[["age", "bp_risk_score"]].dropna())
    ]
})

summary_df.to_csv(
    RESULTS_DIR / "00_指标关系分析汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("指标关系分析汇总")
print("=" * 80)
print(summary_df)


# =========================================================
# 16. 保存带衍生字段的数据
# =========================================================

df.to_csv(
    RESULTS_DIR / "00_添加衍生字段后的数据.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 17. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("指标关系分析图像绘制完成")
print(f"输入数据文件：{DATA_PATH}")
print(f"所有结果已保存到：{RESULTS_DIR}")
print("=" * 80)
