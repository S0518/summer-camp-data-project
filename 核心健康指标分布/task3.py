# -*- coding: utf-8 -*-
"""
任务一：数据探索分析——核心健康指标分布类画像

功能：
基于清洗后的 change.csv 绘制核心健康指标分布图，包括：

1. 年龄分布图
2. 身高分布图
3. 体重分布图
4. BMI 分布图
5. 收缩压分布图
6. 舒张压分布图
7. 胆固醇分布图
8. 血糖分布图

图表类型：
直方图 + KDE 曲线

本版修改：
1. 直方图柱子添加黑色边框；
2. 相邻柱子之间保留轻微间隔；
3. 使每个柱子的边界更加清晰。

输入数据：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data\\change.csv

输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\核心健康指标分布
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
RESULTS_DIR = PROJECT_DIR / "results" / "核心健康指标分布"

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
    "blood_sugar_mg_dl"
]

missing_required_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_required_columns:
    raise ValueError(f"数据中缺少必要字段：{missing_required_columns}")


# =========================================================
# 5. 数值字段处理与 BMI 构建
# =========================================================

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 如果 change.csv 中没有 bmi 字段，则自动计算 BMI
if "bmi" not in df.columns:
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
else:
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

# 避免身高为 0 等情况产生无穷值
df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)


# =========================================================
# 6. 字段配置
# =========================================================

plot_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl"
]

column_name_map = {
    "age": "年龄",
    "height_cm": "身高",
    "weight_kg": "体重",
    "bmi": "BMI",
    "bp_systolic": "收缩压",
    "bp_diastolic": "舒张压",
    "cholesterol_mg_dl": "胆固醇",
    "blood_sugar_mg_dl": "血糖"
}

column_unit_map = {
    "age": "岁",
    "height_cm": "cm",
    "weight_kg": "kg",
    "bmi": "",
    "bp_systolic": "mmHg",
    "bp_diastolic": "mmHg",
    "cholesterol_mg_dl": "mg/dL",
    "blood_sugar_mg_dl": "mg/dL"
}

figure_file_name_map = {
    "age": "01_年龄分布图.png",
    "height_cm": "02_身高分布图.png",
    "weight_kg": "03_体重分布图.png",
    "bmi": "04_BMI分布图.png",
    "bp_systolic": "05_收缩压分布图.png",
    "bp_diastolic": "06_舒张压分布图.png",
    "cholesterol_mg_dl": "07_胆固醇分布图.png",
    "blood_sugar_mg_dl": "08_血糖分布图.png"
}

stat_file_name_map = {
    "age": "01_年龄描述性统计表.csv",
    "height_cm": "02_身高描述性统计表.csv",
    "weight_kg": "03_体重描述性统计表.csv",
    "bmi": "04_BMI描述性统计表.csv",
    "bp_systolic": "05_收缩压描述性统计表.csv",
    "bp_diastolic": "06_舒张压描述性统计表.csv",
    "cholesterol_mg_dl": "07_胆固醇描述性统计表.csv",
    "blood_sugar_mg_dl": "08_血糖描述性统计表.csv"
}


# =========================================================
# 7. KDE 曲线计算函数
# =========================================================

def calculate_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    使用 numpy 手动计算一维高斯核密度估计，避免额外安装 scipy 或 seaborn。

    参数：
    values: 一维数值数组
    grid: KDE 曲线的横坐标网格

    返回：
    density: 每个 grid 点对应的密度值
    """
    values = values[~np.isnan(values)]

    n = len(values)
    if n < 2:
        return np.zeros_like(grid)

    std = np.std(values, ddof=1)
    if std == 0:
        return np.zeros_like(grid)

    # Silverman 经验带宽
    bandwidth = 1.06 * std * (n ** (-1 / 5))

    if bandwidth <= 0:
        return np.zeros_like(grid)

    diff = (grid[:, None] - values[None, :]) / bandwidth
    kernel_values = np.exp(-0.5 * diff ** 2) / np.sqrt(2 * np.pi)
    density = kernel_values.mean(axis=1) / bandwidth

    return density


# =========================================================
# 8. 绘制直方图 + KDE 曲线
# =========================================================

def plot_hist_with_kde(series: pd.Series, col_name: str, output_path: Path):
    """
    绘制单个健康指标的直方图和 KDE 曲线。

    本函数已设置：
    1. edgecolor="black"：柱子边框为黑色；
    2. linewidth=0.8：边框线宽；
    3. rwidth=0.92：柱子宽度略小于组距，使柱间有轻微间隔。
    """
    clean_series = series.dropna()

    if clean_series.empty:
        print(f"字段 {col_name} 没有有效数据，跳过绘图。")
        return

    values = clean_series.to_numpy(dtype=float)

    x_min = values.min()
    x_max = values.max()

    if x_min == x_max:
        x_min = x_min - 1
        x_max = x_max + 1

    grid = np.linspace(x_min, x_max, 300)
    kde_density = calculate_kde(values, grid)

    chinese_name = column_name_map[col_name]
    unit = column_unit_map[col_name]

    if unit:
        x_label = f"{chinese_name}（{unit}）"
    else:
        x_label = chinese_name

    plt.figure(figsize=(8, 5))

    # 直方图：设置柱子边框与间隔，使相邻柱子边界清晰
    plt.hist(
        values,
        bins=30,
        density=True,
        alpha=0.65,
        label="直方图",
        edgecolor="black",
        linewidth=0.8,
        rwidth=0.92
    )

    # KDE 曲线
    if np.any(kde_density > 0):
        plt.plot(
            grid,
            kde_density,
            linewidth=2,
            label="KDE 曲线"
        )

    plt.title(f"{chinese_name}分布图")
    plt.xlabel(x_label)
    plt.ylabel("密度")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# 9. 描述性统计表生成函数
# =========================================================

def build_stat_table(series: pd.Series, col_name: str) -> pd.DataFrame:
    """
    为单个指标生成描述性统计表。
    """
    clean_series = series.dropna()

    if clean_series.empty:
        stat_dict = {
            "指标名称": column_name_map[col_name],
            "字段名": col_name,
            "样本数": 0,
            "缺失值数量": int(series.isna().sum()),
            "均值": np.nan,
            "标准差": np.nan,
            "最小值": np.nan,
            "25%分位数": np.nan,
            "中位数": np.nan,
            "75%分位数": np.nan,
            "最大值": np.nan
        }
    else:
        stat_dict = {
            "指标名称": column_name_map[col_name],
            "字段名": col_name,
            "样本数": int(clean_series.shape[0]),
            "缺失值数量": int(series.isna().sum()),
            "均值": clean_series.mean(),
            "标准差": clean_series.std(),
            "最小值": clean_series.min(),
            "25%分位数": clean_series.quantile(0.25),
            "中位数": clean_series.median(),
            "75%分位数": clean_series.quantile(0.75),
            "最大值": clean_series.max()
        }

    return pd.DataFrame([stat_dict])


# =========================================================
# 10. 逐个指标绘图并导出统计表
# =========================================================

summary_tables = []

for col in plot_columns:
    chinese_name = column_name_map[col]

    print("\n" + "=" * 80)
    print(f"正在处理：{chinese_name}")
    print("=" * 80)

    # 生成描述性统计表
    stat_df = build_stat_table(df[col], col)
    summary_tables.append(stat_df)

    stat_df.to_csv(
        RESULTS_DIR / stat_file_name_map[col],
        index=False,
        encoding="utf-8-sig"
    )

    print(stat_df)

    # 绘制分布图
    plot_hist_with_kde(
        series=df[col],
        col_name=col,
        output_path=RESULTS_DIR / figure_file_name_map[col]
    )


# =========================================================
# 11. 导出核心健康指标描述性统计汇总表
# =========================================================

summary_df = pd.concat(summary_tables, ignore_index=True)

summary_df.to_csv(
    RESULTS_DIR / "00_核心健康指标描述性统计汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("核心健康指标描述性统计汇总")
print("=" * 80)
print(summary_df)


# =========================================================
# 12. 导出核心健康指标分布汇总表
# =========================================================

overall_info_df = pd.DataFrame({
    "项目": [
        "总样本数",
        "指标数量",
        "年龄有效样本数",
        "身高有效样本数",
        "体重有效样本数",
        "BMI有效样本数",
        "收缩压有效样本数",
        "舒张压有效样本数",
        "胆固醇有效样本数",
        "血糖有效样本数"
    ],
    "结果": [
        len(df),
        len(plot_columns),
        int(df["age"].notna().sum()),
        int(df["height_cm"].notna().sum()),
        int(df["weight_kg"].notna().sum()),
        int(df["bmi"].notna().sum()),
        int(df["bp_systolic"].notna().sum()),
        int(df["bp_diastolic"].notna().sum()),
        int(df["cholesterol_mg_dl"].notna().sum()),
        int(df["blood_sugar_mg_dl"].notna().sum())
    ]
})

overall_info_df.to_csv(
    RESULTS_DIR / "00_核心健康指标分布汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("核心健康指标分布汇总")
print("=" * 80)
print(overall_info_df)


# =========================================================
# 13. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("核心健康指标分布类画像绘制完成")
print(f"输入数据文件：{DATA_PATH}")
print(f"所有结果已保存到：{RESULTS_DIR}")
print("=" * 80)
