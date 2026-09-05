# -*- coding: utf-8 -*-
"""
任务一：数据探索分析——异常值与医学逻辑检查类画像

本脚本基于清洗后的 change.csv 绘制以下图表：
1. 年龄箱线图
2. BMI 箱线图
3. 收缩压箱线图
4. 舒张压箱线图
5. 胆固醇箱线图
6. 血糖箱线图
7. 收缩压与舒张压散点图

输入数据：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data\\change.csv

输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\异常值与医学逻辑检查
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
RESULTS_DIR = PROJECT_DIR / "results" / "异常值与医学逻辑检查"

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

# 如果数据中没有 bmi 字段，则自动构建
if "bmi" not in df.columns:
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
else:
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)


# =========================================================
# 6. 字段配置
# =========================================================

boxplot_columns = [
    "age",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl"
]

column_name_map = {
    "age": "年龄",
    "bmi": "BMI",
    "bp_systolic": "收缩压",
    "bp_diastolic": "舒张压",
    "cholesterol_mg_dl": "胆固醇",
    "blood_sugar_mg_dl": "血糖"
}

column_unit_map = {
    "age": "岁",
    "bmi": "",
    "bp_systolic": "mmHg",
    "bp_diastolic": "mmHg",
    "cholesterol_mg_dl": "mg/dL",
    "blood_sugar_mg_dl": "mg/dL"
}

figure_file_name_map = {
    "age": "01_年龄箱线图.png",
    "bmi": "02_BMI箱线图.png",
    "bp_systolic": "03_收缩压箱线图.png",
    "bp_diastolic": "04_舒张压箱线图.png",
    "cholesterol_mg_dl": "05_胆固醇箱线图.png",
    "blood_sugar_mg_dl": "06_血糖箱线图.png"
}

stat_file_name_map = {
    "age": "01_年龄异常值统计表.csv",
    "bmi": "02_BMI异常值统计表.csv",
    "bp_systolic": "03_收缩压异常值统计表.csv",
    "bp_diastolic": "04_舒张压异常值统计表.csv",
    "cholesterol_mg_dl": "05_胆固醇异常值统计表.csv",
    "blood_sugar_mg_dl": "06_血糖异常值统计表.csv"
}

outlier_record_file_name_map = {
    "age": "01_年龄IQR异常记录.csv",
    "bmi": "02_BMI_IQR异常记录.csv",
    "bp_systolic": "03_收缩压IQR异常记录.csv",
    "bp_diastolic": "04_舒张压IQR异常记录.csv",
    "cholesterol_mg_dl": "05_胆固醇IQR异常记录.csv",
    "blood_sugar_mg_dl": "06_血糖IQR异常记录.csv"
}


# =========================================================
# 7. IQR 异常值统计函数
# =========================================================

def calculate_iqr_outlier_stat(series: pd.Series, col_name: str) -> dict:
    """
    使用 IQR 四分位距法计算异常值统计结果。
    """
    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "指标名称": column_name_map[col_name],
            "字段名": col_name,
            "样本数": 0,
            "缺失值数量": int(series.isna().sum()),
            "Q1": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "异常下界": np.nan,
            "异常上界": np.nan,
            "异常值数量": 0,
            "异常值占比(%)": 0
        }

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_rate = outlier_count / len(series) * 100

    return {
        "指标名称": column_name_map[col_name],
        "字段名": col_name,
        "样本数": int(clean_series.shape[0]),
        "缺失值数量": int(series.isna().sum()),
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "异常下界": lower_bound,
        "异常上界": upper_bound,
        "异常值数量": outlier_count,
        "异常值占比(%)": outlier_rate
    }


def get_iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """
    返回某个字段的 IQR 异常值布尔标记。
    """
    clean_series = series.dropna()

    if clean_series.empty:
        return pd.Series(False, index=series.index)

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (series < lower_bound) | (series > upper_bound)


# =========================================================
# 8. 箱线图绘制函数
# =========================================================

def plot_boxplot(series: pd.Series, col_name: str, output_path: Path):
    """
    绘制单个指标箱线图。

    说明：
    当前版本未在 plt.boxplot() 中使用 labels 参数，
    而是使用 plt.xticks([1], [chinese_name]) 设置横坐标标签，
    这样可以兼容更多 matplotlib 版本。
    """
    clean_series = series.dropna()

    if clean_series.empty:
        print(f"字段 {col_name} 没有有效数据，跳过绘图。")
        return

    chinese_name = column_name_map[col_name]
    unit = column_unit_map[col_name]

    if unit:
        y_label = f"{chinese_name}（{unit}）"
    else:
        y_label = chinese_name

    plt.figure(figsize=(6, 5))

    plt.boxplot(
        clean_series,
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

    # 替代 labels=[chinese_name]，避免 matplotlib 版本兼容问题
    plt.xticks([1], [chinese_name])

    plt.title(f"{chinese_name}箱线图")
    plt.ylabel(y_label)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# 9. 绘制 6 个箱线图并导出异常值统计
# =========================================================

outlier_summary_list = []

for col in boxplot_columns:
    chinese_name = column_name_map[col]

    print("\n" + "=" * 80)
    print(f"正在处理：{chinese_name}箱线图")
    print("=" * 80)

    stat = calculate_iqr_outlier_stat(df[col], col)
    stat_df = pd.DataFrame([stat])
    outlier_summary_list.append(stat_df)

    stat_df.to_csv(
        RESULTS_DIR / stat_file_name_map[col],
        index=False,
        encoding="utf-8-sig"
    )

    print(stat_df)

    # 导出当前字段的 IQR 异常记录
    current_outlier_mask = get_iqr_outlier_mask(df[col])
    current_outlier_records = df[current_outlier_mask].copy()

    current_outlier_records.to_csv(
        RESULTS_DIR / outlier_record_file_name_map[col],
        index=False,
        encoding="utf-8-sig"
    )

    # 绘制箱线图
    plot_boxplot(
        series=df[col],
        col_name=col,
        output_path=RESULTS_DIR / figure_file_name_map[col]
    )


# =========================================================
# 10. 汇总异常值统计表
# =========================================================

outlier_summary_df = pd.concat(outlier_summary_list, ignore_index=True)

outlier_summary_df.to_csv(
    RESULTS_DIR / "00_异常值统计汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("异常值统计汇总")
print("=" * 80)
print(outlier_summary_df)


# =========================================================
# 11. 收缩压与舒张压散点图
# =========================================================

bp_df = df[["bp_systolic", "bp_diastolic"]].dropna().copy()

bp_logic_abnormal_mask = bp_df["bp_systolic"] <= bp_df["bp_diastolic"]

normal_bp_df = bp_df[~bp_logic_abnormal_mask]
abnormal_bp_df = bp_df[bp_logic_abnormal_mask]

if len(bp_df) > 0:
    bp_logic_df = pd.DataFrame({
        "血压逻辑状态": [
            "正常记录：收缩压大于舒张压",
            "异常记录：收缩压小于等于舒张压"
        ],
        "记录数量": [
            int(len(normal_bp_df)),
            int(len(abnormal_bp_df))
        ]
    })

    bp_logic_df["占比(%)"] = bp_logic_df["记录数量"] / len(bp_df) * 100
else:
    bp_logic_df = pd.DataFrame({
        "血压逻辑状态": [
            "正常记录：收缩压大于舒张压",
            "异常记录：收缩压小于等于舒张压"
        ],
        "记录数量": [0, 0],
        "占比(%)": [0, 0]
    })

bp_logic_df.to_csv(
    RESULTS_DIR / "07_血压逻辑异常统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

abnormal_bp_df.to_csv(
    RESULTS_DIR / "07_血压逻辑异常记录.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("血压逻辑异常统计")
print("=" * 80)
print(bp_logic_df)


plt.figure(figsize=(7, 6))

plt.scatter(
    normal_bp_df["bp_diastolic"],
    normal_bp_df["bp_systolic"],
    s=12,
    alpha=0.45,
    label="正常记录"
)

plt.scatter(
    abnormal_bp_df["bp_diastolic"],
    abnormal_bp_df["bp_systolic"],
    s=18,
    alpha=0.75,
    label="逻辑异常记录"
)

if len(bp_df) > 0:
    min_bp = min(
        bp_df["bp_diastolic"].min(),
        bp_df["bp_systolic"].min()
    )

    max_bp = max(
        bp_df["bp_diastolic"].max(),
        bp_df["bp_systolic"].max()
    )

    plt.plot(
        [min_bp, max_bp],
        [min_bp, max_bp],
        linestyle="--",
        linewidth=1.2,
        label="收缩压 = 舒张压"
    )

plt.title("收缩压与舒张压散点图")
plt.xlabel("舒张压（mmHg）")
plt.ylabel("收缩压（mmHg）")
plt.legend()
plt.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "07_收缩压与舒张压散点图.png")
plt.close()


# =========================================================
# 12. 汇总表
# =========================================================

summary_df = pd.DataFrame({
    "项目": [
        "总样本数",
        "年龄有效样本数",
        "BMI有效样本数",
        "收缩压有效样本数",
        "舒张压有效样本数",
        "胆固醇有效样本数",
        "血糖有效样本数",
        "血压逻辑检查有效样本数",
        "血压逻辑异常记录数"
    ],
    "结果": [
        len(df),
        int(df["age"].notna().sum()),
        int(df["bmi"].notna().sum()),
        int(df["bp_systolic"].notna().sum()),
        int(df["bp_diastolic"].notna().sum()),
        int(df["cholesterol_mg_dl"].notna().sum()),
        int(df["blood_sugar_mg_dl"].notna().sum()),
        int(len(bp_df)),
        int(len(abnormal_bp_df))
    ]
})

summary_df.to_csv(
    RESULTS_DIR / "00_异常值与医学逻辑检查汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("异常值与医学逻辑检查汇总")
print("=" * 80)
print(summary_df)


# =========================================================
# 13. 保存带 BMI 的数据
# =========================================================

df.to_csv(
    RESULTS_DIR / "00_添加BMI后的检查数据.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 14. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("异常值与医学逻辑检查类画像绘制完成")
print(f"输入数据文件：{DATA_PATH}")
print(f"所有结果已保存到：{RESULTS_DIR}")
print("=" * 80)
