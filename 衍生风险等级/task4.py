# -*- coding: utf-8 -*-
"""
任务一：数据探索分析——衍生风险等级类画像

功能：
基于清洗后的 change.csv 绘制以下图表：
1. BMI 分级人数图
2. 血压风险评分分布图
3. 吸烟风险评分分布图
4. 血糖等级分布图
5. 胆固醇等级分布图

输入数据：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data\\change.csv

输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\衍生风险等级
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
RESULTS_DIR = PROJECT_DIR / "results" / "衍生风险等级"

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
    尝试使用常见编码读取 CSV 文件
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
    "height_cm",
    "weight_kg",
    "bp_systolic",
    "bp_diastolic",
    "smoking_history",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl"
]

missing_required_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_required_columns:
    raise ValueError(f"数据中缺少必要字段：{missing_required_columns}")


# =========================================================
# 5. 基础数据预处理
# =========================================================

numeric_columns = [
    "height_cm",
    "weight_kg",
    "bp_systolic",
    "bp_diastolic",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["smoking_history"] = df["smoking_history"].astype(str).str.strip()

# 构建 BMI
if "bmi" not in df.columns:
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
else:
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)


# =========================================================
# 6. 构建衍生字段
# =========================================================

# ---------- 6.1 BMI 分级 ----------
# 使用中国成人 BMI 分类标准：
# <18.5 偏瘦；18.5-23.9 正常；24.0-27.9 超重；>=28.0 肥胖
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


# ---------- 6.2 血压风险评分 ----------
# 规则：
# 0：正常       收缩压<120 且 舒张压<80
# 1：偏高       120<=收缩压<140 或 80<=舒张压<90
# 2：高血压风险 140<=收缩压<160 或 90<=舒张压<100
# 3：高血压高风险 收缩压>=160 或 舒张压>=100
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


# ---------- 6.3 吸烟风险评分 ----------
# Never=0, Former=1, Current=2
def get_smoking_score(smoking_history):
    value = str(smoking_history).strip().lower()

    if value == "never":
        return 0
    elif value == "former":
        return 1
    elif value == "current":
        return 2
    else:
        return np.nan


df["smoking_score"] = df["smoking_history"].apply(get_smoking_score)


# ---------- 6.4 血糖等级 ----------
# <100 正常；100-125 偏高；>=126 高血糖风险
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


# ---------- 6.5 胆固醇等级 ----------
# <200 理想；200-239 边缘偏高；>=240 偏高
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


# =========================================================
# 7. 统计与绘图辅助函数
# =========================================================

def build_count_table(series: pd.Series, col_name: str, ordered_categories=None):
    """
    生成分类变量统计表
    """
    count_series = series.value_counts(dropna=False)

    if ordered_categories is not None:
        existing_order = [x for x in ordered_categories if x in count_series.index]
        extra_order = [x for x in count_series.index if x not in existing_order]
        count_series = count_series.reindex(existing_order + extra_order)

    count_df = count_series.reset_index()
    count_df.columns = [col_name, "人数"]
    count_df["占比(%)"] = count_df["人数"] / count_df["人数"].sum() * 100

    return count_df


def plot_bar_chart(
    count_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path
):
    """
    绘制柱状图，柱子带黑色边界，边界清晰
    """
    plt.figure(figsize=(8, 5))
    plt.bar(
        count_df[x_col].astype(str),
        count_df[y_col],
        edgecolor="black",
        linewidth=0.8,
        width=0.8
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# 8. BMI 分级人数图
# =========================================================

bmi_order = ["偏瘦", "正常", "超重", "肥胖"]
bmi_level_df = build_count_table(df["bmi_level"], "BMI分级", bmi_order)

bmi_level_df.to_csv(
    RESULTS_DIR / "01_BMI分级统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("BMI 分级统计")
print("=" * 80)
print(bmi_level_df)

plot_bar_chart(
    count_df=bmi_level_df,
    x_col="BMI分级",
    y_col="人数",
    title="BMI 分级人数图",
    x_label="BMI 分级",
    y_label="人数",
    output_path=RESULTS_DIR / "01_BMI分级人数图.png"
)


# =========================================================
# 9. 血压风险评分分布图
# =========================================================

bp_score_order = [0, 1, 2, 3]
bp_score_df = build_count_table(df["bp_risk_score"], "血压风险评分", bp_score_order)

bp_score_df.to_csv(
    RESULTS_DIR / "02_血压风险评分统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("血压风险评分统计")
print("=" * 80)
print(bp_score_df)

plot_bar_chart(
    count_df=bp_score_df,
    x_col="血压风险评分",
    y_col="人数",
    title="血压风险评分分布图",
    x_label="血压风险评分",
    y_label="人数",
    output_path=RESULTS_DIR / "02_血压风险评分分布图.png"
)

bp_level_order = ["正常", "偏高", "高血压风险", "高血压高风险"]
bp_level_df = build_count_table(df["bp_risk_level"], "血压风险等级", bp_level_order)

bp_level_df.to_csv(
    RESULTS_DIR / "03_血压风险等级统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

plot_bar_chart(
    count_df=bp_level_df,
    x_col="血压风险等级",
    y_col="人数",
    title="血压风险等级分布图",
    x_label="血压风险等级",
    y_label="人数",
    output_path=RESULTS_DIR / "03_血压风险等级分布图.png"
)


# =========================================================
# 10. 吸烟风险评分分布图
# =========================================================

smoking_score_order = [0, 1, 2]
smoking_score_df = build_count_table(df["smoking_score"], "吸烟风险评分", smoking_score_order)

smoking_score_df.to_csv(
    RESULTS_DIR / "04_吸烟风险评分统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("吸烟风险评分统计")
print("=" * 80)
print(smoking_score_df)

plot_bar_chart(
    count_df=smoking_score_df,
    x_col="吸烟风险评分",
    y_col="人数",
    title="吸烟风险评分分布图",
    x_label="吸烟风险评分",
    y_label="人数",
    output_path=RESULTS_DIR / "04_吸烟风险评分分布图.png"
)


# =========================================================
# 11. 血糖等级分布图
# =========================================================

glucose_order = ["正常", "偏高", "高血糖风险"]
glucose_level_df = build_count_table(df["glucose_level"], "血糖等级", glucose_order)

glucose_level_df.to_csv(
    RESULTS_DIR / "05_血糖等级统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("血糖等级统计")
print("=" * 80)
print(glucose_level_df)

plot_bar_chart(
    count_df=glucose_level_df,
    x_col="血糖等级",
    y_col="人数",
    title="血糖等级分布图",
    x_label="血糖等级",
    y_label="人数",
    output_path=RESULTS_DIR / "05_血糖等级分布图.png"
)


# =========================================================
# 12. 胆固醇等级分布图
# =========================================================

cholesterol_order = ["理想", "边缘偏高", "偏高"]
cholesterol_level_df = build_count_table(df["cholesterol_level"], "胆固醇等级", cholesterol_order)

cholesterol_level_df.to_csv(
    RESULTS_DIR / "06_胆固醇等级统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("胆固醇等级统计")
print("=" * 80)
print(cholesterol_level_df)

plot_bar_chart(
    count_df=cholesterol_level_df,
    x_col="胆固醇等级",
    y_col="人数",
    title="胆固醇等级分布图",
    x_label="胆固醇等级",
    y_label="人数",
    output_path=RESULTS_DIR / "06_胆固醇等级分布图.png"
)


# =========================================================
# 13. 导出带衍生字段的数据
# =========================================================

df.to_csv(
    RESULTS_DIR / "00_添加衍生风险等级后的数据.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 14. 汇总表
# =========================================================

summary_df = pd.DataFrame({
    "项目": [
        "总样本数",
        "BMI 分级类别数",
        "血压风险评分类别数",
        "血压风险等级类别数",
        "吸烟风险评分类别数",
        "血糖等级类别数",
        "胆固醇等级类别数"
    ],
    "结果": [
        len(df),
        df["bmi_level"].nunique(dropna=True),
        df["bp_risk_score"].nunique(dropna=True),
        df["bp_risk_level"].nunique(dropna=True),
        df["smoking_score"].nunique(dropna=True),
        df["glucose_level"].nunique(dropna=True),
        df["cholesterol_level"].nunique(dropna=True)
    ]
})

summary_df.to_csv(
    RESULTS_DIR / "00_衍生风险等级汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("衍生风险等级汇总")
print("=" * 80)
print(summary_df)


# =========================================================
# 15. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("衍生风险等级类画像绘制完成")
print(f"输入数据文件：{DATA_PATH}")
print(f"所有结果已保存到：{RESULTS_DIR}")
print("=" * 80)
