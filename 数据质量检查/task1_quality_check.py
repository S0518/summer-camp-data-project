# -*- coding: utf-8 -*-
"""
任务一：数据探索分析——数据质量检查类画像

本脚本完成以下内容：
1. 字段缺失值统计图
2. 字段缺失率统计图
3. 患者 ID 重复情况统计图
4. 各字段异常值数量统计图
5. 血压逻辑异常统计图

同时完成：
1. IQR 四分位距异常值判断
2. 正常值范围异常判断
3. 血压逻辑异常判断
4. 删除异常值所在行
5. 将删除异常后的数据保存到 data/change.csv

数据输入目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data

结果输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\quality
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
RESULTS_DIR = PROJECT_DIR / "results" / "quality"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE_NAME = "chronic_patients.csv"
DATA_PATH = DATA_DIR / DATA_FILE_NAME

# 删除异常值后的数据保存路径
CHANGE_DATA_PATH = DATA_DIR / "change.csv"

# 如果 chronic_patients.csv 不存在，则自动读取 data 目录下第一个 csv 文件
if not DATA_PATH.exists():
    csv_files = list(DATA_DIR.glob("*.csv"))
    csv_files = [file for file in csv_files if file.name.lower() != "change.csv"]

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
    "Arial Unicode MS"
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
original_df = df.copy()
original_field_count = df.shape[1]

print("=" * 80)
print("数据读取成功")
print("=" * 80)
print(f"数据文件：{DATA_PATH}")
print(f"原始数据规模：{df.shape[0]} 行，{df.shape[1]} 列")
print("\n字段名称：")
print(df.columns.tolist())
print("\n字段类型：")
print(df.dtypes)
print("\n前 5 行数据：")
print(df.head())


# =========================================================
# 4. 检查必要字段是否存在
# =========================================================

required_columns = [
    "patient_id",
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
# 5. 数值字段转换与 BMI 构建
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

# 构建 BMI
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
df["bmi"] = df["bmi"].replace([np.inf, -np.inf], np.nan)

numeric_columns = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
    "cholesterol_mg_dl",
    "blood_sugar_mg_dl"
]


# =========================================================
# 6. 字段缺失值检查
# =========================================================

missing_count = df.isnull().sum()
missing_rate = df.isnull().mean() * 100

missing_df = pd.DataFrame({
    "字段名": missing_count.index,
    "缺失值数量": missing_count.values,
    "缺失率": missing_rate.values
})

missing_df = missing_df.sort_values(
    by="缺失值数量",
    ascending=False
)

missing_df.to_csv(
    RESULTS_DIR / "01_字段缺失值统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("字段缺失值统计结果")
print("=" * 80)
print(missing_df)


# 图 1：字段缺失值统计图
plt.figure(figsize=(11, 5))
plt.bar(missing_df["字段名"], missing_df["缺失值数量"])
plt.title("字段缺失值统计图")
plt.xlabel("字段名称")
plt.ylabel("缺失值数量")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "01_字段缺失值统计图.png")
plt.close()


# 图 2：字段缺失率统计图
plt.figure(figsize=(11, 5))
plt.bar(missing_df["字段名"], missing_df["缺失率"])
plt.title("字段缺失率统计图")
plt.xlabel("字段名称")
plt.ylabel("缺失率（%）")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_字段缺失率统计图.png")
plt.close()


# =========================================================
# 7. 患者 ID 重复情况检查
# =========================================================

patient_id_duplicate_count = int(df["patient_id"].duplicated().sum())
all_row_duplicate_count = int(df.duplicated().sum())

duplicate_df = pd.DataFrame({
    "检查项目": [
        "患者ID重复记录数",
        "整行完全重复记录数"
    ],
    "重复数量": [
        patient_id_duplicate_count,
        all_row_duplicate_count
    ]
})

duplicate_df.to_csv(
    RESULTS_DIR / "03_重复值统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("重复值统计结果")
print("=" * 80)
print(duplicate_df)


# 图 3：患者 ID 重复情况统计图
plt.figure(figsize=(7, 5))
plt.bar(duplicate_df["检查项目"], duplicate_df["重复数量"])
plt.title("患者 ID 重复情况统计图")
plt.xlabel("检查项目")
plt.ylabel("重复数量")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_患者ID重复情况统计图.png")
plt.close()


# 导出重复 patient_id 的记录
duplicate_patient_records = df[df["patient_id"].duplicated(keep=False)].copy()

duplicate_patient_records.to_csv(
    RESULTS_DIR / "03_重复患者ID记录.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. 异常值判断：IQR + 正常值范围
# =========================================================

"""
异常值判断分为两类：

1. IQR 统计异常：
   小于 Q1 - 1.5 * IQR 或大于 Q3 + 1.5 * IQR。

2. 正常值范围异常：
   小于人工设定的正常下限，或大于人工设定的正常上限。

综合异常：
   只要满足 IQR 异常或正常值范围异常中的任意一种，
   就标记为该字段异常。
"""

normal_ranges = {
    "age": (10, 90),
    "bmi": (12, 60),
    "height_cm": (120, 220),
    "weight_kg": (30, 250),
    "bp_systolic": (70, 250),
    "bp_diastolic": (40, 150),
    "cholesterol_mg_dl": (80, 500),
    "blood_sugar_mg_dl": (40, 500)
}

outlier_result = []

# 用于记录每一行是否存在任意异常
any_numeric_outlier_mask = pd.Series(False, index=df.index)

for col in numeric_columns:
    # ---------- IQR 异常判断 ----------
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    iqr_lower_bound = Q1 - 1.5 * IQR
    iqr_upper_bound = Q3 + 1.5 * IQR

    iqr_outlier_mask = (
        (df[col] < iqr_lower_bound) |
        (df[col] > iqr_upper_bound)
    )

    # ---------- 正常值范围异常判断 ----------
    normal_lower_bound, normal_upper_bound = normal_ranges[col]

    range_outlier_mask = (
        (df[col] < normal_lower_bound) |
        (df[col] > normal_upper_bound)
    )

    # ---------- 当前字段综合异常 ----------
    combined_outlier_mask = iqr_outlier_mask | range_outlier_mask

    any_numeric_outlier_mask = any_numeric_outlier_mask | combined_outlier_mask

    iqr_outlier_count = int(iqr_outlier_mask.sum())
    range_outlier_count = int(range_outlier_mask.sum())
    combined_outlier_count = int(combined_outlier_mask.sum())

    outlier_result.append({
        "字段名": col,

        "IQR_Q1": Q1,
        "IQR_Q3": Q3,
        "IQR": IQR,
        "IQR异常下界": iqr_lower_bound,
        "IQR异常上界": iqr_upper_bound,
        "IQR异常值数量": iqr_outlier_count,
        "IQR异常值占比": iqr_outlier_count / len(df) * 100,

        "正常范围下限": normal_lower_bound,
        "正常范围上限": normal_upper_bound,
        "正常范围异常值数量": range_outlier_count,
        "正常范围异常值占比": range_outlier_count / len(df) * 100,

        "综合异常值数量": combined_outlier_count,
        "综合异常值占比": combined_outlier_count / len(df) * 100
    })

    # 导出当前字段的综合异常记录
    df[combined_outlier_mask].to_csv(
        RESULTS_DIR / f"04_{col}_综合异常记录.csv",
        index=False,
        encoding="utf-8-sig"
    )

outlier_df = pd.DataFrame(outlier_result)

outlier_df.to_csv(
    RESULTS_DIR / "04_各字段异常值统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("各字段异常值统计结果")
print("=" * 80)
print(outlier_df)


# 图 4：各字段异常值数量统计图
plt.figure(figsize=(11, 5))
plt.bar(outlier_df["字段名"], outlier_df["综合异常值数量"])
plt.title("各字段异常值数量统计图")
plt.xlabel("数值字段")
plt.ylabel("异常值数量")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "04_各字段异常值数量统计图.png")
plt.close()


# =========================================================
# 9. 血压逻辑异常检查
# =========================================================

"""
血压逻辑异常判断：
正常情况下，收缩压应大于舒张压。
如果 bp_systolic <= bp_diastolic，则标记为血压逻辑异常。
"""

bp_abnormal_mask = df["bp_systolic"] <= df["bp_diastolic"]

bp_normal_count = int((~bp_abnormal_mask).sum())
bp_abnormal_count = int(bp_abnormal_mask.sum())

bp_logic_df = pd.DataFrame({
    "血压逻辑状态": [
        "正常记录",
        "异常记录：收缩压小于等于舒张压"
    ],
    "记录数量": [
        bp_normal_count,
        bp_abnormal_count
    ]
})

bp_logic_df["占比"] = bp_logic_df["记录数量"] / len(df) * 100

bp_logic_df.to_csv(
    RESULTS_DIR / "05_血压逻辑异常统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("血压逻辑异常统计结果")
print("=" * 80)
print(bp_logic_df)


# 导出血压逻辑异常记录
bp_abnormal_records = df[bp_abnormal_mask].copy()

bp_abnormal_records.to_csv(
    RESULTS_DIR / "05_血压逻辑异常记录.csv",
    index=False,
    encoding="utf-8-sig"
)


# 图 5：血压逻辑异常统计图
plt.figure(figsize=(8, 5))
plt.bar(bp_logic_df["血压逻辑状态"], bp_logic_df["记录数量"])
plt.title("血压逻辑异常统计图")
plt.xlabel("血压逻辑状态")
plt.ylabel("记录数量")
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "05_血压逻辑异常统计图.png")
plt.close()


# =========================================================
# 10. 删除异常值所在行，并保存 change.csv
# =========================================================

"""
删除规则：
1. 若某一行存在任意数值字段综合异常，则删除；
2. 若某一行存在血压逻辑异常，则删除；
3. 缺失值所在行是否删除：这里一并删除。
"""

missing_row_mask = df.isnull().any(axis=1)

delete_mask = (
    any_numeric_outlier_mask |
    bp_abnormal_mask |
    missing_row_mask
)

delete_count = int(delete_mask.sum())
keep_count = int((~delete_mask).sum())

deleted_records = df[delete_mask].copy()
changed_df = df[~delete_mask].copy()

# 导出被删除的记录，方便检查
deleted_records.to_csv(
    RESULTS_DIR / "06_被删除的异常记录.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存删除异常后的数据到 data/change.csv
changed_df.to_csv(
    CHANGE_DATA_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("异常值删除结果")
print("=" * 80)
print(f"原始记录数：{len(df)}")
print(f"删除记录数：{delete_count}")
print(f"保留记录数：{keep_count}")
print(f"删除异常后的数据已保存到：{CHANGE_DATA_PATH}")


# =========================================================
# 11. 数据质量汇总表
# =========================================================

quality_summary = pd.DataFrame({
    "质量检查项目": [
        "原始记录数",
        "原始字段数",
        "构建 BMI 后字段数",
        "总缺失值数量",
        "存在缺失值的行数",
        "患者ID重复记录数",
        "整行完全重复记录数",
        "数值字段综合异常行数",
        "血压逻辑异常记录数",
        "最终删除记录数",
        "最终保留记录数"
    ],
    "结果": [
        len(df),
        original_field_count,
        df.shape[1],
        int(df.isnull().sum().sum()),
        int(missing_row_mask.sum()),
        patient_id_duplicate_count,
        all_row_duplicate_count,
        int(any_numeric_outlier_mask.sum()),
        bp_abnormal_count,
        delete_count,
        keep_count
    ]
})

quality_summary.to_csv(
    RESULTS_DIR / "00_数据质量检查汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("数据质量检查汇总")
print("=" * 80)
print(quality_summary)


# =========================================================
# 12. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("数据质量检查类画像已完成")
print(f"输入数据文件：{DATA_PATH}")
print(f"图表和统计表已保存到：{RESULTS_DIR}")
print(f"删除异常后的数据已保存到：{CHANGE_DATA_PATH}")
print("=" * 80)
