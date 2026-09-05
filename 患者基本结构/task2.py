# -*- coding: utf-8 -*-
"""
任务一：数据探索分析——患者基本结构画像

本脚本基于清洗后的数据 change.csv 绘制以下图表：
1. 性别分布图（柱状图 + 饼图）
2. 吸烟史分布图（柱状图）
3. 疾病类型分布图（横向柱状图）
4. 就诊日期趋势图（按月折线图 + 按月柱状图）

输入数据：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\data\\change.csv

输出目录：
D:\\Programming\\Data\\Pycharm_data\\数据探索\\results\\患者基本结构
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_DIR = Path(r"D:\Programming\Data\Pycharm_data\数据探索")

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "患者基本结构"

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


# =========================================================
# 4. 读取数据
# =========================================================

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
# 5. 检查必要字段
# =========================================================

required_columns = [
    "gender",
    "smoking_history",
    "disease",
    "visit_date"
]

missing_required_columns = [col for col in required_columns if col not in df.columns]

if missing_required_columns:
    raise ValueError(f"数据中缺少必要字段：{missing_required_columns}")


# =========================================================
# 6. 数据预处理
# =========================================================

# 去除字符串前后空格
for col in ["gender", "smoking_history", "disease"]:
    df[col] = df[col].astype(str).str.strip()

# 处理就诊日期
df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

# 衍生字段：按月统计
df["visit_year_month"] = df["visit_date"].dt.to_period("M").astype(str)

# 去除 visit_date 无法解析的记录再做时间分析
visit_valid_df = df.dropna(subset=["visit_date"]).copy()


# =========================================================
# 7. 性别分布分析
# =========================================================

gender_count = df["gender"].value_counts(dropna=False).sort_index()
gender_df = gender_count.reset_index()
gender_df.columns = ["性别", "人数"]
gender_df["占比(%)"] = gender_df["人数"] / gender_df["人数"].sum() * 100

gender_df.to_csv(
    RESULTS_DIR / "01_性别分布统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("性别分布统计")
print("=" * 80)
print(gender_df)

# 图 1：性别分布柱状图
plt.figure(figsize=(7, 5))
plt.bar(gender_df["性别"], gender_df["人数"])
plt.title("性别分布图")
plt.xlabel("性别")
plt.ylabel("人数")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "01_性别分布图_柱状图.png")
plt.close()

# 图 2：性别分布饼图
plt.figure(figsize=(6, 6))
plt.pie(
    gender_df["人数"],
    labels=gender_df["性别"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("性别分布图")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "02_性别分布图_饼图.png")
plt.close()


# =========================================================
# 8. 吸烟史分布分析
# =========================================================

# 设定吸烟史顺序（如存在）
smoking_order = ["Never", "Former", "Current"]

smoking_count = df["smoking_history"].value_counts(dropna=False)

# 按预期顺序排列，若有额外类别则追加到后面
ordered_index = [x for x in smoking_order if x in smoking_count.index]
extra_index = [x for x in smoking_count.index if x not in ordered_index]
smoking_count = smoking_count.reindex(ordered_index + extra_index)

smoking_df = smoking_count.reset_index()
smoking_df.columns = ["吸烟史", "人数"]
smoking_df["占比(%)"] = smoking_df["人数"] / smoking_df["人数"].sum() * 100

smoking_df.to_csv(
    RESULTS_DIR / "03_吸烟史分布统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("吸烟史分布统计")
print("=" * 80)
print(smoking_df)

# 图 3：吸烟史分布图
plt.figure(figsize=(7, 5))
plt.bar(smoking_df["吸烟史"], smoking_df["人数"])
plt.title("吸烟史分布图")
plt.xlabel("吸烟史")
plt.ylabel("人数")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_吸烟史分布图.png")
plt.close()


# =========================================================
# 9. 疾病类型分布分析
# =========================================================

disease_count = df["disease"].value_counts(dropna=False).sort_values(ascending=True)
disease_df = disease_count.reset_index()
disease_df.columns = ["疾病类型", "人数"]
disease_df["占比(%)"] = disease_df["人数"] / disease_df["人数"].sum() * 100

disease_df.to_csv(
    RESULTS_DIR / "04_疾病类型分布统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("疾病类型分布统计")
print("=" * 80)
print(disease_df)

# 图 4：疾病类型分布图（横向柱状图）
plt.figure(figsize=(10, 6))
plt.barh(disease_df["疾病类型"], disease_df["人数"])
plt.title("疾病类型分布图")
plt.xlabel("人数")
plt.ylabel("疾病类型")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "04_疾病类型分布图.png")
plt.close()


# =========================================================
# 10. 就诊日期趋势分析
# =========================================================

visit_month_count = (
    visit_valid_df.groupby("visit_year_month")
    .size()
    .reset_index(name="就诊人数")
)

# 转成真正时间以保证按月份排序
visit_month_count["排序日期"] = pd.to_datetime(visit_month_count["visit_year_month"], format="%Y-%m")
visit_month_count = visit_month_count.sort_values("排序日期").reset_index(drop=True)

visit_month_count.to_csv(
    RESULTS_DIR / "05_就诊日期趋势统计表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("就诊日期趋势统计")
print("=" * 80)
print(visit_month_count[["visit_year_month", "就诊人数"]])

# 图 5：就诊日期趋势图（折线图）
plt.figure(figsize=(11, 5))
plt.plot(
    visit_month_count["visit_year_month"],
    visit_month_count["就诊人数"],
    marker="o"
)
plt.title("就诊日期趋势图")
plt.xlabel("年月")
plt.ylabel("就诊人数")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "05_就诊日期趋势图_折线图.png")
plt.close()

# 图 6：就诊日期趋势图（柱状图）
plt.figure(figsize=(11, 5))
plt.bar(
    visit_month_count["visit_year_month"],
    visit_month_count["就诊人数"]
)
plt.title("就诊日期趋势图")
plt.xlabel("年月")
plt.ylabel("就诊人数")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "06_就诊日期趋势图_柱状图.png")
plt.close()


# =========================================================
# 11. 汇总表
# =========================================================

summary_df = pd.DataFrame({
    "项目": [
        "总样本数",
        "性别类别数",
        "吸烟史类别数",
        "疾病类型类别数",
        "有效就诊日期记录数",
        "月份数"
    ],
    "结果": [
        len(df),
        df["gender"].nunique(dropna=False),
        df["smoking_history"].nunique(dropna=False),
        df["disease"].nunique(dropna=False),
        len(visit_valid_df),
        visit_month_count.shape[0]
    ]
})

summary_df.to_csv(
    RESULTS_DIR / "00_患者基本结构汇总表.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("患者基本结构汇总")
print("=" * 80)
print(summary_df)


# =========================================================
# 12. 完成提示
# =========================================================

print("\n" + "=" * 80)
print("患者基本结构图表绘制完成")
print(f"输入数据文件：{DATA_PATH}")
print(f"所有结果已保存到：{RESULTS_DIR}")
print("=" * 80)
