# -*- coding: utf-8 -*-
"""
慢性病患者健康风险画像与分群分析 — 可视化大屏

基于 Streamlit 构建，单文件实现。
启动: streamlit run dashboard.py
"""

import sys
import warnings
from pathlib import Path

# 抑制 Streamlit 内部的弃用警告（如 use_container_width → width）
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# 0. 页面配置
# ============================================================

st.set_page_config(
    page_title="慢性病患者健康风险画像与分群分析",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. 路径配置（自适应项目根目录）
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# 2. 缓存的工具函数
# ============================================================


@st.cache_data
def load_image(img_path: Path) -> Image.Image | None:
    """加载单张PNG图片，返回PIL Image或None。"""
    if not img_path.exists():
        return None
    try:
        return Image.open(img_path)
    except Exception:
        return None


@st.cache_data
def load_csv(csv_path: Path) -> pd.DataFrame | None:
    """安全读取CSV，自动尝试常见编码。"""
    if not csv_path.exists():
        return None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return None


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """加载全量患者数据 change.csv，用于筛选和KPI计算。"""
    return load_csv(DATA_DIR / "change.csv")


def img(img_path: Path, caption: str = "", stretch: bool = True):
    """安全显示图片：存在则渲染，否则显示占位提示。"""
    image = load_image(img_path)
    if image is not None:
        st.image(image, caption=caption, width="stretch" if stretch else "content")
    else:
        st.caption(f"⚠️ 图片未找到: {img_path.name}")


def img_rel(subpath: str, caption: str = "", stretch: bool = True):
    """使用 results/ 下的相对路径显示图片。"""
    img(RESULTS_DIR / subpath, caption, stretch)


# ============================================================
# 3. 数据筛选函数
# ============================================================


def filter_data(df: pd.DataFrame, diseases: list[str], gender: str, age_range: tuple[int, int]) -> pd.DataFrame:
    """按疾病、性别、年龄范围筛选。"""
    filtered = df.copy()

    if diseases and "全部" not in diseases:
        filtered = filtered[filtered["disease"].isin(diseases)]

    if gender == "男":
        filtered = filtered[filtered["gender"] == "Male"]
    elif gender == "女":
        filtered = filtered[filtered["gender"] == "Female"]

    filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

    return filtered


def compute_kpis(df: pd.DataFrame) -> dict:
    """从筛选后的数据计算核心KPI。"""
    n = len(df)
    if n == 0:
        return {"患者数": 0, "平均年龄": "-", "平均BMI": "-", "平均收缩压": "-", "平均血糖": "-", "疾病种类": 0}

    kpis = {"患者数": n, "疾病种类": df["disease"].nunique()}

    for col, label in [
        ("age", "平均年龄"),
        ("bmi", "平均BMI"),
        ("bp_systolic", "平均收缩压"),
        ("blood_sugar_mg_dl", "平均血糖"),
    ]:
        if col in df.columns:
            val = df[col].mean()
            kpis[label] = round(val, 1)
        else:
            kpis[label] = "-"

    return kpis


# ============================================================
# 4. Sidebar
# ============================================================


def render_sidebar(df: pd.DataFrame):
    """渲染侧边栏：标题、筛选控件、筛选后的数据概览。"""
    with st.sidebar:
        st.markdown("## 🏥 慢性病患者健康风险画像")
        st.markdown("### 与分群分析")
        st.markdown("---")

        # ---- 筛选控件 ----
        st.markdown("### 🔍 数据筛选")

        # 疾病多选
        all_diseases = sorted(df["disease"].dropna().unique().tolist())
        selected_diseases = st.multiselect(
            "疾病类型",
            options=["全部"] + all_diseases,
            default="全部",
        )

        # 性别单选
        selected_gender = st.radio(
            "性别",
            options=["全部", "男", "女"],
            horizontal=True,
        )

        # 年龄范围
        min_age = int(df["age"].min())
        max_age = int(df["age"].max())
        age_range = st.slider(
            "年龄范围",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
            step=1,
        )

        st.markdown("---")

        # ---- 筛选后概览 ----
        st.markdown("### 📋 筛选后概览")

        filtered = filter_data(df, selected_diseases, selected_gender, age_range)
        kpis = compute_kpis(filtered)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("患者数", f"{kpis['患者数']:,}")
            st.metric("平均年龄", kpis["平均年龄"])
            st.metric("平均BMI", kpis["平均BMI"])
        with col_b:
            st.metric("疾病种类", kpis["疾病种类"])
            st.metric("平均收缩压", kpis["平均收缩压"])
            st.metric("平均血糖", kpis["平均血糖"])

        st.markdown("---")
        st.caption("💡 图表为全量数据预生成，筛选仅影响上方KPI数字。")

    return filtered, kpis


# ============================================================
# 5. 通用渲染工具
# ============================================================


def section_title(title: str, description: str = ""):
    """统一的段落标题。"""
    st.markdown(f"### {title}")
    if description:
        st.markdown(f"*{description}*")


def image_grid(image_specs: list[tuple[str, str]], cols: int = 3):
    """按cols列数渲染图片网格。
    image_specs: [(relative_path, caption), ...]
    """
    for i in range(0, len(image_specs), cols):
        row_specs = image_specs[i : i + cols]
        row_cols = st.columns(len(row_specs))
        for col, (rel_path, caption) in zip(row_cols, row_specs):
            with col:
                img_rel(rel_path, caption)


def show_table_in_expander(csv_rel_path: str, label: str = "📋 查看数据表"):
    """在expander中展示CSV数据表。"""
    with st.expander(label):
        df_table = load_csv(RESULTS_DIR / csv_rel_path)
        if df_table is not None:
            st.dataframe(df_table, width="stretch")
        else:
            st.caption("数据表未找到")


# ============================================================
# 6. Tab 1 — 项目概览
# ============================================================


def render_tab_overview(kpis: dict):
    """Tab 1: 项目概览"""
    st.markdown("## 📊 项目概览")

    # NLP KPI 卡片行
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("👥 总患者数", f"{kpis['患者数']:,}")
    with c2:
        st.metric("🦠 疾病种类", kpis["疾病种类"])
    with c3:
        st.metric("📅 平均年龄", kpis["平均年龄"])
    with c4:
        st.metric("⚖️ 平均BMI", kpis["平均BMI"])
    with c5:
        best_f1 = _get_best_f1()
        st.metric("🎯 最佳模型F1", best_f1)

    st.markdown("---")

    # 项目结构 — 三大任务卡片
    st.markdown("### 项目三大任务")
    tc1, tc2, tc3 = st.columns(3)

    with tc1:
        with st.container(border=True):
            st.markdown("#### 🔎 任务一：数据探索分析")
            st.markdown("""
            - 数据质量检查（缺失值/重复值/异常值）
            - 患者基本结构分析
            - 核心健康指标分布
            - 衍生风险等级构建
            - 异常值与医学逻辑检查
            - 指标关系分析
            - 疾病类型健康画像
            """)

    with tc2:
        with st.container(border=True):
            st.markdown("#### 🧩 任务二：无监督学习")
            st.markdown("""
            - K-Means 聚类（K=2~10评估）
            - Kneedle 拐点法选最优K
            - PCA 降维可视化
            - 聚类中心雷达图画像
            - cluster × disease 交叉分析
            """)

    with tc3:
        with st.container(border=True):
            st.markdown("#### 🤖 任务三：监督学习")
            st.markdown("""
            - disease 十分类预测
            - disease_group 四分类预测
            - Logistic Regression / Decision Tree / Random Forest
            - 混淆矩阵 & 特征重要性
            - 两个粒度标签对比分析
            """)

    st.markdown("---")

    # 数据质量快照（2×2）
    st.markdown("### 数据质量快照")
    image_grid(
        [
            ("quality/01_字段缺失值统计图.png", "字段缺失值统计"),
            ("quality/04_各字段异常值数量统计图.png", "各字段异常值数量"),
            ("quality/03_患者ID重复情况统计图.png", "患者ID重复情况"),
            ("quality/05_血压逻辑异常统计图.png", "血压逻辑异常"),
        ],
        cols=4,
    )


def _get_best_f1() -> str:
    """从模型对比表中提取最佳F1值。"""
    cmp = load_csv(RESULTS_DIR / "监督学习/analysis/comparison/task3_all_model_comparison.csv")
    if cmp is not None and "macro_f1" in cmp.columns:
        return str(round(cmp["macro_f1"].max(), 4))
    return "-"


# ============================================================
# 7. Tab 2 — 患者基本结构
# ============================================================


def render_tab_demographics():
    """Tab 2: 患者基本结构"""
    st.markdown("## 👥 患者基本结构")
    st.markdown("*性别、吸烟史、疾病类型、就诊趋势*")

    # 上排：3列
    c1, c2, c3 = st.columns(3)
    with c1:
        img_rel("患者基本结构/01_性别分布图_柱状图.png", "性别分布（柱状图）")
    with c2:
        img_rel("患者基本结构/02_性别分布图_饼图.png", "性别分布（饼图）")
    with c3:
        img_rel("患者基本结构/03_吸烟史分布图.png", "吸烟史分布")

    # 下排：2列（宽图）
    c4, c5 = st.columns(2)
    with c4:
        img_rel("患者基本结构/04_疾病类型分布图.png", "疾病类型分布")
    with c5:
        img_rel("患者基本结构/05_就诊日期趋势图_折线图.png", "就诊日期趋势")

    # 补充图
    img_rel("患者基本结构/06_就诊日期趋势图_柱状图.png", "就诊日期趋势（柱状图）")

    # 数据表
    show_table_in_expander("患者基本结构/00_患者基本结构汇总表.csv", "📋 患者基本结构汇总表")


# ============================================================
# 8. Tab 3 — 核心健康指标分布
# ============================================================


def render_tab_health_metrics(kpis: dict):
    """Tab 3: 核心健康指标分布"""
    st.markdown("## 💪 核心健康指标分布")
    st.markdown("*年龄、身高、体重、BMI、血压、胆固醇、血糖的直方图+KDE曲线*")

    # NaLP KPI 行
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📅 平均年龄", kpis["平均年龄"])
    with c2:
        st.metric("⚖️ 平均BMI", kpis["平均BMI"])
    with c3:
        st.metric("💓 平均收缩压", kpis["平均收缩压"])
    with c4:
        st.metric("🩸 平均血糖", kpis["平均血糖"])

    st.markdown("---")

    # 2行×4列 指标分布图
    specs = [
        ("核心健康指标分布/01_年龄分布图.png", "年龄分布"),
        ("核心健康指标分布/02_身高分布图.png", "身高分布"),
        ("核心健康指标分布/03_体重分布图.png", "体重分布"),
        ("核心健康指标分布/04_BMI分布图.png", "BMI分布"),
        ("核心健康指标分布/05_收缩压分布图.png", "收缩压分布"),
        ("核心健康指标分布/06_舒张压分布图.png", "舒张压分布"),
        ("核心健康指标分布/07_胆固醇分布图.png", "胆固醇分布"),
        ("核心健康指标分布/08_血糖分布图.png", "血糖分布"),
    ]
    image_grid(specs, cols=4)

    # 描述性统计表
    show_table_in_expander(
        "核心健康指标分布/00_核心健康指标描述性统计汇总表.csv",
        "📋 核心健康指标描述性统计汇总表",
    )

    # 血压逻辑异常散点图
    st.markdown("---")
    st.markdown("### 血压逻辑检查")
    img_rel("异常值与医学逻辑检查/07_收缩压与舒张压散点图.png", "收缩压 vs 舒张压（逻辑异常标记）")


# ============================================================
# 9. Tab 4 — 风险分层分析
# ============================================================


def render_tab_risk_tiers():
    """Tab 4: 衍生风险等级 + 异常值检查 + 指标关系"""
    st.markdown("## ⚠️ 风险分层分析")

    # ---- 衍生风险等级 ----
    st.markdown("### 衍生风险等级分布")
    risk_specs = [
        ("衍生风险等级/01_BMI分级人数图.png", "BMI分级"),
        ("衍生风险等级/02_血压风险评分分布图.png", "血压风险评分"),
        ("衍生风险等级/03_血压风险等级分布图.png", "血压风险等级"),
        ("衍生风险等级/04_吸烟风险评分分布图.png", "吸烟风险评分"),
        ("衍生风险等级/05_血糖等级分布图.png", "血糖等级"),
        ("衍生风险等级/06_胆固醇等级分布图.png", "胆固醇等级"),
    ]
    image_grid(risk_specs, cols=3)

    show_table_in_expander("衍生风险等级/00_衍生风险等级汇总表.csv", "📋 衍生风险等级汇总表")

    st.markdown("---")

    # ---- 异常值箱线图 ----
    st.markdown("### 异常值检测（IQR箱线图）")
    outlier_specs = [
        ("异常值与医学逻辑检查/01_年龄箱线图.png", "年龄箱线图"),
        ("异常值与医学逻辑检查/02_BMI箱线图.png", "BMI箱线图"),
        ("异常值与医学逻辑检查/03_收缩压箱线图.png", "收缩压箱线图"),
        ("异常值与医学逻辑检查/04_舒张压箱线图.png", "舒张压箱线图"),
        ("异常值与医学逻辑检查/05_胆固醇箱线图.png", "胆固醇箱线图"),
        ("异常值与医学逻辑检查/06_血糖箱线图.png", "血糖箱线图"),
    ]
    image_grid(outlier_specs, cols=3)

    show_table_in_expander("异常值与医学逻辑检查/00_异常值统计汇总表.csv", "📋 异常值统计汇总表")

    st.markdown("---")

    # ---- 指标相关性 ----
    st.markdown("### 指标关系分析")
    img_rel("指标关系分析/01_数值变量相关性热力图.png", "数值变量相关性热力图")

    # 补充关系图
    st.markdown("#### 指标与血压风险关系")
    c1, c2, c3 = st.columns(3)
    with c1:
        img_rel("指标关系分析/02_胆固醇与血压风险关系图.png", "胆固醇 vs 血压风险")
    with c2:
        img_rel("指标关系分析/03_血糖与血压风险关系图.png", "血糖 vs 血压风险")
    with c3:
        img_rel("指标关系分析/04_年龄与血压风险关系图.png", "年龄 vs 血压风险")

    show_table_in_expander("指标关系分析/00_指标关系分析汇总表.csv", "📋 指标关系分析汇总表")


# ============================================================
# 10. Tab 5 — 疾病类型健康画像
# ============================================================


def render_tab_disease_profiles():
    """Tab 5: 疾病类型健康画像"""
    st.markdown("## 🔬 疾病类型健康画像")
    st.markdown("*不同疾病患者的健康指标对比分析*")

    specs = [
        ("疾病类型健康画像/01_疾病类型与年龄分布画像.png", "疾病 × 年龄"),
        ("疾病类型健康画像/02_疾病类型与BMI分布画像.png", "疾病 × BMI"),
        ("疾病类型健康画像/03_疾病类型与血压风险分布画像.png", "疾病 × 血压风险"),
        ("疾病类型健康画像/04_疾病类型与血糖分布画像.png", "疾病 × 血糖"),
        ("疾病类型健康画像/05_疾病类型与胆固醇分布画像.png", "疾病 × 胆固醇"),
        ("疾病类型健康画像/06_疾病类型与吸烟史结构画像.png", "疾病 × 吸烟史"),
        ("疾病类型健康画像/07_疾病类型与性别结构画像.png", "疾病 × 性别"),
        ("疾病类型健康画像/08_疾病类型健康指标均值热力画像.png", "健康指标均值热力图"),
    ]
    image_grid(specs, cols=4)

    show_table_in_expander("疾病类型健康画像/08_疾病类型健康指标均值统计表.csv", "📋 健康指标均值统计表")


# ============================================================
# 11. Tab 6 — 机器学习模型结果
# ============================================================


def render_tab_ml_results():
    """Tab 6: 机器学习模型结果，使用expander分组避免一次性加载过多图片。"""
    st.markdown("## 🤖 机器学习模型结果")

    # ---- K-Means 聚类 ----
    with st.expander("🧩 K-Means 聚类分析", expanded=False):
        st.markdown("#### K值评估指标")
        image_grid(
            [
                ("聚类/kmeans_sse_elbow.png", "SSE肘部法"),
                ("聚类/kmeans_ch_index.png", "Calinski-Harabasz指数"),
                ("聚类/kmeans_dbi_index.png", "Davies-Bouldin指数"),
                ("聚类/cluster_distribution.png", "聚类人数分布"),
                ("聚类/pca_cluster_scatter.png", "PCA聚类散点图"),
                ("聚类/cluster_radar.png", "聚类雷达图"),
            ],
            cols=3,
        )

        st.markdown("#### 聚类与疾病交叉分析")
        image_grid(
            [
                ("聚类/cluster_centers_heatmap.png", "聚类中心热力图"),
                ("聚类/cluster_disease_stacked_bar.png", "Cluster×疾病堆叠柱状图"),
                ("聚类/disease_cluster_heatmap.png", "疾病×Cluster热力图"),
                ("聚类/disease_cluster_pct_stacked_bar.png", "疾病×Cluster百分比堆叠图"),
                ("聚类/high_risk_cluster_disease.png", "高风险Cluster疾病分析"),
            ],
            cols=3,
        )

    # ---- 监督学习 — 模型对比 ----
    with st.expander("📊 监督学习 — 模型对比总览", expanded=False):
        st.markdown("#### 十分类 vs 四分类 模型效果对比")
        img_rel("监督学习/analysis/comparison/task3_accuracy_comparison.png", "Accuracy对比")
        img_rel("监督学习/analysis/comparison/task3_macro_f1_comparison.png", "Macro F1对比")
        img_rel("监督学习/analysis/comparison/task3_all_metrics_overview.png", "全指标总览")
        show_table_in_expander(
            "监督学习/analysis/comparison/task3_all_model_comparison.csv",
            "📋 六组模型总对比表",
        )

    # ---- 监督学习 — 十分类 ----
    with st.expander("🔟 监督学习 — Disease 十分类", expanded=False):
        st.markdown("#### 样本分布与模型指标")
        c1, c2 = st.columns(2)
        with c1:
            img_rel("监督学习/analysis/disease_10class/distributions/disease_distribution.png", "10类疾病样本分布")
        with c2:
            img_rel("监督学习/analysis/disease_10class/metrics/disease_10class_model_metrics.png", "十分类模型指标对比")

        st.markdown("#### 混淆矩阵（Top Confusion Pairs）")
        c1, c2, c3 = st.columns(3)
        with c1:
            img_rel("监督学习/analysis/disease_10class/confusion/LR_top_confusion_pairs.png", "LR混淆对")
        with c2:
            img_rel("监督学习/analysis/disease_10class/confusion/DT_top_confusion_pairs.png", "DT混淆对")
        with c3:
            img_rel("监督学习/analysis/disease_10class/confusion/RF_top_confusion_pairs.png", "RF混淆对")

        st.markdown("#### 完整混淆矩阵 & 特征重要性")
        c1, c2, c3 = st.columns(3)
        with c1:
            img_rel("监督学习/disease_10class/figures/LR_confusion_matrix.png", "LR混淆矩阵")
            img_rel("监督学习/disease_10class/figures/LR_feature_importance.png", "LR特征重要性")
        with c2:
            img_rel("监督学习/disease_10class/figures/DT_confusion_matrix.png", "DT混淆矩阵")
            img_rel("监督学习/disease_10class/figures/DT_feature_importance.png", "DT特征重要性")
        with c3:
            img_rel("监督学习/disease_10class/figures/RF_confusion_matrix.png", "RF混淆矩阵")
            img_rel("监督学习/disease_10class/figures/RF_feature_importance.png", "RF特征重要性")

        st.markdown("#### LR系数 & 决策树")
        c1, c2 = st.columns(2)
        with c1:
            img_rel("监督学习/analysis/disease_10class/coefficients/disease_10class_LR_coefficients_heatmap.png", "LR系数热力图")
        with c2:
            img_rel("监督学习/analysis/disease_10class/trees/disease_10class_DT_tree_depth3.png", "决策树（depth=3）")

    # ---- 监督学习 — 四分类 ----
    with st.expander("4️⃣ 监督学习 — Disease Group 四分类", expanded=False):
        st.markdown("#### 样本分布与模型指标")
        c1, c2 = st.columns(2)
        with c1:
            img_rel("监督学习/analysis/disease_group_4class/distributions/disease_group_distribution.png", "4类疾病大类样本分布")
        with c2:
            img_rel("监督学习/analysis/disease_group_4class/metrics/disease_group_4class_model_metrics.png", "四分类模型指标对比")

        st.markdown("#### 混淆矩阵（Top Confusion Pairs）")
        c1, c2, c3 = st.columns(3)
        with c1:
            img_rel("监督学习/analysis/disease_group_4class/confusion/LR_top_confusion_pairs.png", "LR混淆对")
        with c2:
            img_rel("监督学习/analysis/disease_group_4class/confusion/DT_top_confusion_pairs.png", "DT混淆对")
        with c3:
            img_rel("监督学习/analysis/disease_group_4class/confusion/RF_top_confusion_pairs.png", "RF混淆对")

        st.markdown("#### 完整混淆矩阵 & 特征重要性")
        c1, c2, c3 = st.columns(3)
        with c1:
            img_rel("监督学习/disease_group_4class/figures/LR_confusion_matrix.png", "LR混淆矩阵")
            img_rel("监督学习/disease_group_4class/figures/LR_feature_importance.png", "LR特征重要性")
        with c2:
            img_rel("监督学习/disease_group_4class/figures/DT_confusion_matrix.png", "DT混淆矩阵")
            img_rel("监督学习/disease_group_4class/figures/DT_feature_importance.png", "DT特征重要性")
        with c3:
            img_rel("监督学习/disease_group_4class/figures/RF_confusion_matrix.png", "RF混淆矩阵")
            img_rel("监督学习/disease_group_4class/figures/RF_feature_importance.png", "RF特征重要性")

        st.markdown("#### LR系数 & 决策树")
        c1, c2 = st.columns(2)
        with c1:
            img_rel("监督学习/analysis/disease_group_4class/coefficients/disease_group_4class_LR_coefficients_heatmap.png", "LR系数热力图")
        with c2:
            img_rel("监督学习/analysis/disease_group_4class/trees/disease_group_4class_DT_tree_depth3.png", "决策树（depth=3）")


# ============================================================
# 12. 主入口
# ============================================================


def main():
    # 加载全量数据
    df = load_raw_data()

    if df is None or df.empty:
        st.error("❌ 无法加载数据文件 data/change.csv，请确认文件存在。")
        st.stop()

    # 渲染Sidebar，获取筛选结果
    filtered_df, kpis = render_sidebar(df)

    # ---- 主内容区：6个Tab ----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 项目概览",
        "👥 患者基本结构",
        "💪 核心健康指标",
        "⚠️ 风险分层分析",
        "🔬 疾病健康画像",
        "🤖 机器学习模型",
    ])

    with tab1:
        render_tab_overview(kpis)

    with tab2:
        render_tab_demographics()

    with tab3:
        render_tab_health_metrics(kpis)

    with tab4:
        render_tab_risk_tiers()

    with tab5:
        render_tab_disease_profiles()

    with tab6:
        render_tab_ml_results()

    # ---- 页脚 ----
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 12px;'>"
        "基于健康指标的慢性病患者风险画像与分群分析 | "
        "数据来源: chronic_patients.csv (n=15,943) | "
        "Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
