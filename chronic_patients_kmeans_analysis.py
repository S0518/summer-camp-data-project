# ------------------- 0. 导入库与全局配置 -------------------
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免 plt.close() 阻塞
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# 结果输出目录
RESULTS_DIR = Path(r'D:\Programming\Data\Pycharm_data\数据探索\results\聚类')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- 1. 加载数据、疾病名称汉化、构建衍生特征 -------------------

# 1.1 加载数据
df = pd.read_csv(r'D:\Programming\Data\Pycharm_data\数据探索\data\change.csv')
print(f"\n数据加载完成，数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"字段列表: {df.columns.tolist()}")

# 1.2 疾病名称汉化
disease_mapping = {
    'Chronic Kidney Disease': '慢性肾脏病',
    'Stroke': '脑卒中',
    'Lung Cancer': '肺癌',
    'Breast Cancer': '乳腺癌',
    'Hypertension': '高血压',
    'Cardiovascular Disease (Heart Attack, Stroke)': '心血管疾病(心梗/脑卒中)',
    'Chronic Respiratory Disease (COPD, Asthma)': '慢性呼吸系统疾病(COPD/哮喘)',
    'Cancer': '恶性肿瘤(其他)',
    'Diabetes': '糖尿病',
    'Coronary Heart Disease': '冠心病'
}
df['disease'] = df['disease'].replace(disease_mapping)
print(f"\n疾病名称汉化完成，现有疾病类型: {df['disease'].unique().tolist()}")

# 1.3 重新计算 BMI（覆盖CSV中可能存在的空值）
df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
print(f"\nBMI 重新计算完成，缺失值: {df['bmi'].isna().sum()}")

# 1.4 构建吸烟二元编码 smoking_binary
# Never=0（从不吸烟）, Former/Current=1（有吸烟史）
df['smoking_binary'] = df['smoking_history'].map({'Never': 0, 'Former': 1, 'Current': 1})
print(f"\nsmoking_binary 构建完成（0=从不吸烟, 1=有吸烟史），分布情况:")
print(df['smoking_binary'].value_counts().sort_index())

# 1.5 性别编码
# Female=0, Male=1
df['gender_code'] = df['gender'].map({'Female': 0, 'Male': 1})
print(f"\ngender_code 构建完成（Female=0, Male=1），分布情况:")
print(df['gender_code'].value_counts().sort_index())

# 1.6 构建血压风险评分 bp_risk_score
# 来源：项目计划书 表格10
# 分别计算收缩压分数和舒张压分数，取较高者（医学上血压分级取两者中较高等级）
systolic = df['bp_systolic'].values
diastolic = df['bp_diastolic'].values

# 收缩压分数: 0=正常(<120), 1=偏高(120-139), 2=高血压风险(140-159), 3=高血压高风险(>=160)
systolic_score = np.select(
    [systolic < 120, systolic <= 139, systolic <= 159],
    [0, 1, 2],
    default=3
)
# 舒张压分数: 0=正常(<80), 1=偏高(80-89), 2=高血压风险(90-99), 3=高血压高风险(>=100)
diastolic_score = np.select(
    [diastolic < 80, diastolic <= 89, diastolic <= 99],
    [0, 1, 2],
    default=3
)
# 取较高者作为最终血压风险评分
df['bp_risk_score'] = np.maximum(systolic_score, diastolic_score)
print(f"\nbp_risk_score 构建完成，分布情况:")
bp_dist = df['bp_risk_score'].value_counts().sort_index()
for score, count in bp_dist.items():
    print(f"  分数 {score}: {count} 人 ({count/len(df)*100:.1f}%)")

# 1.7 数据概况
print(f"\n数据概况（含衍生特征后）:")
print(f"  总记录数: {len(df)}")
print(f"  总字段数: {len(df.columns)}")
print(f"  疾病类型数: {df['disease'].nunique()}")
print(f"  缺失值总数: {df.isnull().sum().sum()}")

# ------------------- 2. 构建特征矩阵并标准化 -------------------


# 2.1 选取6个建模特征
features = ['age', 'bmi', 'bp_risk_score', 'cholesterol_mg_dl', 'blood_sugar_mg_dl', 'smoking_binary']
X = df[features].values
print(f"\n建模特征: {features}")
print(f"特征矩阵形状: {X.shape}")

# 2.2 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n标准化后各特征均值: {X_scaled.mean(axis=0).round(4)}")
print(f"标准化后各特征标准差: {X_scaled.std(axis=0).round(4)}")

# ------------------- 3. 训练K=2~10的K-Means并计算4项指标 -------------------


K_range = list(range(2, 11))
sse_list = []
ch_list = []
dbi_list = []

print("\n开始训练...\n")
for k in K_range:
    print(f"正在训练 K={k} ...", end=" ")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    sse = kmeans.inertia_
    ch = calinski_harabasz_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    sse_list.append(sse)
    ch_list.append(ch)
    dbi_list.append(dbi)

    print(f"SSE={sse:.2f}, CH={ch:.2f}, DBI={dbi:.4f}")

print("\n训练完成！")

# ------------------- 4. K值评估可视化（3张独立折线图） -------------------


# 4.1 K-SSE肘部法折线图
plt.figure(figsize=(8, 5))
plt.plot(K_range, sse_list, marker='o', color='steelblue', linewidth=2, markersize=8)
plt.title('肘部法：K值与SSE关系', fontsize=14)
plt.xlabel('聚类数量 K', fontsize=12)
plt.ylabel('SSE（簇内误差平方和）', fontsize=12)
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "kmeans_sse_elbow.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4.2 K-CH指数折线图
best_k_ch = K_range[np.argmax(ch_list)]
best_ch_val = max(ch_list)
plt.figure(figsize=(8, 5))
plt.plot(K_range, ch_list, marker='^', color='darkorange', linewidth=2, markersize=8)
plt.axvline(x=best_k_ch, color='gray', linestyle='--', alpha=0.5)
plt.scatter([best_k_ch], [best_ch_val], color='red', s=150, zorder=5,
            label=f'最优K={best_k_ch} (CH={best_ch_val:.2f})')
plt.title('K值与Calinski-Harabasz指数关系', fontsize=14)
plt.xlabel('聚类数量 K', fontsize=12)
plt.ylabel('Calinski-Harabasz指数', fontsize=12)
plt.xticks(K_range)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "kmeans_ch_index.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4.3 K-DBI指数折线图
best_k_dbi = K_range[np.argmin(dbi_list)]
best_dbi_val = min(dbi_list)
plt.figure(figsize=(8, 5))
plt.plot(K_range, dbi_list, marker='d', color='crimson', linewidth=2, markersize=8)
plt.axvline(x=best_k_dbi, color='gray', linestyle='--', alpha=0.5)
plt.scatter([best_k_dbi], [best_dbi_val], color='red', s=150, zorder=5,
            label=f'最优K={best_k_dbi} (DBI={best_dbi_val:.4f})')
plt.title('K值与Davies-Bouldin指数关系', fontsize=14)
plt.xlabel('聚类数量 K', fontsize=12)
plt.ylabel('Davies-Bouldin指数（越小越好）', fontsize=12)
plt.xticks(K_range)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "kmeans_dbi_index.png"), dpi=150, bbox_inches="tight")
plt.close()

# ------------------- 5. 选择最优K值：Kneedle 拐点检测 -------------------


# 各指标各自推荐的最优K（仅供参考）
print(f"\n各指标单独推荐的最优K值（仅供参考）:")
print(f"  CH指数推荐 K = {best_k_ch} (值={best_ch_val:.2f})")
print(f"  DBI指数推荐 K = {best_k_dbi} (值={best_dbi_val:.4f})")

# 方法：Kneedle 拐点检测
# 思路：SSE 天然随K增大而递减，直接归一化评分会偏向大K。
#       改为检测 SSE 曲线上"边际收益骤降"的几何拐点（Kneedle 算法），
#       拐点处的K即是最优选择。

sse_arr = np.array(sse_list)
k_arr = np.array(K_range)

# Kneedle — 归一化 K 轴和 SSE 轴，找离对角线 y=1-x 最远的点
k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
sse_norm = (sse_arr - sse_arr.min()) / (sse_arr.max() - sse_arr.min())
distances = sse_norm + k_norm - 1          # 曲线在对角线下方，偏离为负
knee_idx = np.argmin(distances)            # 最远偏离点 = 拐点
optimal_k = K_range[knee_idx]

print(f"\nKneedle 拐点检测 (SSE 曲线归一化后距对角线偏离):")
for i, k in enumerate(K_range):
    marker = " ← 拐点（最终选择）" if k == optimal_k else ""
    print(f"  K={k:>2}: 偏离={distances[i]:.4f}{marker}")
print(f"\n最终选择 K={optimal_k}（Kneedle 拐点，SSE 下降速率开始显著放缓的位置）")

# ------------------- 6. 最终模型训练与PCA可视化 -------------------


# 6.1 训练最终模型
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
final_kmeans.fit(X_scaled)
df['cluster'] = final_kmeans.labels_
print(f"\n最终K-Means模型训练完成，K={optimal_k}")
print(f"最终SSE（簇内误差平方和）: {final_kmeans.inertia_:.2f}")

# 6.2 聚类人数分布
cluster_counts = df['cluster'].value_counts().sort_index()
print(f"\n各聚类患者数量分布:")
for c in range(optimal_k):
    count = cluster_counts.get(c, 0)
    print(f"  Cluster {c}: {count} 人 ({count/len(df)*100:.1f}%)")

# 聚类人数分布柱状图
colors_cluster = sns.color_palette('Set2', optimal_k)
plt.figure(figsize=(8, 5))
bars = plt.bar(range(optimal_k), cluster_counts.values, color=colors_cluster, edgecolor='white')
plt.title(f'各聚类患者数量分布（K={optimal_k}）', fontsize=14)
plt.xlabel('聚类', fontsize=12)
plt.ylabel('患者数量', fontsize=12)
plt.xticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
# 在柱子上标注数值
for bar, count in zip(bars, cluster_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "cluster_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# 6.3 PCA降维可视化
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print(f"\nPCA降维完成:")
print(f"  主成分1解释方差: {explained_var[0]*100:.1f}%")
print(f"  主成分2解释方差: {explained_var[1]*100:.1f}%")
print(f"  累计解释方差: {explained_var.sum()*100:.1f}%")

if explained_var.sum() < 0.50:
    print("  提示：前两个主成分累计解释方差不足50%，二维投影可能无法完全展示聚类分离，")
    print("        聚类质量应以轮廓系数等指标为准，PCA图仅作辅助参考。")

plt.figure(figsize=(10, 8))
# 绘制各簇散点（半透明，展示数据分布密度）
for c in range(optimal_k):
    mask = df['cluster'] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors_cluster[c]], label=f'Cluster {c}',
                alpha=0.4, s=12, edgecolors='none')
# 叠加各簇在PCA空间中的中心点（大号标记+标签，突出簇间相对位置）
pca_centers = np.array([X_pca[df['cluster'] == c].mean(axis=0) for c in range(optimal_k)])
for c in range(optimal_k):
    plt.scatter(pca_centers[c, 0], pca_centers[c, 1],
                c=[colors_cluster[c]], edgecolors='black', linewidths=1.5,
                s=280, marker='D', zorder=10)
    plt.annotate(f'C{c}', (pca_centers[c, 0], pca_centers[c, 1]),
                 fontsize=11, fontweight='bold', ha='center', va='center', zorder=11)
plt.title(f'PCA降维后的患者聚类分布（K={optimal_k}，◇ 为各簇中心）', fontsize=14)
plt.xlabel(f'主成分1（{explained_var[0]*100:.1f}%）', fontsize=12)
plt.ylabel(f'主成分2（{explained_var[1]*100:.1f}%）', fontsize=12)
plt.legend(fontsize=10, markerscale=3)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "pca_cluster_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ------------------- 7. 聚类中心分析 -------------------


# 7.1 聚类中心表（还原到原始量纲）
centers_scaled = final_kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=features)
centers_df.index.name = 'cluster'
centers_df.index = [f'Cluster {i}' for i in range(optimal_k)]
# 添加患者数列
centers_df['患者数'] = cluster_counts.values
print(f"\n各聚类中心（原始量纲）:")
print(centers_df.round(2))

# 7.2 聚类指标画像热力图（min-max归一化后）
# 对每个特征做min-max归一化，使热力图反映各聚类在特征上的相对高低
heatmap_data = (centers_df[features] - centers_df[features].min()) / \
               (centers_df[features].max() - centers_df[features].min())

# 中文特征名映射
feature_names_cn = ['年龄', 'BMI', '血压风险评分', '胆固醇(mg/dL)', '血糖(mg/dL)', '吸烟史(有/无)']

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': '归一化值（0=最低, 1=最高）'})
plt.title(f'各聚类健康指标画像热力图（归一化后, K={optimal_k}）', fontsize=14, pad=15)
plt.xlabel('聚类', fontsize=12)
plt.ylabel('健康指标', fontsize=12)
# 将y轴标签替换为中文
plt.yticks(ticks=np.arange(len(features)) + 0.5, labels=feature_names_cn, rotation=0)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "cluster_centers_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# 7.3 雷达图
categories = ['年龄', 'BMI', '血压风险', '胆固醇', '血糖', '吸烟史']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i in range(optimal_k):
    values = heatmap_data.iloc[i].tolist()
    values += values[:1]  # 闭合
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}',
            color=colors_cluster[i], markersize=6)
    ax.fill(angles, values, alpha=0.1, color=colors_cluster[i])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_title(f'各聚类健康指标雷达图（K={optimal_k}，归一化值）', fontsize=14, pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "cluster_radar.png"), dpi=150, bbox_inches="tight")
plt.close()

# ------------------- 8. 患者群体命名 -------------------

# 命名使用的代谢特征（不含 age 和 gender——age 用于描述、gender 用于画像补充）
naming_features = ['bp_risk_score', 'bmi', 'cholesterol_mg_dl', 'blood_sugar_mg_dl', 'smoking_binary']

# 计算总体均值
overall_means = df[features].mean()
print(f"\n总体均值（{len(features)}个建模特征）:")
for f in features:
    print(f"  {f}: {overall_means[f]:.2f}")

# 各聚类中心与总体均值对比
print(f"\n各聚类中心与总体均值的差异（正值=高于总体，负值=低于总体）:")
for i in range(optimal_k):
    diff = centers_df[features].iloc[i] - overall_means
    print(f"\n  Cluster {i}:")
    for f in features:
        direction = "高于" if diff[f] > 0 else "低于"
        print(f"    {f}: {diff[f]:+.2f}（{direction}总体均值）")

# 根据聚类中心特征进行群体命名
# 使用 Z-score 偏离（>0.5σ）判断命名特征是否"偏高"
# 血糖和胆固醇合并为"糖脂代谢"维度
feat_std = df[features].std()
cluster_names = {}
for i in range(optimal_k):
    row = centers_df[features].iloc[i]
    z_diff = (row - overall_means) / feat_std  # Z-score 偏离

    # 5个命名特征的偏高判断（>0.5σ）
    high_bp = z_diff['bp_risk_score'] > 0.5
    high_bmi = z_diff['bmi'] > 0.5
    high_chol = z_diff['cholesterol_mg_dl'] > 0.5
    high_glu = z_diff['blood_sugar_mg_dl'] > 0.5
    high_smoke = z_diff['smoking_binary'] > 0.5

    # 血糖与胆固醇合并为"糖脂代谢"维度
    high_metabolic = high_glu or high_chol

    # high_count 统计 4 个维度：血压、BMI、糖脂代谢、吸烟
    high_count = sum([high_bp, high_bmi, high_metabolic, high_smoke])

    if high_count >= 3:
        cluster_names[i] = '多指标综合异常型'
    elif high_bp:
        cluster_names[i] = '血压异常突出型'
    elif high_bmi:
        cluster_names[i] = '肥胖代谢异常型'
    elif high_metabolic:
        cluster_names[i] = '糖脂代谢异常型'
    elif high_smoke:
        cluster_names[i] = '吸烟相关风险型'
    else:
        cluster_names[i] = '代谢指标偏低型'

print(f"\n患者群体命名:")
print(f"  判定标准: Z-score > 0.5σ 视为该特征偏高")
print(f"  命名维度: 血压、BMI、糖脂代谢(血糖/胆固醇合并)、吸烟（共4维）")
for i in range(optimal_k):
    row = centers_df[features].iloc[i]
    z_diff = (row - overall_means) / feat_std
    flags_str = ', '.join([
        f"BP={'↑' if z_diff['bp_risk_score']>0.5 else '–'}",
        f"BMI={'↑' if z_diff['bmi']>0.5 else '–'}",
        f"CHOL={'↑' if z_diff['cholesterol_mg_dl']>0.5 else '–'}",
        f"GLU={'↑' if z_diff['blood_sugar_mg_dl']>0.5 else '–'}",
        f"SMOKE={'↑' if z_diff['smoking_binary']>0.5 else '–'}",
    ])
    print(f"  Cluster {i}: {cluster_names[i]:8s} ({cluster_counts[i]} 人)  [{flags_str}]")

# ------------------- 8.5 聚类画像补充信息 -------------------

# 性别比例
gender_pct = df.groupby('cluster')['gender'].value_counts(normalize=True).unstack(fill_value=0) * 100
# 吸烟史比例（原始分类）
smoke_pct = df.groupby('cluster')['smoking_history'].value_counts(normalize=True).unstack(fill_value=0) * 100
# 疾病行百分比（在 9.1 交叉表之后使用，此处提前计算）
_cross_tab = pd.crosstab(df['cluster'], df['disease'])
_cross_pct_row = _cross_tab.div(_cross_tab.sum(axis=1), axis=0) * 100

print(f"\n聚类画像补充信息:")
for i in range(optimal_k):
    avg_age = df[df['cluster'] == i]['age'].mean()
    age_label = '青年' if avg_age < 45 else ('中年' if avg_age < 60 else '老年')

    f_pct = gender_pct.loc[i, 'Female'] if 'Female' in gender_pct.columns else 0
    m_pct = gender_pct.loc[i, 'Male'] if 'Male' in gender_pct.columns else 0

    cur_pct = smoke_pct.loc[i, 'Current'] if 'Current' in smoke_pct.columns else 0
    frm_pct = smoke_pct.loc[i, 'Former'] if 'Former' in smoke_pct.columns else 0
    nev_pct = smoke_pct.loc[i, 'Never'] if 'Never' in smoke_pct.columns else 0

    top3 = _cross_pct_row.loc[i].nlargest(3)

    print(f"  Cluster {i} ({cluster_names[i]}, {cluster_counts[i]}人):")
    print(f"    年龄: {avg_age:.1f}岁 ({age_label}), 性别: {f_pct:.1f}%女/{m_pct:.1f}%男")
    print(f"    吸烟: Current {cur_pct:.1f}%, Former {frm_pct:.1f}%, Never {nev_pct:.1f}%")
    print(f"    主导疾病: {top3.index[0]}({top3.values[0]:.1f}%), {top3.index[1]}({top3.values[1]:.1f}%), {top3.index[2]}({top3.values[2]:.1f}%)")

# ------------------- 9. 聚类与疾病交叉分析 -------------------


# 9.1 交叉表（频数）
cross_tab = pd.crosstab(df['cluster'], df['disease'])
cross_tab['合计'] = cross_tab.sum(axis=1)
print(f"\nCluster x Disease 交叉表（频数）:")
print(cross_tab)

# 9.2 行百分比表（每簇的疾病构成）
cross_pct_row = cross_tab.div(cross_tab['合计'], axis=0) * 100
print(f"\n各聚类疾病构成百分比（按行，%）:")
print(cross_pct_row.round(1))

# 9.3 列百分比表（每疾病的聚类分布）
cross_sum_col = cross_tab.drop(columns='合计').sum(axis=0)
cross_pct_col = cross_tab.drop(columns='合计').div(cross_sum_col, axis=1) * 100
print(f"\n各疾病聚类分布百分比（按列，%）:")
print(cross_pct_col.round(1))

# 9.4 各聚类疾病构成堆叠柱状图（频数）
cross_plot = cross_tab.drop(columns='合计')
cross_plot.plot(kind='bar', stacked=True, colormap='tab10', figsize=(12, 7), edgecolor='white')
plt.title(f'各聚类的疾病类型构成（堆叠柱状图, K={optimal_k}）', fontsize=14)
plt.xlabel('聚类', fontsize=12)
plt.ylabel('患者数量', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='疾病类型', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "cluster_disease_stacked_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# 9.5 各疾病聚类分布百分比堆叠柱状图
cross_pct_col.T.plot(kind='bar', stacked=True, colormap='Set2', figsize=(12, 7), edgecolor='white')
plt.title(f'每种疾病的聚类分布（百分比堆叠柱状图, K={optimal_k}）', fontsize=14)
plt.xlabel('疾病类型', fontsize=12)
plt.ylabel('占比（%）', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='聚类', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "disease_cluster_pct_stacked_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# 9.6 疾病与聚类关联热力图（按列百分比）
plt.figure(figsize=(12, 8))
sns.heatmap(cross_pct_col, annot=True, fmt='.1f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': '占比（%）'})
plt.title(f'疾病类型与患者聚类的关联热力图（按疾病列百分比, K={optimal_k}）', fontsize=14, pad=15)
plt.xlabel('疾病类型', fontsize=12)
plt.ylabel('聚类', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "disease_cluster_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ------------------- 10. 高风险聚类疾病分析 -------------------


# 10.1 识别高风险聚类（聚类中心z-score均值最高）
cluster_risk = final_kmeans.cluster_centers_.mean(axis=1)
high_risk_cluster = np.argmax(cluster_risk)
print(f"\n各聚类综合风险（平均z-score）:")
for i in range(optimal_k):
    print(f"  Cluster {i}: {cluster_risk[i]:.4f}")
print(f"\n综合风险最高的聚类: Cluster {high_risk_cluster}（{cluster_names[high_risk_cluster]}）")

# 10.2 高风险聚类Top疾病横向柱状图
high_risk_df = df[df['cluster'] == high_risk_cluster]
high_risk_disease = high_risk_df['disease'].value_counts()
print(f"\n高风险聚类（Cluster {high_risk_cluster}）中疾病分布:")
for disease_name, count in high_risk_disease.items():
    print(f"  {disease_name}: {count} 人 ({count/len(high_risk_df)*100:.1f}%)")

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(high_risk_disease)), high_risk_disease.values, color='salmon', edgecolor='white')
plt.yticks(range(len(high_risk_disease)), high_risk_disease.index, fontsize=10)
plt.title(f'高风险聚类（Cluster {high_risk_cluster}：{cluster_names[high_risk_cluster]}）中的疾病类型分布', fontsize=14)
plt.xlabel('患者数量', fontsize=12)
plt.gca().invert_yaxis()  # 数量多的在上
for bar, count in zip(bars, high_risk_disease.values):
    plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{count}人 ({count/len(high_risk_df)*100:.1f}%)', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "high_risk_cluster_disease.png"), dpi=150, bbox_inches="tight")
plt.close()

# 10.3 高风险聚类 vs 总体疾病分布对比
overall_disease_pct = (df['disease'].value_counts(normalize=True) * 100).round(1)
high_risk_disease_pct = (high_risk_df['disease'].value_counts(normalize=True) * 100).round(1)
compare_df = pd.DataFrame({
    '高风险聚类占比(%)': high_risk_disease_pct,
    '总体占比(%)': overall_disease_pct
})
compare_df['差异(%)'] = (compare_df['高风险聚类占比(%)'] - compare_df['总体占比(%)']).round(1)
# 按差异降序排列
compare_df = compare_df.sort_values('差异(%)', ascending=False)
print(f"\n高风险聚类 vs 总体疾病分布对比:")
print(compare_df)

# ------------------- 11. 聚类画像汇总与结论 -------------------


# 11.1 构建综合画像表
profile_df = centers_df[features].round(2)
profile_df['患者数'] = [cluster_counts[i] for i in range(optimal_k)]
profile_df['患者占比(%)'] = (profile_df['患者数'] / len(df) * 100).round(1)
profile_df['群体命名'] = [cluster_names[i] for i in range(optimal_k)]

# 每簇Top3疾病
top3_list = []
for i in range(optimal_k):
    cluster_diseases = df[df['cluster'] == i]['disease'].value_counts().head(3)
    top3_str = '、'.join([f"{d}({c}人)" for d, c in cluster_diseases.items()])
    top3_list.append(top3_str)
profile_df['主要疾病(Top3)'] = top3_list

# 性别比和吸烟史比例
female_ratios = []
smoker_ratios = []
for i in range(optimal_k):
    f_ratio = gender_pct.loc[i, 'Female'] if 'Female' in gender_pct.columns else 0
    female_ratios.append(round(f_ratio, 1))
    cur = smoke_pct.loc[i, 'Current'] if 'Current' in smoke_pct.columns else 0
    frm = smoke_pct.loc[i, 'Former'] if 'Former' in smoke_pct.columns else 0
    smoker_ratios.append(round(cur + frm, 1))
profile_df['女性占比(%)'] = female_ratios
profile_df['有吸烟史(%)'] = smoker_ratios

print(f"\n聚类画像汇总表（K={optimal_k}）:")
print(profile_df)

# 11.2 打印分析总结
print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)

print(f"""
1. 基于{optimal_k}类患者分群，各群体健康风险特征差异明显，
   能够有效区分慢性病患者内部的不同风险等级。

2. 聚类群体画像:
""")
for i in range(optimal_k):
    pct = profile_df.loc[f'Cluster {i}', '患者占比(%)']
    print(f"   Cluster {i} - {cluster_names[i]}: {cluster_counts[i]}人 ({pct}%)")
    print(f"     主要疾病: {top3_list[i]}")

print(f"""
3. 高风险群体（Cluster {high_risk_cluster} - {cluster_names[high_risk_cluster]}）
   占全体患者的 {profile_df.loc[f'Cluster {high_risk_cluster}', '患者占比(%)']}%，
   该群体多项健康指标高于总体均值，需要重点关注和随访管理。
""")


print("\n脚本运行完成。所有图表和结果已输出。")

