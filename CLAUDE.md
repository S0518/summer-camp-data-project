# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

慢性病患者数据探索分析 — exploratory data analysis of chronic disease patient records. Input is `data/chronic_patients.csv` (UUID patient IDs, demographics, vitals, disease labels, visit dates). All output goes to `results/`.

## Data Pipeline (must run in order)

1. **`数据质量检查/task1_quality_check.py`** — Reads `data/chronic_patients.csv`, performs missing-value stats, duplicate detection, IQR + normal-range outlier detection, and blood-pressure logic validation (systolic ≥ diastolic). Deletes anomalous rows and writes the cleaned dataset to **`data/change.csv`**. All downstream scripts depend on this output.

2. **`患者基本结构/task2.py`** — Patient demographics: gender pie/bar charts, smoking history, disease type distribution, visit date trends (monthly line + bar).

3. **`核心健康指标分布/task3.py`** — Histograms + KDE for age, height, weight, BMI, systolic/diastolic BP, cholesterol, blood sugar.

4. **`衍生风险等级/task4.py`** — Derived risk tiers: BMI classification (偏瘦/正常/超重/肥胖), BP risk scoring (0–3), smoking risk, glucose level, cholesterol level.

5. **`异常值与医学逻辑检查/task5.py`** — IQR-based box plots for all numeric indicators, BP scatter plot with logic validation.

6. **`指标关系分析/task6.py`** — Numeric correlation heatmap, cholesterol/glucose vs BP risk box plots, age vs BP risk scatter.

Each task script saves charts (`.png`, 300 DPI) and statistical tables (`.csv`) under `results/<category>/`. Category names match the script directory names.

## Standalone Analysis Scripts (root level)

- **`chronic_patients_kmeans_analysis.py`** — K-Means clustering on 6 features (age, BMI, bp_risk_score, cholesterol, blood_sugar, smoking_binary). Evaluates K=2–10 via SSE/CH/DBI, uses Kneedle algorithm for elbow detection, PCA 2D visualization, and per-cluster disease profiling. Outputs to `results/聚类/`.

- **`age_group_analysis.py`** — Age-stratified analysis (青中年<40 / 中年40-59 / 老年60-74 / 高龄≥75). Stacked bar charts for disease/BMI/BP/glucose/cholesterol/smoking by age group, plus chi-squared and Kruskal-Wallis tests. Outputs to `results/优化补充/`.

- **`task3_supervised_pipeline.py`** — Supervised ML pipeline orchestrator. Embeds three sub-scripts as raw strings (`PREPARE_SOURCE`, `TRAIN_SOURCE`, `ANALYSIS_SOURCE`), loads them dynamically via `exec()`, and runs: data preparation → 10-class disease training → 4-class disease-group training → analysis figures. All outputs go under `results/监督学习/`. Accepts CLI flags for skipping steps (`--skip-prepare`, `--train-only`, `--analysis-only`, etc.).

## Common Patterns Across All Scripts

### Hardcoded Path
Nearly every script has `PROJECT_DIR = Path(r"D:\Programming\Data\Pycharm_data\数据探索")`. When moving the project or running elsewhere, this must be updated across all scripts.

### Matplotlib Configuration
```python
matplotlib.use("Agg")  # Non-interactive backend
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
```

### Safe CSV Reader
Most scripts define a `read_csv_safely()` function that tries encodings `["utf-8-sig", "utf-8", "gbk", "gb18030"]` in order.

### Output Convention
- Charts: `plt.savefig(path, dpi=150, bbox_inches="tight")` followed by `plt.close()`
- Tables: `df.to_csv(path, index=False, encoding="utf-8-sig")`
- Summary tables: `00_*` prefix in each results subdirectory

## Key Dependencies

pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

## Data Schema (`data/change.csv`)

| Column | Description |
|---|---|
| `patient_id` | UUID |
| `age` | integer |
| `gender` | Female / Male |
| `smoking_history` | Never / Former / Current |
| `disease` | 10 chronic disease types (Chinese labels after mapping) |
| `height_cm`, `weight_kg` | float |
| `bmi` | float (recomputed from height/weight) |
| `bp_systolic`, `bp_diastolic` | integer (mmHg) |
| `cholesterol_mg_dl` | integer (mg/dL) |
| `blood_sugar_mg_dl` | integer (mg/dL) |
| `visit_date` | YYYY-MM-DD |

## Running a Single Script

```bash
python "数据质量检查/task1_quality_check.py"
```

For the supervised pipeline with step control:
```bash
python task3_supervised_pipeline.py --skip-prepare --train-only
python task3_supervised_pipeline.py --analysis-only
python task3_supervised_pipeline.py --project-root "C:\Users\sg\Desktop\数据探索\数据探索"
```

No test suite, linting, or build system exists in this project. Scripts are run directly and write results to disk.
