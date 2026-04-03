"""
阶段7.1：同期 vs 滞后DML对比（反向因果检验）
—————————————————————————————————
目的：对T1/T2/T3分别用DML估计同期版本(D_t→Y_t)和滞后版本(D_{t-1}→Y_t)，
      对比ATE方向和大小，论证lagged design缓解反向因果的有效性。
输入：data/preprocess_outputs/data1_modeling_ready.xlsx（同期样本）
      results/03_causal_sample/causal_sample_lagged.xlsx（滞后样本）
输出：results/06_endogeneity/concurrent_vs_lagged_comparison.xlsx
      results/06_endogeneity/concurrent_dml_results.json
      results/06_endogeneity/concurrent_dml_weights.pkl
      results/06_endogeneity/fig_concurrent_vs_lagged.png
      results/10_tables_for_paper/endogeneity_concurrent_vs_lagged.xlsx + .docx
"""

import sys
sys.path.insert(0, 'D:/BaiduSyncdisk/lunwen/model0320/scripts')

import numpy as np
import pandas as pd
import json, pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM

# ── 图像配置 ──
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ── 路径 ──
ROOT = 'D:/BaiduSyncdisk/lunwen/model0320'
DATA_ORIG = f'{ROOT}/data/preprocess_outputs/data1_modeling_ready.xlsx'
DATA_LAG  = f'{ROOT}/results/03_causal_sample/causal_sample_lagged.xlsx'
OUT_DIR   = f'{ROOT}/results/06_endogeneity'
TABLE_DIR = f'{ROOT}/results/10_tables_for_paper'
FIG_DIR   = f'{ROOT}/results/11_figures_for_paper'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── 参数 ──
N_FOLDS = 5
N_REP   = 5
SEED    = 42

def get_ml_reg():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)

def get_ml_cls():
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         use_label_encoder=False, eval_metric='logloss')

# ======================================================================
# 1. 构造同期样本
# ======================================================================
print("=" * 60)
print("7.1  同期 vs 滞后 DML 对比")
print("=" * 60)

print("\n[1/5] 读取原始面板数据...")
df_orig = pd.read_excel(DATA_ORIG)
print(f"  原始数据: {df_orig.shape}")

# 变换
df_orig['hhcperc_log'] = np.log1p(df_orig['hhcperc'])
df_orig['fcamt_log']   = np.log1p(df_orig['fcamt'])

# Province编码
le_prov = LabelEncoder()
df_orig['province_enc'] = le_prov.fit_transform(df_orig['province'].astype(str))

# 构造Treatment
df_orig['T2_chronic'] = (df_orig['chronic_disease_count'] >= 2).astype(int)
df_orig['T3_sleep']   = ((df_orig['sleep'] < 6) | (df_orig['sleep'] > 9)).astype(int)

# W变量（同期版本不含lag_cesd10）
W_COMMON_CONC = ['age', 'gender', 'edu', 'marry', 'rural2', 'province_enc',
                 'hhcperc_log', 'ins', 'pension', 'retire', 'bmi', 'smoken',
                 'drinkev', 'family_size', 'hchild', 'wave']

W_T1_CONC = W_COMMON_CONC + ['chronic_disease_count', 'sleep', 'fcamt_log']
W_T2_CONC = W_COMMON_CONC + ['sleep', 'social_activity_index', 'fcamt_log']
W_T3_CONC = W_COMMON_CONC + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']

# 检查缺失
needed_cols = list(set(['cesd10', 'social_activity_index', 'T2_chronic', 'T3_sleep'] +
                       W_T1_CONC + W_T2_CONC + W_T3_CONC))
df_conc = df_orig[needed_cols].dropna()
print(f"  同期样本（无缺失）: {df_conc.shape[0]} 观测")

# ======================================================================
# 2. 读取滞后样本（已有DML结果直接读取）
# ======================================================================
print("\n[2/5] 读取滞后样本DML结果...")
dml_results_path = f'{ROOT}/results/04_dml/dml_results.json'
with open(dml_results_path, 'r', encoding='utf-8') as f:
    lagged_results = json.load(f)

# 提取滞后版本的ATE
lag_summary = {}
key_map = {'T1_xgb': 'T1_social', 'T2_xgb': 'T2_chronic', 'T3_xgb': 'T3_sleep'}
for key, tname in key_map.items():
    r = lagged_results[key]
    if 'ATE' in r:
        lag_summary[tname] = {
            'estimate': r['ATE'],
            'se': r['ATE_se'],
            'ci_lower': r['ATE_ci_lower'],
            'ci_upper': r['ATE_ci_upper'],
            'pvalue': r['ATE_pvalue']
        }
    elif 'theta' in r:
        lag_summary[tname] = {
            'estimate': r['theta'],
            'se': r['se'],
            'ci_lower': r['ci_lower'],
            'ci_upper': r['ci_upper'],
            'pvalue': r['pvalue']
        }

for k, v in lag_summary.items():
    print(f"  滞后 {k}: ATE={v['estimate']:.4f}, SE={v['se']:.4f}, p={v['pvalue']:.4f}")

# ======================================================================
# 3. 运行同期DML
# ======================================================================
print("\n[3/5] 运行同期DML...")
conc_results = {}
conc_models = {}

# --- T1: 社会参与 (PLR, 连续treatment) ---
print("\n  --- T1: 社会参与 (PLR, 同期) ---")
dml_data_t1 = DoubleMLData.from_arrays(
    y=df_conc['cesd10'].values,
    d=df_conc['social_activity_index'].values,
    x=df_conc[W_T1_CONC].values
)
model_t1 = DoubleMLPLR(dml_data_t1, ml_l=get_ml_reg(), ml_m=get_ml_reg(),
                       n_folds=N_FOLDS, n_rep=N_REP, score='partialling out')
model_t1.fit()
conc_results['T1_social'] = {
    'estimate': model_t1.coef[0],
    'se': model_t1.se[0],
    'ci_lower': model_t1.confint().iloc[0, 0],
    'ci_upper': model_t1.confint().iloc[0, 1],
    'pvalue': model_t1.pval[0]
}
conc_models['T1'] = model_t1
print(f"    θ₀ = {model_t1.coef[0]:.4f} (SE={model_t1.se[0]:.4f}, p={model_t1.pval[0]:.4f})")

# --- T2: 多病共存 (IRM, 二元treatment) ---
print("\n  --- T2: 多病共存 (IRM, 同期) ---")
dml_data_t2 = DoubleMLData.from_arrays(
    y=df_conc['cesd10'].values,
    d=df_conc['T2_chronic'].values,
    x=df_conc[W_T2_CONC].values
)
model_t2 = DoubleMLIRM(dml_data_t2, ml_g=get_ml_reg(), ml_m=get_ml_cls(),
                       n_folds=N_FOLDS, n_rep=N_REP, score='ATE',
                       trimming_threshold=0.01)
model_t2.fit()
conc_results['T2_chronic'] = {
    'estimate': model_t2.coef[0],
    'se': model_t2.se[0],
    'ci_lower': model_t2.confint().iloc[0, 0],
    'ci_upper': model_t2.confint().iloc[0, 1],
    'pvalue': model_t2.pval[0]
}
conc_models['T2'] = model_t2
print(f"    ATE = {model_t2.coef[0]:.4f} (SE={model_t2.se[0]:.4f}, p={model_t2.pval[0]:.4f})")

# --- T3: 睡眠异常 (IRM, 二元treatment) ---
print("\n  --- T3: 睡眠异常 (IRM, 同期) ---")
dml_data_t3 = DoubleMLData.from_arrays(
    y=df_conc['cesd10'].values,
    d=df_conc['T3_sleep'].values,
    x=df_conc[W_T3_CONC].values
)
model_t3 = DoubleMLIRM(dml_data_t3, ml_g=get_ml_reg(), ml_m=get_ml_cls(),
                       n_folds=N_FOLDS, n_rep=N_REP, score='ATE',
                       trimming_threshold=0.01)
model_t3.fit()
conc_results['T3_sleep'] = {
    'estimate': model_t3.coef[0],
    'se': model_t3.se[0],
    'ci_lower': model_t3.confint().iloc[0, 0],
    'ci_upper': model_t3.confint().iloc[0, 1],
    'pvalue': model_t3.pval[0]
}
conc_models['T3'] = model_t3
print(f"    ATE = {model_t3.coef[0]:.4f} (SE={model_t3.se[0]:.4f}, p={model_t3.pval[0]:.4f})")

# ======================================================================
# 4. 整合对比表
# ======================================================================
print("\n[4/5] 整合对比表...")

rows = []
treat_labels = {
    'T1_social': 'T1: 社会参与 (连续)',
    'T2_chronic': 'T2: 多病共存≥2 (二元)',
    'T3_sleep':  'T3: 睡眠异常 (二元)'
}

for tkey in ['T1_social', 'T2_chronic', 'T3_sleep']:
    c = conc_results[tkey]
    l = lag_summary[tkey]

    # 判断膨胀/缩小
    abs_conc = abs(c['estimate'])
    abs_lag  = abs(l['estimate'])
    if abs_lag > 0:
        change_pct = (abs_conc - abs_lag) / abs_lag * 100
    else:
        change_pct = np.nan

    rows.append({
        'Treatment': treat_labels[tkey],
        '同期ATE': round(c['estimate'], 4),
        '同期SE': round(c['se'], 4),
        '同期95%CI': f"[{c['ci_lower']:.3f}, {c['ci_upper']:.3f}]",
        '同期p值': round(c['pvalue'], 4),
        '滞后ATE': round(l['estimate'], 4),
        '滞后SE': round(l['se'], 4),
        '滞后95%CI': f"[{l['ci_lower']:.3f}, {l['ci_upper']:.3f}]",
        '滞后p值': round(l['pvalue'], 4),
        '|同期|-|滞后|': round(abs_conc - abs_lag, 4),
        '变化%': round(change_pct, 1),
        '方向一致': '是' if (c['estimate'] * l['estimate'] > 0) else '否'
    })

df_comp = pd.DataFrame(rows)

# 保存xlsx
df_comp.to_excel(f'{OUT_DIR}/concurrent_vs_lagged_comparison.xlsx', index=False)
print(f"  保存: {OUT_DIR}/concurrent_vs_lagged_comparison.xlsx")

# 论文用表格
df_comp.to_excel(f'{TABLE_DIR}/endogeneity_concurrent_vs_lagged.xlsx', index=False)

# docx三线表
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()
doc.add_heading('表：同期设计与滞后设计的DML估计对比', level=2)

# 简化版表格
table_data = []
for tkey in ['T1_social', 'T2_chronic', 'T3_sleep']:
    c = conc_results[tkey]
    l = lag_summary[tkey]

    def stars(p):
        if p < 0.01: return '***'
        if p < 0.05: return '**'
        if p < 0.1:  return '*'
        return ''

    table_data.append({
        'Treatment': treat_labels[tkey],
        '同期ATE': f"{c['estimate']:.3f}{stars(c['pvalue'])}",
        '同期SE': f"({c['se']:.3f})",
        '滞后ATE': f"{l['estimate']:.3f}{stars(l['pvalue'])}",
        '滞后SE': f"({l['se']:.3f})",
        '|Δ|/|滞后|': f"{abs(abs(c['estimate'])-abs(l['estimate']))/abs(l['estimate'])*100:.1f}%"
    })

cols = ['Treatment', '同期ATE', '同期SE', '滞后ATE', '滞后SE', '|Δ|/|滞后|']
tbl = doc.add_table(rows=1+len(table_data), cols=len(cols))
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

for j, col in enumerate(cols):
    tbl.rows[0].cells[j].text = col
for i, row_data in enumerate(table_data):
    for j, col in enumerate(cols):
        tbl.rows[i+1].cells[j].text = str(row_data[col])

doc.add_paragraph('注：*** p<0.01, ** p<0.05, * p<0.1。同期设计：D_t→Y_t；滞后设计：D_{t-1}→Y_t。'
                  '同期样本N=' + str(df_conc.shape[0]) + '，滞后样本N=24616。')

doc.save(f'{TABLE_DIR}/endogeneity_concurrent_vs_lagged.docx')
print(f"  保存: {TABLE_DIR}/endogeneity_concurrent_vs_lagged.docx")

# ======================================================================
# 5. 可视化
# ======================================================================
print("\n[5/5] 绘制对比图...")

fig, ax = plt.subplots(figsize=(10, 5))

treat_names = ['T1: 社会参与', 'T2: 多病共存', 'T3: 睡眠异常']
x = np.arange(len(treat_names))
width = 0.35

conc_vals = [conc_results[k]['estimate'] for k in ['T1_social', 'T2_chronic', 'T3_sleep']]
conc_errs = [1.96 * conc_results[k]['se'] for k in ['T1_social', 'T2_chronic', 'T3_sleep']]
lag_vals  = [lag_summary[k]['estimate'] for k in ['T1_social', 'T2_chronic', 'T3_sleep']]
lag_errs  = [1.96 * lag_summary[k]['se'] for k in ['T1_social', 'T2_chronic', 'T3_sleep']]

bars1 = ax.bar(x - width/2, conc_vals, width, yerr=conc_errs, capsize=5,
               label='同期设计 (D_t→Y_t)', color='#D4573B', alpha=0.8)
bars2 = ax.bar(x + width/2, lag_vals, width, yerr=lag_errs, capsize=5,
               label='滞后设计 (D_{t-1}→Y_t)', color='#2C5F8A', alpha=0.8)

ax.set_ylabel('因果效应估计 (ATE / θ₀)')
ax.set_title('同期设计 vs 滞后设计：DML因果效应估计对比')
ax.set_xticks(x)
ax.set_xticklabels(treat_names)
ax.legend(loc='upper left')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# 标注数值
for bar, val in zip(bars1, conc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * np.sign(bar.get_height()),
            f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
for bar, val in zip(bars2, lag_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * np.sign(bar.get_height()),
            f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig_concurrent_vs_lagged.png')
fig.savefig(f'{FIG_DIR}/endogeneity_concurrent_vs_lagged.png')
plt.close()
print(f"  保存: fig_concurrent_vs_lagged.png")

# ======================================================================
# 6. 保存模型权重和JSON
# ======================================================================
# JSON
conc_json = {}
for k, v in conc_results.items():
    conc_json[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()}
with open(f'{OUT_DIR}/concurrent_dml_results.json', 'w', encoding='utf-8') as f:
    json.dump(conc_json, f, ensure_ascii=False, indent=2)

# Pickle模型权重
with open(f'{OUT_DIR}/concurrent_dml_weights.pkl', 'wb') as f:
    pickle.dump(conc_models, f)
print(f"  保存: concurrent_dml_results.json, concurrent_dml_weights.pkl")

print("\n✅ 7.1 同期 vs 滞后对比完成！")
print("\n结果摘要：")
for tkey in ['T1_social', 'T2_chronic', 'T3_sleep']:
    c = conc_results[tkey]
    l = lag_summary[tkey]
    print(f"  {treat_labels[tkey]}:")
    print(f"    同期: {c['estimate']:.4f} (p={c['pvalue']:.4f})")
    print(f"    滞后: {l['estimate']:.4f} (p={l['pvalue']:.4f})")
    print(f"    |同期|/|滞后| = {abs(c['estimate'])/abs(l['estimate'])*100:.1f}%")
