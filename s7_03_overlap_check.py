"""
阶段7.3：Overlap检验（重叠性假设验证）
—————————————————————————————————
目的：对T2/T3的二元treatment检验propensity score分布的重叠性，
      验证DML条件独立假设的overlap前提。
输入：results/03_causal_sample/causal_sample_lagged.xlsx
输出：results/06_endogeneity/overlap_statistics.xlsx
      results/06_endogeneity/overlap_propensity_T2.png
      results/06_endogeneity/overlap_propensity_T3.png
      results/06_endogeneity/overlap_detailed_stats.xlsx
"""

import sys
sys.path.insert(0, 'D:/BaiduSyncdisk/lunwen/model0320/scripts')

import numpy as np
import pandas as pd
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier

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
DATA_LAG  = f'{ROOT}/results/03_causal_sample/causal_sample_lagged.xlsx'
OUT_DIR   = f'{ROOT}/results/06_endogeneity'
TABLE_DIR = f'{ROOT}/results/10_tables_for_paper'
FIG_DIR   = f'{ROOT}/results/11_figures_for_paper'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

# ======================================================================
# 1. 读取因果样本
# ======================================================================
print("=" * 60)
print("7.3  Overlap检验（重叠性假设验证）")
print("=" * 60)

print("\n[1/4] 读取因果样本...")
df = pd.read_excel(DATA_LAG)
print(f"  因果样本: {df.shape}")

# Province编码
le_prov = LabelEncoder()
df['province_enc'] = le_prov.fit_transform(df['province'].astype(str))

# W变量（与DML一致）
W_COMMON = ['age', 'gender', 'edu', 'marry', 'rural2', 'province_enc',
            'hhcperc_log', 'ins', 'pension', 'retire', 'bmi', 'smoken',
            'drinkev', 'family_size', 'hchild', 'lag_cesd10', 'wave']

W_T2 = W_COMMON + ['sleep', 'social_activity_index', 'fcamt_log']
W_T3 = W_COMMON + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']

# ======================================================================
# 2. 估计Propensity Score
# ======================================================================
print("\n[2/4] 估计Propensity Score...")

treatments = {
    'T2_chronic': {'label': 'T2: 多病共存(≥2种)', 'W': W_T2, 'col': 'T2_chronic'},
    'T3_sleep':   {'label': 'T3: 睡眠异常', 'W': W_T3, 'col': 'T3_sleep'}
}

ps_results = {}
ps_all = {}

for tkey, tinfo in treatments.items():
    print(f"\n  --- {tinfo['label']} ---")
    D = df[tinfo['col']].values
    X = df[tinfo['W']].values

    # 5-fold CV propensity score
    clf = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0,
                        use_label_encoder=False, eval_metric='logloss')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ps = cross_val_predict(clf, X, D, cv=cv, method='predict_proba')[:, 1]

    ps_all[tkey] = ps

    # 分组统计
    ps_treated = ps[D == 1]
    ps_control = ps[D == 0]

    # Trimming分析
    trim_thresholds = [0.01, 0.02, 0.05, 0.10]
    trim_stats = {}
    for th in trim_thresholds:
        trimmed = ((ps < th) | (ps > 1 - th)).sum()
        trim_stats[f'trim_{th}'] = {
            'trimmed_n': int(trimmed),
            'trimmed_pct': round(trimmed / len(ps) * 100, 2)
        }

    result = {
        'treatment': tinfo['label'],
        'N_total': len(ps),
        'N_treated': int(D.sum()),
        'N_control': int((1-D).sum()),
        'treatment_rate': round(D.mean() * 100, 1),
        'ps_mean_all': round(ps.mean(), 4),
        'ps_std_all': round(ps.std(), 4),
        'ps_min': round(ps.min(), 4),
        'ps_max': round(ps.max(), 4),
        'ps_mean_treated': round(ps_treated.mean(), 4),
        'ps_mean_control': round(ps_control.mean(), 4),
        'ps_median_treated': round(np.median(ps_treated), 4),
        'ps_median_control': round(np.median(ps_control), 4),
        'ps_p5': round(np.percentile(ps, 5), 4),
        'ps_p95': round(np.percentile(ps, 95), 4),
        'overlap_region': f"[{max(ps_treated.min(), ps_control.min()):.4f}, "
                          f"{min(ps_treated.max(), ps_control.max()):.4f}]",
        'in_overlap_pct': round(((ps >= max(ps_treated.min(), ps_control.min())) &
                                  (ps <= min(ps_treated.max(), ps_control.max()))).mean() * 100, 1),
    }
    # 加入trimming统计
    for th in trim_thresholds:
        result[f'trim_{th}_n'] = trim_stats[f'trim_{th}']['trimmed_n']
        result[f'trim_{th}_pct'] = trim_stats[f'trim_{th}']['trimmed_pct']

    ps_results[tkey] = result

    print(f"    N: {result['N_total']} (处理={result['N_treated']}, "
          f"对照={result['N_control']}, 处理率={result['treatment_rate']}%)")
    print(f"    PS范围: [{result['ps_min']}, {result['ps_max']}]")
    print(f"    PS均值: 处理组={result['ps_mean_treated']}, 对照组={result['ps_mean_control']}")
    print(f"    重叠区域: {result['overlap_region']} ({result['in_overlap_pct']}%样本)")
    print(f"    Trimming (0.02): {trim_stats['trim_0.02']['trimmed_n']}条 "
          f"({trim_stats['trim_0.02']['trimmed_pct']}%)")

# ======================================================================
# 3. 可视化
# ======================================================================
print("\n[3/4] 绘制Propensity Score分布图...")

for tkey, tinfo in treatments.items():
    ps = ps_all[tkey]
    D = df[tinfo['col']].values
    r = ps_results[tkey]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a. 重叠直方图
    ax = axes[0]
    ax.hist(ps[D == 0], bins=50, alpha=0.6, label=f'对照组 (N={r["N_control"]})',
            color='#2C5F8A', density=True, edgecolor='white')
    ax.hist(ps[D == 1], bins=50, alpha=0.6, label=f'处理组 (N={r["N_treated"]})',
            color='#D4573B', density=True, edgecolor='white')
    ax.axvline(x=0.02, color='gray', linestyle='--', linewidth=0.8, label='Trim=0.02')
    ax.axvline(x=0.98, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('密度')
    ax.set_title(f'{tinfo["label"]}: Propensity Score分布')
    ax.legend(fontsize=9)

    # 3b. 核密度图
    ax = axes[1]
    from scipy.stats import gaussian_kde
    ps0 = ps[D == 0]
    ps1 = ps[D == 1]
    x_range = np.linspace(0, 1, 500)
    kde0 = gaussian_kde(ps0)
    kde1 = gaussian_kde(ps1)
    ax.fill_between(x_range, kde0(x_range), alpha=0.4, color='#2C5F8A', label='对照组')
    ax.fill_between(x_range, kde1(x_range), alpha=0.4, color='#D4573B', label='处理组')
    ax.plot(x_range, kde0(x_range), color='#2C5F8A', linewidth=1.5)
    ax.plot(x_range, kde1(x_range), color='#D4573B', linewidth=1.5)
    ax.axvline(x=0.02, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.98, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('密度')
    ax.set_title(f'{tinfo["label"]}: 核密度估计')
    ax.legend(fontsize=9)

    # 添加统计文本
    text = (f"PS range: [{r['ps_min']:.3f}, {r['ps_max']:.3f}]\n"
            f"Overlap: {r['in_overlap_pct']}%\n"
            f"Trim(0.02): {r[f'trim_0.02_n']}条 ({r[f'trim_0.02_pct']}%)")
    ax.text(0.98, 0.95, text, transform=ax.transAxes, fontsize=8,
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/overlap_propensity_{tkey}.png')
    fig.savefig(f'{FIG_DIR}/overlap_propensity_{tkey}.png')
    plt.close()
    print(f"  保存: overlap_propensity_{tkey}.png")

# ======================================================================
# 4. 保存统计表
# ======================================================================
print("\n[4/4] 保存统计表...")

# 详细统计
df_stats = pd.DataFrame(ps_results).T
df_stats.to_excel(f'{OUT_DIR}/overlap_statistics.xlsx')
df_stats.to_excel(f'{TABLE_DIR}/endogeneity_overlap.xlsx')

# 简洁版用于论文
summary_rows = []
for tkey in ['T2_chronic', 'T3_sleep']:
    r = ps_results[tkey]
    summary_rows.append({
        'Treatment': r['treatment'],
        '样本量': r['N_total'],
        '处理组N': r['N_treated'],
        '对照组N': r['N_control'],
        '处理率(%)': r['treatment_rate'],
        'PS最小值': r['ps_min'],
        'PS最大值': r['ps_max'],
        'PS均值(处理)': r['ps_mean_treated'],
        'PS均值(对照)': r['ps_mean_control'],
        '重叠区域': r['overlap_region'],
        '重叠率(%)': r['in_overlap_pct'],
        'Trim(0.02)裁剪数': r['trim_0.02_n'],
        'Trim(0.02)裁剪率(%)': r['trim_0.02_pct']
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_excel(f'{OUT_DIR}/overlap_detailed_stats.xlsx', index=False)

# docx
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = Document()
doc.add_heading('表：Propensity Score重叠性检验统计', level=2)

cols = ['Treatment', '样本量', '处理率(%)', 'PS最小值', 'PS最大值',
        'PS均值(处理)', 'PS均值(对照)', '重叠率(%)', 'Trim裁剪率(%)']
tbl = doc.add_table(rows=1+len(summary_rows), cols=len(cols))
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

for j, col in enumerate(cols):
    tbl.rows[0].cells[j].text = col
for i, row_data in enumerate(summary_rows):
    for j, col in enumerate(cols):
        val = row_data.get(col, row_data.get('Trim(0.02)裁剪率(%)', ''))
        if col == 'Trim裁剪率(%)':
            val = row_data['Trim(0.02)裁剪率(%)']
        tbl.rows[i+1].cells[j].text = str(val)

doc.add_paragraph('注：Propensity Score通过5折交叉验证的XGBoost分类器估计。'
                  'Trim阈值为0.02，即裁剪P(D=1|W)<0.02或P(D=1|W)>0.98的观测。'
                  '重叠率=处理组和对照组PS分布重叠区域内的样本比例。')
doc.save(f'{TABLE_DIR}/endogeneity_overlap.docx')

# 保存propensity scores到pkl
import pickle
with open(f'{OUT_DIR}/overlap_propensity_scores.pkl', 'wb') as f:
    pickle.dump(ps_all, f)

print(f"  保存: overlap_statistics.xlsx, overlap_detailed_stats.xlsx, endogeneity_overlap.docx")

print("\n✅ 7.3 Overlap检验完成！")
print("\n结果摘要：")
for tkey in ['T2_chronic', 'T3_sleep']:
    r = ps_results[tkey]
    print(f"  {r['treatment']}:")
    print(f"    PS范围: [{r['ps_min']}, {r['ps_max']}]")
    print(f"    重叠率: {r['in_overlap_pct']}%")
    print(f"    Trim(0.02)裁剪: {r['trim_0.02_n']}条 ({r['trim_0.02_pct']}%)")
    overlap_ok = "✅ 良好" if r['in_overlap_pct'] > 90 else "⚠️ 需注意"
    print(f"    重叠性评估: {overlap_ok}")
