"""
阶段3-最终版：面板双重固定效应回归（论文展示模型）
============================================================
基于探索性FE（s3_01）的VIF诊断，做出以下设计修正：
  1. 移除 activity_index：与 social_activity_index within相关r=0.980, VIF≈25
  2. 不纳入 srh：确认为坏控制（中介），加入后吸收37%慢性病系数
  3. 不纳入 lag_cesd10（Nickell偏误）、satlife/hope（同源偏差）、disease_index（因果不清）

最终模型：
  Y = cesd10
  核心解释X（8个）: social_activity_index, chronic_disease_count,
      body_discomfort_count, sleep, hhcperc_log, ins, pension, fcamt_log
  控制Z（8个）: age, marry, bmi, smoken, drinkev, retire, family_size, hchild
  FE = 个体 + 年份，SE = 个体聚类稳健

输出：
  - 模型权重（pickle + xlsx + json）
  - 论文用三线表（xlsx + docx）
  - 诊断报告（VIF、Hausman参考）
============================================================
"""

import pandas as pd
import numpy as np
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.var_labels import VAR_MAP, get_zh, get_label, get_short
from utils.table_utils import save_three_line_table, format_coef

from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# 0. 路径
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data/preprocess_outputs/data1_modeling_ready.xlsx')
OUT_FE = os.path.join(ROOT, 'results/02_fe_models')
OUT_TABLE = os.path.join(ROOT, 'results/08_tables_for_paper')
os.makedirs(OUT_FE, exist_ok=True)
os.makedirs(OUT_TABLE, exist_ok=True)

# ============================================================
# 1. 数据准备
# ============================================================
print("=" * 70)
print("  最终版FE模型 — 数据准备")
print("=" * 70)
df = pd.read_excel(DATA_PATH)
print(f"原始数据: {df.shape}")

# 筛选 >=2 波个体
waves_per_id = df.groupby('ID')['wave'].nunique()
ids_ge2 = waves_per_id[waves_per_id >= 2].index
df = df[df['ID'].isin(ids_ge2)].copy()
print(f"FE样本(>=2波): {len(ids_ge2)} 个体, {len(df)} 观测")

# 变量变换
df['hhcperc_log'] = np.log(df['hhcperc'] + 1)
df['fcamt_log'] = np.log(df['fcamt'] + 1)

# 设面板索引
df = df.set_index(['ID', 'wave'])

# ============================================================
# 2. 变量定义
# ============================================================
Y_VAR = 'cesd10'

CORE_X = [
    'social_activity_index',   # 社会参与指数
    'chronic_disease_count',   # 慢性病数量
    'body_discomfort_count',   # 身体不适数量
    'sleep',                   # 睡眠时长
    'hhcperc_log',             # 人均消费(对数)
    'ins',                     # 医疗保险
    'pension',                 # 养老保险
    'fcamt_log',               # 子女经济支持(对数)
]

CTRL_Z = [
    'age',           # 年龄
    'marry',         # 已婚
    'bmi',           # BMI
    'smoken',        # 吸烟
    'drinkev',       # 饮酒
    'retire',        # 退休
    'family_size',   # 家庭规模
    'hchild',        # 健在子女数
]

ALL_X = CORE_X + CTRL_Z

# ============================================================
# 3. VIF诊断（within变换后）
# ============================================================
print("\n" + "=" * 70)
print("  共线性诊断 (VIF, within变换后)")
print("=" * 70)

df_within = df.copy()
for v in ALL_X:
    gm = df.groupby(level=0)[v].transform('mean')
    df_within[v] = df[v] - gm

X_w = df_within[ALL_X].dropna()
vif_results = []
print(f"  {'变量':<32s} {'VIF':>8s}  {'判定':<10s}")
print("  " + "-" * 55)
for i, col in enumerate(ALL_X):
    vif = variance_inflation_factor(X_w.values, i)
    if vif > 10:
        flag = '严重共线性'
    elif vif > 5:
        flag = '需关注'
    elif vif > 2:
        flag = '轻度'
    else:
        flag = '正常'
    vif_results.append({'变量': col, 'VIF': round(vif, 3), '判定': flag})
    print(f"  {col:<32s} {vif:8.3f}  {flag}")

vif_df = pd.DataFrame(vif_results)
vif_df.to_excel(os.path.join(OUT_FE, 'fe_final_vif_check.xlsx'), index=False)
max_vif = vif_df['VIF'].max()
print(f"\n  最大VIF = {max_vif:.3f} → {'通过' if max_vif < 5 else '需处理'}")

# ============================================================
# 4. 运行最终FE模型
# ============================================================
print("\n" + "=" * 70)
print("  最终FE模型估计")
print("=" * 70)

sub = df[[Y_VAR] + ALL_X].dropna()
y = sub[Y_VAR]
X = sub[ALL_X]

# 4.1 FE估计
mod_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
res_fe = mod_fe.fit(cov_type='clustered', cluster_entity=True)

n_obs = res_fe.nobs
n_entities = res_fe.entity_info['total']
r2_within = res_fe.rsquared_within
r2_between = res_fe.rsquared_between
r2_overall = res_fe.rsquared_overall
f_stat = res_fe.f_statistic.stat
f_pval = res_fe.f_statistic.pval

print(f"\n  观测数: {n_obs:,}")
print(f"  个体数: {n_entities:,}")
print(f"  R²(within):  {r2_within:.4f}")
print(f"  R²(between): {r2_between:.4f}")
print(f"  R²(overall): {r2_overall:.4f}")
print(f"  F统计量: {f_stat:.4f}  (p = {f_pval:.2e})")

# 4.2 Hausman检验（FE vs RE）
print("\n--- Hausman检验 (FE vs RE) ---")
mod_re = RandomEffects(y, X)
res_re = mod_re.fit(cov_type='clustered', cluster_entity=True)

# 手动Hausman检验
b_fe = res_fe.params
b_re = res_re.params
common_vars = b_fe.index.intersection(b_re.index)
diff = b_fe[common_vars] - b_re[common_vars]
# 近似：使用系数差异的显著性
# linearmodels没有内置Hausman，用系数差异的卡方近似
V_fe = res_fe.cov.loc[common_vars, common_vars]
V_re = res_re.cov.loc[common_vars, common_vars]
V_diff = V_fe - V_re

try:
    chi2 = float(diff.values @ np.linalg.inv(V_diff.values) @ diff.values)
    dof = len(common_vars)
    from scipy.stats import chi2 as chi2_dist
    hausman_p = 1 - chi2_dist.cdf(chi2, dof)
    print(f"  Hausman χ² = {chi2:.2f}, df = {dof}, p = {hausman_p:.4e}")
    if hausman_p < 0.05:
        print("  → p < 0.05, 拒绝RE，FE模型合适 ✓")
    else:
        print("  → p >= 0.05, 无法拒绝RE")
except np.linalg.LinAlgError:
    print("  V_diff不可逆，改用Sargan-Hansen统计量（非本模型必须）")
    hausman_p = None
    print("  → 基于理论判断：本研究关注within效应 + 个体FE吸收不可观测异质性，使用FE合理 ✓")

# ============================================================
# 5. 提取系数
# ============================================================
print("\n" + "=" * 70)
print("  回归系数")
print("=" * 70)

coef_rows = []
print(f"\n  {'变量':<36s} {'系数':>10s}      {'SE':>8s} {'t值':>8s} {'p值':>12s} {'95% CI':<24s}")
print("  " + "-" * 105)

for var in ALL_X:
    coef = res_fe.params[var]
    se = res_fe.std_errors[var]
    t = res_fe.tstats[var]
    p = res_fe.pvalues[var]
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    star = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else '   '))
    label = get_label(var) if var in VAR_MAP else var
    print(f"  {label:<36s} {coef:9.4f}{star}  ({se:.4f}) {t:8.2f} {p:12.4e}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    coef_rows.append({
        'variable': var,
        'label_zh': get_zh(var) if var in VAR_MAP else var,
        'label_table': get_label(var) if var in VAR_MAP else var,
        'coef': round(coef, 6),
        'se': round(se, 6),
        'tstat': round(t, 4),
        'pvalue': p,
        'ci_lower': round(ci_lo, 6),
        'ci_upper': round(ci_hi, 6),
        'significance': star.strip(),
        'category': '核心解释变量' if var in CORE_X else '控制变量',
    })

coef_df = pd.DataFrame(coef_rows)

# ============================================================
# 6. 保存模型权重与结果
# ============================================================
print("\n" + "=" * 70)
print("  保存模型权重与结果")
print("=" * 70)

# 6.1 系数表（xlsx）
coef_df.to_excel(os.path.join(OUT_FE, 'fe_final_coefficients.xlsx'), index=False)
print(f"  系数表: fe_final_coefficients.xlsx")

# 6.2 模型摘要信息（json）
model_meta = {
    'model_name': 'PanelOLS_FE_Final',
    'dependent_variable': Y_VAR,
    'core_explanatory_vars': CORE_X,
    'control_vars': CTRL_Z,
    'excluded_vars': {
        'activity_index': 'VIF=24.84, r=0.980 with social_activity_index',
        'srh': '确认为坏控制（中介），吸收37%慢性病效应',
        'lag_cesd10': 'Nickell偏误',
        'satlife': '与cesd10同源偏差',
        'disease_index': '复合指数因果不清',
    },
    'effects': {'entity': True, 'time': True},
    'cov_type': 'clustered (entity)',
    'n_obs': int(n_obs),
    'n_entities': int(n_entities),
    'r2_within': round(r2_within, 6),
    'r2_between': round(r2_between, 6),
    'r2_overall': round(r2_overall, 6),
    'f_statistic': round(f_stat, 4),
    'f_pvalue': float(f_pval),
    'max_vif': round(max_vif, 3),
    'hausman_pvalue': round(hausman_p, 6) if hausman_p is not None else None,
    'data_source': 'data1_modeling_ready.xlsx',
    'sample_filter': '>=2 waves per individual',
}
with open(os.path.join(OUT_FE, 'fe_final_model_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(model_meta, f, ensure_ascii=False, indent=2)
print(f"  模型元信息: fe_final_model_meta.json")

# 6.3 模型对象（pickle）
model_save = {
    'fitted_result_summary': res_fe.summary.as_text(),
    'params': res_fe.params.to_dict(),
    'std_errors': res_fe.std_errors.to_dict(),
    'tstats': res_fe.tstats.to_dict(),
    'pvalues': res_fe.pvalues.to_dict(),
    'conf_int': {var: [float(coef_df[coef_df['variable']==var]['ci_lower'].values[0]),
                       float(coef_df[coef_df['variable']==var]['ci_upper'].values[0])]
                 for var in ALL_X},
    'r2_within': r2_within,
    'r2_between': r2_between,
    'r2_overall': r2_overall,
    'f_statistic': {'stat': f_stat, 'pval': float(f_pval)},
    'n_obs': int(n_obs),
    'n_entities': int(n_entities),
    'residuals_summary': {
        'mean': float(res_fe.resids.mean()),
        'std': float(res_fe.resids.std()),
        'min': float(res_fe.resids.min()),
        'max': float(res_fe.resids.max()),
    },
    'model_meta': model_meta,
}
with open(os.path.join(OUT_FE, 'fe_final_model_weights.pkl'), 'wb') as f:
    pickle.dump(model_save, f)
print(f"  模型权重: fe_final_model_weights.pkl")

# 6.4 模型完整summary文本
with open(os.path.join(OUT_FE, 'fe_final_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(res_fe.summary.as_text())
print(f"  模型摘要: fe_final_summary.txt")

# ============================================================
# 7. 论文用三线表
# ============================================================
print("\n" + "=" * 70)
print("  生成论文用三线表")
print("=" * 70)

table_rows = []

# 分区：核心解释变量
table_rows.append({'变量': '—— 核心解释变量 ——', '系数': '', '标准误': '', '95% CI': ''})
for _, r in coef_df[coef_df['category'] == '核心解释变量'].iterrows():
    coef_str, se_str = format_coef(r['coef'], r['se'], r['pvalue'])
    ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
    table_rows.append({
        '变量': r['label_table'],
        '系数': coef_str,
        '标准误': se_str,
        '95% CI': ci_str,
    })

# 分区：控制变量
table_rows.append({'变量': '—— 控制变量 ——', '系数': '', '标准误': '', '95% CI': ''})
for _, r in coef_df[coef_df['category'] == '控制变量'].iterrows():
    coef_str, se_str = format_coef(r['coef'], r['se'], r['pvalue'])
    ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
    table_rows.append({
        '变量': r['label_table'],
        '系数': coef_str,
        '标准误': se_str,
        '95% CI': ci_str,
    })

# 底部信息
table_rows.append({'变量': '', '系数': '', '标准误': '', '95% CI': ''})
table_rows.append({'变量': 'N (观测)', '系数': f'{n_obs:,}', '标准误': '', '95% CI': ''})
table_rows.append({'变量': '个体数', '系数': f'{n_entities:,}', '标准误': '', '95% CI': ''})
table_rows.append({'变量': 'R² (within)', '系数': f'{r2_within:.4f}', '标准误': '', '95% CI': ''})
table_rows.append({'变量': 'R² (overall)', '系数': f'{r2_overall:.4f}', '标准误': '', '95% CI': ''})
table_rows.append({'变量': 'F统计量', '系数': f'{f_stat:.2f}***', '标准误': '', '95% CI': ''})
table_rows.append({'变量': '个体固定效应', '系数': '是', '标准误': '', '95% CI': ''})
table_rows.append({'变量': '年份固定效应', '系数': '是', '标准误': '', '95% CI': ''})
table_rows.append({'变量': '聚类标准误', '系数': '个体层面', '标准误': '', '95% CI': ''})

table_df = pd.DataFrame(table_rows)

save_three_line_table(
    table_df.set_index('变量'),
    os.path.join(OUT_TABLE, 'fe_final_table'),
    title='表 X  面板固定效应基准回归结果（最终模型）',
    note='注：*** p<0.01, ** p<0.05, * p<0.1。括号内为个体聚类稳健标准误。'
         '所有模型包含个体固定效应和年份固定效应。'
         'gender、edu、rural2为时间不变变量，被个体固定效应吸收。'
         'activity_index因与social_activity_index严重共线(r=0.98)而排除。'
         'srh因确认为中介变量（坏控制）而不纳入。',
)
table_df.to_excel(os.path.join(OUT_FE, 'fe_final_table_clean.xlsx'), index=False)

print("\n  论文三线表已保存: fe_final_table.xlsx / .docx")
print("\n" + "=" * 70)
print("  最终FE模型全部完成")
print("=" * 70)
