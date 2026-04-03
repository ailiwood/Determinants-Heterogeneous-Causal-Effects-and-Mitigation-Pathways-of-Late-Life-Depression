"""
阶段8.3：模型/Learner替换稳健性检验
—————————————————————————————————
(a) DML XGBoost → RF（读取已有结果）
(b) FE → Mundlak CRE（加组均值项）
(c) CRF参数敏感性（n_estimators=1000, min_leaf=50）
"""

import sys
sys.path.insert(0, 'D:/BaiduSyncdisk/lunwen/model0320/scripts')

import numpy as np
import pandas as pd
import json, os, pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

ROOT = 'D:/BaiduSyncdisk/lunwen/model0320'
DATA_LAG = f'{ROOT}/results/03_causal_sample/causal_sample_lagged.xlsx'
DATA_ORIG = f'{ROOT}/data/preprocess_outputs/data1_modeling_ready.xlsx'
OUT_DIR  = f'{ROOT}/results/07_robustness'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

print("=" * 60)
print("8.3  模型/Learner替换稳健性检验")
print("=" * 60)

results_all = []

# ======================================================================
# (a) DML: XGBoost → RF（读取已有Stage 5结果）
# ======================================================================
print("\n[a] DML Learner替换: XGBoost → Random Forest")

with open(f'{ROOT}/results/04_dml/dml_results.json', 'r') as f:
    dml_data = json.load(f)

for key_xgb, key_rf, label in [
    ('T1_xgb', 'T1_rf', 'T1社会参与'),
    ('T2_xgb', 'T2_rf', 'T2多病共存'),
    ('T3_xgb', 'T3_rf', 'T3睡眠异常')]:

    rx = dml_data[key_xgb]
    rr = dml_data[key_rf]

    if 'theta' in rx:
        est_x, se_x, p_x = rx['theta'], rx['se'], rx['pvalue']
        est_r, se_r, p_r = rr['theta'], rr['se'], rr['pvalue']
    else:
        est_x, se_x, p_x = rx['ATE'], rx['ATE_se'], rx['ATE_pvalue']
        est_r, se_r, p_r = rr['ATE'], rr['ATE_se'], rr['ATE_pvalue']

    def stars(p): return '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))

    results_all.append({'检验项': f'{label} DML(XGB)', '模型': 'DML-XGB',
                        'ATE/θ': round(est_x, 4), 'SE': round(se_x, 4),
                        'p值': round(p_x, 4), '显著性': stars(p_x)})
    results_all.append({'检验项': f'{label} DML(RF)', '模型': 'DML-RF',
                        'ATE/θ': round(est_r, 4), 'SE': round(se_r, 4),
                        'p值': round(p_r, 4), '显著性': stars(p_r)})
    print(f"  {label}: XGB={est_x:.4f}(p={p_x:.4f}), RF={est_r:.4f}(p={p_r:.4f})")

df_rf = pd.DataFrame(results_all)
df_rf.to_excel(f'{OUT_DIR}/rob_learner_rf.xlsx', index=False)

# ======================================================================
# (b) FE → Mundlak CRE
# ======================================================================
print("\n[b] FE → Mundlak CRE")

df_panel = pd.read_excel(DATA_ORIG)
df_panel['hhcperc_log'] = np.log1p(df_panel['hhcperc'])
df_panel['fcamt_log'] = np.log1p(df_panel['fcamt'])
df_panel['T2_chronic'] = (df_panel['chronic_disease_count'] >= 2).astype(int)
df_panel['T3_sleep'] = ((df_panel['sleep'] < 6) | (df_panel['sleep'] > 9)).astype(int)

# 保留≥2波个体
id_counts = df_panel.groupby('ID').size()
panel_ids = id_counts[id_counts >= 2].index
df_cre = df_panel[df_panel['ID'].isin(panel_ids)].copy()
print(f"  CRE样本: {df_cre.shape[0]} obs, {df_cre['ID'].nunique()} individuals")

# FE控制变量
fe_controls = ['age', 'edu', 'marry', 'hhcperc_log', 'ins', 'pension',
               'retire', 'bmi', 'smoken', 'drinkev', 'family_size',
               'hchild', 'activity_index']

# Mundlak CRE：加入时变变量的个体均值
time_varying = ['age', 'edu', 'marry', 'hhcperc_log', 'ins', 'pension',
                'retire', 'bmi', 'smoken', 'drinkev', 'family_size',
                'hchild', 'activity_index', 'social_activity_index',
                'chronic_disease_count', 'sleep']

# 计算组均值
for var in time_varying:
    df_cre[f'{var}_mean'] = df_cre.groupby('ID')[var].transform('mean')

mean_vars = [f'{v}_mean' for v in time_varying]

df_cre = df_cre.set_index(['ID', 'wave'])

cre_results = []
treatments_fe = [
    ('T1 CRE', 'social_activity_index'),
    ('T2 CRE', 'T2_chronic'),
    ('T3 CRE', 'T3_sleep'),
]

for label, tvar in treatments_fe:
    print(f"\n  {label}")
    exog_vars = [tvar] + fe_controls + mean_vars
    # Remove tvar_mean from mean_vars to avoid collinearity
    exog_clean = [v for v in exog_vars if v != f'{tvar}_mean']
    exog = sm.add_constant(df_cre[exog_clean])
    try:
        model = PanelOLS(df_cre['cesd10'], exog, entity_effects=True, drop_absorbed=True)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        coef = res.params[tvar]
        se_val = res.std_errors[tvar]
        pval = res.pvalues[tvar]
        print(f"    β = {coef:.4f} (SE={se_val:.4f}, p={pval:.4f})")
        cre_results.append({'检验项': label, '模型': 'Mundlak-CRE',
                           'ATE/θ': round(coef, 4), 'SE': round(se_val, 4),
                           'p值': round(pval, 4),
                           '显著性': '***' if pval<0.01 else ('**' if pval<0.05 else ('*' if pval<0.1 else ''))})
    except Exception as e:
        print(f"    ERROR: {e}")
        cre_results.append({'检验项': label, '模型': 'Mundlak-CRE',
                           'ATE/θ': 'ERROR', 'SE': '', 'p值': '', '显著性': ''})

df_cre_out = pd.DataFrame(cre_results)
df_cre_out.to_excel(f'{OUT_DIR}/rob_mundlak_cre.xlsx', index=False)

# ======================================================================
# (c) CRF参数敏感性
# ======================================================================
print("\n[c] CRF参数敏感性: n_estimators=1000, min_leaf=50")

from econml.dml import CausalForestDML

df_lag = pd.read_excel(DATA_LAG)
le = LabelEncoder()
df_lag['province_enc'] = le.fit_transform(df_lag['province'].astype(str))

X_VARS = ['gender', 'rural2', 'region', 'age', 'edu', 'hhcperc_log', 'marry']
# Encode region
le_reg = LabelEncoder()
df_lag['region_enc'] = le_reg.fit_transform(df_lag['region'].astype(str))
X_VARS_enc = ['gender', 'rural2', 'region_enc', 'age', 'edu', 'hhcperc_log', 'marry']

W_COMMON_CRF = ['province_enc', 'ins', 'pension', 'retire', 'bmi', 'smoken',
                'drinkev', 'family_size', 'hchild', 'lag_cesd10', 'wave']

def get_model_y():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)
def get_model_t():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)

Y_lag = df_lag['cesd10_t'].values
X_lag = df_lag[X_VARS_enc].values

crf_configs = [
    {'label': 'CRF主模型(2000,30)', 'n_est': 2000, 'min_leaf': 30},
    {'label': 'CRF替代(1000,30)', 'n_est': 1000, 'min_leaf': 30},
    {'label': 'CRF替代(2000,50)', 'n_est': 2000, 'min_leaf': 50},
]

crf_treatments = [
    ('T1', 'social_activity_index', W_COMMON_CRF + ['chronic_disease_count', 'sleep', 'fcamt_log']),
    ('T2', 'T2_chronic', W_COMMON_CRF + ['sleep', 'social_activity_index', 'fcamt_log']),
    ('T3', 'T3_sleep', W_COMMON_CRF + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']),
]

crf_results = []

for cfg in crf_configs:
    for tname, tcol, W_cols in crf_treatments:
        print(f"\n  {cfg['label']} - {tname}")
        D = df_lag[tcol].values
        W = df_lag[W_cols].values

        crf = CausalForestDML(
            model_y=get_model_y(),
            model_t=get_model_t(),
            n_estimators=cfg['n_est'],
            min_samples_leaf=cfg['min_leaf'],
            max_depth=None,
            cv=5,
            random_state=SEED
        )
        crf.fit(Y_lag, D, X=X_lag, W=W)
        ate_inf = crf.ate_inference(X=X_lag)

        ate = ate_inf.mean_point
        se_val = ate_inf.stderr_mean
        ci = ate_inf.conf_int_mean(alpha=0.05)
        pval_arr = ate_inf.pvalue(value=0)
        pval = float(pval_arr) if np.isscalar(pval_arr) else float(pval_arr[0])

        print(f"    ATE={ate:.4f} (SE={se_val:.4f}, p={pval:.4f})")

        def stars(p): return '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))

        crf_results.append({
            '检验项': f'{tname} {cfg["label"]}',
            '模型': f'CRF({cfg["n_est"]},{cfg["min_leaf"]})',
            'ATE/θ': round(float(ate), 4),
            'SE': round(float(se_val), 4),
            'p值': round(pval, 4),
            '显著性': stars(pval)
        })

df_crf = pd.DataFrame(crf_results)
df_crf.to_excel(f'{OUT_DIR}/rob_crf_sensitivity.xlsx', index=False)

# 合并所有Learner替换结果
all_learner = results_all + cre_results + crf_results
df_all = pd.DataFrame(all_learner)
df_all.to_excel(f'{OUT_DIR}/rob_learner_all.xlsx', index=False)

print(f"\n保存: rob_learner_rf.xlsx, rob_mundlak_cre.xlsx, rob_crf_sensitivity.xlsx, rob_learner_all.xlsx")
print("\n✅ 8.3 模型/Learner替换完成！")
