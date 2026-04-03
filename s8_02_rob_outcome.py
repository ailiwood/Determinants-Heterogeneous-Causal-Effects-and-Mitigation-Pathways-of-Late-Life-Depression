"""
阶段8.2：Outcome口径替换稳健性检验
—————————————————————————————————
主因变量：cesd10（连续）→ 替代：cesd10≥10（二值化）
对三个treatment用DML-IRM重估
对FE用LPM-FE重估
"""

import sys
sys.path.insert(0, 'D:/BaiduSyncdisk/lunwen/model0320/scripts')

import numpy as np
import pandas as pd
import json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLIRM
import statsmodels.api as sm

ROOT = 'D:/BaiduSyncdisk/lunwen/model0320'
DATA_LAG = f'{ROOT}/results/03_causal_sample/causal_sample_lagged.xlsx'
DATA_ORIG = f'{ROOT}/data/preprocess_outputs/data1_modeling_ready.xlsx'
OUT_DIR  = f'{ROOT}/results/07_robustness'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_REP = 5

def get_ml_reg():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)
def get_ml_cls():
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         use_label_encoder=False, eval_metric='logloss')

print("=" * 60)
print("8.2  Outcome口径替换稳健性检验")
print("=" * 60)

# 读取因果样本
df = pd.read_excel(DATA_LAG)
le = LabelEncoder()
df['province_enc'] = le.fit_transform(df['province'].astype(str))
print(f"因果样本: {df.shape}")

# 二值化outcome
df['depressed_t'] = (df['cesd10_t'] >= 10).astype(int)
print(f"抑郁二值化: {df['depressed_t'].sum()} ({df['depressed_t'].mean()*100:.1f}%)")

W_COMMON = ['age', 'gender', 'edu', 'marry', 'rural2', 'province_enc',
            'hhcperc_log', 'ins', 'pension', 'retire', 'bmi', 'smoken',
            'drinkev', 'family_size', 'hchild', 'lag_cesd10', 'wave']

W_T1 = W_COMMON + ['chronic_disease_count', 'sleep', 'fcamt_log']
W_T2 = W_COMMON + ['sleep', 'social_activity_index', 'fcamt_log']
W_T3 = W_COMMON + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']

results = []
Y_bin = df['depressed_t'].values

# === DML-IRM for all three treatments with binary outcome ===
treatments_dml = [
    ('T1社会参与(连续→IRM)', 'social_activity_index', W_T1, 'PLR'),
    ('T2多病共存(≥2)', 'T2_chronic', W_T2, 'IRM'),
    ('T3睡眠异常(<6或>9)', 'T3_sleep', W_T3, 'IRM'),
]

# T1 with binary outcome: still use PLR since treatment is continuous
# But outcome is binary, which PLR can handle (it's still a valid partially linear model)
print("\n--- DML with Binary Outcome (cesd10≥10) ---")

# T1: PLR with binary outcome
print("\n  T1: 社会参与 (PLR, Y=binary)")
data_t1 = DoubleMLData.from_arrays(
    y=Y_bin, d=df['social_activity_index'].values, x=df[W_T1].values)
m_t1 = dml.DoubleMLPLR(data_t1, ml_l=get_ml_reg(), ml_m=get_ml_reg(),
                        n_folds=N_FOLDS, n_rep=N_REP, score='partialling out')
m_t1.fit()
print(f"    θ = {m_t1.coef[0]:.4f} (SE={m_t1.se[0]:.4f}, p={m_t1.pval[0]:.4f})")
results.append({'检验项': 'T1 DML(Y=二值)', '模型': 'PLR', 'ATE/θ': round(m_t1.coef[0], 4),
                'SE': round(m_t1.se[0], 4), 'p值': round(m_t1.pval[0], 4),
                '显著性': '***' if m_t1.pval[0]<0.01 else ('**' if m_t1.pval[0]<0.05 else ('*' if m_t1.pval[0]<0.1 else ''))})

# T2: IRM with binary outcome — ml_g用回归器(预测概率值)，避免DoubleML的binary check报错
print("\n  T2: 多病共存 (IRM, Y=binary)")
data_t2 = DoubleMLData.from_arrays(
    y=Y_bin, d=df['T2_chronic'].values, x=df[W_T2].values)
m_t2 = dml.DoubleMLIRM(data_t2, ml_g=get_ml_reg(), ml_m=get_ml_cls(),
                        n_folds=N_FOLDS, n_rep=N_REP, score='ATE',
                        trimming_threshold=0.01)
m_t2.fit()
print(f"    ATE = {m_t2.coef[0]:.4f} (SE={m_t2.se[0]:.4f}, p={m_t2.pval[0]:.4f})")
results.append({'检验项': 'T2 DML(Y=二值)', '模型': 'IRM', 'ATE/θ': round(m_t2.coef[0], 4),
                'SE': round(m_t2.se[0], 4), 'p值': round(m_t2.pval[0], 4),
                '显著性': '***' if m_t2.pval[0]<0.01 else ('**' if m_t2.pval[0]<0.05 else ('*' if m_t2.pval[0]<0.1 else ''))})

# T3: IRM with binary outcome
print("\n  T3: 睡眠异常 (IRM, Y=binary)")
data_t3 = DoubleMLData.from_arrays(
    y=Y_bin, d=df['T3_sleep'].values, x=df[W_T3].values)
m_t3 = dml.DoubleMLIRM(data_t3, ml_g=get_ml_reg(), ml_m=get_ml_cls(),
                        n_folds=N_FOLDS, n_rep=N_REP, score='ATE',
                        trimming_threshold=0.01)
m_t3.fit()
print(f"    ATE = {m_t3.coef[0]:.4f} (SE={m_t3.se[0]:.4f}, p={m_t3.pval[0]:.4f})")
results.append({'检验项': 'T3 DML(Y=二值)', '模型': 'IRM', 'ATE/θ': round(m_t3.coef[0], 4),
                'SE': round(m_t3.se[0], 4), 'p值': round(m_t3.pval[0], 4),
                '显著性': '***' if m_t3.pval[0]<0.01 else ('**' if m_t3.pval[0]<0.05 else ('*' if m_t3.pval[0]<0.1 else ''))})

# === LPM-FE with binary outcome ===
print("\n--- LPM-FE with Binary Outcome ---")

df_panel = pd.read_excel(DATA_ORIG)
df_panel['hhcperc_log'] = np.log1p(df_panel['hhcperc'])
df_panel['fcamt_log'] = np.log1p(df_panel['fcamt'])
df_panel['T2_chronic'] = (df_panel['chronic_disease_count'] >= 2).astype(int)
df_panel['T3_sleep'] = ((df_panel['sleep'] < 6) | (df_panel['sleep'] > 9)).astype(int)
df_panel['depressed'] = (df_panel['cesd10'] >= 10).astype(int)

# 面板数据需要至少2波
id_counts = df_panel.groupby('ID').size()
panel_ids = id_counts[id_counts >= 2].index
df_fe = df_panel[df_panel['ID'].isin(panel_ids)].copy()
df_fe = df_fe.set_index(['ID', 'wave'])
print(f"FE面板样本: {df_fe.shape[0]} obs, {df_fe.index.get_level_values(0).nunique()} individuals")

from linearmodels.panel import PanelOLS

# FE控制变量（同阶段3）
fe_controls = ['age', 'edu', 'marry', 'hhcperc_log', 'ins', 'pension',
               'retire', 'bmi', 'smoken', 'drinkev', 'family_size',
               'hchild', 'activity_index']

fe_treatments = [
    ('T1 LPM-FE(Y=二值)', 'social_activity_index'),
    ('T2 LPM-FE(Y=二值)', 'T2_chronic'),
    ('T3 LPM-FE(Y=二值)', 'T3_sleep'),
]

for label, tvar in fe_treatments:
    print(f"\n  {label}")
    exog_vars = [tvar] + fe_controls
    exog = sm.add_constant(df_fe[exog_vars])
    try:
        model = PanelOLS(df_fe['depressed'], exog, entity_effects=True, drop_absorbed=True)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        coef = res.params[tvar]
        se = res.std_errors[tvar]
        pval = res.pvalues[tvar]
        print(f"    β = {coef:.4f} (SE={se:.4f}, p={pval:.4f})")
        results.append({'检验项': label, '模型': 'LPM-FE', 'ATE/θ': round(coef, 4),
                        'SE': round(se, 4), 'p值': round(pval, 4),
                        '显著性': '***' if pval<0.01 else ('**' if pval<0.05 else ('*' if pval<0.1 else ''))})
    except Exception as e:
        print(f"    ERROR: {e}")
        results.append({'检验项': label, '模型': 'LPM-FE', 'ATE/θ': 'ERROR',
                        'SE': '', 'p值': '', '显著性': ''})

df_out = pd.DataFrame(results)
df_out.to_excel(f'{OUT_DIR}/rob_outcome_binary.xlsx', index=False)
print(f"\n保存: rob_outcome_binary.xlsx")
print("\n结果：")
print(df_out.to_string(index=False))
print("\n✅ 8.2 Outcome口径替换完成！")
