"""
阶段8.1：Treatment定义替换稳健性检验
—————————————————————————————————
T1: 连续social_activity_index → 二值化(参与≥1种=1 vs 不参与=0)
T2: ≥2种慢性病 → (a)≥3种 (b)连续chronic_disease_count
T3: sleep<6或>9 → 仅短睡眠(<6)
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
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM

ROOT = 'D:/BaiduSyncdisk/lunwen/model0320'
DATA_LAG = f'{ROOT}/results/03_causal_sample/causal_sample_lagged.xlsx'
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
print("8.1  Treatment定义替换稳健性检验")
print("=" * 60)

# 读取因果样本
df = pd.read_excel(DATA_LAG)
le = LabelEncoder()
df['province_enc'] = le.fit_transform(df['province'].astype(str))
print(f"因果样本: {df.shape}")

# W变量集
W_COMMON = ['age', 'gender', 'edu', 'marry', 'rural2', 'province_enc',
            'hhcperc_log', 'ins', 'pension', 'retire', 'bmi', 'smoken',
            'drinkev', 'family_size', 'hchild', 'lag_cesd10', 'wave']

# 读取主模型结果
with open(f'{ROOT}/results/04_dml/dml_results.json', 'r') as f:
    main_results = json.load(f)

results = []

def extract_main(key):
    r = main_results[key]
    if 'theta' in r:
        return r['theta'], r['se'], r['pvalue']
    return r['ATE'], r['ATE_se'], r['ATE_pvalue']

def run_dml_plr(y, d, W_cols, label):
    print(f"\n  --- {label} (PLR) ---")
    W = df[W_cols].values
    data = DoubleMLData.from_arrays(y=y, d=d, x=W)
    model = DoubleMLPLR(data, ml_l=get_ml_reg(), ml_m=get_ml_reg(),
                        n_folds=N_FOLDS, n_rep=N_REP, score='partialling out')
    model.fit()
    print(f"    θ = {model.coef[0]:.4f} (SE={model.se[0]:.4f}, p={model.pval[0]:.4f})")
    return model.coef[0], model.se[0], model.pval[0]

def run_dml_irm(y, d, W_cols, label):
    print(f"\n  --- {label} (IRM) ---")
    W = df[W_cols].values
    data = DoubleMLData.from_arrays(y=y, d=d, x=W)
    model = DoubleMLIRM(data, ml_g=get_ml_reg(), ml_m=get_ml_cls(),
                        n_folds=N_FOLDS, n_rep=N_REP, score='ATE',
                        trimming_threshold=0.01)
    model.fit()
    print(f"    ATE = {model.coef[0]:.4f} (SE={model.se[0]:.4f}, p={model.pval[0]:.4f})")
    return model.coef[0], model.se[0], model.pval[0]

Y = df['cesd10_t'].values

# === T1: 连续 → 二值化 ===
W_T1 = W_COMMON + ['chronic_disease_count', 'sleep', 'fcamt_log']
est_m, se_m, p_m = extract_main('T1_xgb')
results.append({'检验项': 'T1主模型(连续)', '模型': 'PLR', 'ATE/θ': round(est_m, 4),
                'SE': round(se_m, 4), 'p值': round(p_m, 4),
                '显著性': '***' if p_m<0.01 else ('**' if p_m<0.05 else ('*' if p_m<0.1 else ''))})

df['T1_binary'] = (df['social_activity_index'] >= 1).astype(int)
print(f"\nT1二值化: 参与≥1种={df['T1_binary'].sum()} ({df['T1_binary'].mean()*100:.1f}%)")
est, se, p = run_dml_irm(Y, df['T1_binary'].values, W_T1, 'T1替代(二值化)')
results.append({'检验项': 'T1替代(二值化≥1)', '模型': 'IRM', 'ATE/θ': round(est, 4),
                'SE': round(se, 4), 'p值': round(p, 4),
                '显著性': '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))})

# === T2: ≥2 → ≥3; 连续 ===
W_T2 = W_COMMON + ['sleep', 'social_activity_index', 'fcamt_log']
est_m, se_m, p_m = extract_main('T2_xgb')
results.append({'检验项': 'T2主模型(≥2种)', '模型': 'IRM', 'ATE/θ': round(est_m, 4),
                'SE': round(se_m, 4), 'p值': round(p_m, 4),
                '显著性': '***' if p_m<0.01 else ('**' if p_m<0.05 else ('*' if p_m<0.1 else ''))})

# T2替代a: ≥3种
df['T2_chronic3'] = (df['chronic_disease_count'] >= 3).astype(int)
print(f"\nT2替代(≥3种): N_treated={df['T2_chronic3'].sum()} ({df['T2_chronic3'].mean()*100:.1f}%)")
est, se, p = run_dml_irm(Y, df['T2_chronic3'].values, W_T2, 'T2替代(≥3种)')
results.append({'检验项': 'T2替代(≥3种)', '模型': 'IRM', 'ATE/θ': round(est, 4),
                'SE': round(se, 4), 'p值': round(p, 4),
                '显著性': '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))})

# T2替代b: 连续
est, se, p = run_dml_plr(Y, df['chronic_disease_count'].values, W_T2, 'T2替代(连续)')
results.append({'检验项': 'T2替代(连续)', '模型': 'PLR', 'ATE/θ': round(est, 4),
                'SE': round(se, 4), 'p值': round(p, 4),
                '显著性': '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))})

# === T3: sleep<6或>9 → 仅<6 ===
W_T3 = W_COMMON + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']
est_m, se_m, p_m = extract_main('T3_xgb')
results.append({'检验项': 'T3主模型(<6或>9)', '模型': 'IRM', 'ATE/θ': round(est_m, 4),
                'SE': round(se_m, 4), 'p值': round(p_m, 4),
                '显著性': '***' if p_m<0.01 else ('**' if p_m<0.05 else ('*' if p_m<0.1 else ''))})

df['T3_short'] = (df['sleep'] < 6).astype(int)
print(f"\nT3替代(仅<6h): N_treated={df['T3_short'].sum()} ({df['T3_short'].mean()*100:.1f}%)")
est, se, p = run_dml_irm(Y, df['T3_short'].values, W_T3, 'T3替代(仅短睡眠<6h)')
results.append({'检验项': 'T3替代(仅短睡眠<6h)', '模型': 'IRM', 'ATE/θ': round(est, 4),
                'SE': round(se, 4), 'p值': round(p, 4),
                '显著性': '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))})

df_out = pd.DataFrame(results)
df_out.to_excel(f'{OUT_DIR}/rob_treatment_definition.xlsx', index=False)
print(f"\n保存: rob_treatment_definition.xlsx")
print("\n结果：")
print(df_out.to_string(index=False))
print("\n✅ 8.1 Treatment定义替换完成！")
