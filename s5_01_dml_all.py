"""
阶段5：DML平均因果效应估计（T1 / T2 / T3）
============================================================
T1: 社会参与（连续, DoubleMLPLR）→ θ₀ 边际因果效应
T2: 高慢病负担≥2（二元, DoubleMLIRM）→ ATE + ATTE
T3: 睡眠异常<6或>9（二元, DoubleMLIRM）→ ATE + ATTE

Learner: XGBoost (主) / RF (稳健性)
交叉拟合: 5-fold
重复: 5次取中位数（消除样本分割随机性）
============================================================
"""

import pandas as pd
import numpy as np
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_config import *
from utils.var_labels import get_zh, get_label
from utils.table_utils import save_three_line_table, format_coef

import doubleml as dml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

# ============================================================
# 0. 路径与参数
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAUSAL_PATH = os.path.join(ROOT, 'results/03_causal_sample/causal_sample_lagged.xlsx')
OUT_DML = os.path.join(ROOT, 'results/04_dml')
OUT_TABLE = os.path.join(ROOT, 'results/08_tables_for_paper')
OUT_FIG = os.path.join(ROOT, 'results/09_figures_for_paper')
for d in [OUT_DML, OUT_TABLE, OUT_FIG]:
    os.makedirs(d, exist_ok=True)

N_FOLDS = 5
N_REP = 5  # 重复次数
SEED = 42

# ============================================================
# 1. 读取因果样本
# ============================================================
print("=" * 70)
print("  阶段5：DML因果效应估计")
print("=" * 70)
df = pd.read_excel(CAUSAL_PATH)
print(f"因果样本: {df.shape}, 个体: {df['ID'].nunique()}")

# province → 数值编码（用于DML的W）
from sklearn.preprocessing import LabelEncoder
le_prov = LabelEncoder()
df['province_enc'] = le_prov.fit_transform(df['province'].astype(str))

# ============================================================
# 2. 定义W变量集（按研究设计 §3.3.3）
# ============================================================
# 通用安全控制集
W_COMMON = ['age', 'gender', 'edu', 'marry', 'rural2', 'province_enc',
            'hhcperc_log', 'ins', 'pension', 'retire', 'bmi',
            'smoken', 'drinkev', 'family_size', 'hchild',
            'lag_cesd10', 'wave']

# Treatment-specific W
W_T1 = W_COMMON + ['chronic_disease_count', 'sleep', 'fcamt_log']
# T1时不纳入social相关（构成重叠）

W_T2 = W_COMMON + ['sleep', 'social_activity_index', 'fcamt_log']
# T2时不纳入disease_index, body_discomfort_count（共线/中介）

W_T3 = W_COMMON + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']
# T3时不纳入srh（主观同源）

Y_VAR = 'cesd10_t'

# ============================================================
# 3. Learner定义
# ============================================================
def get_xgb_reg():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)

def get_xgb_cls():
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         use_label_encoder=False, eval_metric='logloss')

def get_rf_reg():
    return RandomForestRegressor(n_estimators=500, max_depth=None,
                                 random_state=SEED, n_jobs=-1)

def get_rf_cls():
    return RandomForestClassifier(n_estimators=500, max_depth=None,
                                  random_state=SEED, n_jobs=-1)

# ============================================================
# 4. DML估计函数
# ============================================================
def run_dml_plr(df, y_col, d_col, w_cols, learner='xgb', n_folds=N_FOLDS, n_rep=N_REP):
    """DoubleMLPLR: 连续treatment, 输出θ₀"""
    data = df[[y_col, d_col] + w_cols].dropna()
    print(f"  PLR样本: {len(data)}")

    obj = dml.DoubleMLData(data, y_col=y_col, d_cols=d_col, x_cols=w_cols)

    if learner == 'xgb':
        ml_l, ml_m = get_xgb_reg(), get_xgb_reg()
    else:
        ml_l, ml_m = get_rf_reg(), get_rf_reg()

    model = dml.DoubleMLPLR(obj, ml_l=ml_l, ml_m=ml_m,
                             n_folds=n_folds, n_rep=n_rep,
                             score='partialling out')
    model.fit()

    theta = model.coef[0]
    se = model.se[0]
    ci = model.confint(level=0.95).values[0]
    pval = model.pval[0]

    return {
        'model_type': 'DoubleMLPLR',
        'treatment': d_col,
        'learner': learner,
        'theta': theta,
        'se': se,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'pvalue': pval,
        'n_obs': len(data),
        'n_folds': n_folds,
        'n_rep': n_rep,
        'model_obj': model,
    }


def run_dml_irm(df, y_col, d_col, w_cols, learner='xgb', n_folds=N_FOLDS, n_rep=N_REP):
    """DoubleMLIRM: 二元treatment, 输出ATE和ATTE"""
    data = df[[y_col, d_col] + w_cols].dropna()
    n_treat = data[d_col].sum()
    n_ctrl = len(data) - n_treat
    print(f"  IRM样本: {len(data)} (治疗={n_treat}, 对照={n_ctrl})")

    obj = dml.DoubleMLData(data, y_col=y_col, d_cols=d_col, x_cols=w_cols)

    if learner == 'xgb':
        ml_g, ml_m = get_xgb_reg(), get_xgb_cls()
    else:
        ml_g, ml_m = get_rf_reg(), get_rf_cls()

    # ATE
    model_ate = dml.DoubleMLIRM(obj, ml_g=ml_g, ml_m=ml_m,
                                 n_folds=n_folds, n_rep=n_rep,
                                 score='ATE', trimming_threshold=0.01)
    model_ate.fit()

    ate = model_ate.coef[0]
    ate_se = model_ate.se[0]
    ate_ci = model_ate.confint(level=0.95).values[0]
    ate_pval = model_ate.pval[0]

    # ATTE
    if learner == 'xgb':
        ml_g2, ml_m2 = get_xgb_reg(), get_xgb_cls()
    else:
        ml_g2, ml_m2 = get_rf_reg(), get_rf_cls()

    model_atte = dml.DoubleMLIRM(obj, ml_g=ml_g2, ml_m=ml_m2,
                                  n_folds=n_folds, n_rep=n_rep,
                                  score='ATTE', trimming_threshold=0.01)
    model_atte.fit()

    atte = model_atte.coef[0]
    atte_se = model_atte.se[0]
    atte_ci = model_atte.confint(level=0.95).values[0]
    atte_pval = model_atte.pval[0]

    return {
        'model_type': 'DoubleMLIRM',
        'treatment': d_col,
        'learner': learner,
        'ATE': ate, 'ATE_se': ate_se,
        'ATE_ci_lower': ate_ci[0], 'ATE_ci_upper': ate_ci[1],
        'ATE_pvalue': ate_pval,
        'ATTE': atte, 'ATTE_se': atte_se,
        'ATTE_ci_lower': atte_ci[0], 'ATTE_ci_upper': atte_ci[1],
        'ATTE_pvalue': atte_pval,
        'n_obs': len(data),
        'n_treated': int(n_treat),
        'n_control': int(n_ctrl),
        'n_folds': n_folds,
        'n_rep': n_rep,
        'model_ate_obj': model_ate,
        'model_atte_obj': model_atte,
    }


def plot_propensity(model_obj, d_col, save_path):
    """绘制propensity score分布图"""
    # 从模型提取propensity scores
    # DoubleMLIRM stores nuisance predictions
    try:
        # 获取propensity scores（ml_m的预测）
        prop_scores = model_obj.nuisance_targets['ml_m0']
        if prop_scores is not None:
            pass  # use it
    except:
        pass

    # 替代方案：用model的内部数据重新获取
    try:
        props = model_obj.models['ml_m']['rep_0'][0].predict_proba(
            model_obj._dml_data.x)[:, 1]
    except:
        print(f"  ⚠ 无法提取propensity scores for {d_col}")
        return

    d = model_obj._dml_data.d.flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(props[d == 0], bins=50, alpha=0.6, label='对照组 (D=0)',
            color=COLORS['primary'], density=True)
    ax.hist(props[d == 1], bins=50, alpha=0.6, label='治疗组 (D=1)',
            color=COLORS['secondary'], density=True)
    ax.set_xlabel('倾向得分 P(D=1|W)')
    ax.set_ylabel('密度')
    ax.set_title(f'倾向得分分布: {get_zh(d_col)}')
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Propensity图已保存: {os.path.basename(save_path)}")


# ============================================================
# 5. 运行T1: 社会参与（连续，PLR）
# ============================================================
print("\n" + "#" * 70)
print("# T1: 社会参与（连续treatment, DoubleMLPLR）")
print("#" * 70)

print("\n--- XGBoost learner ---")
t1_xgb = run_dml_plr(df, Y_VAR, 'T1_social', W_T1, learner='xgb')
print(f"  θ₀ = {t1_xgb['theta']:.4f} (SE={t1_xgb['se']:.4f}), "
      f"95%CI=[{t1_xgb['ci_lower']:.4f}, {t1_xgb['ci_upper']:.4f}], p={t1_xgb['pvalue']:.4e}")

print("\n--- RF learner (稳健性) ---")
t1_rf = run_dml_plr(df, Y_VAR, 'T1_social', W_T1, learner='rf')
print(f"  θ₀ = {t1_rf['theta']:.4f} (SE={t1_rf['se']:.4f}), "
      f"95%CI=[{t1_rf['ci_lower']:.4f}, {t1_rf['ci_upper']:.4f}], p={t1_rf['pvalue']:.4e}")

# ============================================================
# 6. 运行T2: 高慢病负担（二元，IRM）
# ============================================================
print("\n" + "#" * 70)
print("# T2: 高慢病负担≥2（二元treatment, DoubleMLIRM）")
print("#" * 70)

print("\n--- XGBoost learner ---")
t2_xgb = run_dml_irm(df, Y_VAR, 'T2_chronic', W_T2, learner='xgb')
print(f"  ATE  = {t2_xgb['ATE']:.4f} (SE={t2_xgb['ATE_se']:.4f}), "
      f"95%CI=[{t2_xgb['ATE_ci_lower']:.4f}, {t2_xgb['ATE_ci_upper']:.4f}], p={t2_xgb['ATE_pvalue']:.4e}")
print(f"  ATTE = {t2_xgb['ATTE']:.4f} (SE={t2_xgb['ATTE_se']:.4f}), "
      f"95%CI=[{t2_xgb['ATTE_ci_lower']:.4f}, {t2_xgb['ATTE_ci_upper']:.4f}], p={t2_xgb['ATTE_pvalue']:.4e}")

print("\n--- RF learner (稳健性) ---")
t2_rf = run_dml_irm(df, Y_VAR, 'T2_chronic', W_T2, learner='rf')
print(f"  ATE  = {t2_rf['ATE']:.4f} (SE={t2_rf['ATE_se']:.4f}), p={t2_rf['ATE_pvalue']:.4e}")
print(f"  ATTE = {t2_rf['ATTE']:.4f} (SE={t2_rf['ATTE_se']:.4f}), p={t2_rf['ATTE_pvalue']:.4e}")

# Propensity plot
plot_propensity(t2_xgb['model_ate_obj'], 'T2_chronic',
                os.path.join(OUT_FIG, 'propensity_T2_chronic.png'))

# ============================================================
# 7. 运行T3: 睡眠异常（二元，IRM）
# ============================================================
print("\n" + "#" * 70)
print("# T3: 睡眠异常（二元treatment, DoubleMLIRM）")
print("#" * 70)

print("\n--- XGBoost learner ---")
t3_xgb = run_dml_irm(df, Y_VAR, 'T3_sleep', W_T3, learner='xgb')
print(f"  ATE  = {t3_xgb['ATE']:.4f} (SE={t3_xgb['ATE_se']:.4f}), "
      f"95%CI=[{t3_xgb['ATE_ci_lower']:.4f}, {t3_xgb['ATE_ci_upper']:.4f}], p={t3_xgb['ATE_pvalue']:.4e}")
print(f"  ATTE = {t3_xgb['ATTE']:.4f} (SE={t3_xgb['ATTE_se']:.4f}), "
      f"95%CI=[{t3_xgb['ATTE_ci_lower']:.4f}, {t3_xgb['ATTE_ci_upper']:.4f}], p={t3_xgb['ATTE_pvalue']:.4e}")

print("\n--- RF learner (稳健性) ---")
t3_rf = run_dml_irm(df, Y_VAR, 'T3_sleep', W_T3, learner='rf')
print(f"  ATE  = {t3_rf['ATE']:.4f} (SE={t3_rf['ATE_se']:.4f}), p={t3_rf['ATE_pvalue']:.4e}")
print(f"  ATTE = {t3_rf['ATTE']:.4f} (SE={t3_rf['ATTE_se']:.4f}), p={t3_rf['ATTE_pvalue']:.4e}")

# Propensity plot
plot_propensity(t3_xgb['model_ate_obj'], 'T3_sleep',
                os.path.join(OUT_FIG, 'propensity_T3_sleep.png'))

# ============================================================
# 8. 汇总表
# ============================================================
print("\n" + "=" * 70)
print("  DML结果汇总")
print("=" * 70)

summary_rows = []

# T1
summary_rows.append({
    'Treatment': 'T1: 社会参与 (连续)',
    'DML模型': 'DoubleMLPLR',
    '估计量': 'θ₀ (边际效应)',
    'Learner': 'XGBoost',
    '点估计': round(t1_xgb['theta'], 4),
    'SE': round(t1_xgb['se'], 4),
    '95% CI下界': round(t1_xgb['ci_lower'], 4),
    '95% CI上界': round(t1_xgb['ci_upper'], 4),
    'p值': t1_xgb['pvalue'],
    'N': t1_xgb['n_obs'],
})
summary_rows.append({
    'Treatment': 'T1: 社会参与 (连续)',
    'DML模型': 'DoubleMLPLR',
    '估计量': 'θ₀ (边际效应)',
    'Learner': 'RF',
    '点估计': round(t1_rf['theta'], 4),
    'SE': round(t1_rf['se'], 4),
    '95% CI下界': round(t1_rf['ci_lower'], 4),
    '95% CI上界': round(t1_rf['ci_upper'], 4),
    'p值': t1_rf['pvalue'],
    'N': t1_rf['n_obs'],
})

# T2
for est_type in ['ATE', 'ATTE']:
    for name, res in [('XGBoost', t2_xgb), ('RF', t2_rf)]:
        summary_rows.append({
            'Treatment': 'T2: 多病共存≥2 (二元)',
            'DML模型': 'DoubleMLIRM',
            '估计量': est_type,
            'Learner': name,
            '点估计': round(res[est_type], 4),
            'SE': round(res[f'{est_type}_se'], 4),
            '95% CI下界': round(res[f'{est_type}_ci_lower'], 4),
            '95% CI上界': round(res[f'{est_type}_ci_upper'], 4),
            'p值': res[f'{est_type}_pvalue'],
            'N': res['n_obs'],
        })

# T3
for est_type in ['ATE', 'ATTE']:
    for name, res in [('XGBoost', t3_xgb), ('RF', t3_rf)]:
        summary_rows.append({
            'Treatment': 'T3: 睡眠异常 (二元)',
            'DML模型': 'DoubleMLIRM',
            '估计量': est_type,
            'Learner': name,
            '点估计': round(res[est_type], 4),
            'SE': round(res[f'{est_type}_se'], 4),
            '95% CI下界': round(res[f'{est_type}_ci_lower'], 4),
            '95% CI上界': round(res[f'{est_type}_ci_upper'], 4),
            'p值': res[f'{est_type}_pvalue'],
            'N': res['n_obs'],
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(os.path.join(OUT_DML, 'dml_summary.xlsx'), index=False)
print("\n汇总表已保存: dml_summary.xlsx")
print(summary_df.to_string(index=False))

# ============================================================
# 9. 论文用三线表（仅XGBoost主结果）
# ============================================================
paper_rows = []

# T1
c, s = format_coef(t1_xgb['theta'], t1_xgb['se'], t1_xgb['pvalue'])
paper_rows.append({
    '变量': 'T1: 社会参与指数 (连续)',
    '模型': 'PLR',
    '估计量': 'θ₀',
    '点估计': c, '标准误': s,
    '95% CI': f"[{t1_xgb['ci_lower']:.3f}, {t1_xgb['ci_upper']:.3f}]",
    'N': f"{t1_xgb['n_obs']:,}",
})

# T2
for est in ['ATE', 'ATTE']:
    val = t2_xgb[est]
    se = t2_xgb[f'{est}_se']
    p = t2_xgb[f'{est}_pvalue']
    ci_l = t2_xgb[f'{est}_ci_lower']
    ci_u = t2_xgb[f'{est}_ci_upper']
    c, s = format_coef(val, se, p)
    paper_rows.append({
        '变量': 'T2: 多病共存≥2种 (二元)',
        '模型': 'IRM',
        '估计量': est,
        '点估计': c, '标准误': s,
        '95% CI': f"[{ci_l:.3f}, {ci_u:.3f}]",
        'N': f"{t2_xgb['n_obs']:,}",
    })

# T3
for est in ['ATE', 'ATTE']:
    val = t3_xgb[est]
    se = t3_xgb[f'{est}_se']
    p = t3_xgb[f'{est}_pvalue']
    ci_l = t3_xgb[f'{est}_ci_lower']
    ci_u = t3_xgb[f'{est}_ci_upper']
    c, s = format_coef(val, se, p)
    paper_rows.append({
        '变量': 'T3: 睡眠异常 (二元)',
        '模型': 'IRM',
        '估计量': est,
        '点估计': c, '标准误': s,
        '95% CI': f"[{ci_l:.3f}, {ci_u:.3f}]",
        'N': f"{t3_xgb['n_obs']:,}",
    })

paper_df = pd.DataFrame(paper_rows)

save_three_line_table(
    paper_df.set_index('变量'),
    os.path.join(OUT_TABLE, 'dml_ate_summary'),
    title='表 X  双重机器学习因果效应估计结果',
    note='注：*** p<0.01, ** p<0.05, * p<0.1。标准误由DML Neyman正交化推断得到。'
         '第一阶段learner为XGBoost(n_estimators=500, max_depth=5)。'
         '交叉拟合5折，重复5次取中位数。'
         'Treatment均取t-1期值，Outcome取t期值（lagged结构）。'
         'PLR: 部分线性回归模型(连续treatment)；IRM: 交互回归模型(二元treatment)。'
         'ATE: 平均处理效应；ATTE: 处理组平均处理效应。',
)

# ============================================================
# 10. 保存模型权重
# ============================================================
dml_weights = {
    'T1_xgb': {k: v for k, v in t1_xgb.items() if k != 'model_obj'},
    'T1_rf': {k: v for k, v in t1_rf.items() if k != 'model_obj'},
    'T2_xgb': {k: v for k, v in t2_xgb.items() if 'model' not in k},
    'T2_rf': {k: v for k, v in t2_rf.items() if 'model' not in k},
    'T3_xgb': {k: v for k, v in t3_xgb.items() if 'model' not in k},
    'T3_rf': {k: v for k, v in t3_rf.items() if 'model' not in k},
    'W_sets': {'T1': W_T1, 'T2': W_T2, 'T3': W_T3},
}
with open(os.path.join(OUT_DML, 'dml_model_weights.pkl'), 'wb') as f:
    pickle.dump(dml_weights, f)

# JSON版
dml_json = {}
for key in ['T1_xgb','T1_rf','T2_xgb','T2_rf','T3_xgb','T3_rf']:
    dml_json[key] = {k: (float(v) if isinstance(v, (np.floating, float)) else
                         int(v) if isinstance(v, (np.integer, int)) else str(v))
                     for k, v in dml_weights[key].items()}
dml_json['W_sets'] = dml_weights['W_sets']
with open(os.path.join(OUT_DML, 'dml_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dml_json, f, ensure_ascii=False, indent=2)

print(f"\n模型权重已保存: dml_model_weights.pkl, dml_results.json")
print("\n" + "=" * 70)
print("  阶段5完成")
print("=" * 70)
