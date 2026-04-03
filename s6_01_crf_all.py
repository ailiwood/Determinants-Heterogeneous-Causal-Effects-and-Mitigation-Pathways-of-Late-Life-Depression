"""
阶段6：CRF异质性因果效应估计（T1 / T2 / T3）
============================================================
T1: 社会参与（连续, CausalForestDML）→ ITE分布 + GATE + BLP
T2: 高慢病负担≥2（二元, CausalForestDML）→ ITE分布 + GATE + BLP
T3: 睡眠异常（二元, CausalForestDML）→ ITE分布 + GATE + BLP

X (effect modifiers): gender, rural2, region, age, edu, hhcperc_log, marry
W (controls): treatment-specific安全控制集（除X外）
model_y: XGBRegressor, model_t: XGBClassifier/XGBRegressor
n_estimators=2000, min_samples_leaf=30, cv=5
============================================================
"""

import pandas as pd
import numpy as np
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_config import *
from utils.var_labels import get_zh, get_label, VAR_MAP
from utils.table_utils import save_three_line_table, format_coef

from econml.dml import CausalForestDML
from xgboost import XGBRegressor, XGBClassifier
from scipy import stats

# ============================================================
# 0. 路径与参数
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAUSAL_PATH = os.path.join(ROOT, 'results/03_causal_sample/causal_sample_lagged.xlsx')
OUT_CRF = os.path.join(ROOT, 'results/05_crf')
OUT_TABLE = os.path.join(ROOT, 'results/08_tables_for_paper')
OUT_FIG = os.path.join(ROOT, 'results/09_figures_for_paper')

SEED = 42
N_ESTIMATORS = 2000
MIN_LEAF = 30
CV = 5

# ============================================================
# 1. 读取因果样本
# ============================================================
print("=" * 70)
print("  阶段6：CRF异质性因果效应估计")
print("=" * 70)

df = pd.read_excel(CAUSAL_PATH)
print(f"因果样本: {df.shape}, 个体: {df['ID'].nunique()}")

# province → 数值编码
from sklearn.preprocessing import LabelEncoder
le_prov = LabelEncoder()
df['province_enc'] = le_prov.fit_transform(df['province'].astype(str))

# ============================================================
# 2. 定义X和W变量集
# ============================================================
# X: 异质性特征（effect modifiers, 用于forest分裂）
X_VARS = ['gender', 'rural2', 'region', 'age', 'edu', 'hhcperc_log', 'marry']

# W: 控制变量（残差化, 不在X中的通用安全控制集 + treatment-specific）
W_COMMON_CRF = ['province_enc', 'ins', 'pension', 'retire', 'bmi',
                'smoken', 'drinkev', 'family_size', 'hchild',
                'lag_cesd10', 'wave']

# Treatment-specific W
W_T1 = W_COMMON_CRF + ['chronic_disease_count', 'sleep', 'fcamt_log']
W_T2 = W_COMMON_CRF + ['sleep', 'social_activity_index', 'fcamt_log']
W_T3 = W_COMMON_CRF + ['chronic_disease_count', 'social_activity_index', 'fcamt_log']

Y_VAR = 'cesd10_t'

# 分组标签
GROUP_LABELS = {
    'gender': {0: '女性', 1: '男性'},
    'rural2': {0: '农村户口', 1: '城镇户口'},
    'region': {1: '东部', 2: '中部', 3: '西部'},
}

# ============================================================
# 3. Learner工厂
# ============================================================
def get_model_y():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)

def get_model_t_cls():
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         eval_metric='logloss')

def get_model_t_reg():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)


# ============================================================
# 4. CRF估计函数
# ============================================================
def run_crf(df, y_col, d_col, x_cols, w_cols, treatment_type='binary',
            out_dir=None, treatment_name=''):
    """
    运行CausalForestDML并输出全部结果。

    Parameters
    ----------
    treatment_type : 'binary' or 'continuous'
    """
    print(f"\n{'#' * 70}")
    print(f"# CRF: {treatment_name} ({treatment_type} treatment)")
    print(f"{'#' * 70}")

    # 准备数据
    all_cols = [y_col, d_col] + x_cols + w_cols
    all_cols = list(dict.fromkeys(all_cols))  # 去重保序
    data = df[all_cols].dropna()
    print(f"  有效样本: {len(data)}")

    Y = data[y_col].values
    T = data[d_col].values
    X = data[x_cols].values
    W = data[w_cols].values

    # CausalForestDML使用R-learner框架，model_t始终为回归器
    # （即使treatment为二元，也通过E[T|W]残差化而非倾向得分）
    model_t = get_model_t_reg()

    # 拟合CRF
    print(f"  拟合CausalForestDML (n_estimators={N_ESTIMATORS}, min_leaf={MIN_LEAF})...")
    est = CausalForestDML(
        model_y=get_model_y(),
        model_t=model_t,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_LEAF,
        max_depth=None,
        cv=CV,
        random_state=SEED,
        n_jobs=-1,
    )
    est.fit(Y, T, X=X, W=W)
    print("  拟合完成")

    # --- 4.1 ITE估计 ---
    ite = est.effect(X).flatten()
    ite_inf = est.effect_inference(X)
    ite_lb = ite_inf.conf_int(alpha=0.05)[0].flatten()
    ite_ub = ite_inf.conf_int(alpha=0.05)[1].flatten()

    print(f"\n  --- ITE分布 ---")
    print(f"  Mean ITE:   {ite.mean():.4f}")
    print(f"  Median ITE: {np.median(ite):.4f}")
    print(f"  Std ITE:    {ite.std():.4f}")
    print(f"  Range:      [{ite.min():.4f}, {ite.max():.4f}]")

    # --- 4.2 ATE（从CRF） ---
    ate_inf = est.ate_inference(X)
    ate = float(ate_inf.mean_point)
    ate_se = float(ate_inf.stderr_mean)
    ate_ci = ate_inf.conf_int_mean(alpha=0.05)
    ate_pval = float(ate_inf.pvalue(value=0))
    print(f"\n  --- ATE (CRF) ---")
    print(f"  ATE = {ate:.4f} (SE={ate_se:.4f}), "
          f"95%CI=[{ate_ci[0]:.4f}, {ate_ci[1]:.4f}], p={float(ate_pval):.4e}")

    # --- 4.3 CATE分位数 ---
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    cate_q = np.quantile(ite, quantiles)
    print(f"\n  --- CATE分位数 ---")
    for q, v in zip(quantiles, cate_q):
        print(f"  P{int(q*100):02d}: {v:.4f}")

    # --- 4.4 分组GATE ---
    print(f"\n  --- 分组GATE ---")
    gate_rows = []
    for g_idx, g_var in enumerate(['gender', 'rural2', 'region']):
        g_vals = data[g_var].values
        for g_val, g_label in GROUP_LABELS[g_var].items():
            mask = g_vals == g_val
            if mask.sum() < 30:
                continue
            g_ite = ite[mask]
            g_mean = g_ite.mean()
            g_se = g_ite.std() / np.sqrt(mask.sum())
            g_ci_l = g_mean - 1.96 * g_se
            g_ci_u = g_mean + 1.96 * g_se
            # t-test: GATE != 0
            t_stat = g_mean / g_se if g_se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            gate_rows.append({
                '分组变量': get_zh(g_var) if g_var in VAR_MAP else g_var,
                '分组': g_label,
                'N': int(mask.sum()),
                'GATE': round(g_mean, 4),
                'SE': round(g_se, 4),
                '95% CI下界': round(g_ci_l, 4),
                '95% CI上界': round(g_ci_u, 4),
                'p值': p_val,
            })
            print(f"  {g_var}={g_label}: GATE={g_mean:.4f} (SE={g_se:.4f}), "
                  f"95%CI=[{g_ci_l:.4f}, {g_ci_u:.4f}], N={mask.sum()}")

    gate_df = pd.DataFrame(gate_rows)

    # --- 4.5 组间差异检验 ---
    print(f"\n  --- 组间差异检验 ---")
    diff_rows = []
    for g_var in ['gender', 'rural2', 'region']:
        g_vals = data[g_var].values
        labels = GROUP_LABELS[g_var]
        groups_ite = {}
        for g_val, g_label in labels.items():
            mask = g_vals == g_val
            if mask.sum() >= 30:
                groups_ite[g_label] = ite[mask]

        if g_var in ['gender', 'rural2']:
            # 两组t检验
            keys = list(groups_ite.keys())
            if len(keys) == 2:
                t_stat, p_val = stats.ttest_ind(groups_ite[keys[0]], groups_ite[keys[1]])
                diff = groups_ite[keys[0]].mean() - groups_ite[keys[1]].mean()
                print(f"  {g_var} ({keys[0]} vs {keys[1]}): diff={diff:.4f}, t={t_stat:.3f}, p={p_val:.4e}")
                diff_rows.append({
                    '分组变量': get_zh(g_var) if g_var in VAR_MAP else g_var,
                    '比较': f"{keys[0]} vs {keys[1]}",
                    '效应差': round(diff, 4),
                    't统计量': round(t_stat, 3),
                    'p值': p_val,
                })
        elif g_var == 'region':
            # 三组ANOVA
            groups = [v for v in groups_ite.values()]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                print(f"  {g_var} (东/中/西): F={f_stat:.3f}, p={p_val:.4e}")
                diff_rows.append({
                    '分组变量': get_zh(g_var) if g_var in VAR_MAP else g_var,
                    '比较': '东/中/西 ANOVA',
                    '效应差': np.nan,
                    't统计量': round(f_stat, 3),
                    'p值': p_val,
                })
    diff_df = pd.DataFrame(diff_rows) if diff_rows else pd.DataFrame()

    # --- 4.6 BLP检验 (Best Linear Projection) ---
    print(f"\n  --- BLP异质性检验 ---")
    # BLP: regress ITE on X to test if heterogeneity is systematic
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm

    X_centered = X - X.mean(axis=0)
    X_blp = sm.add_constant(X_centered)
    blp_model = sm.OLS(ite, X_blp).fit(cov_type='HC1')

    # F-test on all X coefficients (excluding constant)
    r_matrix = np.eye(len(x_cols) + 1)[1:]  # exclude intercept
    f_test = blp_model.f_test(r_matrix)
    blp_f = float(f_test.fvalue)
    blp_p = float(f_test.pvalue)
    print(f"  BLP F-test (H0: 无系统异质性): F={blp_f:.3f}, p={blp_p:.4e}")
    print(f"  ATE (BLP intercept): {blp_model.params[0]:.4f} (SE={blp_model.bse[0]:.4f})")

    blp_results = pd.DataFrame({
        '变量': ['常数项 (ATE)'] + x_cols,
        '系数': blp_model.params.round(4),
        'SE': blp_model.bse.round(4),
        't值': blp_model.tvalues.round(3),
        'p值': blp_model.pvalues,
    })
    print("\n  BLP系数:")
    for _, row in blp_results.iterrows():
        sig = '***' if row['p值'] < 0.01 else '**' if row['p值'] < 0.05 else '*' if row['p值'] < 0.1 else ''
        print(f"    {row['变量']:20s}: {row['系数']:+.4f}{sig} (SE={row['SE']:.4f})")

    # --- 4.7 变量重要度 ---
    print(f"\n  --- 变量重要度 (X feature importances) ---")
    fi = est.feature_importances_
    fi_df = pd.DataFrame({
        '变量': x_cols,
        '变量(中文)': [get_zh(v) if v in VAR_MAP else v for v in x_cols],
        '重要度': fi.round(4),
    }).sort_values('重要度', ascending=False)
    print(fi_df.to_string(index=False))

    # --- 4.8 高受益人群画像 (ITE前20%) ---
    print(f"\n  --- 高受益/高风险人群画像 (ITE前20%) ---")
    # 对于保护性treatment(T1)，受益最大=ITE最负；对于风险性(T2,T3)，受害最大=ITE最大
    if treatment_type == 'continuous' or 'social' in d_col.lower():
        # 保护效应：ITE最负的20%
        threshold = np.percentile(ite, 20)
        top_mask = ite <= threshold
        profile_label = "保护效应最强20%"
    else:
        # 风险效应：ITE最大的20%
        threshold = np.percentile(ite, 80)
        top_mask = ite >= threshold
        profile_label = "风险效应最强20%"

    top_data = data[x_cols].iloc[top_mask]
    all_data = data[x_cols]
    profile_rows = []
    for v in x_cols:
        top_mean = top_data[v].mean()
        all_mean = all_data[v].mean()
        profile_rows.append({
            '变量': get_zh(v) if v in VAR_MAP else v,
            f'{profile_label}均值': round(top_mean, 3),
            '全样本均值': round(all_mean, 3),
            '差异': round(top_mean - all_mean, 3),
        })
    profile_df = pd.DataFrame(profile_rows)
    print(profile_df.to_string(index=False))

    # ============================================================
    # 5. 保存结果
    # ============================================================
    os.makedirs(out_dir, exist_ok=True)

    # 5.1 ITE分布图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ite, bins=60, color=COLORS['primary'], alpha=0.7, edgecolor='white', density=True)
    ax.axvline(ite.mean(), color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'Mean ITE = {ite.mean():.3f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('个体处理效应 (ITE)')
    ax.set_ylabel('密度')
    ax.set_title(f'ITE分布: {treatment_name}')
    ax.legend()
    ite_fig_path = os.path.join(out_dir, f'ite_distribution.png')
    fig.savefig(ite_fig_path)
    plt.close(fig)
    print(f"\n  ITE分布图: {ite_fig_path}")

    # 同时保存到论文图表文件夹
    ite_paper_path = os.path.join(OUT_FIG, f'crf_ite_distribution_{d_col}.png')
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(ite, bins=60, color=COLORS['primary'], alpha=0.7, edgecolor='white', density=True)
    ax2.axvline(ite.mean(), color=COLORS['secondary'], linestyle='--', linewidth=2,
                label=f'Mean ITE = {ite.mean():.3f}')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('个体处理效应 (ITE)')
    ax2.set_ylabel('密度')
    ax2.set_title(f'ITE分布: {treatment_name}')
    ax2.legend()
    fig2.savefig(ite_paper_path)
    plt.close(fig2)

    # 5.2 GATE分组柱状图
    for g_var in ['gender', 'rural2', 'region']:
        g_gate = gate_df[gate_df['分组变量'] == (get_zh(g_var) if g_var in VAR_MAP else g_var)]
        if g_gate.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        x_pos = range(len(g_gate))
        bars = ax.bar(x_pos, g_gate['GATE'].values, color=COLOR_LIST[:len(g_gate)],
                      alpha=0.8, edgecolor='white', width=0.6)
        # 误差棒
        yerr_lower = g_gate['GATE'].values - g_gate['95% CI下界'].values
        yerr_upper = g_gate['95% CI上界'].values - g_gate['GATE'].values
        ax.errorbar(x_pos, g_gate['GATE'].values,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='none', color='black', capsize=5, linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(g_gate['分组'].values)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('GATE (分组平均处理效应)')
        g_var_zh = get_zh(g_var) if g_var in VAR_MAP else g_var
        ax.set_title(f'{treatment_name} — 按{g_var_zh}的GATE')
        fig.savefig(os.path.join(out_dir, f'gate_by_{g_var}.png'))
        fig.savefig(os.path.join(OUT_FIG, f'crf_gate_{d_col}_by_{g_var}.png'))
        plt.close(fig)

    # 5.3 变量重要度图
    fig, ax = plt.subplots(figsize=(7, 5))
    fi_sorted = fi_df.sort_values('重要度', ascending=True)
    ax.barh(range(len(fi_sorted)), fi_sorted['重要度'].values,
            color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(fi_sorted)))
    ax.set_yticklabels(fi_sorted['变量(中文)'].values)
    ax.set_xlabel('变量重要度')
    ax.set_title(f'{treatment_name} — X变量重要度')
    fig.savefig(os.path.join(out_dir, 'feature_importance.png'))
    fig.savefig(os.path.join(OUT_FIG, f'crf_feature_importance_{d_col}.png'))
    plt.close(fig)

    # 5.4 保存Excel表格
    gate_df.to_excel(os.path.join(out_dir, 'gate_results.xlsx'), index=False)
    fi_df.to_excel(os.path.join(out_dir, 'feature_importance.xlsx'), index=False)
    blp_results.to_excel(os.path.join(out_dir, 'blp_results.xlsx'), index=False)
    profile_df.to_excel(os.path.join(out_dir, 'high_benefit_profile.xlsx'), index=False)
    if not diff_df.empty:
        diff_df.to_excel(os.path.join(out_dir, 'gate_diff_test.xlsx'), index=False)

    # CATE分位数表
    cate_q_df = pd.DataFrame({
        '分位数': [f'P{int(q*100)}' for q in quantiles],
        'CATE': cate_q.round(4),
    })
    cate_q_df.to_excel(os.path.join(out_dir, 'cate_quantiles.xlsx'), index=False)

    # ITE个体数据（带CI）
    ite_df = pd.DataFrame({
        'ITE': ite.round(4),
        'CI_lower': ite_lb.round(4),
        'CI_upper': ite_ub.round(4),
    })
    for v in x_cols:
        ite_df[v] = data[v].values
    ite_df.to_excel(os.path.join(out_dir, 'ite_individual.xlsx'), index=False)

    # 5.5 汇总结果dict
    results = {
        'treatment': d_col,
        'treatment_name': treatment_name,
        'treatment_type': treatment_type,
        'n_obs': len(data),
        'ATE_crf': float(ate),
        'ATE_se': float(ate_se),
        'ATE_ci': [float(ate_ci[0]), float(ate_ci[1])],
        'ATE_pvalue': float(ate_pval),
        'ITE_mean': float(ite.mean()),
        'ITE_median': float(np.median(ite)),
        'ITE_std': float(ite.std()),
        'ITE_min': float(ite.min()),
        'ITE_max': float(ite.max()),
        'CATE_quantiles': {f'P{int(q*100)}': float(v) for q, v in zip(quantiles, cate_q)},
        'BLP_F': blp_f,
        'BLP_p': blp_p,
        'feature_importances': {v: float(fi_df[fi_df['变量']==v]['重要度'].values[0])
                                 for v in x_cols},
        'X_vars': x_cols,
        'W_vars': w_cols,
    }

    # 保存JSON
    with open(os.path.join(out_dir, 'crf_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  所有结果已保存到: {out_dir}")

    return results, gate_df, diff_df, blp_results, fi_df, profile_df, est


# ============================================================
# 6. 运行三个Treatment
# ============================================================

# --- T1: 社会参与（连续） ---
t1_results, t1_gate, t1_diff, t1_blp, t1_fi, t1_profile, t1_est = run_crf(
    df, Y_VAR, 'T1_social', X_VARS, W_T1,
    treatment_type='continuous',
    out_dir=os.path.join(OUT_CRF, 'crf_T1_social'),
    treatment_name='T1: 社会参与 (连续)'
)

# --- T2: 多病共存（二元） ---
t2_results, t2_gate, t2_diff, t2_blp, t2_fi, t2_profile, t2_est = run_crf(
    df, Y_VAR, 'T2_chronic', X_VARS, W_T2,
    treatment_type='binary',
    out_dir=os.path.join(OUT_CRF, 'crf_T2_chronic'),
    treatment_name='T2: 多病共存≥2种 (二元)'
)

# --- T3: 睡眠异常（二元, 扩展） ---
t3_results, t3_gate, t3_diff, t3_blp, t3_fi, t3_profile, t3_est = run_crf(
    df, Y_VAR, 'T3_sleep', X_VARS, W_T3,
    treatment_type='binary',
    out_dir=os.path.join(OUT_CRF, 'crf_T3_sleep'),
    treatment_name='T3: 睡眠异常 (二元)'
)

# ============================================================
# 7. 论文用GATE汇总三线表
# ============================================================
print("\n" + "=" * 70)
print("  生成论文用GATE汇总表")
print("=" * 70)

# 合并三个treatment的GATE
all_gate_rows = []
for d_col, t_name, gate, diff in [
    ('T1_social', 'T1: 社会参与', t1_gate, t1_diff),
    ('T2_chronic', 'T2: 多病共存', t2_gate, t2_diff),
    ('T3_sleep', 'T3: 睡眠异常', t3_gate, t3_diff),
]:
    for _, row in gate.iterrows():
        c, s = format_coef(row['GATE'], row['SE'], row['p值'])
        all_gate_rows.append({
            'Treatment': t_name,
            '分组变量': row['分组变量'],
            '分组': row['分组'],
            'GATE': c,
            'SE': s,
            '95% CI': f"[{row['95% CI下界']:.3f}, {row['95% CI上界']:.3f}]",
            'N': f"{row['N']:,}",
        })

all_gate_df = pd.DataFrame(all_gate_rows)

save_three_line_table(
    all_gate_df.set_index('Treatment'),
    os.path.join(OUT_TABLE, 'crf_gate_summary'),
    title='表 X  因果随机森林分组平均处理效应（GATE）',
    note='注：*** p<0.01, ** p<0.05, * p<0.1。'
         'GATE为各分组内ITE的均值，标准误由ITE的组内标准差/√N计算。'
         '95%置信区间基于正态近似。'
         'CausalForestDML(n_estimators=2000, min_samples_leaf=30, cv=5)。'
         '第一阶段learner: XGBRegressor/XGBClassifier(n_estimators=500)。'
         'Treatment均取t-1期值，Outcome取t期值（lagged结构）。',
)

# ============================================================
# 8. BLP汇总表
# ============================================================
blp_summary_rows = []
for d_col, t_name, res in [
    ('T1_social', 'T1: 社会参与', t1_results),
    ('T2_chronic', 'T2: 多病共存', t2_results),
    ('T3_sleep', 'T3: 睡眠异常', t3_results),
]:
    blp_summary_rows.append({
        'Treatment': t_name,
        'BLP F值': round(res['BLP_F'], 3),
        'BLP p值': f"{res['BLP_p']:.4e}",
        '异质性显著': '是***' if res['BLP_p'] < 0.01 else '是**' if res['BLP_p'] < 0.05 else '是*' if res['BLP_p'] < 0.1 else '否',
        'ATE (CRF)': round(res['ATE_crf'], 4),
        'ITE标准差': round(res['ITE_std'], 4),
    })

blp_summary_df = pd.DataFrame(blp_summary_rows)
blp_summary_df.to_excel(os.path.join(OUT_CRF, 'blp_summary.xlsx'), index=False)
print("\nBLP异质性检验汇总:")
print(blp_summary_df.to_string(index=False))

# ============================================================
# 9. 保存模型权重
# ============================================================
crf_weights = {
    'T1': t1_results,
    'T2': t2_results,
    'T3': t3_results,
    'X_vars': X_VARS,
    'W_sets': {'T1': W_T1, 'T2': W_T2, 'T3': W_T3},
    'params': {
        'n_estimators': N_ESTIMATORS,
        'min_samples_leaf': MIN_LEAF,
        'cv': CV,
        'seed': SEED,
    },
}
with open(os.path.join(OUT_CRF, 'crf_model_weights.pkl'), 'wb') as f:
    pickle.dump(crf_weights, f)

with open(os.path.join(OUT_CRF, 'crf_results_all.json'), 'w', encoding='utf-8') as f:
    json.dump(crf_weights, f, ensure_ascii=False, indent=2)

print(f"\n模型权重已保存: crf_model_weights.pkl, crf_results_all.json")

# ============================================================
# 10. 最终汇总
# ============================================================
print("\n" + "=" * 70)
print("  阶段6：CRF异质性因果效应 —— 完成")
print("=" * 70)

print("\n各Treatment CRF-ATE:")
for name, res in [('T1 社会参与', t1_results), ('T2 多病共存', t2_results), ('T3 睡眠异常', t3_results)]:
    sig = '***' if res['ATE_pvalue'] < 0.01 else '**' if res['ATE_pvalue'] < 0.05 else '*' if res['ATE_pvalue'] < 0.1 else ''
    print(f"  {name}: ATE={res['ATE_crf']:.4f}{sig} (SE={res['ATE_se']:.4f}), "
          f"ITE std={res['ITE_std']:.4f}, BLP F={res['BLP_F']:.3f} (p={res['BLP_p']:.4e})")

print("\n产出文件:")
print(f"  results/05_crf/crf_T1_social/  (ITE, GATE, BLP, 变量重要度, 人群画像)")
print(f"  results/05_crf/crf_T2_chronic/ (同上)")
print(f"  results/05_crf/crf_T3_sleep/   (同上)")
print(f"  results/05_crf/crf_model_weights.pkl")
print(f"  results/05_crf/crf_results_all.json")
print(f"  results/08_tables_for_paper/crf_gate_summary.xlsx/.docx")
print(f"  results/09_figures_for_paper/crf_ite_distribution_*.png")
print(f"  results/09_figures_for_paper/crf_gate_*_by_*.png")
print(f"  results/09_figures_for_paper/crf_feature_importance_*.png")
