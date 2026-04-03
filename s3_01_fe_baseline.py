"""
阶段3：面板双重固定效应基准回归（FE-1 / FE-2 / FE-3）
============================================================
数据：data1_modeling_ready.xlsx（异常值已清洗+XGB插补）
模型：linearmodels.PanelOLS，个体FE + 年份FE，个体聚类稳健SE
FE-1：全部核心解释变量 + 控制变量
FE-2：FE-1 + srh（检验srh是否为坏控制）
FE-3：仅3个treatment变量 + 控制（便于与DML对比）
============================================================
"""

import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.var_labels import VAR_MAP, get_zh, get_label, get_short
from utils.table_utils import save_three_line_table, format_coef

from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# ============================================================
# 0. 路径与参数
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
print("=== 1. 数据准备 ===")
df = pd.read_excel(DATA_PATH)
print(f"原始数据: {df.shape}")

# 1.1 筛选 >=2 波个体（FE可用）
waves_per_id = df.groupby('ID')['wave'].nunique()
ids_ge2 = waves_per_id[waves_per_id >= 2].index
df = df[df['ID'].isin(ids_ge2)].copy()
print(f"FE样本(>=2波): {len(ids_ge2)}个体, {len(df)}观测")

# 1.2 变量变换
df['hhcperc_log'] = np.log(df['hhcperc'] + 1)
df['fcamt_log'] = np.log(df['fcamt'] + 1)

# 1.3 province → 哑变量（province在FE中若不随时间变化会被吸收，
#     但部分个体可能跨省迁移，保留为控制更安全）
#     由于province是时间不变的（绑定户籍地），在个体FE中被吸收，无需加入
#     验证：
province_change = df.groupby('ID')['province'].nunique()
n_change = (province_change > 1).sum()
print(f"跨省迁移个体: {n_change} ({n_change/len(ids_ge2)*100:.2f}%)")
# 如果几乎不变，FE吸收省份效应，无需手动控制

# 1.4 设置面板索引
df = df.set_index(['ID', 'wave'])
print(f"面板索引已设置: {df.index.names}")

# ============================================================
# 2. 变量定义
# ============================================================
Y_VAR = 'cesd10'

# FE-1 核心解释变量（研究设计 §2.2.1）
CORE_X = [
    'social_activity_index',   # 社会参与
    'chronic_disease_count',   # 慢性病数量
    'body_discomfort_count',   # 身体不适
    'sleep',                   # 睡眠时长
    'hhcperc_log',             # 人均消费（对数）
    'ins',                     # 医疗保险
    'pension',                 # 养老保险
    'fcamt_log',               # 子女经济支持（对数）
    'activity_index',          # 身体活动指数
]

# 控制变量（时变，研究设计 §2.3.1）
# gender/edu/rural2被个体FE吸收，不纳入
# province基本不变，被个体FE吸收
# wave通过 time_effects=True 纳入年份FE
CTRL_Z = [
    'age',           # 年龄（within：同一个体随时间增长）
    'marry',         # 婚姻（可能因丧偶变化）
    'bmi',           # BMI
    'smoken',        # 吸烟
    'drinkev',       # 饮酒
    'retire',        # 退休
    'family_size',   # 家庭规模
    'hchild',        # 健在子女数
]

# FE-2 扩展变量
EXT_SRH = ['srh']

# FE-3 精简核心（仅3个treatment对应变量）
CORE_X_SLIM = [
    'social_activity_index',
    'chronic_disease_count',
    'sleep',
]


# ============================================================
# 3. 运行FE模型
# ============================================================
def run_fe(df, y_var, x_vars, model_name):
    """运行PanelOLS个体+年份FE，聚类稳健SE"""
    all_vars = [y_var] + x_vars
    sub = df[all_vars].dropna()
    n_obs = len(sub)
    n_ids = sub.index.get_level_values(0).nunique()

    formula = f"{y_var} ~ " + " + ".join([f"1" if False else v for v in x_vars])
    # 使用 EntityEffects + TimeEffects
    y = sub[y_var]
    X = sub[x_vars]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)

    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"观测: {n_obs}, 个体: {n_ids}")
    print(f"R²(within): {res.rsquared_within:.4f}")
    print(f"R²(between): {res.rsquared_between:.4f}")
    print(f"R²(overall): {res.rsquared_overall:.4f}")
    print(f"F-stat: {res.f_statistic.stat:.2f} (p={res.f_statistic.pval:.4e})")

    # 提取结果
    results = []
    for var in x_vars:
        coef = res.params[var]
        se = res.std_errors[var]
        t = res.tstats[var]
        p = res.pvalues[var]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        results.append({
            'variable': var,
            'coef': coef,
            'se': se,
            'tstat': t,
            'pvalue': p,
            'ci_low': ci_low,
            'ci_high': ci_high,
        })
        star = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f"  {var:30s}: {coef:8.4f}{star:3s}  ({se:.4f})  t={t:.2f}")

    res_df = pd.DataFrame(results)
    res_df['model'] = model_name
    res_df['n_obs'] = n_obs
    res_df['n_ids'] = n_ids
    res_df['r2_within'] = res.rsquared_within
    res_df['f_stat'] = res.f_statistic.stat
    res_df['f_pval'] = res.f_statistic.pval

    return res_df, res


# --- FE-1: 基准模型 ---
print("\n" + "#"*60)
print("# FE-1: 基准模型（全部核心解释变量 + 控制）")
print("#"*60)
fe1_df, fe1_res = run_fe(df, Y_VAR, CORE_X + CTRL_Z, 'FE-1(基准)')

# --- FE-2: +srh 扩展模型 ---
print("\n" + "#"*60)
print("# FE-2: 扩展模型（+srh）")
print("#"*60)
fe2_df, fe2_res = run_fe(df, Y_VAR, CORE_X + CTRL_Z + EXT_SRH, 'FE-2(+srh)')

# --- FE-3: 精简模型 ---
print("\n" + "#"*60)
print("# FE-3: 精简模型（仅3个treatment变量 + 控制）")
print("#"*60)
fe3_df, fe3_res = run_fe(df, Y_VAR, CORE_X_SLIM + CTRL_Z, 'FE-3(精简)')


# ============================================================
# 4. 保存原始结果
# ============================================================
all_results = pd.concat([fe1_df, fe2_df, fe3_df], ignore_index=True)
all_results.to_excel(os.path.join(OUT_FE, 'fe_regression_all_models.xlsx'), index=False)
print(f"\n原始结果已保存: fe_regression_all_models.xlsx")


# ============================================================
# 5. 生成论文用三线表
# ============================================================
print("\n=== 生成论文用回归表 ===")

# 合并所有变量（按出现顺序）
all_vars_order = CORE_X + EXT_SRH + CTRL_Z  # srh在核心和控制之间

# 构建表格
table_rows = []
for var in all_vars_order:
    row = {'变量': get_label(var)}
    for model_name, res_df in [('FE-1', fe1_df), ('FE-2', fe2_df), ('FE-3', fe3_df)]:
        match = res_df[res_df['variable'] == var]
        if len(match) > 0:
            r = match.iloc[0]
            coef_str, se_str = format_coef(r['coef'], r['se'], r['pvalue'])
            row[f'{model_name}'] = coef_str
            row[f'{model_name}_SE'] = se_str
        else:
            row[f'{model_name}'] = ''
            row[f'{model_name}_SE'] = ''
    table_rows.append(row)

table_df = pd.DataFrame(table_rows)

# 添加底部信息行
for model_name, res_df in [('FE-1', fe1_df), ('FE-2', fe2_df), ('FE-3', fe3_df)]:
    r = res_df.iloc[0]
    n_obs = int(r['n_obs'])
    n_ids = int(r['n_ids'])
    r2w = r['r2_within']
    f_stat = r['f_stat']

    # N行
    obs_row = {'变量': 'N (观测)'}
    obs_row[f'{model_name}'] = f'{n_obs:,}'
    obs_row[f'{model_name}_SE'] = ''
    # 查找或追加
    found = False
    for tr in table_rows:
        if tr['变量'] == 'N (观测)':
            tr.update(obs_row)
            found = True
    if not found:
        table_rows.append(obs_row)

    ids_row = {'变量': '个体数'}
    ids_row[f'{model_name}'] = f'{n_ids:,}'
    ids_row[f'{model_name}_SE'] = ''
    found = False
    for tr in table_rows:
        if tr['变量'] == '个体数':
            tr.update(ids_row)
            found = True
    if not found:
        table_rows.append(ids_row)

    r2_row = {'变量': 'R²(within)'}
    r2_row[f'{model_name}'] = f'{r2w:.4f}'
    r2_row[f'{model_name}_SE'] = ''
    found = False
    for tr in table_rows:
        if tr['变量'] == 'R²(within)':
            tr.update(r2_row)
            found = True
    if not found:
        table_rows.append(r2_row)

    f_row = {'变量': 'F统计量'}
    f_row[f'{model_name}'] = f'{f_stat:.2f}'
    f_row[f'{model_name}_SE'] = ''
    found = False
    for tr in table_rows:
        if tr['变量'] == 'F统计量':
            tr.update(f_row)
            found = True
    if not found:
        table_rows.append(f_row)

# 重建DataFrame
table_df = pd.DataFrame(table_rows)
# 填充空值
table_df = table_df.fillna('')

# 重排列顺序
col_order = ['变量', 'FE-1', 'FE-1_SE', 'FE-2', 'FE-2_SE', 'FE-3', 'FE-3_SE']
for c in col_order:
    if c not in table_df.columns:
        table_df[c] = ''
table_df = table_df[col_order]

# 保存
save_three_line_table(
    table_df.set_index('变量'),
    os.path.join(OUT_TABLE, 'fe_baseline_table'),
    title='表 X  面板固定效应基准回归结果',
    note='注：*** p<0.01, ** p<0.05, * p<0.1。括号内为个体聚类稳健标准误。'
         '所有模型包含个体固定效应和年份固定效应。'
         'gender、edu、rural2被个体固定效应吸收，不在表中显示。'
)

# 额外保存一份纯净xlsx
table_df.to_excel(os.path.join(OUT_FE, 'fe_baseline_table_clean.xlsx'), index=False)
print("论文用回归表已保存")


# ============================================================
# 6. srh坏控制检验
# ============================================================
print("\n=== srh坏控制检验（FE-1 vs FE-2系数对比）===")
compare_vars = CORE_X
print(f"{'变量':<35s} {'FE-1系数':>10s} {'FE-2系数':>10s} {'变化%':>8s} {'方向':>6s}")
print("-" * 75)
for var in compare_vars:
    c1 = fe1_df[fe1_df['variable']==var]['coef'].values[0]
    c2 = fe2_df[fe2_df['variable']==var]['coef'].values[0]
    if abs(c1) > 1e-6:
        pct = (c2 - c1) / abs(c1) * 100
    else:
        pct = 0
    direction = '↓' if abs(c2) < abs(c1) else '↑'
    print(f"  {get_label(var):<33s} {c1:10.4f} {c2:10.4f} {pct:7.1f}% {direction:>4s}")

srh_coef = fe2_df[fe2_df['variable']=='srh']['coef'].values[0]
srh_p = fe2_df[fe2_df['variable']=='srh']['pvalue'].values[0]
print(f"\n  srh自身系数: {srh_coef:.4f} (p={srh_p:.4e})")
print("  若加入srh后其他核心变量系数大幅缩小，说明srh可能吸收了treatment通过主观感知的中介路径（坏控制）。")

print("\n=== 阶段3完成 ===")
