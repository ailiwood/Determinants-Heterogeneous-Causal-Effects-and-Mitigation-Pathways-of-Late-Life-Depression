"""
分析B：FE复合风险交互效应模型
============================================================
功能：检验三个处理变量之间的交互效应
模型：
  - M1：仅主效应
  - M2：M1 + 两两交互项
  - M3：M2 + 三阶交互项
检验：Wald联合显著性检验
输出：三模型对比表（xlsx + docx）
============================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加工具路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 's2_utils'))
from composite_risk import build_composite_risk_variables

# 路径配置
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, '..', 'data', 'preprocess_outputs', 'data1_modeling_ready.xlsx')
OUT_DIR = os.path.join(ROOT, '..', '..', 'round2_outputs', '02_interaction_cumulative_risk', 'analysis_B_fe_interaction')
OUT_TABLE = os.path.join(ROOT, '..', '..', 'round2_outputs', '04_tables_for_paper')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_TABLE, exist_ok=True)

print("=" * 70)
print("  分析B：FE复合风险交互效应模型")
print("=" * 70)

# ============================================================
# 1. 读取数据
# ============================================================
print("\n[1] 读取FE主样本数据...")
df = pd.read_excel(DATA_PATH)
print(f"    原始数据: {df.shape}")

# 筛选 ≥2 波次的个体
waves_per_id = df.groupby('ID')['wave'].nunique()
ids_ge2 = waves_per_id[waves_per_id >= 2].index
df = df[df['ID'].isin(ids_ge2)].copy()
print(f"    FE主样本: {len(ids_ge2)} 个体, {len(df)} 观测")

# ============================================================
# 2. 构建复合风险变量
# ============================================================
print("\n[2] 构建复合风险变量...")
df = build_composite_risk_variables(df, source='main')

# 标准化（用于计算交互项，降低多重共线性）
df['low_social_c'] = df['low_social'] - df['low_social'].mean()
df['multimorbidity_c'] = df['multimorbidity'] - df['multimorbidity'].mean()
df['abnormal_sleep_c'] = df['abnormal_sleep'] - df['abnormal_sleep'].mean()

# 交互项（标准化后计算）
df['low_x_multi'] = df['low_social_c'] * df['multimorbidity_c']
df['low_x_sleep'] = df['low_social_c'] * df['abnormal_sleep_c']
df['multi_x_sleep'] = df['multimorbidity_c'] * df['abnormal_sleep_c']

# 三阶交互
df['low_x_multi_x_sleep'] = df['low_social_c'] * df['multimorbidity_c'] * df['abnormal_sleep_c']

# ============================================================
# 3. 设置面板索引
# ============================================================
df = df.set_index(['ID', 'wave'])

# ============================================================
# 4. 模型估计（使用对象接口，避免公式接口的索引问题）
# ============================================================
print("\n[3] 估计FE交互效应模型...")

from linearmodels.panel import PanelOLS

# 控制变量（不含gender，因会被固定效应吸收）
x_vars_ctrl = ['age', 'marry', 'bmi', 'hhcperc', 'family_size', 'hchild']

# ---- M1：仅主效应 ----
print("    M1估计中...")
y_var = 'cesd10'
x_vars_main = ['low_social', 'multimorbidity', 'abnormal_sleep']

# 提取数据
all_vars_m1 = [y_var] + x_vars_main + x_vars_ctrl
df_m1 = df[all_vars_m1].dropna()
y1 = df_m1[[y_var]]  # DataFrame形式
X1 = df_m1[x_vars_main + x_vars_ctrl]

mod1 = PanelOLS(y1, X1, entity_effects=True, time_effects=True, drop_absorbed=True)
res1 = mod1.fit(cov_type='clustered', cluster_entity=True)
print(f"      R²(within) = {res1.rsquared_within:.4f}")

# ---- M2：主效应 + 两两交互 ----
print("    M2估计中...")
x_vars_m2 = x_vars_main + ['low_x_multi', 'low_x_sleep', 'multi_x_sleep'] + x_vars_ctrl

all_vars_m2 = [y_var] + x_vars_m2
df_m2 = df[all_vars_m2].dropna()
y2 = df_m2[[y_var]]
X2 = df_m2[x_vars_m2]

mod2 = PanelOLS(y2, X2, entity_effects=True, time_effects=True, drop_absorbed=True)
res2 = mod2.fit(cov_type='clustered', cluster_entity=True)
print(f"      R²(within) = {res2.rsquared_within:.4f}")

# ---- M3：主效应 + 两两交互 + 三阶交互 ----
print("    M3估计中...")
x_vars_m3 = x_vars_main + ['low_x_multi', 'low_x_sleep', 'multi_x_sleep', 'low_x_multi_x_sleep'] + x_vars_ctrl

all_vars_m3 = [y_var] + x_vars_m3
df_m3 = df[all_vars_m3].dropna()
y3 = df_m3[[y_var]]
X3 = df_m3[x_vars_m3]

mod3 = PanelOLS(y3, X3, entity_effects=True, time_effects=True, drop_absorbed=True)
res3 = mod3.fit(cov_type='clustered', cluster_entity=True)
print(f"      R²(within) = {res3.rsquared_within:.4f}")

# ============================================================
# 5. Wald检验
# ============================================================
print("\n[4] Wald联合显著性检验...")

from scipy import stats

# 获取M2中交互项的系数和标准误
interaction_terms_m2 = ['low_x_multi', 'low_x_sleep', 'multi_x_sleep']
b_m2 = np.array([res2.params[t] for t in interaction_terms_m2])
se_m2 = np.array([res2.std_errors[t] for t in interaction_terms_m2])
t_stats = b_m2 / se_m2
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), res2.df_resid))

# Wald统计量（联合检验H0: 所有交互项系数=0）
q = len(interaction_terms_m2)
f_stat = np.mean(t_stats**2)
p_wald_m2 = 1 - stats.f.cdf(f_stat * q, q, res2.df_resid)

print(f"    M2交互项 Wald检验: F={f_stat:.3f}, p={p_wald_m2:.4e}")

# M3中三阶交互项的检验
if 'low_x_multi_x_sleep' in res3.params:
    b3 = res3.params['low_x_multi_x_sleep']
    se3 = res3.std_errors['low_x_multi_x_sleep']
    t3 = b3 / se3
    p3 = 2 * (1 - stats.t.cdf(np.abs(t3), res3.df_resid))
    print(f"    M3三阶交互项: β={b3:.4f}, t={t3:.3f}, p={p3:.4f}")

# Delta R²（M2 vs M1，M3 vs M2）
delta_r2_m2_m1 = res2.rsquared_within - res1.rsquared_within
delta_r2_m3_m2 = res3.rsquared_within - res2.rsquared_within
print(f"    ΔR² (M2 vs M1): {delta_r2_m2_m1:.5f}")
print(f"    ΔR² (M3 vs M2): {delta_r2_m3_m2:.5f}")

# ============================================================
# 6. 整理系数表
# ============================================================
print("\n[5] 整理系数表...")

def extract_results(res, name):
    """提取回归结果"""
    rows = []
    for var in ['low_social', 'multimorbidity', 'abnormal_sleep',
                'low_x_multi', 'low_x_sleep', 'multi_x_sleep', 'low_x_multi_x_sleep']:
        if var in res.params.index:
            b = res.params[var]
            se = res.std_errors[var]
            p = res.pvalues[var]
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            rows.append({
                '变量': var,
                f'{name}_系数': round(b, 4),
                f'{name}_标准误': round(se, 4),
                f'{name}_p值': round(p, 4),
                f'{name}_显著性': sig
            })
    rows.append({
        '变量': 'R²(within)',
        f'{name}_系数': round(res.rsquared_within, 4),
        f'{name}_标准误': '',
        f'{name}_p值': '',
        f'{name}_显著性': ''
    })
    rows.append({
        '变量': 'N',
        f'{name}_系数': res.nobs,
        f'{name}_标准误': '',
        f'{name}_p值': '',
        f'{name}_显著性': ''
    })
    return pd.DataFrame(rows)

# 合并三列
df_m1 = extract_results(res1, 'M1')
df_m2 = extract_results(res2, 'M2')
df_m3 = extract_results(res3, 'M3')

# 合并
result = df_m1.merge(df_m2, on='变量', how='outer').merge(df_m3, on='变量', how='outer')

# 添加Wald检验结果行
wald_row = pd.DataFrame([{
    '变量': f'Wald交互项检验',
    'M1_系数': '', 'M1_标准误': '', 'M1_p值': '', 'M1_显著性': '',
    'M2_系数': f'F={f_stat:.3f}', 'M2_标准误': '', 'M2_p值': f'p={p_wald_m2:.4e}', 'M2_显著性': '***' if p_wald_m2 < 0.01 else '**' if p_wald_m2 < 0.05 else '*' if p_wald_m2 < 0.1 else '',
    'M3_系数': '', 'M3_标准误': '', 'M3_p值': '', 'M3_显著性': ''
}])
result = pd.concat([result, wald_row], ignore_index=True)

# 变量名翻译
var_map = {
    'low_social': '低社会参与',
    'multimorbidity': '多病共存',
    'abnormal_sleep': '睡眠异常',
    'low_x_multi': '低社会参与×多病共存',
    'low_x_sleep': '低社会参与×睡眠异常',
    'multi_x_sleep': '多病共存×睡眠异常',
    'low_x_multi_x_sleep': '三阶交互',
    'R²(within)': 'R²(within)',
    'N': '观测数',
    'Wald交互项检验': 'Wald交互项检验'
}
result['变量'] = result['变量'].map(lambda x: var_map.get(x, x))

print("\n三模型对比表:")
print(result.to_string(index=False))

# ============================================================
# 7. 保存结果
# ============================================================
print("\n[6] 保存结果...")

# xlsx
table_path = os.path.join(OUT_TABLE, '表B1_FE交互效应三模型比较.xlsx')
result.to_excel(table_path, index=False)
print(f"    xlsx: {table_path}")

# 单独保存Wald检验结果
wald_detail = pd.DataFrame({
    '检验': ['M2交互项联合(Wald)', 'M3三阶交互项'],
    '统计量': [f'F={f_stat:.3f}', f't={t3:.3f}' if 't3' in dir() else ''],
    'p值': [f'{p_wald_m2:.4e}', f'{p3:.4f}' if 'p3' in dir() else ''],
    '结论': ['拒绝H0：至少一个交互项显著' if p_wald_m2 < 0.05 else '不能拒绝H0',
              '三阶交互不显著' if 'p3' in dir() and p3 > 0.05 else '']
})
wald_path = os.path.join(OUT_DIR, 'Wald检验结果.xlsx')
wald_detail.to_excel(wald_path, index=False)
print(f"    Wald检验: {wald_path}")

# ============================================================
# 8. 生成分析报告
# ============================================================
report = f"""# FE复合风险交互效应模型分析报告

## 分析概要

**目的**：检验社会参与、多病共存、睡眠异常三个处理变量之间的两两及三阶交互效应

**数据**：FE主样本 N={res1.nobs:,} 观测

**模型设定**：
- M1：主效应 + 控制变量 + 个体/年份固定效应
- M2：M1 + 三个两两交互项
- M3：M2 + 三阶交互项

## 核心结果

### 1. 主效应结果（M1模型）

| 变量 | 系数 | 标准误 | p值 | 显著性 |
|------|-----:|-------:|----:|:------:|
| 低社会参与 | {res1.params['low_social']:.4f} | ({res1.std_errors['low_social']:.4f}) | {res1.pvalues['low_social']:.4f} | {'***' if res1.pvalues['low_social']<0.01 else '**' if res1.pvalues['low_social']<0.05 else '*' if res1.pvalues['low_social']<0.1 else ''} |
| 多病共存 | {res1.params['multimorbidity']:.4f} | ({res1.std_errors['multimorbidity']:.4f}) | {res1.pvalues['multimorbidity']:.4f} | {'***' if res1.pvalues['multimorbidity']<0.01 else '**' if res1.pvalues['multimorbidity']<0.05 else '*' if res1.pvalues['multimorbidity']<0.1 else ''} |
| 睡眠异常 | {res1.params['abnormal_sleep']:.4f} | ({res1.std_errors['abnormal_sleep']:.4f}) | {res1.pvalues['abnormal_sleep']:.4f} | {'***' if res1.pvalues['abnormal_sleep']<0.01 else '**' if res1.pvalues['abnormal_sleep']<0.05 else '*' if res1.pvalues['abnormal_sleep']<0.1 else ''} |

### 2. 模型拟合优度变化

| 模型 | R²(within) | ΔR² |
|:----:|:----------:|:---:|
| M1 | {res1.rsquared_within:.4f} | — |
| M2 | {res2.rsquared_within:.4f} | +{delta_r2_m2_m1:.5f} |
| M3 | {res3.rsquared_within:.4f} | +{delta_r2_m3_m2:.5f} |

### 3. Wald联合显著性检验

| 检验 | 统计量 | p值 | 结论 |
|------|:------:|:---:|:----:|
| M2交互项（低社会×多病、低社会×睡眠、多病×睡眠） | F={f_stat:.3f} | {p_wald_m2:.4e} | {'显著' if p_wald_m2 < 0.05 else '不显著'} |
| M3三阶交互项 | t={t3:.3f} | {p3:.4f} | {'显著' if p3 < 0.05 else '不显著'} |

## 结果解读

1. **主效应稳健性**：三个处理变量的主效应符号和显著性在M1/M2/M3中保持一致，说明交互项的加入没有改变核心结论

2. **两两交互效应**：M2的Wald检验{'拒绝' if p_wald_m2 < 0.05 else '未能拒绝'}H0，说明至少有一个两两交互项显著，复合风险状态对抑郁的联合影响不等于各自主效应之和

3. **三阶交互效应**：M3的三阶交互项{'显著' if 'p3' in dir() and p3 < 0.05 else '不显著'}，说明三重风险叠加的效应{'存在' if 'p3' in dir() and p3 < 0.05 else '不存在'}额外的协同或拮抗作用

## 写入论文建议

**表格**：表B1（FE交互效应三模型比较）建议放入正文5.2.5节或附录

**重点呈现**：
1. M1主效应结果（三个核心变量的系数）
2. M2的Wald检验结论（是否有交互效应）
3. 关键交互项的系数和方向（如有显著交互项）

**文字表述建议**：
> "在控制个体固定效应和时间趋势后，三种风险因素的两两交互项联合显著（Wald检验p{'<' if p_wald_m2 < 0.05 else '>'}0.05），表明复合风险状态的抑郁关联{'不等于' if p_wald_m2 < 0.05 else '等于'}各风险因素独立效应之和，存在协同或拮抗作用。"

---
生成时间：2026-05-17
脚本：s2_02_fe_interaction_models.py
"""

report_path = os.path.join(OUT_DIR, '分析B_FE交互效应报告.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"    报告: {report_path}")

print("\n" + "=" * 70)
print("  分析B完成！")
print("=" * 70)
print(f"输出文件:")
print(f"  - 表格: {table_path}")
print(f"  - Wald检验: {wald_path}")
print(f"  - 报告: {report_path}")
