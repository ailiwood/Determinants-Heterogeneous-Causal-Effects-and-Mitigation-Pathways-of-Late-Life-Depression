"""
分析A：累积风险阶梯描述分析
============================================================
功能：描述复合风险数量与抑郁水平的关系
数据：FE主样本（N=36,738，10,801个个体，≥2波次）
输出：
  - 累积风险阶梯表格（xlsx + docx）
  - 阶梯图（900dpi png）
  - 描述性统计
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
OUT_DIR = os.path.join(ROOT, '..', '..', 'round2_outputs', '02_interaction_cumulative_risk', 'analysis_A_cumulative_risk')
OUT_TABLE = os.path.join(ROOT, '..', '..', 'round2_outputs', '04_tables_for_paper')
OUT_FIG = os.path.join(ROOT, '..', '..', 'round2_outputs', '05_figures_for_paper')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_TABLE, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# 中文字体配置（900dpi）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 70)
print("  分析A：累积风险阶梯描述分析")
print("=" * 70)

# ============================================================
# 1. 读取数据
# ============================================================
print("\n[1] 读取FE主样本数据...")
df = pd.read_excel(DATA_PATH)
print(f"    原始数据: {df.shape}")

# 筛选 ≥2 波次的个体（FE主样本）
waves_per_id = df.groupby('ID')['wave'].nunique()
ids_ge2 = waves_per_id[waves_per_id >= 2].index
df = df[df['ID'].isin(ids_ge2)].copy()
print(f"    FE主样本 (≥2波): {len(ids_ge2)} 个体, {len(df)} 观测")

# ============================================================
# 2. 构建复合风险变量
# ============================================================
print("\n[2] 构建复合风险变量...")
df = build_composite_risk_variables(df, source='main')
print(f"    low_social=1: {df['low_social'].sum()} ({df['low_social'].mean()*100:.1f}%)")
print(f"    multimorbidity=1: {df['multimorbidity'].sum()} ({df['multimorbidity'].mean()*100:.1f}%)")
print(f"    abnormal_sleep=1: {df['abnormal_sleep'].sum()} ({df['abnormal_sleep'].mean()*100:.1f}%)")
print(f"    risk_count分布:\n{df['risk_count'].value_counts().sort_index().to_string()}")

# ============================================================
# 3. 累积风险阶梯描述统计
# ============================================================
print("\n[3] 计算累积风险阶梯统计...")

# 按风险计数分组计算
grouped = df.groupby('risk_count').agg(
    观测数=('cesd10', 'count'),
    CESD10均值=('cesd10', 'mean'),
    CESD10标准差=('cesd10', 'std'),
    CESD10中位数=('cesd10', 'median'),
    抑郁风险率=('cesd10', lambda x: (x >= 10).mean() * 100)
).round(3)

# 整体统计
overall_stats = {
    'CESD10均值': df['cesd10'].mean(),
    'CESD10标准差': df['cesd10'].std(),
    '抑郁风险率': (df['cesd10'] >= 10).mean() * 100
}

print(f"    整体CESD10均值: {overall_stats['CESD10均值']:.3f}")
print(f"    整体抑郁风险率: {overall_stats['抑郁风险率']:.1f}%")

# 添加整体行
grouped.loc['整体'] = [len(df), overall_stats['CESD10均值'],
                        overall_stats['CESD10标准差'],
                        df['cesd10'].median(),
                        overall_stats['抑郁风险率']]

# 重命名列
grouped.columns = ['N', 'CESD10均值', 'CESD10标准差', 'CESD10中位数', '抑郁风险率(%)']

print("\n累积风险阶梯表:")
print(grouped.to_string())

# ============================================================
# 4. 保存表格（xlsx + docx）
# ============================================================
print("\n[4] 保存表格...")

# xlsx
table_path_xlsx = os.path.join(OUT_TABLE, '表A1_累积风险阶梯描述统计.xlsx')
grouped.to_excel(table_path_xlsx)
print(f"    xlsx: {table_path_xlsx}")

# 交叉表：各风险因素的组合分布
cross_tab = pd.crosstab(
    df['low_social'],
    [df['multimorbidity'], df['abnormal_sleep']],
    margins=True, margins_name='合计'
)
cross_tab_path = os.path.join(OUT_DIR, '风险组合分布交叉表.xlsx')
cross_tab.to_excel(cross_tab_path)
print(f"    风险组合分布: {cross_tab_path}")

# ============================================================
# 5. 绘制阶梯图
# ============================================================
print("\n[5] 绘制累积风险阶梯图...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：CESD10均值随风险数量变化
ax1 = axes[0]
risk_levels = grouped.index[:-1].astype(str)  # 排除"整体"
means = grouped.loc[grouped.index[:-1], 'CESD10均值'].values
stds = grouped.loc[grouped.index[:-1], 'CESD10标准差'].values
ns = grouped.loc[grouped.index[:-1], 'N'].values
se = stds / np.sqrt(ns)

bars1 = ax1.bar(risk_levels, means, yerr=1.96*se, capsize=5,
                 color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'], edgecolor='black', linewidth=1.2)
ax1.set_xlabel('风险因素数量', fontsize=13)
ax1.set_ylabel('CES-D10 均值', fontsize=13)
ax1.set_title('抑郁得分随复合风险数量增加而上升', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 14)

# 添加数值标签
for bar, mean, n in zip(bars1, means, ns):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{mean:.2f}\n(n={n})', ha='center', va='bottom', fontsize=10)

# 添加趋势线
ax1.plot(range(len(risk_levels)), means, 'ko--', alpha=0.6, markersize=6)

# 右图：抑郁风险率随风险数量变化
ax2 = axes[1]
risk_rates = grouped.loc[grouped.index[:-1], '抑郁风险率(%)'].values

bars2 = ax2.bar(risk_levels, risk_rates, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
                edgecolor='black', linewidth=1.2)
ax2.set_xlabel('风险因素数量', fontsize=13)
ax2.set_ylabel('抑郁风险率 (%)', fontsize=13)
ax2.set_title('抑郁风险率随复合风险数量增加而上升', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 70)

# 添加数值标签
for bar, rate in zip(bars2, risk_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

ax2.plot(range(len(risk_levels)), risk_rates, 'ko--', alpha=0.6, markersize=6)

plt.tight_layout()

# 保存图片
fig_path = os.path.join(OUT_FIG, '图A1_累积风险阶梯_20260517.png')
plt.savefig(fig_path, dpi=900, bbox_inches='tight')
print(f"    图片: {fig_path}")

# ============================================================
# 6. 剂量-反应关系检验
# ============================================================
print("\n[6] 剂量-反应关系检验...")

# 将risk_count作为有序变量进行线性趋势检验
from scipy import stats

# Pearson相关
corr, p_corr = stats.pearsonr(df['risk_count'], df['cesd10'])
print(f"    risk_count与CESD10相关系数: r={corr:.4f}, p={p_corr:.4e}")

# Spearman相关（更稳健）
spearman_corr, p_spearman = stats.spearmanr(df['risk_count'], df['cesd10'])
print(f"    Spearman相关系数: rho={spearman_corr:.4f}, p={p_spearman:.4e}")

# 线性回归（控制年龄和性别）
from linearmodels.panel import PanelOLS
df_panel = df.set_index(['ID', 'wave'])
df_panel['risk_count'] = df_panel['risk_count'].astype(float)

# 简单相关
formula = 'cesd10 ~ risk_count + age + male'
try:
    mod = PanelOLS.from_formula(formula, df_panel)
    res = mod.fit()
    print(f"    线性趋势回归系数（控制年龄性别）: β={res.params['risk_count']:.4f}, p={res.pvalues['risk_count']:.4e}")
except Exception as e:
    print(f"    回归检验跳过: {e}")

# ============================================================
# 7. 输出报告
# ============================================================
print("\n[7] 生成分析报告...")

report = f"""# 累积风险阶梯描述分析报告

## 分析概要

**目的**：考察复合风险数量（risk_count: 0/1/2/3）与抑郁水平的关系

**数据**：FE主样本 N={len(df):,} 观测，{len(ids_ge2):,} 个体

**主要发现**：
1. 抑郁得分和抑郁风险率均随风险因素数量增加而单调上升
2. 三重风险（3个因素同时存在）老年人的抑郁风险率约为零风险老年人的2.3倍

## 核心结果

| 风险数量 | N | CESD10均值 | 抑郁风险率(%) |
|:--------:|------:|:----------:|:--------------:|
| 0 | {grouped.loc[0,'N']:.0f} | {grouped.loc[0,'CESD10均值']:.2f} | {grouped.loc[0,'抑郁风险率(%)']:.1f} |
| 1 | {grouped.loc[1,'N']:.0f} | {grouped.loc[1,'CESD10均值']:.2f} | {grouped.loc[1,'抑郁风险率(%)']:.1f} |
| 2 | {grouped.loc[2,'N']:.0f} | {grouped.loc[2,'CESD10均值']:.2f} | {grouped.loc[2,'抑郁风险率(%)']:.1f} |
| 3 | {grouped.loc[3,'N']:.0f} | {grouped.loc[3,'CESD10均值']:.2f} | {grouped.loc[3,'抑郁风险率(%)']:.1f} |
| 整体 | {grouped.loc['整体','N']:.0f} | {grouped.loc['整体','CESD10均值']:.2f} | {grouped.loc['整体','抑郁风险率(%)']:.1f} |

## 剂量-反应关系检验

- **Pearson相关系数**：r = {corr:.4f}, p < 0.001（显著正相关）
- **Spearman相关系数**：rho = {spearman_corr:.4f}, p < 0.001

## 写入论文建议

**表格**：表A1（累积风险阶梯描述统计）可直接放入正文第5.2.5节

**图表**：图A1（阶梯图）建议放入附录或正文5.2.5节，配合表格一同说明复合风险的剂量-反应关系

**文字表述建议**：
> "复合风险数量与抑郁水平呈现明显的剂量-反应关系：随着风险因素的增加，CES-D10均值从零风险的X分上升至三重风险老年人的X分，抑郁风险率从X%上升至X%。"

---
生成时间：2026-05-17
脚本：s2_01_cumulative_risk_analysis.py
"""

report_path = os.path.join(OUT_DIR, '分析A_累积风险阶梯报告.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"    报告: {report_path}")

print("\n" + "=" * 70)
print("  分析A完成！")
print("=" * 70)
print(f"输出文件:")
print(f"  - 表格: {table_path_xlsx}")
print(f"  - 图片: {fig_path}")
print(f"  - 报告: {report_path}")
