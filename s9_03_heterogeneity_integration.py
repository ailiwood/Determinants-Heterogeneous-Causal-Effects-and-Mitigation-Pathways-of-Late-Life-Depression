#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 9 - Step 3: 三层异质性证据整合
整合分组FE + 分组DML + CRF-GATE结果，输出对比表和可视化
对应写作框架 6.3.1（比较框架）和 6.3.6（三证据一致性与差异解释）
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

# === 中文字体配置 ===
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

warnings.filterwarnings('ignore')

# === 路径 ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from utils.var_labels import get_zh, get_label
from utils.table_utils import save_three_line_table, format_coef

OUT_DIR = os.path.join(ROOT, 'results/08_heterogeneity')
PAPER_TABLE_DIR = os.path.join(ROOT, 'results/10_tables_for_paper')
PAPER_FIG_DIR = os.path.join(ROOT, 'results/11_figures_for_paper')
os.makedirs(PAPER_TABLE_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# === CRF GATE路径 ===
CRF_DIRS = {
    'T1': os.path.join(ROOT, 'results/05_crf/crf_T1_social'),
    'T2': os.path.join(ROOT, 'results/05_crf/crf_T2_chronic'),
    'T3': os.path.join(ROOT, 'results/05_crf/crf_T3_sleep'),
}

# 分组变量中文→英文映射
SG_ZH_TO_EN = {
    '性别(男=1)': 'gender',
    '城镇户口': 'rural2',
    '区域(东/中/西)': 'region',
}

# 子组标签映射
GROUP_MAP = {
    'gender': {'女性': '女性', '男性': '男性'},
    'rural2': {'农村户口': '农村户口', '城镇户口': '城镇户口'},
    'region': {'东部': '东部', '中部': '中部', '西部': '西部'},
}

TREATMENT_LABELS = {
    'T1': '社会参与',
    'T2': '多病共存',
    'T3': '睡眠异常',
}


def load_fe_results():
    """加载分组FE结果"""
    path = os.path.join(OUT_DIR, 'subgroup_fe_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dml_results():
    """加载分组DML结果"""
    path = os.path.join(OUT_DIR, 'subgroup_dml_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_crf_gate():
    """加载CRF-GATE结果"""
    all_gate = {}
    for t_key, crf_dir in CRF_DIRS.items():
        gate_path = os.path.join(crf_dir, 'gate_results.xlsx')
        if os.path.exists(gate_path):
            df_gate = pd.read_excel(gate_path)
            for _, row in df_gate.iterrows():
                sg_zh = row['分组变量']
                sg_en = SG_ZH_TO_EN.get(sg_zh, sg_zh)
                group_label = row['分组']
                key = f"{t_key}_{group_label}"
                all_gate[key] = {
                    'GATE': float(row['GATE']),
                    'SE': float(row['SE']),
                    'CI_lower': float(row['95% CI下界']),
                    'CI_upper': float(row['95% CI上界']),
                    'p_value': float(row['p值']),
                    'N': int(row['N']),
                    'sg_var': sg_en,
                }
    return all_gate


def build_integration_table(fe_data, dml_data, crf_data):
    """构建三证据整合表"""
    rows = []
    subgroups = [
        ('gender', ['女性', '男性']),
        ('rural2', ['农村户口', '城镇户口']),
        ('region', ['东部', '中部', '西部']),
    ]

    for t_key in ['T1', 'T2', 'T3']:
        for sg_var, sg_groups in subgroups:
            for glabel in sg_groups:
                key = f"{t_key}_{glabel}"

                row = {
                    'Treatment': t_key,
                    'Treatment_label': TREATMENT_LABELS[t_key],
                    '分组维度': get_zh(sg_var) if sg_var != 'region' else '区域(东/中/西)',
                    '子组': glabel,
                }

                # FE
                fe = fe_data.get(key, {})
                if fe and fe.get('coef') is not None:
                    row['FE_coef'] = fe['coef']
                    row['FE_se'] = fe['se']
                    row['FE_p'] = fe['p_value']
                    row['FE_formatted'] = format_coef(fe['coef'], fe['se'], fe['p_value'])
                    row['FE_N'] = fe['n_obs']
                else:
                    row['FE_coef'] = np.nan
                    row['FE_formatted'] = '-'

                # DML
                dml = dml_data.get(key, {})
                if dml:
                    row['DML_ATE'] = dml['ATE']
                    row['DML_se'] = dml['SE']
                    row['DML_p'] = dml['p_value']
                    row['DML_formatted'] = format_coef(dml['ATE'], dml['SE'], dml['p_value'])
                    row['DML_N'] = dml['N']
                else:
                    row['DML_ATE'] = np.nan
                    row['DML_formatted'] = '-'

                # CRF-GATE
                crf = crf_data.get(key, {})
                if crf:
                    row['CRF_GATE'] = crf['GATE']
                    row['CRF_se'] = crf['SE']
                    row['CRF_p'] = crf['p_value']
                    row['CRF_formatted'] = format_coef(crf['GATE'], crf['SE'], crf['p_value'])
                    row['CRF_N'] = crf['N']
                else:
                    row['CRF_GATE'] = np.nan
                    row['CRF_formatted'] = '-'

                # 一致性判断
                # 注意：T3的FE用sleep(连续,负=保护)，DML/CRF用T3_sleep(二元,正=致郁)
                # 概念一致但符号相反，需要翻转FE符号后比较
                fe_val = row.get('FE_coef')
                dml_val = row.get('DML_ATE')
                crf_val = row.get('CRF_GATE')

                if t_key == 'T3' and fe_val is not None and not np.isnan(fe_val):
                    fe_sign = np.sign(-fe_val)  # 翻转：sleep负(保护) ↔ T3_sleep正(致郁)
                else:
                    fe_sign = np.sign(fe_val) if fe_val is not None and not np.isnan(fe_val) else None

                signs = []
                if fe_sign is not None: signs.append(fe_sign)
                if dml_val is not None and not np.isnan(dml_val): signs.append(np.sign(dml_val))
                if crf_val is not None and not np.isnan(crf_val): signs.append(np.sign(crf_val))

                if len(signs) >= 2:
                    row['方向一致'] = '是' if len(set(signs)) == 1 else '否'
                else:
                    row['方向一致'] = '-'

                rows.append(row)

    return pd.DataFrame(rows)


def plot_forest_comparison(df_int):
    """绘制三证据对比森林图"""
    for t_key in ['T1', 'T2', 'T3']:
        df_t = df_int[df_int['Treatment'] == t_key].copy()
        if df_t.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        y_labels = []
        y_positions = []
        pos = 0
        prev_sg = None

        for idx, row in df_t.iterrows():
            sg = row['分组维度']
            if prev_sg is not None and sg != prev_sg:
                pos -= 0.5  # 分组间留间隔
            prev_sg = sg

            label = f"{row['子组']}"
            y_labels.append(label)

            # 三个方法的点和CI
            methods = [
                ('FE', row.get('FE_coef'), row.get('FE_se'), row.get('FE_p'), '#2196F3'),
                ('DML', row.get('DML_ATE'), row.get('DML_se'), row.get('DML_p'), '#FF9800'),
                ('CRF', row.get('CRF_GATE'), row.get('CRF_se'), row.get('CRF_p'), '#4CAF50'),
            ]

            offsets = [-0.15, 0, 0.15]
            for (mname, coef, se, pval, color), offset in zip(methods, offsets):
                if coef is not None and not np.isnan(coef):
                    ci_low = coef - 1.96 * se if se and not np.isnan(se) else coef
                    ci_high = coef + 1.96 * se if se and not np.isnan(se) else coef
                    marker = 'o' if pval is not None and not np.isnan(pval) and pval < 0.05 else 'x'
                    ax.errorbar(coef, pos + offset, xerr=[[coef - ci_low], [ci_high - coef]],
                               fmt=marker, color=color, markersize=7, capsize=3, linewidth=1.5,
                               label=mname if pos == 0 else '')

            y_positions.append(pos)
            pos -= 1

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('效应值 (FE系数 / DML-ATE / CRF-GATE)')
        ax.set_title(f'{TREATMENT_LABELS[t_key]} ({t_key}) — 三层异质性证据对比')

        # 添加分组维度标注
        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        ax.invert_yaxis()
        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, f'forest_comparison_{t_key}.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"  森林图已保存: forest_comparison_{t_key}.png")

        # 同时保存到论文图目录
        fig_paper = os.path.join(PAPER_FIG_DIR, f'heterogeneity_forest_{t_key}.png')
        import shutil
        shutil.copy2(fig_path, fig_paper)


def assess_consistency(df_int):
    """评估三层证据的一致性"""
    print("\n" + "=" * 60)
    print("三层异质性证据一致性评估")
    print("=" * 60)

    for t_key in ['T1', 'T2', 'T3']:
        df_t = df_int[df_int['Treatment'] == t_key]
        total = len(df_t)
        consistent = (df_t['方向一致'] == '是').sum()
        print(f"\n{t_key} ({TREATMENT_LABELS[t_key]}):")
        print(f"  方向一致: {consistent}/{total}")

        # 按分组维度分析
        for sg in df_t['分组维度'].unique():
            df_sg = df_t[df_t['分组维度'] == sg]
            print(f"  [{sg}]:")
            for _, row in df_sg.iterrows():
                fe_str = f"FE={row['FE_coef']:.4f}" if not np.isnan(row.get('FE_coef', np.nan)) else "FE=N/A"
                dml_str = f"DML={row['DML_ATE']:.4f}" if not np.isnan(row.get('DML_ATE', np.nan)) else "DML=N/A"
                crf_str = f"CRF={row['CRF_GATE']:.4f}" if not np.isnan(row.get('CRF_GATE', np.nan)) else "CRF=N/A"
                print(f"    {row['子组']}: {fe_str}, {dml_str}, {crf_str} → {row['方向一致']}")


def main():
    print("=" * 60)
    print("Stage 9 Step 3: 三层异质性证据整合")
    print("=" * 60)

    # 加载三层结果
    print("\n加载数据...")
    fe_data = load_fe_results()
    print(f"  FE: {len(fe_data)} 条记录")
    dml_data = load_dml_results()
    print(f"  DML: {len(dml_data)} 条记录")
    crf_data = load_crf_gate()
    print(f"  CRF-GATE: {len(crf_data)} 条记录")

    # 构建整合表
    df_int = build_integration_table(fe_data, dml_data, crf_data)
    print(f"\n整合表: {len(df_int)} 行")

    # 保存完整整合表
    df_int.to_excel(os.path.join(OUT_DIR, 'heterogeneity_integration_full.xlsx'), index=False)

    # === 论文用表格（简化版）===
    df_paper = df_int[['Treatment', 'Treatment_label', '分组维度', '子组',
                       'FE_formatted', 'FE_N',
                       'DML_formatted', 'DML_N',
                       'CRF_formatted', 'CRF_N',
                       '方向一致']].copy()
    df_paper.columns = ['Treatment', '处理变量', '分组维度', '子组',
                        'FE系数', 'FE_N', 'DML-ATE', 'DML_N',
                        'CRF-GATE', 'CRF_N', '方向一致']

    save_three_line_table(
        df_paper,
        os.path.join(OUT_DIR, 'heterogeneity_integration_summary'),
        title='表 三层异质性证据整合对比',
        note='注：***p<0.01, **p<0.05, *p<0.1。FE为面板固定效应回归系数（聚类稳健标准误）；'
             'DML-ATE为双重机器学习因果效应估计；CRF-GATE为因果随机森林分组平均处理效应。'
             '方向一致指三种方法的效应方向是否相同。'
    )

    # 论文表格也保存到paper目录
    import shutil
    for ext in ['.xlsx', '.docx']:
        src = os.path.join(OUT_DIR, f'heterogeneity_integration_summary{ext}')
        dst = os.path.join(PAPER_TABLE_DIR, f'heterogeneity_integration{ext}')
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # 绘制森林图
    print("\n绘制三证据对比森林图...")
    plot_forest_comparison(df_int)

    # 一致性评估
    assess_consistency(df_int)

    # === JSON汇总 ===
    summary = {
        'total_comparisons': len(df_int),
        'direction_consistent': int((df_int['方向一致'] == '是').sum()),
        'direction_inconsistent': int((df_int['方向一致'] == '否').sum()),
    }
    for t_key in ['T1', 'T2', 'T3']:
        df_t = df_int[df_int['Treatment'] == t_key]
        summary[t_key] = {
            'total': len(df_t),
            'consistent': int((df_t['方向一致'] == '是').sum()),
        }

    with open(os.path.join(OUT_DIR, 'heterogeneity_integration_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("三层异质性证据整合完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
