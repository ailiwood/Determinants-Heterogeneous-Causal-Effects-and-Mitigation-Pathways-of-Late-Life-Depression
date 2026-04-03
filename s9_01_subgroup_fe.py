#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 9 - Step 1: 分组固定效应回归 (Subgroup FE)
按 gender / rural2 / region 分组，对每个子样本跑 PanelOLS FE 回归
输出各子组的核心解释变量系数，用于与分组DML和CRF-GATE三角验证
"""

import os, sys, json, warnings, pickle
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from datetime import datetime

warnings.filterwarnings('ignore')

# === 路径 ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from utils.var_labels import VAR_MAP, get_zh, get_label
from utils.table_utils import save_three_line_table, format_coef

DATA_PATH = os.path.join(ROOT, 'data/preprocess_outputs/data1_modeling_ready.xlsx')
OUT_DIR = os.path.join(ROOT, 'results/08_heterogeneity')
os.makedirs(OUT_DIR, exist_ok=True)

# === 变量定义 ===
Y_VAR = 'cesd10'

# 核心解释变量（对应三个Treatment方向）
CORE_VARS = {
    'T1': 'social_activity_index',   # 社会参与（连续）
    'T2': 'chronic_disease_count',    # 慢性病数量（连续）
    'T3': 'sleep',                    # 睡眠时长（连续）
}

# 其他核心变量 + 控制变量
OTHER_CORE = ['body_discomfort_count', 'hhcperc_log', 'ins', 'pension', 'fcamt_log']
CTRL_Z = ['age', 'marry', 'bmi', 'smoken', 'drinkev', 'retire', 'family_size', 'hchild']
ALL_X = list(CORE_VARS.values()) + OTHER_CORE + CTRL_Z

# 分组变量定义
SUBGROUPS = {
    'gender': {
        'var': 'gender',
        'groups': {0: '女性', 1: '男性'},
    },
    'rural2': {
        'var': 'rural2',
        'groups': {0: '农村户口', 1: '城镇户口'},
    },
    'region': {
        'var': 'region',
        'groups': {1: '东部', 2: '中部', 3: '西部'},
    },
}


# === 省份→区域映射 ===
EAST = ['北京','北京市','天津','天津市','河北省','辽宁省','上海市',
        '江苏省','浙江省','福建省','山东省','广东省','海南省']
CENTRAL = ['山西省','吉林省','黑龙江省','安徽省','江西省','河南省',
           '湖北省','湖南省']
WEST = ['内蒙古自治区','广西省','广西壮族自治区','重庆市','四川省','贵州省',
        '云南省','西藏自治区','陕西省','甘肃省','青海省','宁夏回族自治区',
        '新疆维吾尔自治区']

def map_region(prov):
    if prov in EAST: return 1
    elif prov in CENTRAL: return 2
    elif prov in WEST: return 3
    else: return np.nan


def load_and_prepare():
    """加载数据并准备面板结构"""
    print("=" * 60)
    print("加载数据...")
    df = pd.read_excel(DATA_PATH)
    print(f"  原始数据: {df.shape}")

    # 对数变换
    df['hhcperc_log'] = np.log(df['hhcperc'] + 1)
    df['fcamt_log'] = np.log(df['fcamt'] + 1)

    # 区域映射
    df['region'] = df['province'].apply(map_region)
    unmapped = df['region'].isna().sum()
    if unmapped > 0:
        print(f"  警告: {unmapped}条未映射到区域，已排除")
        df = df.dropna(subset=['region'])
    df['region'] = df['region'].astype(int)
    print(f"  区域分布: 东部{(df['region']==1).sum()}, 中部{(df['region']==2).sum()}, 西部{(df['region']==3).sum()}")

    # 筛选 >=2 波次
    waves_per_id = df.groupby('ID')['wave'].nunique()
    ids_ge2 = waves_per_id[waves_per_id >= 2].index
    df = df[df['ID'].isin(ids_ge2)].copy()
    print(f"  >=2波次: {df.shape}, {df['ID'].nunique()} 个体")

    return df


def run_fe_subgroup(df_sub, group_name, group_label, n_individuals):
    """对子样本跑FE回归，返回核心变量系数"""
    df_panel = df_sub.set_index(['ID', 'wave'])

    y = df_panel[Y_VAR]
    # 去掉分组变量本身（如果在X中且是时不变的）
    x_cols = [c for c in ALL_X if c in df_panel.columns]
    X = df_panel[x_cols]
    X = X.assign(const=1)

    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)

        results = []
        for t_key, var_name in CORE_VARS.items():
            if var_name in res.params.index:
                coef = res.params[var_name]
                se = res.std_errors[var_name]
                pval = res.pvalues[var_name]
                ci = res.conf_int().loc[var_name]
                results.append({
                    'Treatment': t_key,
                    'Variable': var_name,
                    'Subgroup': group_label,
                    'N_obs': len(df_sub),
                    'N_individuals': n_individuals,
                    'Coef': coef,
                    'SE': se,
                    'p_value': pval,
                    'CI_lower': ci.iloc[0],
                    'CI_upper': ci.iloc[1],
                    'Coef_formatted': format_coef(coef, se, pval),
                })
            else:
                results.append({
                    'Treatment': t_key,
                    'Variable': var_name,
                    'Subgroup': group_label,
                    'N_obs': len(df_sub),
                    'N_individuals': n_individuals,
                    'Coef': np.nan,
                    'SE': np.nan,
                    'p_value': np.nan,
                    'CI_lower': np.nan,
                    'CI_upper': np.nan,
                    'Coef_formatted': 'absorbed',
                })

        # 保存模型对象
        model_path = os.path.join(OUT_DIR, f'fe_subgroup_{group_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(res, f)

        return results, res.rsquared_overall

    except Exception as e:
        print(f"    ERROR in {group_label}: {e}")
        return [], None


def main():
    df = load_and_prepare()

    all_results = []
    r2_records = []

    for sg_name, sg_info in SUBGROUPS.items():
        var = sg_info['var']
        groups = sg_info['groups']
        print(f"\n{'='*60}")
        print(f"分组变量: {var} ({get_zh(var)})")
        print(f"{'='*60}")

        for gval, glabel in groups.items():
            df_sub = df[df[var] == gval].copy()
            n_ids = df_sub['ID'].nunique()
            print(f"\n  [{glabel}] N_obs={len(df_sub)}, N_id={n_ids}")

            group_key = f"{sg_name}_{gval}"
            results, r2 = run_fe_subgroup(df_sub, group_key, glabel, n_ids)
            all_results.extend(results)

            if r2 is not None:
                r2_records.append({
                    'Subgroup_var': sg_name,
                    'Subgroup': glabel,
                    'R2_overall': r2,
                })

            for r in results:
                stars = ''
                if r['p_value'] < 0.01: stars = '***'
                elif r['p_value'] < 0.05: stars = '**'
                elif r['p_value'] < 0.1: stars = '*'
                print(f"    {r['Treatment']}({r['Variable']}): coef={r['Coef']:.4f}{stars}, SE={r['SE']:.4f}, p={r['p_value']:.4f}")

    # === 汇总输出 ===
    df_results = pd.DataFrame(all_results)
    df_results.to_excel(os.path.join(OUT_DIR, 'subgroup_fe_results.xlsx'), index=False)
    print(f"\n结果已保存: subgroup_fe_results.xlsx")

    # === 生成论文友好的表格 ===
    # 透视表：行=Treatment+Subgroup_var，列=各子组
    pivot_rows = []
    for sg_name, sg_info in SUBGROUPS.items():
        var = sg_info['var']
        groups = sg_info['groups']
        for t_key, t_var in CORE_VARS.items():
            row = {
                '分组维度': get_zh(var),
                'Treatment': t_key,
                '变量': get_label(t_var),
            }
            for gval, glabel in groups.items():
                match = df_results[
                    (df_results['Treatment'] == t_key) &
                    (df_results['Subgroup'] == glabel)
                ]
                if len(match) > 0:
                    m = match.iloc[0]
                    row[glabel] = m['Coef_formatted']
                    row[f'{glabel}_N'] = int(m['N_obs'])
                else:
                    row[glabel] = '-'
            pivot_rows.append(row)

    df_pivot = pd.DataFrame(pivot_rows)
    save_three_line_table(
        df_pivot,
        os.path.join(OUT_DIR, 'subgroup_fe_summary'),
        title='表 分组固定效应回归结果',
        note='注：***p<0.01, **p<0.05, *p<0.1。括号内为聚类稳健标准误。所有模型控制个体固定效应和时间固定效应。'
    )
    print("论文三线表已保存: subgroup_fe_summary.xlsx/.docx")

    # === JSON输出 ===
    json_out = {}
    for _, r in df_results.iterrows():
        key = f"{r['Treatment']}_{r['Subgroup']}"
        json_out[key] = {
            'coef': float(r['Coef']) if not np.isnan(r['Coef']) else None,
            'se': float(r['SE']) if not np.isnan(r['SE']) else None,
            'p_value': float(r['p_value']) if not np.isnan(r['p_value']) else None,
            'ci_lower': float(r['CI_lower']) if not np.isnan(r['CI_lower']) else None,
            'ci_upper': float(r['CI_upper']) if not np.isnan(r['CI_upper']) else None,
            'n_obs': int(r['N_obs']),
            'n_individuals': int(r['N_individuals']),
        }
    with open(os.path.join(OUT_DIR, 'subgroup_fe_results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("分组FE分析完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
