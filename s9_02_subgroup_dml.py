#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 9 - Step 2: 分组DML因果效应估计 (Subgroup DML)
按 gender / rural2 / region 分组，对每个子样本分别跑 DML
T1: DoubleMLPLR (连续treatment)
T2/T3: DoubleMLIRM (二元treatment)
输出各子组ATE及置信区间，用于与分组FE和CRF-GATE三角验证
"""

import os, sys, json, warnings, pickle
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# === 路径 ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from utils.var_labels import get_zh, get_label
from utils.table_utils import save_three_line_table, format_coef

CAUSAL_PATH = os.path.join(ROOT, 'results/03_causal_sample/causal_sample_lagged.xlsx')
OUT_DIR = os.path.join(ROOT, 'results/08_heterogeneity')
os.makedirs(OUT_DIR, exist_ok=True)

# === 参数 ===
SEED = 42
N_FOLDS = 5
N_REP = 3   # 分组样本较小，3次重复足够
np.random.seed(SEED)

# === 变量定义 ===
Y_VAR = 'cesd10_t'

TREATMENTS = {
    'T1': {'var': 'social_activity_index', 'type': 'continuous'},
    'T2': {'var': 'T2_chronic',            'type': 'binary'},
    'T3': {'var': 'T3_sleep',              'type': 'binary'},
}

W_COMMON = ['age', 'gender', 'edu', 'marry', 'rural2', 'province_enc',
            'hhcperc_log', 'ins', 'pension', 'retire', 'bmi',
            'smoken', 'drinkev', 'family_size', 'hchild',
            'lag_cesd10', 'wave']

# Treatment-specific controls (exclude the treatment itself and related vars)
W_EXTRA = {
    'T1': ['chronic_disease_count', 'sleep', 'fcamt_log'],
    'T2': ['sleep', 'social_activity_index', 'fcamt_log'],
    'T3': ['chronic_disease_count', 'social_activity_index', 'fcamt_log'],
}

# === 省份→区域 ===
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

# === 分组定义 ===
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

# === Learner工厂 ===
def get_xgb_reg():
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1,
                        random_state=SEED, n_jobs=-1, verbosity=0)

def get_xgb_cls():
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         eval_metric='logloss')


def load_and_prepare():
    """加载因果样本并准备"""
    print("=" * 60)
    print("加载因果样本...")
    df = pd.read_excel(CAUSAL_PATH)
    print(f"  原始: {df.shape}")

    # 对数变换
    df['hhcperc_log'] = np.log(df['hhcperc'] + 1)
    df['fcamt_log'] = np.log(df['fcamt'] + 1)

    # 区域映射
    df['region'] = df['province'].apply(map_region)
    unmapped = df['region'].isna().sum()
    if unmapped > 0:
        print(f"  警告: {unmapped}条未映射区域，已排除")
        df = df.dropna(subset=['region'])
    df['region'] = df['region'].astype(int)

    # 省份编码
    le_prov = LabelEncoder()
    df['province_enc'] = le_prov.fit_transform(df['province'].astype(str))

    # Treatment变量构造
    df['T2_chronic'] = (df['chronic_disease_count'] >= 2).astype(int)
    df['T3_sleep'] = ((df['sleep'] < 6) | (df['sleep'] > 9)).astype(int)

    print(f"  最终样本: {df.shape}")
    print(f"  区域分布: 东部{(df['region']==1).sum()}, 中部{(df['region']==2).sum()}, 西部{(df['region']==3).sum()}")

    return df


def run_dml_subgroup(df_sub, t_key, t_info, group_label):
    """对子样本跑DML"""
    d_var = t_info['var']
    t_type = t_info['type']

    # 构建W
    w_cols = [c for c in W_COMMON + W_EXTRA[t_key] if c in df_sub.columns]
    # 去掉分组变量本身（避免共线性或无变异）
    # 如果按gender分组，gender在子样本中无变异，需要排除
    for sg_info in SUBGROUPS.values():
        sg_var = sg_info['var']
        if df_sub[sg_var].nunique() <= 1 and sg_var in w_cols:
            w_cols.remove(sg_var)

    # 构建分析数据
    all_cols = [Y_VAR, d_var] + w_cols
    all_cols = list(dict.fromkeys(all_cols))
    data = df_sub[all_cols].dropna()

    if len(data) < 200:
        print(f"    样本过小({len(data)}), 跳过")
        return None

    # 检查treatment有足够变异
    if t_type == 'binary':
        treat_mean = data[d_var].mean()
        if treat_mean < 0.05 or treat_mean > 0.95:
            print(f"    Treatment极端分布({treat_mean:.3f}), 跳过")
            return None

    try:
        dml_data = DoubleMLData.from_arrays(
            y=data[Y_VAR].values,
            d=data[d_var].values,
            x=data[w_cols].values,
        )

        if t_type == 'continuous':
            # PLR
            model = DoubleMLPLR(
                dml_data,
                ml_l=get_xgb_reg(),
                ml_m=get_xgb_reg(),
                n_folds=N_FOLDS,
                n_rep=N_REP,
                score='partialling out',
            )
            model.fit()
            theta = model.coef[0]
            se = model.se[0]
            ci = model.confint().iloc[0]
            pval = model.pval[0]

            return {
                'Treatment': t_key,
                'Subgroup': group_label,
                'N': len(data),
                'ATE': theta,
                'SE': se,
                'CI_lower': ci['2.5 %'],
                'CI_upper': ci['97.5 %'],
                'p_value': pval,
                'model_type': 'PLR',
            }

        else:
            # IRM
            model = DoubleMLIRM(
                dml_data,
                ml_g=get_xgb_reg(),
                ml_m=get_xgb_cls(),
                n_folds=N_FOLDS,
                n_rep=N_REP,
                score='ATE',
                trimming_threshold=0.01,
            )
            model.fit()
            ate = model.coef[0]
            se = model.se[0]
            ci = model.confint().iloc[0]
            pval = model.pval[0]

            return {
                'Treatment': t_key,
                'Subgroup': group_label,
                'N': len(data),
                'ATE': ate,
                'SE': se,
                'CI_lower': ci['2.5 %'],
                'CI_upper': ci['97.5 %'],
                'p_value': pval,
                'model_type': 'IRM',
            }

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    df = load_and_prepare()

    all_results = []
    all_models = {}

    for sg_name, sg_info in SUBGROUPS.items():
        var = sg_info['var']
        groups = sg_info['groups']
        print(f"\n{'='*60}")
        print(f"分组变量: {var}")
        print(f"{'='*60}")

        for gval, glabel in groups.items():
            df_sub = df[df[var] == gval].copy()
            print(f"\n  [{glabel}] N={len(df_sub)}")

            for t_key, t_info in TREATMENTS.items():
                print(f"    {t_key} ({t_info['var']}, {t_info['type']})...")
                result = run_dml_subgroup(df_sub, t_key, t_info, glabel)
                if result:
                    all_results.append(result)
                    stars = ''
                    if result['p_value'] < 0.01: stars = '***'
                    elif result['p_value'] < 0.05: stars = '**'
                    elif result['p_value'] < 0.1: stars = '*'
                    print(f"      ATE={result['ATE']:.4f}{stars}, SE={result['SE']:.4f}, "
                          f"p={result['p_value']:.4f}, 95%CI=[{result['CI_lower']:.4f}, {result['CI_upper']:.4f}]")

    # === 汇总输出 ===
    df_results = pd.DataFrame(all_results)
    df_results['ATE_formatted'] = df_results.apply(
        lambda r: format_coef(r['ATE'], r['SE'], r['p_value']), axis=1
    )
    df_results.to_excel(os.path.join(OUT_DIR, 'subgroup_dml_results.xlsx'), index=False)
    print(f"\n结果已保存: subgroup_dml_results.xlsx")

    # === 论文三线表 ===
    pivot_rows = []
    for sg_name, sg_info in SUBGROUPS.items():
        var = sg_info['var']
        groups = sg_info['groups']
        for t_key in TREATMENTS:
            row = {
                '分组维度': get_zh(var),
                'Treatment': t_key,
            }
            for gval, glabel in groups.items():
                match = df_results[
                    (df_results['Treatment'] == t_key) &
                    (df_results['Subgroup'] == glabel)
                ]
                if len(match) > 0:
                    m = match.iloc[0]
                    row[glabel] = m['ATE_formatted']
                    row[f'{glabel}_N'] = int(m['N'])
                else:
                    row[glabel] = '-'
            pivot_rows.append(row)

    df_pivot = pd.DataFrame(pivot_rows)
    save_three_line_table(
        df_pivot,
        os.path.join(OUT_DIR, 'subgroup_dml_summary'),
        title='表 分组DML因果效应估计结果',
        note='注：***p<0.01, **p<0.05, *p<0.1。括号内为标准误。T1使用DoubleMLPLR，T2/T3使用DoubleMLIRM。XGBoost作为基础学习器，5折交叉拟合，3次重复。'
    )
    print("论文三线表已保存: subgroup_dml_summary.xlsx/.docx")

    # === JSON输出 ===
    json_out = {}
    for _, r in df_results.iterrows():
        key = f"{r['Treatment']}_{r['Subgroup']}"
        json_out[key] = {
            'ATE': float(r['ATE']),
            'SE': float(r['SE']),
            'p_value': float(r['p_value']),
            'CI_lower': float(r['CI_lower']),
            'CI_upper': float(r['CI_upper']),
            'N': int(r['N']),
            'model_type': r['model_type'],
        }
    with open(os.path.join(OUT_DIR, 'subgroup_dml_results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("分组DML分析完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
