"""
阶段4：构造lagged因果样本
============================================================
逻辑：D_{t-1} → Y_t，相邻波次配对
配对：(wave1→2), (wave2→3), (wave3→4), (wave4→5)
      即 (2011→2013), (2013→2015), (2015→2018), (2018→2020)

输出字段：
  Y_t      = cesd10 在 t 期
  lag_cesd10 = cesd10 在 t-1 期
  所有X/D 取 t-1 期值
  T1 = social_activity_index_{t-1}（连续）
  T1_binary = I(social_activity_index_{t-1} >= 1)（稳健性用）
  T2 = I(chronic_disease_count_{t-1} >= 2)
  T3 = I(sleep_{t-1} < 6 or sleep_{t-1} > 9)
  region = province → 东/中/西
============================================================
"""

import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data/preprocess_outputs/data1_modeling_ready.xlsx')
OUT = os.path.join(ROOT, 'results/03_causal_sample')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. 读取数据
# ============================================================
print("=== 阶段4：构造lagged因果样本 ===")
df = pd.read_excel(DATA_PATH)
print(f"原始数据: {df.shape}")

# ============================================================
# 2. 省份→区域映射
# ============================================================
EAST = ['北京','北京市','天津','天津市','河北省','辽宁省','上海市',
        '江苏省','浙江省','福建省','山东省','广东省','海南省']
CENTRAL = ['山西省','吉林省','黑龙江省','安徽省','江西省','河南省',
           '湖北省','湖南省']
WEST = ['内蒙古自治区','广西省','广西壮族自治区','重庆市','四川省','贵州省',
        '云南省','西藏自治区','陕西省','甘肃省','青海省','宁夏回族自治区',
        '新疆维吾尔自治区']

def map_region(prov):
    if prov in EAST: return 1  # 东部
    elif prov in CENTRAL: return 2  # 中部
    elif prov in WEST: return 3  # 西部
    else: return np.nan

df['region'] = df['province'].apply(map_region)
unmapped = df['region'].isna().sum()
if unmapped > 0:
    print(f"  ⚠ {unmapped}条未映射到区域")
    print(f"    未映射省份: {df[df['region'].isna()]['province'].unique()}")
else:
    print(f"  区域映射完成: 东部{(df['region']==1).sum()}, 中部{(df['region']==2).sum()}, 西部{(df['region']==3).sum()}")

# ============================================================
# 3. 构造lagged配对
# ============================================================
print("\n=== 构造lagged配对 ===")
df = df.sort_values(['ID', 'wave'])

# 对每个ID，配对相邻波次
pairs = []
for uid, grp in df.groupby('ID'):
    grp = grp.sort_values('wave')
    waves = grp['wave'].values
    for i in range(len(waves) - 1):
        w_prev = waves[i]
        w_curr = waves[i + 1]
        # 只配对相邻波次（差1）
        if w_curr - w_prev == 1:
            row_prev = grp[grp['wave'] == w_prev].iloc[0]
            row_curr = grp[grp['wave'] == w_curr].iloc[0]
            pairs.append((uid, w_prev, w_curr, row_prev, row_curr))

print(f"  有效配对数: {len(pairs)}")

# 波次对应年份
wave_year = {1: 2011, 2: 2013, 3: 2015, 4: 2018, 5: 2020}

# ============================================================
# 4. 构建因果样本DataFrame
# ============================================================
# 需要的变量列表
# DML的W变量（通用安全控制集 + treatment-specific）
w_vars = ['age', 'gender', 'edu', 'marry', 'rural2',
          'hhcperc', 'ins', 'pension', 'retire', 'bmi',
          'smoken', 'drinkev', 'family_size', 'hchild',
          'social_activity_index', 'chronic_disease_count',
          'body_discomfort_count', 'sleep', 'fcamt',
          'activity_index']

records = []
for uid, w_prev, w_curr, row_prev, row_curr in pairs:
    rec = {
        'ID': uid,
        'wave_prev': w_prev,
        'wave_curr': w_curr,
        'year_prev': wave_year[w_prev],
        'year_curr': wave_year[w_curr],
        # Y_t: 当期因变量
        'cesd10_t': row_curr['cesd10'],
        # lag_cesd10: 上期因变量
        'lag_cesd10': row_prev['cesd10'],
        # province和region（取t-1期，通常不变）
        'province': row_prev['province'],
        'region': row_prev['region'],
    }
    # 所有W变量取t-1期
    for v in w_vars:
        rec[v] = row_prev[v]
    # wave标记（用于DML的W）
    rec['wave'] = w_curr  # 使用当期wave作为时间标记

    records.append(rec)

causal_df = pd.DataFrame(records)
print(f"  因果样本: {causal_df.shape}")

# ============================================================
# 5. 变量变换
# ============================================================
print("\n=== 变量变换 ===")

# 5.1 ln(x+1) 变换
causal_df['hhcperc_log'] = np.log(causal_df['hhcperc'] + 1)
causal_df['fcamt_log'] = np.log(causal_df['fcamt'] + 1)

# 5.2 构造Treatment变量
# T1: 社会参与（连续）
causal_df['T1_social'] = causal_df['social_activity_index']
# T1_binary: 参与≥1种活动（稳健性用）
causal_df['T1_binary'] = (causal_df['social_activity_index'] >= 1).astype(int)

# T2: 高慢病负担（二元）
causal_df['T2_chronic'] = (causal_df['chronic_disease_count'] >= 2).astype(int)

# T3: 睡眠异常（二元）
causal_df['T3_sleep'] = ((causal_df['sleep'] < 6) | (causal_df['sleep'] > 9)).astype(int)

print(f"  T1(社会参与): mean={causal_df['T1_social'].mean():.3f}, 参与率={causal_df['T1_binary'].mean():.3f}")
print(f"  T2(多病共存≥2): prevalence={causal_df['T2_chronic'].mean():.3f}")
print(f"  T3(睡眠异常): prevalence={causal_df['T3_sleep'].mean():.3f}")

# ============================================================
# 6. 删除关键变量缺失行
# ============================================================
key_cols = ['cesd10_t', 'lag_cesd10', 'T1_social', 'T2_chronic', 'T3_sleep',
            'age', 'gender', 'edu', 'marry', 'rural2', 'hhcperc_log',
            'ins', 'pension', 'bmi', 'region']
n_before = len(causal_df)
causal_df = causal_df.dropna(subset=key_cols)
n_after = len(causal_df)
print(f"\n  关键变量缺失删除: {n_before} → {n_after} (删{n_before - n_after})")

# ============================================================
# 7. 保存
# ============================================================
save_path = os.path.join(OUT, 'causal_sample_lagged.xlsx')
causal_df.to_excel(save_path, index=False)
print(f"\n  因果样本已保存: {save_path}")
print(f"  最终维度: {causal_df.shape}")

# ============================================================
# 8. 样本摘要
# ============================================================
print("\n=== 因果样本摘要 ===")
summary_rows = [
    {'指标': '总配对数', '值': len(causal_df)},
    {'指标': '唯一个体数', '值': causal_df['ID'].nunique()},
    {'指标': 'cesd10_t 均值', '值': round(causal_df['cesd10_t'].mean(), 3)},
    {'指标': 'cesd10_t 标准差', '值': round(causal_df['cesd10_t'].std(), 3)},
    {'指标': 'lag_cesd10 均值', '值': round(causal_df['lag_cesd10'].mean(), 3)},
    {'指标': 'T1(社会参与) 均值', '值': round(causal_df['T1_social'].mean(), 3)},
    {'指标': 'T1_binary 参与率', '值': round(causal_df['T1_binary'].mean(), 3)},
    {'指标': 'T2(多病共存≥2) 比例', '值': round(causal_df['T2_chronic'].mean(), 3)},
    {'指标': 'T3(睡眠异常) 比例', '值': round(causal_df['T3_sleep'].mean(), 3)},
    {'指标': '年龄均值', '值': round(causal_df['age'].mean(), 1)},
    {'指标': '男性比例', '值': round(causal_df['gender'].mean(), 3)},
    {'指标': '城镇户口比例', '值': round(causal_df['rural2'].mean(), 3)},
    {'指标': '东部占比', '值': round((causal_df['region']==1).mean(), 3)},
    {'指标': '中部占比', '值': round((causal_df['region']==2).mean(), 3)},
    {'指标': '西部占比', '值': round((causal_df['region']==3).mean(), 3)},
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(os.path.join(OUT, 'sample_summary.xlsx'), index=False)
print(summary_df.to_string(index=False))

# 按波次对统计
print("\n按配对波次分布:")
pair_dist = causal_df.groupby(['wave_prev','wave_curr']).size().reset_index(name='count')
pair_dist['年份对'] = pair_dist.apply(lambda r: f"{wave_year[r['wave_prev']]}→{wave_year[r['wave_curr']]}", axis=1)
print(pair_dist[['年份对','count']].to_string(index=False))

print("\n=== 阶段4完成 ===")
