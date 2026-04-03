"""
阶段1补充：异常值 → NaN → XGBoost迭代插补 → 保存正式建模数据
============================================================
异常值规则：
  BMI = 0 或 BMI > 50       → NaN  (55条：1+27+27)
  sleep < 0 或 sleep = 0    → NaN  (155条：79+76)
  sleep > 16                → NaN  (4条)
  age > 100                 → NaN  (6条)
插补方法：
  XGBoost迭代插补（逐变量预测缺失值，迭代至收敛）
输出：
  data/preprocess_outputs/data1_modeling_ready.xlsx  ← 正式建模数据
  results/01_data_audit/anomaly_imputation_log.xlsx  ← 插补日志
"""

import pandas as pd
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 0. 读取原始数据
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(ROOT, 'data/preprocess_outputs/data1_preprocessed_for_screening.xlsx'))
print(f"原始数据: {df.shape}")

# ============================================================
# 1. 标记异常值 → NaN
# ============================================================
anomaly_log = []

def mark_anomaly(df, col, condition, reason):
    """将满足condition的值设为NaN，记录日志"""
    mask = condition(df[col])
    n = mask.sum()
    if n > 0:
        vals = df.loc[mask, col].unique()
        df.loc[mask, col] = np.nan
        anomaly_log.append({
            '变量': col,
            '条件': reason,
            '设NaN数量': n,
            '涉及值': str(vals[:10])
        })
        print(f"  {col}: {reason} → {n}条设NaN")
    return df

print("\n=== 标记异常值 ===")
# BMI
df = mark_anomaly(df, 'bmi', lambda x: x == 0, 'BMI=0 (无效值)')
df = mark_anomaly(df, 'bmi', lambda x: x > 50, 'BMI>50 (极端异常)')

# sleep
df = mark_anomaly(df, 'sleep', lambda x: x < 0, 'sleep<0 (无效值-1)')
df = mark_anomaly(df, 'sleep', lambda x: x == 0, 'sleep=0 (录入异常)')
df = mark_anomaly(df, 'sleep', lambda x: x > 16, 'sleep>16 (不合理)')

# age
df = mark_anomaly(df, 'age', lambda x: x > 100, 'age>100 (极端高龄)')

# 汇总
total_nan_cells = df[['bmi', 'sleep', 'age']].isna().sum().sum()
affected_rows = df[df[['bmi', 'sleep', 'age']].isna().any(axis=1)]
print(f"\n共标记 {total_nan_cells} 个NaN单元格，涉及 {len(affected_rows)} 行（占总样本 {len(affected_rows)/len(df)*100:.2f}%）")

# ============================================================
# 2. XGBoost迭代插补
# ============================================================
print("\n=== XGBoost迭代插补 ===")

# 需要插补的变量
impute_cols = ['bmi', 'sleep', 'age']

# 选择用于插补的特征（数值型，排除ID和字符串列）
# province是字符串，先编码
le_province = LabelEncoder()
df['province_enc'] = le_province.fit_transform(df['province'].astype(str))

# 所有数值列作为潜在特征（排除ID类和province原始列）
exclude_cols = ['ID', 'householdID', 'communityID', 'province']
feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
                and c not in impute_cols]

print(f"插补特征数: {len(feature_cols)}")

# 迭代插补
MAX_ITER = 5
for iteration in range(MAX_ITER):
    print(f"\n--- 迭代 {iteration+1}/{MAX_ITER} ---")
    changes = 0

    for col in impute_cols:
        mask = df[col].isna()
        n_miss = mask.sum()
        if n_miss == 0:
            continue

        # 构建特征矩阵（使用其他非缺失变量）
        other_impute = [c for c in impute_cols if c != col]
        use_features = feature_cols + other_impute
        # 排除当前特征中也有NaN的列（第一轮时其他impute_cols可能也有NaN）
        X_all = df[use_features].copy()

        # 对于特征中仍有NaN的列，用中位数临时填充（仅用于预测，不改原数据）
        for fc in X_all.columns:
            if X_all[fc].isna().any():
                X_all[fc] = X_all[fc].fillna(X_all[fc].median())

        X_train = X_all[~mask]
        y_train = df.loc[~mask, col]
        X_pred = X_all[mask]

        # XGBoost插补
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_pred)

        # 对插补值做合理性约束
        if col == 'bmi':
            predicted = np.clip(predicted, 15, 40)
        elif col == 'sleep':
            predicted = np.clip(predicted, 1, 14)
        elif col == 'age':
            predicted = np.clip(predicted, 60, 100)

        # 记录变化
        old_vals = df.loc[mask, col].copy()
        df.loc[mask, col] = predicted
        changes += n_miss

        print(f"  {col}: 插补{n_miss}条, 范围[{predicted.min():.2f}, {predicted.max():.2f}], "
              f"均值={predicted.mean():.2f}")

    if changes == 0:
        print("无变化，提前终止")
        break

# 删除临时编码列
df.drop(columns=['province_enc'], inplace=True)

# ============================================================
# 3. 保存
# ============================================================
out_data = os.path.join(ROOT, 'data/preprocess_outputs/data1_modeling_ready.xlsx')
df.to_excel(out_data, index=False)
print(f"\n正式建模数据已保存: {out_data}")
print(f"维度: {df.shape}")

out_log = os.path.join(ROOT, 'results/01_data_audit/anomaly_imputation_log.xlsx')
log_df = pd.DataFrame(anomaly_log)
log_df.to_excel(out_log, index=False)
print(f"插补日志已保存: {out_log}")

# ============================================================
# 4. 插补后验证
# ============================================================
print("\n=== 插补后验证 ===")
for col in impute_cols:
    s = df[col]
    print(f"  {col}: NaN={s.isna().sum()}, mean={s.mean():.3f}, sd={s.std():.3f}, "
          f"min={s.min():.2f}, max={s.max():.2f}")
print(f"\n全数据NaN总数: {df.isna().sum().sum()}")
