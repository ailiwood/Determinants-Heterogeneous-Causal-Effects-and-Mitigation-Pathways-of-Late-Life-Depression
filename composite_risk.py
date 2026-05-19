"""
复合风险变量定义工具
============================================================
功能：根据CLAUDE.md Section 6.1统一定义复合风险变量
口径：
  - low_social: 社会参与指数==0（二值化，仅用于交互/累积分析）
  - multimorbidity: 慢性病数量>=2
  - abnormal_sleep: 睡眠<6或>9小时
  - risk_count: 三者之和（0-3）
============================================================
"""

import pandas as pd
import numpy as np
import os

def build_composite_risk_variables(df, source='main'):
    """
    为数据框添加复合风险变量

    Parameters
    ----------
    df : pd.DataFrame
        原始数据框，需包含 social_activity_index, chronic_disease_count, sleep 列
    source : str
        'main' = FE主样本（使用social_activity_index）
        'causal' = 因果样本（使用滞后处理变量）

    Returns
    -------
    pd.DataFrame
        添加了复合风险变量的数据框
    """
    # 社会参与：完全不参与 = 1，否则 = 0
    # 注意：因果样本中列名为 social_activity_index_lag1
    if source == 'causal':
        sa_col = 'social_activity_index_lag1'
    else:
        sa_col = 'social_activity_index'

    df['low_social'] = (df[sa_col] == 0).astype(int)

    # 多病共存：慢性病数量>=2
    df['multimorbidity'] = (df['chronic_disease_count'] >= 2).astype(int)

    # 睡眠异常：<6小时或>9小时
    df['abnormal_sleep'] = ((df['sleep'] < 6) | (df['sleep'] > 9)).astype(int)

    # 风险计数：三者之和
    df['risk_count'] = df['low_social'] + df['multimorbidity'] + df['abnormal_sleep']

    return df


def get_risk_labels():
    """返回复合风险变量的中英文标签"""
    return {
        'low_social': {
            'zh': '完全不参与社会活动',
            'en': 'Zero Social Participation',
            'description': '社会参与指数为0'
        },
        'multimorbidity': {
            'zh': '多病共存',
            'en': 'Multimorbidity (≥2 chronic diseases)',
            'description': '慢性病数量≥2'
        },
        'abnormal_sleep': {
            'zh': '睡眠异常',
            'en': 'Abnormal Sleep (<6h or >9h)',
            'description': '睡眠<6小时或>9小时'
        },
        'risk_count': {
            'zh': '风险计数',
            'en': 'Composite Risk Count',
            'description': '0/1/2/3个风险因素'
        }
    }


def risk_distribution_summary(df):
    """
    生成复合风险分布描述统计

    Parameters
    ----------
    df : pd.DataFrame
        包含复合风险变量的数据框

    Returns
    -------
    pd.DataFrame
        风险分布统计表
    """
    # 各风险状态的人数和比例
    n_total = len(df)

    # 风险计数分布
    risk_dist = df['risk_count'].value_counts().sort_index()
    risk_pct = (risk_dist / n_total * 100).round(1)

    result = pd.DataFrame({
        '风险因素数量': risk_dist.index,
        '人数': risk_dist.values,
        '占比(%)': risk_pct.values
    })

    # 各二分变量的分布
    binary_vars = ['low_social', 'multimorbidity', 'abnormal_sleep']
    for var in binary_vars:
        count_1 = df[var].sum()
        pct_1 = count_1 / n_total * 100
        result[var] = [f'{count_1} ({pct_1:.1f}%)' if i == 0 else '' for i in range(len(result))]

    return result


if __name__ == '__main__':
    # 测试代码
    print("composite_risk.py 工具已加载")
    print("函数: build_composite_risk_variables(df, source='main')")
    print("函数: risk_distribution_summary(df)")
    print("函数: get_risk_labels()")