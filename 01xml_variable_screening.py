# -*- coding: utf-8 -*-
"""
脚本名称：xml_variable_screening_from_preprocessed.py
用途：
1. 直接读取“已经预处理后的正式建模数据”，不再重复做样本删除、删变量、RF插补与指标构造
2. 基于可解释机器学习（XML）完成变量筛选：
   - LASSO（GroupKFold + GridSearchCV）
   - Random Forest（GroupKFold + RandomizedSearchCV，小搜索空间）
   - XGBoost（GroupKFold + RandomizedSearchCV，小搜索空间）
3. 输出与上一版保持一致：
   - 三个模型的最佳超参数与性能汇总（1个xlsx）
   - 三个算法的最佳模型PKL文件
   - LASSO变量筛选结果（xlsx）
   - RF / XGB变量重要性结果（各1个xlsx）
   - 三算法整合排序结果（1个xlsx）
   - 各算法柱状图与蜂群图

依赖：
pip install pandas numpy scikit-learn xgboost shap openpyxl matplotlib joblib
"""

from __future__ import annotations

import json
import math
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError as e:
    raise ImportError("未检测到 shap，请先执行：pip install shap") from e

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("未检测到 xgboost，请先执行：pip install xgboost") from e


# =========================================================
# 一、配置区
# =========================================================
CONFIG = {
    # ===== 输入文件：这里直接读取预处理后的正式建模数据 =====
    "data_path" : r"D:\BaiduSyncdisk\lunwen\model0320\data\preprocess_outputs\data1_preprocessed_for_screening.xlsx" ,
    "data_sheet" :0 ,
    # 说明：输出统一使用数据中的原始变量名/预处理后变量名，不做中文重命名
    "label_path" : r"D:\BaiduSyncdisk\lunwen\model0320\data\preprocess_outputs\labels_updated_for_preprocessed_data.xlsx",

    # ===== 输出目录 =====
    "output_dir" : r"D:\BaiduSyncdisk\lunwen\model0320\results\xml_variable_screening",

    # ===== 核心字段 =====
    "target": "cesd10",
    "id_col": "ID",
    "group_split_col": "ID",

    # ===== 通用参数 =====
    "random_state": 20260322,
    "test_size": 0.20,
    "top_k": 20,
    "max_shap_sample": 1500,
    "max_category_levels": 40,
    "cv_n_splits": 5,
    "cv_scoring": "neg_root_mean_squared_error",

    # ===== 小搜索空间：控制运行时间 =====
    "lasso_alpha_grid": list(np.logspace(-3, 0.5, 12)),
    "rf_n_iter": 8,
    "xgb_n_iter": 10,
}


# =========================================================
# 二、候选变量池（基于“已预处理后数据”重新整理）
# =========================================================
DEMOGRAPHIC_VARS = [
    "age", "gender", "edu", "marry", "nation", "rural", "rural2", "province"
]

SOCIOECONOMIC_VARS = [
    "income_total", "hhcperc", "ins", "pension", "retire", "water",
    "family_size", "hchild", "fcamt", "tcamt", "communication",
    "insurance_coverage_count"
]

HEALTH_VARS = [
    "srh", "adlab_c", "iadl", "bmi", "disability",
    "body_discomfort_count", "chronic_disease_count",
    "hibpe", "diabe", "cancre", "lunge", "hearte", "stroke",
    "arthre", "dyslipe", "livere", "kidneye", "digeste", "asthmae",
    "memrye", "hear",
    # 兼容若预处理后仍保留的旧综合指标
    "disease_index"
]

PSYCHO_COG_VARS = [
    "total_cognition", "memeory", "executive", "satlife"
]

BEHAVIOR_VARS = [
    "sleep", "exercise", "vgact_c", "mdact_c", "ltact_c", "totmet",
    "smoken", "smokev", "drinkl", "drinkev"
]

SOCIAL_SUPPORT_VARS = [
    # 预处理后新增或保留的综合变量
    "activity_index", "social_activity_index", "social_index",
    # 保留原始社交活动变量和社会支持变量（若存在）
    "social1", "social2", "social3", "social4", "social5", "social6",
    "social7", "social8", "social9", "social10", "social11",
]

BIOMARKER_VARS = [
    "tyg", "tyg_bmi", "bl_wbc", "bl_glu", "bl_crea", "bl_cho", "bl_tg", "bl_hdl",
    "bl_ldl", "bl_crp", "bl_hbalc", "bl_ua", "bl_hct", "bl_hgb", "bl_mcv",
    "bl_plt", "bl_rdw", "bl_top_coding_tg"
]

MEDICAL_USE_VARS = [
    "hospital", "hospital_time", "hspnite", "doctor", "doctor_time",
    "oophos1y", "tothos1y", "oopdoc1m", "totdoc1m"
]

GROUP_VARS = ["gender", "rural2", "marry", "province"]

LEAKAGE_OR_EXCLUDE_VARS = [
    # ID 与明显泄漏变量
    "ID", "householdID", "communityID", "wave", "iwy", "iwm", "city",
    "cesd10", "depressed", "depression", "cesd",
    # 研究范围上已经处理/不应再进模型的变量
    "psyche",
    # 预处理阶段已经应被删除的原始变量
    "wspeed", "wspeed1", "wspeed2", "puff", "rgrip", "lgrip", "diasto", "systo", "hope",
    # 预处理阶段构造后通常不再保留的原始变量
    "da042s1", "da042s2", "da042s3", "da042s4", "da042s5", "da042s6", "da042s7", "da042s8",
    "da042s9", "da042s10", "da042s11", "da042s12", "da042s13", "da042s14", "da042s15",
    "teeth", "hip",
    "act_1", "act_2", "act_3", "act_4", "act_5", "act_6", "act_7", "act_8",
    "ea001s1", "ea001s2", "ea001s3", "ea001s4", "ea001s5", "ea001s6", "ea001s7", "ea001s8", "ea001s11",
    "ea001s9", "ea001s10",
]

FORCE_CATEGORICAL_VARS = [
    "gender", "marry", "nation", "rural", "rural2", "edu", "ins", "pension",
    "retire", "exercise", "drinkl", "drinkev", "smoken", "smokev", "water",
    "province"
]

MANUAL_ADD_VARS: List[str] = []
MANUAL_DROP_VARS: List[str] = []


# =========================================================
# 三、基础工具函数
# =========================================================
def setup_chinese_font() -> None:
    preferred_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_print(msg: str, logs: List[str]) -> None:
    print(msg)
    logs.append(str(msg))


def save_log(log_path: Path, logs: List[str]) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        for line in logs:
            f.write(str(line) + "\n")


def read_table(file_path: str, sheet=0) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"暂不支持该文件格式：{suffix}")


def save_single_sheet_xlsx(df: pd.DataFrame, out_path: Path, sheet_name: str = "Sheet1") -> None:
    ensure_dir(out_path.parent)
    if df is None or df.empty:
        df = pd.DataFrame({"message": ["暂无数据"]})
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def clean_categorical_series(s: pd.Series) -> pd.Series:
    """将类别变量统一清洗为纯字符串列，避免 float / str 混型。"""
    s = s.copy()
    s = s.replace(r"^\s*$", np.nan, regex=True)
    s = s.astype("object")
    s = s.where(~pd.isna(s), "__MISSING__")
    s = s.astype(str)
    return s


def normalize_categorical_columns(X_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for c in categorical_cols:
        if c in X_train.columns:
            X_train[c] = clean_categorical_series(X_train[c])
        if c in X_test.columns:
            X_test[c] = clean_categorical_series(X_test[c])
    return X_train, X_test


def build_feature_pool(df: pd.DataFrame) -> List[str]:
    base_pool = (
        DEMOGRAPHIC_VARS
        + SOCIOECONOMIC_VARS
        + HEALTH_VARS
        + PSYCHO_COG_VARS
        + BEHAVIOR_VARS
        + SOCIAL_SUPPORT_VARS
        + BIOMARKER_VARS
        + MEDICAL_USE_VARS
    )
    base_pool = list(dict.fromkeys(base_pool + MANUAL_ADD_VARS))
    base_pool = [c for c in base_pool if c not in LEAKAGE_OR_EXCLUDE_VARS]
    base_pool = [c for c in base_pool if c not in MANUAL_DROP_VARS]
    base_pool = [c for c in base_pool if c in df.columns]
    return base_pool


def infer_variable_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    categorical_cols = []
    numeric_cols = []
    for c in feature_cols:
        if c in FORCE_CATEGORICAL_VARS:
            categorical_cols.append(c)
        elif pd.api.types.is_object_dtype(df[c]) or str(df[c].dtype).startswith("category") or pd.api.types.is_bool_dtype(df[c]):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    return numeric_cols, categorical_cols


def remove_bad_categorical_levels(df: pd.DataFrame, feature_cols: List[str], logs: List[str]) -> List[str]:
    keep = []
    for c in feature_cols:
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        is_cat = (c in FORCE_CATEGORICAL_VARS) or pd.api.types.is_object_dtype(s) or str(s.dtype).startswith("category") or pd.api.types.is_bool_dtype(s)
        if is_cat and nunique > CONFIG["max_category_levels"]:
            log_print(f"[DROP] {c}: 类别水平过多（{nunique} > {CONFIG['max_category_levels']}）", logs)
            continue
        if nunique <= 1:
            log_print(f"[DROP] {c}: 常数或全缺失", logs)
            continue
        keep.append(c)
    return keep


def group_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    valid = df[target].notna()
    work = df.loc[valid, feature_cols + [target] + ([group_col] if group_col in df.columns else [])].copy()
    X = work[feature_cols].copy()
    y = pd.to_numeric(work[target], errors="coerce")

    if group_col in work.columns:
        groups = work[group_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        return (
            X.iloc[train_idx].copy(),
            X.iloc[test_idx].copy(),
            y.iloc[train_idx].copy(),
            y.iloc[test_idx].copy(),
            groups.iloc[train_idx].copy(),
            groups.iloc[test_idx].copy(),
        )

    rng = np.random.RandomState(CONFIG["random_state"])
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = int(len(idx) * CONFIG["test_size"])
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
        pd.Series(np.arange(len(train_idx)), index=X.iloc[train_idx].index),
        pd.Series(np.arange(len(test_idx)), index=X.iloc[test_idx].index),
    )


def build_cv(groups: Optional[pd.Series]):
    if groups is None:
        return KFold(n_splits=5, shuffle=True, random_state=CONFIG["random_state"]), False
    n_groups = int(pd.Series(groups).nunique())
    if n_groups >= 3:
        n_splits = min(CONFIG["cv_n_splits"], n_groups)
        return GroupKFold(n_splits=n_splits), True
    return KFold(n_splits=3, shuffle=True, random_state=CONFIG["random_state"]), False


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}


def evaluate_model(name: str, model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_m = regression_metrics(y_train, train_pred)
    test_m = regression_metrics(y_test, test_pred)
    return pd.DataFrame([
        {"model": name, "sample": "train", **train_m, "n": len(y_train)},
        {"model": name, "sample": "test", **test_m, "n": len(y_test)},
    ])


# =========================================================
# 四、模型预处理与调参
# =========================================================
def build_preprocessors(numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[ColumnTransformer, ColumnTransformer]:
    # LASSO：数值标准化；类别先补缺再 one-hot
    num_lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    lasso_pre = ColumnTransformer(
        transformers=[
            ("num", num_lasso, numeric_cols),
            ("cat", cat_lasso, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # 树模型：数值先补中位数；类别先补缺再编码
    num_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    tree_pre = ColumnTransformer(
        transformers=[
            ("num", num_tree, numeric_cols),
            ("cat", cat_tree, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return lasso_pre, tree_pre


def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series, groups: Optional[pd.Series], lasso_pre: ColumnTransformer):
    pipe = Pipeline([
        ("preprocessor", lasso_pre),
        ("model", Lasso(max_iter=50000, random_state=CONFIG["random_state"])),
    ])
    param_grid = {"model__alpha": CONFIG["lasso_alpha_grid"]}
    cv, used_group_cv = build_cv(groups)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=CONFIG["cv_scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    if used_group_cv:
        gs.fit(X_train, y_train, groups=groups)
    else:
        gs.fit(X_train, y_train)
    return gs.best_estimator_, {"best_params": gs.best_params_, "best_score": gs.best_score_, "used_group_cv": used_group_cv}


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series, groups: Optional[pd.Series], tree_pre: ColumnTransformer):
    base = RandomForestRegressor(random_state=CONFIG["random_state"], n_jobs=1)
    pipe = Pipeline([
        ("preprocessor", tree_pre),
        ("model", base),
    ])
    param_dist = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 10],
        "model__min_samples_split": [2, 8],
        "model__min_samples_leaf": [2, 4],
        "model__max_features": ["sqrt", 0.5],
        "model__bootstrap": [True],
    }
    cv, used_group_cv = build_cv(groups)
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=CONFIG["rf_n_iter"],
        scoring=CONFIG["cv_scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=CONFIG["random_state"],
        return_train_score=False,
    )
    if used_group_cv:
        rs.fit(X_train, y_train, groups=groups)
    else:
        rs.fit(X_train, y_train)
    return rs.best_estimator_, {"best_params": rs.best_params_, "best_score": rs.best_score_, "used_group_cv": used_group_cv}


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, groups: Optional[pd.Series], tree_pre: ColumnTransformer):
    base = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=CONFIG["random_state"],
        n_jobs=1,
    )
    pipe = Pipeline([
        ("preprocessor", tree_pre),
        ("model", base),
    ])
    param_dist = {
        "model__n_estimators": [180, 280],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.03, 0.08],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        "model__min_child_weight": [1, 3],
        "model__gamma": [0.0, 0.05],
        "model__reg_alpha": [0.0, 0.1],
        "model__reg_lambda": [1.0, 2.0],
    }
    cv, used_group_cv = build_cv(groups)
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=CONFIG["xgb_n_iter"],
        scoring=CONFIG["cv_scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=CONFIG["random_state"],
        return_train_score=False,
    )
    if used_group_cv:
        rs.fit(X_train, y_train, groups=groups)
    else:
        rs.fit(X_train, y_train)
    return rs.best_estimator_, {"best_params": rs.best_params_, "best_score": rs.best_score_, "used_group_cv": used_group_cv}


# =========================================================
# 五、变量重要性与图表
# =========================================================
def save_model_pickle(model, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    dump(model, out_path)


def back_to_original_feature(name: str, categorical_cols: List[str]) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        rest = name.replace("cat__", "", 1)
        for col in sorted(categorical_cols, key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        return rest.split("_")[0]
    return name


def get_lasso_importance(lasso_model, categorical_cols: List[str]) -> pd.DataFrame:
    feature_names = list(lasso_model.named_steps["preprocessor"].get_feature_names_out())
    coef = lasso_model.named_steps["model"].coef_
    out = pd.DataFrame({
        "transformed_feature": feature_names,
        "coef_signed": coef,
        "coef_abs": np.abs(coef),
    })
    out["variable"] = out["transformed_feature"].apply(lambda x: back_to_original_feature(x, categorical_cols))
    agg = out.groupby("variable", as_index=False).agg(
        coef_abs=("coef_abs", "sum"),
        coef_signed=("coef_signed", "sum"),
        n_dummy_terms=("transformed_feature", "count"),
    )
    agg["selected_by_lasso"] = agg["coef_abs"] > 1e-12
    agg = agg.sort_values(["selected_by_lasso", "coef_abs", "variable"], ascending=[False, False, True]).reset_index(drop=True)
    agg["rank_lasso"] = np.arange(1, len(agg) + 1)
    return agg


def get_tree_shap_importance(tree_model, X_reference: pd.DataFrame, model_name: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    pre = tree_model.named_steps["preprocessor"]
    model = tree_model.named_steps["model"]

    if len(X_reference) > CONFIG["max_shap_sample"]:
        X_ref = X_reference.sample(CONFIG["max_shap_sample"], random_state=CONFIG["random_state"]).copy()
    else:
        X_ref = X_reference.copy()

    X_t = pre.transform(X_ref)
    X_t_df = pd.DataFrame(np.asarray(X_t), columns=list(X_reference.columns), index=X_ref.index)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_t_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_mean = np.abs(shap_values).mean(axis=0)

    imp_df = pd.DataFrame({
        "variable": list(X_reference.columns),
        f"importance_{model_name}": shap_mean,
    }).sort_values(f"importance_{model_name}", ascending=False).reset_index(drop=True)
    imp_df[f"rank_{model_name}"] = np.arange(1, len(imp_df) + 1)
    return imp_df, shap_values, X_t_df


def minmax_scale(s: pd.Series) -> pd.Series:
    s = s.fillna(0)
    if s.empty:
        return s
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def combine_rankings(lasso_df: pd.DataFrame, rf_df: pd.DataFrame, xgb_df: pd.DataFrame) -> pd.DataFrame:
    all_vars = sorted(set(lasso_df["variable"]).union(set(rf_df["variable"])).union(set(xgb_df["variable"])))
    out = pd.DataFrame({"variable": all_vars})
    out = out.merge(lasso_df[["variable", "coef_abs", "rank_lasso", "selected_by_lasso"]], on="variable", how="left")
    out = out.merge(rf_df[["variable", "importance_rf", "rank_rf"]], on="variable", how="left")
    out = out.merge(xgb_df[["variable", "importance_xgb", "rank_xgb"]], on="variable", how="left")
    out["selected_by_lasso"] = out["selected_by_lasso"].fillna(False)
    out["score_lasso"] = minmax_scale(out["coef_abs"])
    out["score_rf"] = minmax_scale(out["importance_rf"])
    out["score_xgb"] = minmax_scale(out["importance_xgb"])
    out["ensemble_score"] = out[["score_lasso", "score_rf", "score_xgb"]].sum(axis=1)
    out["top20_count"] = (
        out["rank_lasso"].fillna(999).le(20).astype(int)
        + out["rank_rf"].fillna(999).le(20).astype(int)
        + out["rank_xgb"].fillna(999).le(20).astype(int)
    )
    out = out.sort_values(["ensemble_score", "selected_by_lasso", "top20_count", "variable"], ascending=[False, False, False, True]).reset_index(drop=True)
    out["final_rank"] = np.arange(1, len(out) + 1)
    return out


def save_horizontal_bar(df: pd.DataFrame, value_col: str, label_col: str, title: str, out_path: Path, top_n: int = 20) -> None:
    ensure_dir(out_path.parent)
    plot_df = df.head(top_n).iloc[::-1].copy()
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lasso_beeswarm_like(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    ensure_dir(out_path.parent)
    plot_df = df.head(top_n).iloc[::-1].copy()
    y_pos = np.arange(len(plot_df))
    rng = np.random.RandomState(CONFIG["random_state"])
    jitter = rng.uniform(-0.12, 0.12, size=len(plot_df))

    plt.figure(figsize=(10, 7))
    plt.scatter(plot_df["coef_signed"], y_pos + jitter, s=60)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y_pos, plot_df["variable"])
    plt.title("LASSO变量筛选结果（蜂群样式图）")
    plt.xlabel("聚合后有符号系数")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_shap_beeswarm(shap_values: np.ndarray, X_transformed_df: pd.DataFrame, title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_transformed_df, show=False, max_display=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 六、主程序
# =========================================================
def main() -> None:
    setup_chinese_font()

    out_dir = Path(CONFIG["output_dir"])
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    model_dir = out_dir / "models"
    log_dir = out_dir / "logs"
    for p in [out_dir, table_dir, figure_dir, model_dir, log_dir]:
        ensure_dir(p)

    logs: List[str] = []
    log_file = log_dir / "run_log.txt"

    try:
        # 1. 读取预处理后的正式建模数据
        log_print(f"[START] 读取预处理后的正式建模数据：{CONFIG['data_path']}", logs)
        df = read_table(CONFIG["data_path"], sheet=CONFIG["data_sheet"])
        log_print(f"[INFO] 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列", logs)
        save_log(log_file, logs)

        if CONFIG["target"] not in df.columns:
            raise KeyError(f"因变量 {CONFIG['target']} 不存在，请检查预处理后的数据。")
        if CONFIG["group_split_col"] not in df.columns:
            log_print(f"[WARN] 分组列 {CONFIG['group_split_col']} 不存在，将退化为普通随机切分。", logs)

        # 2. 构建候选变量池
        feature_pool = build_feature_pool(df)
        log_print(f"[INFO] 初始候选变量数：{len(feature_pool)}", logs)
        feature_pool = remove_bad_categorical_levels(df, feature_pool, logs)
        numeric_cols, categorical_cols = infer_variable_types(df, feature_pool)
        log_print(f"[INFO] 过滤后变量数：{len(feature_pool)}", logs)
        log_print(f"[INFO] 数值变量：{len(numeric_cols)} 个；类别变量：{len(categorical_cols)} 个", logs)
        save_log(log_file, logs)

        # 3. 训练/测试集划分
        X_train, X_test, y_train, y_test, g_train, g_test = group_train_test_split(
            df=df,
            feature_cols=feature_pool,
            target=CONFIG["target"],
            group_col=CONFIG["group_split_col"],
        )
        X_train, X_test = normalize_categorical_columns(X_train, X_test, categorical_cols)
        log_print(f"[INFO] 训练集：{len(X_train)}；测试集：{len(X_test)}", logs)
        log_print(f"[INFO] 训练集唯一组数：{pd.Series(g_train).nunique()}；测试集唯一组数：{pd.Series(g_test).nunique()}", logs)
        save_log(log_file, logs)

        # 4. 预处理器
        lasso_pre, tree_pre = build_preprocessors(numeric_cols, categorical_cols)

        summary_rows = []

        # 5. LASSO
        log_print("[MODEL] 开始拟合 LASSO ...", logs)
        lasso_model, lasso_info = fit_lasso(X_train, y_train, g_train, lasso_pre)
        lasso_pkl = model_dir / "best_lasso.pkl"
        save_model_pickle(lasso_model, lasso_pkl)
        log_print(f"[SAVE] 已保存 LASSO 最佳模型：{lasso_pkl.resolve()}", logs)

        lasso_perf = evaluate_model("LASSO", lasso_model, X_train, y_train, X_test, y_test)
        lasso_imp = get_lasso_importance(lasso_model, categorical_cols)
        save_single_sheet_xlsx(lasso_imp, table_dir / "lasso_selected_variables.xlsx", sheet_name="lasso")
        save_horizontal_bar(lasso_imp, "coef_abs", "variable", "LASSO变量筛选结果（Top20）", figure_dir / "lasso_bar_top20.png", top_n=20)
        save_lasso_beeswarm_like(lasso_imp, figure_dir / "lasso_beeswarm_top20.png", top_n=20)
        save_log(log_file, logs)

        lasso_test = lasso_perf.query("sample == 'test'").iloc[0]
        summary_rows.append({
            "model": "LASSO",
            "best_params_json": json.dumps(lasso_info["best_params"], ensure_ascii=False, default=str),
            "best_cv_rmse": -float(lasso_info["best_score"]),
            "train_r2": float(lasso_perf.query("sample=='train'").iloc[0]["r2"]),
            "train_mse": float(lasso_perf.query("sample=='train'").iloc[0]["mse"]),
            "train_rmse": float(lasso_perf.query("sample=='train'").iloc[0]["rmse"]),
            "train_mae": float(lasso_perf.query("sample=='train'").iloc[0]["mae"]),
            "test_r2": float(lasso_test["r2"]),
            "test_mse": float(lasso_test["mse"]),
            "test_rmse": float(lasso_test["rmse"]),
            "test_mae": float(lasso_test["mae"]),
            "model_file": str(lasso_pkl.resolve()),
        })
        save_single_sheet_xlsx(pd.DataFrame(summary_rows), table_dir / "best_hyperparameters.xlsx", sheet_name="summary")

        # 6. Random Forest
        log_print("[MODEL] 开始拟合 Random Forest ...", logs)
        rf_model, rf_info = fit_random_forest(X_train, y_train, g_train, tree_pre)
        rf_pkl = model_dir / "best_rf.pkl"
        save_model_pickle(rf_model, rf_pkl)
        log_print(f"[SAVE] 已保存 RF 最佳模型：{rf_pkl.resolve()}", logs)

        rf_perf = evaluate_model("RF", rf_model, X_train, y_train, X_test, y_test)
        rf_imp, rf_shap_values, rf_x_trans = get_tree_shap_importance(rf_model, X_train, model_name="rf")
        save_single_sheet_xlsx(rf_imp, table_dir / "rf_variable_importance.xlsx", sheet_name="rf")
        save_horizontal_bar(rf_imp, "importance_rf", "variable", "Random Forest变量重要性（Top20）", figure_dir / "rf_bar_top20.png", top_n=20)
        save_shap_beeswarm(rf_shap_values, rf_x_trans, "Random Forest SHAP蜂群图（Top20显示）", figure_dir / "rf_beeswarm_top20.png")
        save_log(log_file, logs)

        rf_test = rf_perf.query("sample == 'test'").iloc[0]
        summary_rows.append({
            "model": "RF",
            "best_params_json": json.dumps(rf_info["best_params"], ensure_ascii=False, default=str),
            "best_cv_rmse": -float(rf_info["best_score"]),
            "train_r2": float(rf_perf.query("sample=='train'").iloc[0]["r2"]),
            "train_mse": float(rf_perf.query("sample=='train'").iloc[0]["mse"]),
            "train_rmse": float(rf_perf.query("sample=='train'").iloc[0]["rmse"]),
            "train_mae": float(rf_perf.query("sample=='train'").iloc[0]["mae"]),
            "test_r2": float(rf_test["r2"]),
            "test_mse": float(rf_test["mse"]),
            "test_rmse": float(rf_test["rmse"]),
            "test_mae": float(rf_test["mae"]),
            "model_file": str(rf_pkl.resolve()),
        })
        save_single_sheet_xlsx(pd.DataFrame(summary_rows), table_dir / "best_hyperparameters.xlsx", sheet_name="summary")

        # 7. XGBoost
        log_print("[MODEL] 开始拟合 XGBoost ...", logs)
        xgb_model, xgb_info = fit_xgboost(X_train, y_train, g_train, tree_pre)
        xgb_pkl = model_dir / "best_xgb.pkl"
        save_model_pickle(xgb_model, xgb_pkl)
        log_print(f"[SAVE] 已保存 XGB 最佳模型：{xgb_pkl.resolve()}", logs)

        xgb_perf = evaluate_model("XGB", xgb_model, X_train, y_train, X_test, y_test)
        xgb_imp, xgb_shap_values, xgb_x_trans = get_tree_shap_importance(xgb_model, X_train, model_name="xgb")
        save_single_sheet_xlsx(xgb_imp, table_dir / "xgb_variable_importance.xlsx", sheet_name="xgb")
        save_horizontal_bar(xgb_imp, "importance_xgb", "variable", "XGBoost变量重要性（Top20）", figure_dir / "xgb_bar_top20.png", top_n=20)
        save_shap_beeswarm(xgb_shap_values, xgb_x_trans, "XGBoost SHAP蜂群图（Top20显示）", figure_dir / "xgb_beeswarm_top20.png")
        save_log(log_file, logs)

        xgb_test = xgb_perf.query("sample == 'test'").iloc[0]
        summary_rows.append({
            "model": "XGB",
            "best_params_json": json.dumps(xgb_info["best_params"], ensure_ascii=False, default=str),
            "best_cv_rmse": -float(xgb_info["best_score"]),
            "train_r2": float(xgb_perf.query("sample=='train'").iloc[0]["r2"]),
            "train_mse": float(xgb_perf.query("sample=='train'").iloc[0]["mse"]),
            "train_rmse": float(xgb_perf.query("sample=='train'").iloc[0]["rmse"]),
            "train_mae": float(xgb_perf.query("sample=='train'").iloc[0]["mae"]),
            "test_r2": float(xgb_test["r2"]),
            "test_mse": float(xgb_test["mse"]),
            "test_rmse": float(xgb_test["rmse"]),
            "test_mae": float(xgb_test["mae"]),
            "model_file": str(xgb_pkl.resolve()),
        })
        summary_df = pd.DataFrame(summary_rows)
        save_single_sheet_xlsx(summary_df, table_dir / "best_hyperparameters.xlsx", sheet_name="summary")

        # 8. 三算法整合排序
        combined_df = combine_rankings(lasso_imp, rf_imp, xgb_imp)
        save_single_sheet_xlsx(combined_df, table_dir / "combined_variable_ranking.xlsx", sheet_name="combined")

        # 9. 结束提示
        log_print("\n================ 运行完成 ================", logs)
        log_print(f"[SAVE] 最佳超参数汇总：{(table_dir / 'best_hyperparameters.xlsx').resolve()}", logs)
        log_print(f"[SAVE] LASSO结果：{(table_dir / 'lasso_selected_variables.xlsx').resolve()}", logs)
        log_print(f"[SAVE] RF结果：{(table_dir / 'rf_variable_importance.xlsx').resolve()}", logs)
        log_print(f"[SAVE] XGB结果：{(table_dir / 'xgb_variable_importance.xlsx').resolve()}", logs)
        log_print(f"[SAVE] 整合排序：{(table_dir / 'combined_variable_ranking.xlsx').resolve()}", logs)
        log_print(f"[SAVE] 模型目录：{model_dir.resolve()}", logs)
        log_print(f"[SAVE] 图表目录：{figure_dir.resolve()}", logs)
        save_log(log_file, logs)

    except Exception as e:
        log_print(f"[FATAL] 程序运行失败：{e}", logs)
        log_print(traceback.format_exc(), logs)
        save_log(log_file, logs)
        raise


if __name__ == "__main__":
    main()
