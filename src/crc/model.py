"""Pipeline factory and evaluation metrics - same definitions as notebook Steps 7, 9 and 11b."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], verbose_feature_names_out=False)


def make_selector(use=True, C=0.05):
    if not use:
        return "passthrough"
    return SelectFromModel(LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=300), threshold=1e-6)


def make_classifier(name, params, seed, neg_pos_ratio=1.0):
    """The notebook's model definitions; `params` (from params.yaml) override the defaults."""
    p = dict(params or {})
    if name == "Gradient Boosting":
        base = dict(n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.8)
        return GradientBoostingClassifier(**{**base, **p}, random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        base = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg_pos_ratio)
        return XGBClassifier(**{**base, **p}, eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced")
        return LGBMClassifier(**{**base, **p}, verbose=-1, n_jobs=-1, random_state=seed)
    if name == "Random Forest":
        base = dict(n_estimators=300, max_depth=14, min_samples_leaf=30, class_weight="balanced_subsample")
        return RandomForestClassifier(**{**base, **p}, n_jobs=-1, random_state=seed)
    if name == "Logistic Regression":
        return LogisticRegression(**{"max_iter": 2000, "class_weight": "balanced", **p}, random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


def build_pipeline(clf, num_cols, cat_cols, use_selection=True, selector_C=0.05):
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("select", make_selector(use_selection, selector_C)), ("clf", clf)])


def selected_names(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    sel = pipe.named_steps["select"]
    return list(names) if isinstance(sel, str) else list(names[sel.get_support()])


def origin_of(n, all_cols, cat_cols):
    """Map an encoded column name (e.g. 'Stage_Group_IV') back to its clinical variable."""
    n = n.replace("missingindicator_", "")
    if n in all_cols:
        return n
    for c in sorted(cat_cols, key=len, reverse=True):
        if n.startswith(c + "_"):
            return c
    return n


def calib_int_slope(y, p):
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    a = 0.0
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(a + lp))); step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12); a += step
        if abs(step) < 1e-10: break
    X1 = np.column_stack([np.ones_like(lp), lp]); beta = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-X1 @ beta)); W = mu * (1 - mu)
        step = np.linalg.solve(X1.T @ (X1 * W[:, None]) + 1e-9 * np.eye(2), X1.T @ (y - mu)); beta += step
        if np.max(np.abs(step)) < 1e-10: break
    return a, beta[1]


def metric_set(y, p, thr=0.5):
    y = np.asarray(y).astype(int); pred = (np.asarray(p) >= thr).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0)); fn = np.sum((pred == 0) & (y == 1))
    out = {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p), "Accuracy": (tp + tn) / len(y),
           "Sensitivity": tp / max(tp + fn, 1), "Specificity": tn / max(tn + fp, 1), "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn, 1),
           "F1": 2 * tp / max(2 * tp + fp + fn, 1), "Brier": brier_score_loss(y, p)}
    out["Calib_intercept"], out["Calib_slope"] = calib_int_slope(y, p)
    return {k: float(v) for k, v in out.items()}


def bootstrap_ci(y, p, n_boot=1000, seed=42, thr=0.5):
    """Point estimate + percentile 95% CI for every metric."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); rng = np.random.default_rng(seed); n = len(y)
    idx = [b for b in (rng.integers(0, n, n) for _ in range(n_boot)) if 0 < y[b].sum() < n]
    est = metric_set(y, p, thr)
    bs = pd.DataFrame([metric_set(y[b], p[b], thr) for b in idx])
    return {k: {"estimate": v, "ci_low": float(bs[k].quantile(.025)), "ci_high": float(bs[k].quantile(.975))} for k, v in est.items()}
