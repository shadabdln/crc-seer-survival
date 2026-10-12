"""Stage 2 - train the final model (+ calibrator, + core model), save artifacts, log to MLflow, register a new model version.

Usage:  python src/train.py            (reads params.yaml)
Outputs models/ (crc_model.joblib, crc_calibrator.joblib, model_meta.json, core model) and models/mlflow_run.json.
"""
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
import yaml
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc import tracking  # noqa: E402
from crc.data import CORE_CAT, CORE_NUM, feats_for  # noqa: E402
from crc.model import build_pipeline, make_classifier, origin_of, selected_names  # noqa: E402

TIMEPOINT = {"A": "diagnosis / clinical staging", "B": "post-surgical pathological assessment", "C": "after treatment decisions"}


def oof_calibrator(pipe, X, y, folds, seed):
    oof = cross_val_predict(clone(pipe), X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)


def main():
    P = yaml.safe_load(open("params.yaml")); T = P["train"]; seed = P["seed"]
    M = P["mlflow"]; tracking.setup(M["experiment"])   # connect (and fail on bad credentials) before the long training
    train = pd.read_pickle("data/processed/train.pkl")
    num, cat = feats_for(T["feature_set"]); feats = num + cat
    X, y = train[feats], train["y5"].astype(int)
    neg_pos = float((y == 0).sum() / max((y == 1).sum(), 1))

    clf = make_classifier(T["model"], T["model_params"], seed, neg_pos)
    pipe = build_pipeline(clf, num, cat, T["use_feature_selection"], T["selector_C"])
    t0 = time.time(); pipe.fit(X, y); fit_s = time.time() - t0
    print(f"Trained {T['model']} on {len(X):,} rows in {fit_s:.1f}s")

    out = Path("models"); out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / "crc_model.joblib")
    if T["calibrate"]:
        joblib.dump(oof_calibrator(pipe, X, y, T["calibration_folds"], seed), out / "crc_calibrator.joblib")

    sel = selected_names(pipe)
    meta = {
        "project": "Explainable AI-driven 5-year survival prediction in colorectal cancer (SEER)",
        "model_name": T["model"], "feature_set": T["feature_set"], "prediction_timepoint": TIMEPOINT[T["feature_set"]],
        "target": "y5 = 1 if SEER survival months >= 60 (alive or dead at >= 60 mo); 0 if died (all causes) < 60 mo; alive < 60 mo = censored, excluded",
        "positive_class": "survived >= 5 years", "threshold": T["threshold"],
        "threshold_applies_to": "uncalibrated predict_proba()[:, 1] of crc_model.joblib",
        "calibrator": "crc_calibrator.joblib (isotonic, out-of-fold on training set)" if T["calibrate"] else None,
        "feature_cols": feats,
        "numeric": {c: {"min": float(train[c].min()), "max": float(train[c].max()), "median": float(train[c].median())} for c in num},
        "categorical": {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in cat},
        "selected_feature_names": sel, "feature_origin": {n: origin_of(n, feats, cat) for n in sel},
        "model_params": T["model_params"], "n_train": int(len(X)),
        "trained_on": datetime.now().isoformat(timespec="seconds"), "trained_by": "src/train.py",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    if T["train_core_model"]:
        core = build_pipeline(make_classifier(T["model"], T["model_params"], seed, neg_pos), CORE_NUM, CORE_CAT, T["use_feature_selection"], T["selector_C"])
        core.fit(train[CORE_NUM + CORE_CAT], y)
        joblib.dump(core, out / "crc_model_core_transportable.joblib")
        if T["calibrate"]:
            joblib.dump(oof_calibrator(core, train[CORE_NUM + CORE_CAT], y, T["calibration_folds"], seed), out / "crc_calibrator_core.joblib")
        meta["core_model"] = {"file": "crc_model_core_transportable.joblib", "calibrator": "crc_calibrator_core.joblib", "feature_cols": CORE_NUM + CORE_CAT}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- MLflow: params, artifacts, registered model version ----
    with mlflow.start_run(run_name=f"train_{T['model'].replace(' ', '_')}_set{T['feature_set']}") as run:
        mlflow.log_params({"model": T["model"], "feature_set": T["feature_set"], "seed": seed, "threshold": T["threshold"],
                           "use_feature_selection": T["use_feature_selection"], "selector_C": T["selector_C"],
                           "n_train": len(X), "n_selected_features": len(sel),
                           **{f"clf_{k}": v for k, v in T["model_params"].items()}})
        mlflow.log_metric("fit_time_s", fit_s)
        mlflow.set_tags({"stage": "train", "data": "SEER 17 (Nov 2025 sub), local only", "prediction_timepoint": TIMEPOINT[T["feature_set"]]})
        mlflow.log_artifact(str(out / "model_meta.json"))
        if T["calibrate"]:
            mlflow.log_artifact(str(out / "crc_calibrator.joblib"))
        version = tracking.log_and_register(pipe, meta, M["registered_model"])
        if version:
            client = mlflow.MlflowClient()
            try:
                client.set_registered_model_alias(M["registered_model"], "candidate", version)
                client.set_model_version_tag(M["registered_model"], version, "feature_set", T["feature_set"])
                print(f"Registered '{M['registered_model']}' version {version} with alias @candidate")
            except Exception as e:  # some hosted MLflow servers do not support aliases yet
                print(f"Registered version {version}; could not set alias/tag on this server ({type(e).__name__}: {e})")
        (out / "mlflow_run.json").write_text(json.dumps({"run_id": run.info.run_id, "registered_model": M["registered_model"], "version": version}, indent=2))
    print("Saved:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
