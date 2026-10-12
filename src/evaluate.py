"""Stage 3 - evaluate the trained model on the held-out test set, log metrics to MLflow, and promote it if it earns it.

Usage:
  python src/evaluate.py                                   # model from models/ (the DVC pipeline)
  python src/evaluate.py --model-uri models:/crc-5y-survival@champion   # any registered version
Outputs reports/metrics.json (DVC metrics) and reports/metrics_95ci.json.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc import tracking  # noqa: E402
from crc.model import bootstrap_ci, metric_set  # noqa: E402


def champion_auc(client, name):
    try:
        mv = client.get_model_version_by_alias(name, "champion")
        return mv.version, float(mv.tags.get("test_roc_auc", "nan"))
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-uri", default=None, help="e.g. models:/crc-5y-survival@champion (default: models/crc_model.joblib)")
    args = ap.parse_args()

    P = yaml.safe_load(open("params.yaml")); E = P["evaluate"]; M = P["mlflow"]
    meta = json.loads(Path("models/model_meta.json").read_text())
    test = pd.read_pickle("data/processed/test.pkl")
    X, y = test[meta["feature_cols"]], test["y5"].astype(int)

    tracking.setup(M["experiment"])
    if args.model_uri:
        model = mlflow.sklearn.load_model(args.model_uri)
    else:
        model = joblib.load("models/crc_model.joblib")
    p = model.predict_proba(X)[:, 1]
    res = bootstrap_ci(y.values, p, n_boot=E["n_boot"], seed=P["seed"], thr=meta["threshold"])
    flat = {f"test_{k.lower()}": round(v["estimate"], 4) for k, v in res.items()}
    cal_path = Path("models/crc_calibrator.joblib")
    if cal_path.exists() and not args.model_uri:
        flat["test_brier_calibrated"] = round(metric_set(y.values, joblib.load(cal_path).predict(p))["Brier"], 4)
    flat["n_test"] = int(len(y))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(flat, indent=2))
    Path("reports/metrics_95ci.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:16s} {v['estimate']:.4f}  (95% CI {v['ci_low']:.4f}-{v['ci_high']:.4f})")

    # ---- MLflow: attach the test metrics to the training run and the registered version ----
    run_info = json.loads(Path("models/mlflow_run.json").read_text()) if Path("models/mlflow_run.json").exists() else {}
    client = mlflow.MlflowClient()
    resume = run_info.get("run_id") if not args.model_uri else None
    with (mlflow.start_run(run_id=resume) if resume else mlflow.start_run(run_name="evaluate")):
        mlflow.log_metrics({k: v for k, v in flat.items() if k != "n_test"})
        mlflow.log_metrics({"test_roc_auc_ci_low": round(res["ROC_AUC"]["ci_low"], 4), "test_roc_auc_ci_high": round(res["ROC_AUC"]["ci_high"], 4)})
        mlflow.log_artifact("reports/metrics_95ci.json")

    name, version = run_info.get("registered_model"), run_info.get("version")
    if name and version and not args.model_uri:
        try:
            auc = flat["test_roc_auc"]
            client.set_model_version_tag(name, version, "test_roc_auc", str(auc))
            if E["promote"]:
                champ_v, champ_auc = champion_auc(client, name)
                beats_champion = champ_auc is None or pd.isna(champ_auc) or auc >= champ_auc - E["max_auc_drop_vs_champion"]
                ok = auc >= E["min_auc_for_champion"] and beats_champion
                if ok:
                    client.set_registered_model_alias(name, "champion", version)
                    print(f"PROMOTED: {name} v{version} is now @champion (test AUC {auc:.4f}; previous champion: {champ_v or 'none'})")
                else:
                    print(f"NOT promoted: v{version} test AUC {auc:.4f} vs rule min {E['min_auc_for_champion']} / champion v{champ_v} AUC {champ_auc}")
        except Exception as e:  # some hosted MLflow servers do not support aliases yet
            print(f"Registry update skipped on this server ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
