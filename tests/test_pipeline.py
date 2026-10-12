"""End-to-end test on SYNTHETIC data: prepare -> train -> evaluate -> Model Registry -> Flask app.

Runs in CI (GitHub Actions) and locally with `pytest -q`. The real SEER file is never needed or touched.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import synthetic_seer  # noqa: E402


def _run(script, cwd, env):
    r = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_pipeline_and_app(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    synthetic_seer.make(4000, seed=1).to_csv(tmp_path / "data" / "raw" / "Survival_Data.csv", index=False)
    P = yaml.safe_load(open(ROOT / "params.yaml"))
    P["train"]["model_params"] = {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8}
    P["train"]["calibration_folds"] = 3
    P["evaluate"]["n_boot"] = 50
    P["evaluate"]["min_auc_for_champion"] = 0.5
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(P, sort_keys=False))
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}", "MLFLOW_DISABLE_AGENT_HINT": "1"}

    _run("prepare.py", tmp_path, env)
    _run("train.py", tmp_path, env)
    out = _run("evaluate.py", tmp_path, env)

    m = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert 0.5 < m["test_roc_auc"] <= 1.0
    run = json.loads((tmp_path / "models" / "mlflow_run.json").read_text())
    assert run["version"] is not None
    assert "PROMOTED" in out

    import mlflow
    mlflow.set_tracking_uri(env["MLFLOW_TRACKING_URI"])
    mv = mlflow.MlflowClient().get_model_version_by_alias(P["mlflow"]["registered_model"], "champion")
    assert str(mv.version) == str(run["version"])

    # the registered model loads back and predicts
    import mlflow.sklearn
    meta = json.loads((tmp_path / "models" / "model_meta.json").read_text())
    from crc.tracking import synthetic_example
    model = mlflow.sklearn.load_model(f"models:/{P['mlflow']['registered_model']}@champion")
    assert 0 <= model.predict_proba(synthetic_example(meta))[0, 1] <= 1

    # the Flask app serves the trained model
    app_dir = tmp_path / "models"
    os.environ["MODEL_DIR"] = str(app_dir)
    spec = importlib.util.spec_from_file_location("crc_app", ROOT / "app" / "app.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    client = mod.app.test_client()
    assert client.get("/health").get_json()["status"] == "ok"
    payload = {f: (meta["numeric"][f]["median"] if f in meta["numeric"] else meta["categorical"][f][0]) for f in meta["feature_cols"]}
    res = client.post("/predict", json=payload).get_json()
    assert 0 <= res["survival_probability"] <= 1
    shutil.rmtree(tmp_path / "data", ignore_errors=True)
