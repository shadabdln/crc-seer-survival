"""MLflow connection. Local by default; online (e.g. DagsHub) when MLFLOW_TRACKING_URI is set."""
import os

import mlflow
import numpy as np
import pandas as pd


def setup(experiment):
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    print(f"MLflow tracking -> {uri} | experiment '{experiment}'")
    return uri


def synthetic_example(meta):
    """A MADE-UP patient (numeric medians + first category of each variable) used as the MLflow input example.

    Never log real SEER rows to MLflow: on an online server that would publish patient-level data.
    """
    row = {}
    for f in meta["feature_cols"]:
        if f in meta["numeric"]:
            row[f] = float(meta["numeric"][f]["median"])
        else:
            row[f] = str(meta["categorical"][f][0])
    return pd.DataFrame([row], columns=meta["feature_cols"])


def serialization_kwargs(pipe):
    """MLflow saves sklearn models with skops, which only loads types we list as trusted.

    We trained this pipeline ourselves, so its own types are trusted. Falls back to cloudpickle if skops cannot handle it.
    """
    try:
        from skops.io import dumps, get_untrusted_types
        return {"serialization_format": "skops", "skops_trusted_types": get_untrusted_types(data=dumps(pipe))}
    except Exception:
        return {"serialization_format": "cloudpickle"}


def log_and_register(pipe, meta, registered_name, artifact_name="model"):
    """Log the sklearn pipeline as an MLflow model and register a new version. Returns the version (str) or None."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    example = synthetic_example(meta)
    signature = infer_signature(example, np.asarray(pipe.predict_proba(example)[:, 1]))
    ser = serialization_kwargs(pipe)
    try:
        info = mlflow.sklearn.log_model(sk_model=pipe, name=artifact_name, signature=signature, input_example=example,
                                        registered_model_name=registered_name, **ser)
        version = getattr(info, "registered_model_version", None)
        return str(version) if version is not None else None
    except Exception as e:  # older tracking servers do not know the MLflow-3 "logged model" API
        print(f"log_model failed ({type(e).__name__}: {e}); falling back to artifact upload + register_model")
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / artifact_name
            mlflow.sklearn.save_model(pipe, str(path), signature=signature, input_example=example, **ser)
            mlflow.log_artifacts(str(path), artifact_path=artifact_name)
        run_id = mlflow.active_run().info.run_id
        mv = mlflow.register_model(f"runs:/{run_id}/{artifact_name}", registered_name)
        return str(mv.version)
