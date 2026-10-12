"""Copy the notebook's final model choice and tuned settings into params.yaml.

Usage:  python src/sync_params.py notebooks/artifacts/model_meta.json
Run it once after the notebook, so `src/train.py` trains exactly the model the notebook selected.
"""
import json
import sys
from pathlib import Path

import yaml


def _num(v):
    """model_meta.json may store numpy numbers as strings ("250") - turn them back into int / float."""
    if isinstance(v, str):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
    return v


def main():
    meta = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/artifacts/model_meta.json").read_text())
    P = yaml.safe_load(open("params.yaml"))
    best = meta.get("best_params") or {}
    clf = {k.replace("clf__", ""): _num(v) for k, v in best.items() if k.startswith("clf__")}
    P["train"]["model"] = meta["model_name"]
    P["train"]["feature_set"] = meta.get("feature_set", P["train"]["feature_set"])
    if clf:
        P["train"]["model_params"] = clf
    if "select__estimator__C" in best:
        P["train"]["selector_C"] = float(_num(best["select__estimator__C"]))
    if "threshold" in meta:
        P["train"]["threshold"] = meta["threshold"]
    Path("params.yaml").write_text(yaml.safe_dump(P, sort_keys=False))
    print("params.yaml updated:", json.dumps(P["train"], indent=2))


if __name__ == "__main__":
    main()
