"""Stage 1 - SEER export -> cleaned cohort -> labelled train / test split.

Usage:  python src/prepare.py            (reads params.yaml)
Outputs data/processed/train.pkl, data/processed/test.pkl and reports/cohort_attrition.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc.data import build_cohort, split  # noqa: E402


def main():
    P = yaml.safe_load(open("params.yaml"))
    raw = Path(P["prepare"]["raw_path"])
    if not raw.exists():
        sys.exit(f"{raw} not found. Put your SEER export there (it never leaves your computer) - see MLOPS_GUIDE.md.")
    df_raw = pd.read_csv(raw, low_memory=False)
    C, attrition = build_cohort(df_raw, horizon=P["prepare"]["horizon_months"])
    train, test = split(C, test_size=P["prepare"]["test_size"], seed=P["seed"])

    out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
    train.to_pickle(out / "train.pkl"); test.to_pickle(out / "test.pkl")
    Path("reports").mkdir(exist_ok=True)
    summary = {"attrition": attrition.to_dict(orient="records"), "n_cohort": int(len(C)),
               "n_train_labelled": int(len(train)), "n_test_labelled": int(len(test)),
               "survived_5y_rate_train": float(train.y5.mean()), "survived_5y_rate_test": float(test.y5.mean())}
    Path("reports/cohort_attrition.json").write_text(json.dumps(summary, indent=2))
    print(attrition.to_string(index=False))
    print(f"labelled train {len(train):,} | labelled test {len(test):,}")


if __name__ == "__main__":
    main()
