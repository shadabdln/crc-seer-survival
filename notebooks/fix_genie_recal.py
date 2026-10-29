"""Fixes the recalibration cell of CRC_GENIE_External_Validation_v2.ipynb (28 Oct 2026).

lifelines' CoxPHFitter uses the delayed entry (entry_col) when it estimates the slope, but not when it estimates the
baseline hazard, so its predicted survival is too high in left-truncated data. This script replaces that step with a
Breslow estimate whose risk sets respect the entry times. Only the v2 copy is changed; the original notebook is not read.

Run it in the notebooks folder, then Restart & Run All on CRC_GENIE_External_Validation_v2.ipynb:
    python fix_genie_recal.py
"""
import json
from pathlib import Path

P = Path("CRC_GENIE_External_Validation_v2.ipynb")
OLD_FIT = ('def fit_recal(df_): return CoxPHFitter().fit(df_[COLS], duration_col="time", event_col="event", entry_col="entry_mos")\n'
           'def s60(m, df_): return m.predict_survival_function(df_[["lp"]], times=[HORIZON]).values.ravel()\n'
           'full = fit_recal(r)\n')
NEW_FIT = ('def fit_recal(df_):\n'
           '    m = CoxPHFitter().fit(df_[COLS], duration_col="time", event_col="event", entry_col="entry_mos"); b = float(m.params_["lp"])\n'
           '    # lifelines ignores delayed entry in its baseline hazard, so use Breslow with left-truncated risk sets (entry < t <= time)\n'
           '    ev = df_[(df_.event == 1) & (df_.time <= HORIZON)].groupby("time").size()\n'
           '    w = np.exp(b * df_.lp.values); en, ti = df_.entry_mos.values, df_.time.values\n'
           '    H0 = float(sum(k / w[(en < u) & (ti >= u)].sum() for u, k in ev.items()))\n'
           '    return m, b, H0\n'
           'def s60(fit_, df_): m, b, H0 = fit_; return np.exp(-H0 * np.exp(b * df_.lp.values))\n'
           'full = fit_recal(r)[0]\n')

if not P.exists():
    raise SystemExit(f"{P} not found - run this in the folder that holds the v2 notebooks.")
nb = json.loads(P.read_text(encoding="utf-8"))
src = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
done = [c for c in nb["cells"] if c["cell_type"] == "code" and "left-truncated risk sets" in src(c)]
if done:
    raise SystemExit("Already fixed - just Restart & Run All the v2 notebook.")
hits = [c for c in nb["cells"] if c["cell_type"] == "code" and OLD_FIT in src(c)]
if len(hits) != 1:
    raise SystemExit(f"Expected one recalibration cell, found {len(hits)}. Nothing was written.")
hits[0]["source"] = src(hits[0]).replace(OLD_FIT, NEW_FIT).splitlines(keepends=True)
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"fixed {P} - now Restart & Run All on it")
