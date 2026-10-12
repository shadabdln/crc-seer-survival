"""CRC 5-year survival (SEER) - clinical decision-support PROTOTYPE (not a diagnostic device)."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
ART = Path(os.environ.get("MODEL_DIR", BASE / "artifacts"))       # MODEL_DIR lets tests / scripts point to models/
pipe = joblib.load(ART / "crc_model.joblib")
meta = json.loads((ART / "model_meta.json").read_text())

FEATURES, NUMERIC, CATEGORICAL = meta["feature_cols"], meta["numeric"], meta["categorical"]
NAMES, ORIGIN = meta["selected_feature_names"], meta["feature_origin"]
prep = Pipeline(pipe.steps[:-1])
CAL = joblib.load(ART / "crc_calibrator.joblib") if (ART / "crc_calibrator.joblib").exists() else None
THRESHOLD = meta.get("threshold", 0.5)
clf = pipe.steps[-1][1]

DISCLAIMER = ("Research prototype for clinical decision support, trained on SEER registry data. Not a diagnostic device; "
              "it must not replace clinical judgement.")

explainer = None
try:
    import shap
    if clf.__class__.__name__ in ("RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"):
        explainer = shap.TreeExplainer(clf)
except Exception:
    explainer = None

# SVM has neither trees nor coefficients -> model-agnostic KernelExplainer (a few seconds per patient)
kernel_explainer = None
BG = ART / "background.npy"
if explainer is None and not hasattr(clf, "coef_") and BG.exists():
    try:
        import shap
        _f = clf.decision_function if hasattr(clf, "decision_function") else (lambda a: clf.predict_proba(a)[:, 1])
        kernel_explainer = shap.KernelExplainer(_f, np.load(BG))
    except Exception:
        kernel_explainer = None


def explain(xrow):
    """Top-5 patient-level contributions, summed back to the original clinical variable."""
    if explainer is not None:
        sv = explainer.shap_values(xrow)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        contrib = (sv[:, :, 1] if sv.ndim == 3 else sv)[0]
    elif hasattr(clf, "coef_"):
        contrib = clf.coef_[0] * xrow[0]
    elif kernel_explainer is not None:
        sv = kernel_explainer.shap_values(xrow, nsamples=200, silent=True)
        contrib = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(-1)
    else:
        return []
    agg = {}
    for n, v in zip(NAMES, contrib):
        agg[ORIGIN[n]] = agg.get(ORIGIN[n], 0.0) + float(v)
    top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return [{"feature": k, "impact": round(v, 4),
             "direction": "towards survival" if v > 0 else "towards non-survival"} for k, v in top]


def predict_one(payload):
    row = {}
    for f in FEATURES:
        v = payload.get(f)
        if f in NUMERIC:
            row[f] = float(v) if v not in (None, "") else np.nan
        else:
            row[f] = str(v) if v not in (None, "") else "Unknown"
    Xt = prep.transform(pd.DataFrame([row], columns=FEATURES))
    if hasattr(clf, "predict_proba"):
        p_raw = float(clf.predict_proba(Xt)[0, 1])
    else:
        p_raw = float(1.0 / (1.0 + np.exp(-clf.decision_function(Xt)[0])))
    p = float(CAL.predict([p_raw])[0]) if CAL is not None else p_raw          # calibrated probability shown to the user
    return {"model": meta["model_name"], "survival_probability": round(p, 4), "uncalibrated_probability": round(p_raw, 4),
            "prediction": ("Predicted 5-year survival at or above the model threshold" if p_raw >= THRESHOLD
                           else "Predicted 5-year survival below the model threshold (higher risk)"),
            "top_factors": explain(Xt), "disclaimer": DISCLAIMER}


PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CRC 5-year survival (SEER prototype)</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b1b1b}
 .warn{background:#fff4e5;border-left:5px solid #f59e0b;padding:10px 14px;margin:12px 0;font-size:.92rem}
 form{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
 label{font-size:.82rem;font-weight:600;display:block;margin-bottom:3px}
 input,select{width:100%;padding:7px;border:1px solid #bbb;border-radius:6px;box-sizing:border-box}
 button{grid-column:1/-1;padding:12px;background:#1d3557;color:#fff;border:0;border-radius:8px;font-size:1rem;cursor:pointer}
 #out{margin-top:20px;padding:16px;border:1px solid #ddd;border-radius:10px;display:none}
 .bar{height:14px;background:#e5e7eb;border-radius:7px;overflow:hidden}.bar>div{height:100%;background:#2a9d8f}
 li.pos{color:#1b7f6d} li.neg{color:#c2410c}
</style></head><body>
<h2>Colorectal cancer &ndash; 5-year survival prediction (SEER-trained)</h2>
<div class="warn">{{ disclaimer }}</div>
<form id="f">
{% for c, r in numeric.items() %}
 <div><label>{{ c }} (blank = unknown; typical {{ r.min|int }}-{{ r.max|int }})</label><input type="number" step="any" name="{{ c }}" value="{{ r.median|int }}"></div>
{% endfor %}
{% for c, opts in categorical.items() %}
 <div><label>{{ c }}</label><select name="{{ c }}">{% for o in opts %}<option {% if o == 'Unknown' %}selected{% endif %}>{{ o }}</option>{% endfor %}</select></div>
{% endfor %}
 <button type="submit">Predict</button>
</form>
<div id="out"></div>
<script>
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.target).entries());
  const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const d = await r.json(), out = document.getElementById('out'); out.style.display = 'block';
  if (!r.ok) { out.innerHTML = '<b>Error:</b> ' + d.error; return; }
  const pct = (d.survival_probability * 100).toFixed(1);
  out.innerHTML = '<h3>' + d.prediction + '</h3><div class="bar"><div style="width:' + pct + '%"></div></div>' +
    '<p>Estimated (calibrated) 5-year overall-survival probability: <b>' + pct + '%</b> (model: ' + d.model + ')</p>' +
    '<b>Top contributing factors</b><ul>' + d.top_factors.map(t => '<li class="' + (t.impact > 0 ? 'pos' : 'neg') + '">' +
    t.feature + ' &mdash; ' + t.direction + ' (' + t.impact + ')</li>').join('') + '</ul><small>' + d.disclaimer + '</small>';
});
</script></body></html>
"""

app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string(PAGE, numeric=NUMERIC, categorical=CATEGORICAL, disclaimer=DISCLAIMER)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(predict_one(payload))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": meta["model_name"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)