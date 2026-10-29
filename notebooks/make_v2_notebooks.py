"""Step 4 of strengthening_roadmap.md: makes v2 COPIES of the two notebooks. The originals are only read, never written.

Run it in the folder that holds CRC_SEER_Survival_Prediction.ipynb and CRC_GENIE_External_Validation.ipynb:
    python make_v2_notebooks.py

CRC_SEER_Survival_Prediction_v2.ipynb
  * writes to outputs_v2/ and artifacts_v2/ (the original outputs/ and artifacts/ stay as they are)
  * saves every figure at 600 dpi as PNG and also as a vector PDF
  * logs MLflow runs only to a local file (mlflow_v2.db), never to an online server
  * skips the three cells that write and test app.py, requirements.txt and the Dockerfile
  * adds: cancer-specific sensitivity analysis (only if Survival_COD.csv is present), a check that this run
    reproduces outputs/, and claude_upload_seer_v2.zip with aggregate results and figures only

CRC_GENIE_External_Validation_v2.ipynb
  * uses the ORIGINAL frozen model in artifacts/ (as in the paper) and writes to outputs_genie_v2/
  * saves every figure at 600 dpi as PNG and also as a vector PDF
  * adds: a GENIE characteristics table (levels with fewer than 11 patients merged), a 10-fold cross-validated
    recalibration with delayed entry, and claude_upload_genie_v2.zip with aggregate results and figures only
"""
import json, re, uuid
from pathlib import Path

SEER_IN, SEER_OUT = Path("CRC_SEER_Survival_Prediction.ipynb"), Path("CRC_SEER_Survival_Prediction_v2.ipynb")
GENIE_IN, GENIE_OUT = Path("CRC_GENIE_External_Validation.ipynb"), Path("CRC_GENIE_External_Validation_v2.ipynb")


def load(p):
    if not p.exists():
        raise SystemExit(f"{p} not found - run this script in the folder that holds the notebooks.")
    return json.loads(p.read_text(encoding="utf-8"))


def src(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def patch(nb, old, new):
    """Replace `old` with `new` in exactly one code cell (stops if the notebook is not the expected version)."""
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and old in src(c)]
    if len(hits) != 1:
        raise SystemExit(f"Expected one cell containing {old[:60]!r}, found {len(hits)}. Nothing was written.")
    set_src(hits[0], src(hits[0]).replace(old, new))


def drop(nb, start):
    before = len(nb["cells"])
    nb["cells"] = [c for c in nb["cells"] if not (c["cell_type"] == "code" and src(c).lstrip().startswith(start))]
    if len(nb["cells"]) != before - 1:
        raise SystemExit(f"Expected one cell starting with {start!r}. Nothing was written.")


def new_cell(text, kind="code"):
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def clear_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None


# ================================================================ SEER v2
seer = load(SEER_IN)
clear_outputs(seer)
patch(seer, 'OUT = Path("outputs"); FIG = OUT / "figures"; ART = Path("artifacts")',
      'OUT = Path("outputs_v2"); FIG = OUT / "figures"; ART = Path("artifacts_v2")          # v2: new folders')
patch(seer, 'def savefig(name): plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")',
      'def savefig(name):                                                                  # v2: 600 dpi PNG + vector PDF\n'
      '    plt.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight"); plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")')
# the tracking line differs between copies of the notebook, so match any form of it
_hits = [c for c in seer["cells"] if c["cell_type"] == "code" and "mlflow.set_tracking_uri(" in src(c)]
if len(_hits) != 1:
    raise SystemExit(f"Expected one cell containing 'mlflow.set_tracking_uri(', found {len(_hits)}. Nothing was written.")
set_src(_hits[0], re.sub(r'mlflow\.set_tracking_uri\((?:os\.environ\.get\([^)]*\)|[^)]*)\)',
                         'mlflow.set_tracking_uri("sqlite:///mlflow_v2.db")  # v2: local only', src(_hits[0]), count=1))
drop(seer, "%%writefile app.py")
drop(seer, "from importlib.metadata import version")
drop(seer, "# automatic test of the web app")

seer["cells"] += [
    new_cell("## v2 additions (28 Oct 2026): cancer-specific sensitivity analysis, reproducibility check, upload file",
             kind="markdown"),
    new_cell('''
# ---------- v2: cancer-specific five-year survival (sensitivity analysis) ----------
COD_PATH = Path("Survival_COD.csv")   # SEER*Stat case listing: Patient ID, Sequence number, SEER cause-specific death classification
if not COD_PATH.exists():
    print(f"{COD_PATH} not found -> cancer-specific sensitivity analysis skipped.")
else:
    cod = pd.read_csv(COD_PATH, low_memory=False)
    cod = cod[cod["Sequence number"].isin(["One primary only", "1st of 2 or more primaries"])].copy()
    cod["pid"] = pd.to_numeric(cod["Patient ID"], errors="coerce")
    cod = cod.drop_duplicates("pid").set_index("pid")["SEER cause-specific death classification"].astype(str)
    print("Cause-specific death classification in the export:"); print(cod.value_counts().to_string())
    cause = pd.to_numeric(C["Patient_ID"], errors="coerce").map(cod)
    print(f"Matched {int(cause.notna().sum()):,} of {len(C):,} cohort patients")
    crc_death = cause.str.contains("attributable to this cancer", case=False, na=False)
    y5_css = pd.Series(np.where(C.Time >= HORIZON, 1.0, np.where((C.Event == 1) & crc_death, 0.0, np.nan)), index=C.index)
    n_other = int(((C.Time < HORIZON) & (C.Event == 1) & ~crc_death).sum())
    print(f"Cancer-specific label: {int((y5_css == 0).sum()):,} CRC deaths < 60 months, {int((y5_css == 1).sum()):,} "
          f"survivors >= 60 months; {n_other:,} deaths from other or unknown causes < 60 months excluded")
    tr_idx = X_train.index[y5_css.loc[X_train.index].notna().values]
    te_idx = X_test.index[y5_css.loc[X_test.index].notna().values]
    y_te_css = y5_css.loc[te_idx].astype(int)
    m_css = clone(tuned_pipe).fit(X_train.loc[tr_idx], y5_css.loc[tr_idx].astype(int))
    css_ci, _ = bootstrap_table(y_te_css.values, {
        f"{final_name} trained on all-cause death": (final_model.predict_proba(X_test.loc[te_idx])[:, 1], True),
        f"{final_name} refitted on cancer-specific death": (m_css.predict_proba(X_test.loc[te_idx])[:, 1], True)},
        n_boot=min(N_BOOT, 500))
    css_ci.insert(0, "test n", len(te_idx)); css_ci.insert(1, "CRC deaths < 60 months", int((y_te_css == 0).sum()))
    css_ci.insert(2, "other-cause deaths excluded (whole cohort)", n_other)
    css_ci.to_csv(OUT / "css_sensitivity_95ci.csv")
    display(css_ci[["test n", "CRC deaths < 60 months", "ROC-AUC", "Brier", "Calib. slope"]])
'''),
    new_cell('''
# ---------- v2: does this run reproduce the original outputs/ folder? ----------
rows = []
for f in sorted(OUT.glob("*.csv")):
    g = Path("outputs") / f.name
    if not g.exists():
        continue
    a, b = pd.read_csv(g), pd.read_csv(f)
    keep = [c for c in a.columns if "time" not in c.lower()]          # fit times always differ
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        rows.append((f.name, "shape or columns differ", np.nan)); continue
    a, b = a[keep], b[keep]
    na_, nb_ = a.select_dtypes("number"), b.select_dtypes("number")
    diff = float((na_ - nb_).abs().max().max()) if na_.shape[1] else 0.0
    same_text = a.select_dtypes(exclude="number").fillna("").astype(str).equals(b.select_dtypes(exclude="number").fillna("").astype(str))
    rows.append((f.name, "identical" if (diff == 0 and same_text) else ("text differs" if not same_text else "numbers differ"), diff))
rep = pd.DataFrame(rows, columns=["file", "result", "max abs numeric difference"])
rep.to_csv(OUT / "reproducibility_check.csv", index=False); display(rep)
'''),
    new_cell('''
# ---------- v2: aggregate results and figures for Claude (no patient-level rows) ----------
import zipfile
with zipfile.ZipFile("claude_upload_seer_v2.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob("*.csv")):
        z.write(f)
    for f in sorted(FIG.glob("*")):
        if "patient" not in f.name:                    # single-patient explanation plots stay on this PC
            z.write(f)
print("Created claude_upload_seer_v2.zip - upload this file in the project thread.")
'''),
]

# ================================================================ GENIE v2
genie = load(GENIE_IN)
clear_outputs(genie)
patch(genie, 'OUTG = Path("outputs_genie")', 'OUTG = Path("outputs_genie_v2")')
patch(genie, 'def savefig(name): plt.savefig(OUTG / "figures" / f"{name}.png", dpi=200, bbox_inches="tight")',
      'def savefig(name):                                                                  # v2: 600 dpi PNG + vector PDF\n'
      '    plt.savefig(OUTG / "figures" / f"{name}.png", dpi=600, bbox_inches="tight"); plt.savefig(OUTG / "figures" / f"{name}.pdf", bbox_inches="tight")')

genie["cells"] += [
    new_cell("## v2 additions (28 Oct 2026): cohort characteristics, recalibration, upload file", kind="markdown"),
    new_cell('''
# ---------- v2: characteristics of the GENIE cohort (aggregate; levels with fewer than 11 patients are merged) ----------
def counts(series, name):
    s_ = pd.Series(series).fillna("Unknown").astype(str); vc = s_.value_counts()
    big, small = vc[vc >= 11], vc[vc < 11]
    out = [(name, lev, int(n), f"{100 * n / len(s_):.1f}") for lev, n in big.items()]
    if len(small):
        n = int(small.sum())
        out.append((name, "other levels (each < 11)", n if n >= 11 else "<11", f"{100 * n / len(s_):.1f}" if n >= 11 else ""))
    return out

rows = [("Patients", "", len(d), "100.0"),
        ("Age at diagnosis, years", "median (IQR)", f"{d.age_dx.median():.0f} ({d.age_dx.quantile(.25):.0f}-{d.age_dx.quantile(.75):.0f})", "")]
for c in ["Sex", "Race", "Site", "Histology", "Grade", "Summary_Stage", "Stage_Group", "T_Stage", "N_Stage", "M_Stage", "CEA"]:
    rows += counts(X.loc[d.index, c], c)
rows += counts(d.institution, "Institution")
status = np.where(d.y5.eq(1), "survived >= 5 years", np.where(d.y5.eq(0), "died < 5 years", "alive, < 5 years follow-up"))
rows += counts(pd.Series(status, index=d.index), "Five-year status")
rows.append(("Months from diagnosis to first sequencing report", "median (IQR)",
             f"{d.entry_mos.median():.1f} ({d.entry_mos.quantile(.25):.1f}-{d.entry_mos.quantile(.75):.1f})", ""))
g1 = pd.DataFrame(rows, columns=["Variable", "Level", "n", "%"]); g1.to_csv(OUTG / "genie_table1.csv", index=False); display(g1)
'''),
    new_cell('''
# ---------- v2: does local recalibration fix the GENIE calibration? (10-fold cross-validated, delayed entry) ----------
# The frozen prediction is turned into its log cumulative hazard and used as the only covariate of a Cox model with
# delayed entry. A coefficient of 1 means the spread of the predictions is right; the refitted baseline corrects the
# overall level. Out-of-fold predictions avoid judging the recalibration on the patients it was fitted to.
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
r = d[d.entry_mos.notna() & (d.entry_mos < d.time)].copy()
r["lp"] = np.log(-np.log(np.clip(r.p_surv5.astype(float), 1e-6, 1 - 1e-6)))
COLS = ["time", "event", "entry_mos", "lp"]
def fit_recal(df_):
    m = CoxPHFitter().fit(df_[COLS], duration_col="time", event_col="event", entry_col="entry_mos"); b = float(m.params_["lp"])
    # lifelines ignores delayed entry in its baseline hazard, so use Breslow with left-truncated risk sets (entry < t <= time)
    ev = df_[(df_.event == 1) & (df_.time <= HORIZON)].groupby("time").size()
    w = np.exp(b * df_.lp.values); en, ti = df_.entry_mos.values, df_.time.values
    H0 = float(sum(k / w[(en < u) & (ti >= u)].sum() for u, k in ev.items()))
    return m, b, H0
def s60(fit_, df_): m, b, H0 = fit_; return np.exp(-H0 * np.exp(b * df_.lp.values))
full = fit_recal(r)[0]
b_, (b_lo, b_hi) = float(full.params_["lp"]), full.confidence_intervals_.loc["lp"].values
print(f"Cox recalibration slope: {b_:.3f} (95% CI {b_lo:.3f}-{b_hi:.3f}); 1 = predictions neither too extreme nor too flat")
r["p_recal"] = np.nan
for tr, te in KFold(10, shuffle=True, random_state=SEED).split(r):
    r.iloc[te, r.columns.get_loc("p_recal")] = s60(fit_recal(r.iloc[tr]), r.iloc[te])

kmf = KaplanMeierFitter(); km_rows = []
for g, part in r.groupby("risk_tertile", observed=True):              # same tertiles as Figure 4 (frozen predictions)
    kmf.fit(part.time, part.event, entry=part.entry_mos); obs = float(kmf.predict(HORIZON))
    ci_ = kmf.confidence_interval_survival_function_; lo, hi = ci_.iloc[(np.abs(ci_.index - HORIZON)).argmin()]
    km_rows.append((g, len(part), part.p_surv5.mean(), part.p_recal.mean(), obs, lo, hi))
kmr = pd.DataFrame(km_rows, columns=["tertile", "n", "mean predicted S(60), frozen", "mean predicted S(60), recalibrated (CV)",
                                     "KM observed S(60)", "KM 95% low", "KM 95% high"])
kmr.to_csv(OUTG / "genie_recalibration_km.csv", index=False); display(kmr.round(3))

det_r = r[r.y5.notna()]; yy = det_r.y5.astype(int).values; bin_rows = []
for lab_, p_ in [("frozen model", det_r.p_surv5.values), ("recalibrated (10-fold CV)", det_r.p_recal.values)]:
    a_, s_ = calib_int_slope(yy, p_)
    bin_rows.append((lab_, len(yy), roc_auc_score(yy, p_), brier_score_loss(yy, p_), a_, s_))
recb = pd.DataFrame(bin_rows, columns=["predictions", "n determinate", "AUC", "Brier", "Calib. intercept", "Calib. slope"])
recb["Cox recalibration slope (95% CI), all patients"] = f"{b_:.3f} ({b_lo:.3f}-{b_hi:.3f})"
recb.to_csv(OUTG / "genie_recalibration_binary.csv", index=False); display(recb.round(3))

obs_, err_ = kmr["KM observed S(60)"], [kmr["KM observed S(60)"] - kmr["KM 95% low"], kmr["KM 95% high"] - kmr["KM observed S(60)"]]
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.errorbar(kmr["mean predicted S(60), frozen"], obs_, yerr=err_, fmt="o", capsize=3, label="frozen model")
ax.errorbar(kmr["mean predicted S(60), recalibrated (CV)"], obs_, yerr=err_, fmt="s", capsize=3, label="recalibrated (10-fold CV)")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("mean predicted 5-year survival"); ax.set_ylabel("observed 5-year survival (KM, delayed entry)"); ax.legend()
ax.set_title("GENIE: calibration before and after recalibration"); plt.tight_layout(); savefig("genie_recalibration"); plt.show()
'''),
    new_cell('''
# ---------- v2: aggregate results and figures for Claude (no patient-level rows) ----------
import zipfile
KEEP = ["genie_attrition.csv", "genie_binary_metrics_95ci.csv", "genie_km_calibration.csv", "genie_subgroups.csv",
        "seer_genie_feature_mapping.csv", "genie_table1.csv", "genie_recalibration_km.csv", "genie_recalibration_binary.csv",
        "GENIE_validation_report.md"]
with zipfile.ZipFile("claude_upload_genie_v2.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in KEEP:
        if (OUTG / n).exists():
            z.write(OUTG / n)
    for f in sorted((OUTG / "figures").glob("*")):
        if "patient" not in f.name:
            z.write(f)
print("Created claude_upload_genie_v2.zip - upload this file in the project thread.")
'''),
]

# ================================================================ write the copies
for p_in, p_out, nb in [(SEER_IN, SEER_OUT, seer), (GENIE_IN, GENIE_OUT, genie)]:
    assert p_out != p_in
    p_out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", p_out, "| original", p_in, "unchanged")
