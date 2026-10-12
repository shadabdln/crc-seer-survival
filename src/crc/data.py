"""Cohort building and feature engineering - a script version of notebook Steps 3-5 and 7.

Every rule here mirrors the notebook so that `src/train.py` reproduces the notebook's model.
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AGE_COL = "Age recode with <1 year olds and 90+"
STAGE_COL = "Combined Summary Stage with Expanded Regional Codes (2004+)"
BLANKS = ("Blank(s)",)

GRADE_OLD = {"Well differentiated; Grade I": "I", "Moderately differentiated; Grade II": "II",
             "Poorly differentiated; Grade III": "III", "Undifferentiated; anaplastic; Grade IV": "IV"}
GRADE_NEW = {"Site-specific grade system category (1)": "I", "Site-specific grade system category (2)": "II",
             "Site-specific grade system category (3)": "III", "Site-specific grade system category (4)": "IV",
             "Well differentiated": "I", "Moderately differentiated": "II", "Poorly differentiated": "III",
             "Undifferentiated and anaplastic": "IV"}
STAGE_MAP = {"Localized only": "Localized", "Regional by direct extension only": "Regional (extension)",
             "Regional lymph nodes involved only": "Regional (nodes)",
             "Regional by both direct extension and lymph node involvement": "Regional (both)",
             "Distant site(s)/node(s) involved": "Distant", "Unknown/unstaged/unspecified/DCO": "Unknown"}

NUM_A = ["Age_years", "Income_USD", "RUCC", "Tumor_Size_mm"]
CAT_A = ["Sex", "Race", "Marital_Status", "Site", "Histology", "Grade", "Summary_Stage", "Stage_Group", "T_Stage", "N_Stage", "M_Stage", "CEA"]
NUM_B = ["Nodes_Examined", "Nodes_Positive", "Node_Ratio"]
CAT_B = ["Perineural_Invasion", "Tumor_Deposits"]
CAT_C = ["Surgery", "Radiation", "Chemotherapy"]
CORE_NUM = ["Age_years"]
CORE_CAT = ["Sex", "Race", "Site", "Histology", "Grade", "Summary_Stage", "Stage_Group", "T_Stage", "N_Stage", "M_Stage", "CEA"]
ID_COLS = ["Patient_ID", "Year", "Time", "Event", "y5"]


def feats_for(fs):
    """(numeric, categorical) columns for feature set A (diagnosis), B (+ pathology) or C (+ treatment)."""
    num = NUM_A + (NUM_B if fs in ("B", "C") else [])
    cat = CAT_A + (CAT_B if fs in ("B", "C") else []) + (CAT_C if fs == "C" else [])
    return num, cat


def age_to_years(s):
    nums = re.findall(r"\d+", str(s))
    if str(s).startswith("90+") or "+" in str(s): return float(nums[0]) + 2
    if len(nums) >= 2: return (float(nums[0]) + float(nums[1])) / 2
    return float(nums[0]) if nums else np.nan


def first_valid(frame, cols):
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    for c in cols:
        s = frame[c].astype(object)
        ok = out.isna() & s.notna() & ~s.isin(BLANKS)
        out[ok] = s[ok]
    return out


def norm_stage(v):
    s = str(v).strip().upper()
    if s in ("", "NAN", "BLANK(S)", "UNK STAGE", "99", "88"): return "Unknown"
    if s == "0": return "0"
    s = s.replace("NOS", "")
    if s[:1] in "1234" and s[:1] != "": return {"1": "I", "2": "II", "3": "III", "4": "IV"}[s[0]]
    for r in ("IV", "III", "II", "I"):
        if s.startswith(r): return r
    return "Unknown"


def norm_tnm(v, letter):
    s = str(v).strip().upper()
    if s in ("", "NAN", "BLANK(S)", "NOT APPLICABLE", "88"): return "Unknown"
    s = re.sub(rf"^[CP{letter}]", "", s)
    if letter == "T":
        if s.startswith("IS"): return "Tis"
        return {"0": "T0", "1": "T1", "2": "T2", "3": "T3", "4": "T4"}.get(s[:1], "Unknown")
    if letter == "N": return {"0": "N0", "1": "N1", "2": "N2"}.get(s[:1], "Unknown")
    return {"0": "M0", "1": "M1"}.get(s[:1], "Unknown")


def income_to_num(v):
    s = str(v).replace(",", "")
    if s.startswith("Unknown"): return np.nan
    m = re.findall(r"\d+", s)
    if s.startswith("<") and m: return float(m[0]) - 2500
    if s.endswith("+") and m: return float(m[0]) + 5000
    if len(m) >= 2: return (float(m[0]) + float(m[1])) / 2
    return np.nan


def rucc_ord(v):
    s = str(v)
    if s.startswith("Counties in metropolitan areas ge 1 million"): return 0
    if "250,000 to 1 million" in s: return 1
    if "lt 250 thousand" in s: return 2
    if s.startswith("Nonmetropolitan counties adjacent"): return 3
    if s.startswith("Nonmetropolitan counties not adjacent"): return 4
    return np.nan


def tumour_mm(v):
    s = str(v)
    if s.isdigit():
        x = int(s); return float(x) if 1 <= x <= 400 else np.nan
    if s.startswith("990"): return 0.5
    if s.startswith("000"): return 0.0
    return np.nan


def build_cohort(df_raw, horizon=60):
    """SEER*Stat export -> (engineered cohort C, attrition table). Same rules as notebook Steps 3-5."""
    all_blank = [c for c in df_raw.columns if df_raw[c].astype(str).str.startswith("Blank").all()]
    d = df_raw.drop(columns=all_blank).copy()
    attr = [("Exported records", len(d))]
    d = d[d["Sequence number"].isin(["One primary only", "1st of 2 or more primaries"])]; attr.append(("First / only primary cancer", len(d)))
    d = d[~d["Type of Reporting Source"].isin(["Death certificate only", "Autopsy only"])]; attr.append(("Excl. death-certificate / autopsy-only", len(d)))
    d["Time"] = pd.to_numeric(d["Survival months"], errors="coerce")
    d = d[d["Time"].notna()]; attr.append(("Known survival time", len(d)))
    d["Age_years"] = d[AGE_COL].map(age_to_years)
    d = d[d["Age_years"] >= 20]; attr.append(("Adults (>= 20 years)", len(d)))
    d = d[d[STAGE_COL] != "In situ"]; attr.append(("Excl. in-situ stage", len(d)))
    d["Event"] = d["Vital status recode (study cutoff used)"].eq("Dead").astype(int)
    d["y5"] = np.where(d["Time"] >= horizon, 1.0, np.where(d["Event"] == 1, 0.0, np.nan))

    C = pd.DataFrame(index=d.index)
    C["Patient_ID"], C["Year"], C["Time"], C["Event"], C["y5"] = d["Patient ID"], d["Year of diagnosis"], d["Time"], d["Event"], d["y5"]
    C["Time"] = C["Time"].clip(lower=0.5)
    # ---- A: diagnosis & staging ----
    C["Age_years"] = d["Age_years"]
    C["Sex"] = d["Sex"]
    C["Race"] = d["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"].str.replace("Non-Hispanic", "NH")
    C["Marital_Status"] = d["Marital status at diagnosis"]
    C["Income_USD"] = d["Median household income inflation adj to 2024"].map(income_to_num)
    C["RUCC"] = d["Rural-Urban Continuum Code"].map(rucc_ord)
    C["Site"] = d["Site recode ICD-O-3/WHO 2008"]
    h = d["Histology recode - broad groupings"].str.replace(r"^\d{4}-\d{4}:\s*", "", regex=True)
    keep = h.value_counts(normalize=True); keep = keep[keep >= 0.01].index
    C["Histology"] = h.where(h.isin(keep), "Other")
    C["Grade"] = d["Grade Recode (thru 2017)"].map(GRADE_OLD).fillna(d["Derived Summary Grade 2018 (2018+)"].map(GRADE_NEW)).fillna("Unknown")
    C["Summary_Stage"] = d[STAGE_COL].map(STAGE_MAP).fillna("Unknown")
    C["Stage_Simple"] = C["Summary_Stage"].map(lambda s: "Regional" if s.startswith("Regional") else s)
    C["Stage_Group"] = first_valid(d, ["Derived AJCC Stage Group, 7th ed (2010-2015)", "7th Edition Stage Group Recode (2016-2017)",
                                       "Derived EOD 2018 Stage Group Recode (2018+)"]).map(norm_stage)
    C["T_Stage"] = first_valid(d, ["Derived AJCC T, 7th ed (2010-2015)", "Derived SEER Combined T (2016-2017)", "Derived EOD 2018 T Recode (2018+)"]).map(lambda v: norm_tnm(v, "T"))
    C["N_Stage"] = first_valid(d, ["Derived AJCC N, 7th ed (2010-2015)", "Derived SEER Combined N (2016-2017)", "Derived EOD 2018 N Recode (2018+)"]).map(lambda v: norm_tnm(v, "N"))
    C["M_Stage"] = first_valid(d, ["Derived AJCC M, 7th ed (2010-2015)", "Derived SEER Combined M (2016-2017)", "Derived EOD 2018 M Recode (2018+)"]).map(lambda v: norm_tnm(v, "M"))
    C["Tumor_Size_mm"] = d["Tumor Size Over Time Recode (1988+)"].map(tumour_mm)
    cea = d["CEA Pretreatment Interpretation Recode (2010+)"].astype(str).str.lower()
    C["CEA"] = np.select([cea.str.contains("positive"), cea.str.contains("negative"), cea.str.contains("borderline")], ["Positive", "Negative", "Borderline"], "Unknown")
    # ---- B: surgical pathology ----
    ne = pd.to_numeric(d["Regional nodes examined (1988+)"], errors="coerce"); ne = ne.where(ne.between(0, 90))
    npos = pd.to_numeric(d["Regional nodes positive (1988+)"], errors="coerce"); npos = npos.where(npos.between(0, 90))
    C["Nodes_Examined"], C["Nodes_Positive"] = ne, npos
    C["Node_Ratio"] = (npos / ne).where(ne > 0)
    pni = d["Perineural Invasion Recode (2010+)"].astype(str).str.lower()
    C["Perineural_Invasion"] = np.select([pni.str.contains("identified/present"), pni.str.contains("not identified")], ["Present", "Absent"], "Unknown")
    td = d["Tumor Deposits Recode (2010+)"].astype(str)
    C["Tumor_Deposits"] = np.select([td.eq("No tumor deposits"), td.str.isdigit() | td.str.startswith("Tumor Deposits identified") | td.str.startswith("81 or more")],
                                    ["None", "Present"], "Unknown")
    # ---- C: treatment ----
    sc = d["RX Summ--Surg Prim Site (1998-2022)"]
    C["Surgery"] = np.select([sc.eq(0), sc.eq(99)], ["No", "Unknown"], "Yes")
    C["Radiation"] = np.where(d["Radiation recode"].astype(str).str.contains("beam|isotope|implant|NOS", case=False, regex=True), "Yes", "No/Unknown")
    C["Chemotherapy"] = d["Chemotherapy recode (yes, no/unk)"]

    _, cat_all = feats_for("C")
    for c in cat_all + ["Stage_Simple"]:
        C[c] = C[c].astype(str).replace({"nan": "Unknown"})
    attrition = pd.DataFrame(attr, columns=["Step", "N"])
    attrition["Removed"] = (-attrition["N"].diff()).fillna(0).astype(int)
    return C, attrition


def split(C, test_size=0.2, seed=42):
    """Notebook Step 7: one stratified split on y5 (censored kept as their own stratum) -> labelled train / test."""
    assert C["Patient_ID"].is_unique, "Patient IDs must be unique after the first-primary filter"
    idx_tr, idx_te = train_test_split(C.index, test_size=test_size, stratify=C["y5"].fillna(-1), random_state=seed)
    tr, te = C.loc[idx_tr], C.loc[idx_te]
    return tr[tr.y5.notna()], te[te.y5.notna()]
