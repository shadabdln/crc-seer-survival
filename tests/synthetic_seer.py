"""SYNTHETIC SEER-shaped data for tests and CI. NOT real patients; the numbers mean nothing.

Same column names as the SEER*Stat export, so the whole pipeline can be tested without the real (restricted) data.
"""
import sys

import numpy as np
import pandas as pd


def make(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ages = ["20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years", "50-54 years", "55-59 years", "60-64 years",
            "65-69 years", "70-74 years", "75-79 years", "80-84 years", "85-89 years", "90+ years"]
    age_i = rng.choice(len(ages), n, p=np.r_[[0.01] * 5, [0.05, 0.08, 0.09, 0.11, 0.12, 0.12, 0.11, 0.09, 0.06, 0.02]] / np.r_[[0.01] * 5, [0.05, 0.08, 0.09, 0.11, 0.12, 0.12, 0.11, 0.09, 0.06, 0.02]].sum())
    year = rng.integers(2010, 2019, n)
    stage = rng.choice(["Localized only", "Regional by direct extension only", "Regional lymph nodes involved only",
                        "Regional by both direct extension and lymph node involvement", "Distant site(s)/node(s) involved", "Unknown/unstaged/unspecified/DCO", "In situ"],
                       n, p=[.36, .08, .07, .25, .2, .035, .005])
    dist = stage.startswith("Distant") if False else np.char.startswith(stage.astype(str), "Distant")
    reg = np.char.startswith(stage.astype(str), "Regional")
    risk = -3.0 + 0.06 * (age_i * 5 + 22) + 2.4 * dist + 0.8 * reg + rng.normal(0, 0.8, n)
    months = np.where(rng.random(n) < 1 / (1 + np.exp(-risk)), rng.integers(0, 60, n), rng.integers(60, 167, n))
    followup_max = (2023 - year) * 12 + rng.integers(0, 12, n)
    dead = (months < followup_max) & (months < 60) | (rng.random(n) < 0.15)
    months = np.minimum(months, followup_max)
    cens_short = rng.random(n) < 0.025
    dead = np.where(cens_short, False, dead); months = np.where(cens_short, rng.integers(5, 59, n), months)

    def blank_unless(mask, vals):
        return np.where(mask, vals, "Blank(s)")

    old = year <= 2015; mid = (year >= 2016) & (year <= 2017); new = year >= 2018
    sg = np.where(dist, "IV", np.where(reg, rng.choice(["IIIA", "IIIB", "IIIC", "IIA"], n), rng.choice(["I", "IIA"], n)))
    T = rng.choice(["T1", "T2", "T3", "T4a"], n); N = np.where(reg, rng.choice(["N1a", "N1b", "N2a"], n), "N0"); M = np.where(dist, "M1a", "M0")
    surg = np.where(dist & (rng.random(n) < .5), 0, rng.choice([30, 40, 50], n))
    ne = np.where(surg == 0, 0, rng.integers(5, 30, n)); npos = np.where(reg, np.minimum(ne, rng.integers(1, 6, n)), 0)
    df = pd.DataFrame({
        "Age recode with <1 year olds and 90+": np.array(ages)[age_i], "Sex": rng.choice(["Male", "Female"], n), "Year of diagnosis": year,
        "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": rng.choice(["Non-Hispanic White", "Non-Hispanic Black", "Hispanic (All Races)", "Non-Hispanic Asian or Pacific Islander"], n),
        "Site recode ICD-O-3/WHO 2008": rng.choice(["Rectum", "Sigmoid Colon", "Cecum", "Ascending Colon", "Transverse Colon"], n),
        "Behavior code ICD-O-3": "Malignant", "Primary Site - labeled": "C20.9-Rectum, NOS",
        "Grade Recode (thru 2017)": np.where(new, "Blank(s)", rng.choice(["Moderately differentiated; Grade II", "Poorly differentiated; Grade III", "Well differentiated; Grade I", "Unknown"], n)),
        "Derived Summary Grade 2018 (2018+)": np.where(new, rng.choice(["Site-specific grade system category (2)", "Site-specific grade system category (3)"], n), "Blank(s)"),
        "Histology recode - broad groupings": rng.choice(["8140-8389: adenomas and adenocarcinomas", "8440-8499: cystic, mucinous and serous neoplasms", "8010-8049: epithelial neoplasms, NOS"], n, p=[.85, .1, .05]),
        "Combined Summary Stage with Expanded Regional Codes (2004+)": stage,
        "Survival months": np.char.zfill(months.astype(str), 4), "Survival months flag": "Complete dates are available and there are more than 0 days of survival",
        "Vital status recode (study cutoff used)": np.where(dead, "Dead", "Alive"),
        "Sequence number": rng.choice(["One primary only", "1st of 2 or more primaries", "2nd of 2 or more primaries"], n, p=[.8, .1, .1]),
        "Median household income inflation adj to 2024": rng.choice(["$75,000 - $79,999", "$90,000 - $94,999", "$60,000 - $64,999", "$120,000+"], n),
        "Rural-Urban Continuum Code": rng.choice(["Counties in metropolitan areas ge 1 million pop", "Counties in metropolitan areas of 250,000 to 1 million pop", "Nonmetropolitan counties adjacent to a metropolitan area"], n),
        "Patient ID": np.arange(n) + 1000,
        "Type of Reporting Source": rng.choice(["Hospital inpatient/outpatient or clinic", "Death certificate only"], n, p=[.99, .01]),
        "Marital status at diagnosis": rng.choice(["Married (including common law)", "Single (never married)", "Widowed", "Divorced", "Unknown"], n),
        "RX Summ--Surg Prim Site (1998-2022)": surg,
        "Reason no cancer-directed surgery": "Surgery performed", "Radiation recode": rng.choice(["None/Unknown", "Beam radiation"], n),
        "Chemotherapy recode (yes, no/unk)": rng.choice(["Yes", "No/Unknown"], n),
        "CEA Pretreatment Interpretation Recode (2010+)": rng.choice(["CEA negative/normal; within normal limits", "CEA positive/elevated", "Test ordered, results not in chart"], n),
        "Perineural Invasion Recode (2010+)": rng.choice(["Perineural invasion not identified/not present", "Perineural invasion identified/present", "Not documented; Indeterminate"], n),
        "Response to Neoadjuvant Therapy Recode (2010+)": "Blank(s)",
        "Tumor Deposits Recode (2010+)": rng.choice(["No tumor deposits", "Tumor Deposits identified, number unknown", "3", "Unknown"], n),
        "Derived EOD 2018 T Recode (2018+)": blank_unless(new, T), "Derived EOD 2018 N Recode (2018+)": blank_unless(new, N),
        "Derived EOD 2018 M Recode (2018+)": blank_unless(new, M), "Derived EOD 2018 Stage Group Recode (2018+)": blank_unless(new, sg),
        "Derived AJCC Stage Group, 7th ed (2010-2015)": blank_unless(old, sg), "Derived AJCC T, 7th ed (2010-2015)": blank_unless(old, T),
        "Derived AJCC N, 7th ed (2010-2015)": blank_unless(old, N), "Derived AJCC M, 7th ed (2010-2015)": blank_unless(old, M),
        "7th Edition Stage Group Recode (2016-2017)": blank_unless(mid, sg),
        "Derived SEER Combined T (2016-2017)": blank_unless(mid, np.char.add("p", np.char.lstrip(T.astype(str), "T"))),
        "Derived SEER Combined N (2016-2017)": blank_unless(mid, np.char.add("p", np.char.lstrip(N.astype(str), "N"))),
        "Derived SEER Combined M (2016-2017)": blank_unless(mid, np.char.add("c", np.char.lstrip(M.astype(str), "M"))),
        "Tumor Size Over Time Recode (1988+)": np.char.zfill(rng.integers(5, 120, n).astype(str), 3),
        "Regional nodes examined (1988+)": ne, "Regional nodes positive (1988+)": np.where(ne == 0, 98, npos),
        "Lymph-vascular Invasion (2004+ varying by schema)": "Blank(s)",
    })
    df.loc[df.sample(frac=0.01, random_state=1).index, "Derived AJCC Stage Group, 7th ed (2010-2015)"] = np.nan
    return df


if __name__ == "__main__":
    make(int(sys.argv[2]) if len(sys.argv) > 2 else 4000).to_csv(sys.argv[1], index=False)
