# Explainable machine learning for five-year overall survival in colorectal cancer (SEER)

M.Tech research project: Shadab Ansari, M.Tech (CS & IT), MANUU Hyderabad. Guide: Dr. Mohd Omar.

* **Primary task:** five-year overall-survival classification (survived ≥ 60 months vs. died < 60 months, all causes). Patients alive with < 60 months of follow-up are censored and excluded from the classifiers.
* **Secondary analysis:** time-to-event survival modelling (Cox, XGBoost-Cox, RSF, GBSA, CoxNet) with IPCW.
* **Prediction time point:** the primary model (feature set B) is a *post-surgical pathological-assessment* model. Feature set A is the diagnosis-time model.
* **Validation:** stratified 5-fold CV, a held-out 20% test set, strict temporal validation (selection and tuning only within 2010–2015, tested on 2016–2018), and external validation on AACR Project GENIE BPC CRC v2.0-public (separate notebook).
* **Explainability:** SHAP (main), with partial dependence and permutation importance as support. LIME is supplementary.

> Research prototype for patient-level survival prediction and explanation. It is **not** a diagnostic device.

## Main results
| Model | Data | ROC-AUC (95% CI) | Brier | Calibration slope |
|---|---|---|---|---|
| Gradient Boosting, feature set B (21 features) | SEER held-out test set | 0.877 (0.874–0.880) | 0.138 | 1.02 |
| Core transportable model (12 features) | SEER held-out test set | 0.863 (0.860–0.866) | 0.146 | 1.02 |
| Core transportable model (frozen) | GENIE BPC CRC, external (n = 695) | 0.790 (0.755–0.825) | 0.183 | 0.73 |

On GENIE, Harrell's C-index on all 1,136 eligible patients was 0.735. Discrimination transported well, but calibration did not: the model over-estimates survival in the low- and intermediate-risk groups, so local recalibration would be needed before clinical use.

## Repository layout
| Path | What it is |
|---|---|
| `notebooks/CRC_SEER_Survival_Prediction.ipynb` | Full SEER pipeline: cohort, features, models, CIs, calibration, decision curves, temporal validation, survival models, SHAP, model export |
| `notebooks/CRC_GENIE_External_Validation.ipynb` | Applies the frozen SEER model to GENIE BPC CRC (no training on GENIE) |
| `app/app.py` | Flask web app: calibrated 5-year survival probability plus the top SHAP factors |
| `Dockerfile`, `requirements.txt` | Container for the web app |
| `src/prepare.py`, `src/train.py`, `src/evaluate.py` | Script version of the final pipeline: cohort and split, training with MLflow Model Registry, test metrics with bootstrap CIs and `@champion` promotion |
| `src/crc/` | Shared code: cohort rules, features, pipeline, metrics, MLflow helpers |
| `params.yaml`, `dvc.yaml` | All settings and the DVC pipeline (`prepare → train → evaluate`) |
| `tests/` | End-to-end test on synthetic data (pipeline, registry, web app) |
| `.github/workflows/ci.yml` | GitHub Actions: no-patient-data check, tests, Docker build |
| `k8s/crc-survival.yaml` | Kubernetes (Minikube) Deployment and Service |

## Data (not included)
* **SEER:** SEER Research Data, 17 Registries, Nov 2025 Sub (2000–2023), exported from SEER*Stat as `Survival_Data.csv`. Access requires a signed SEER Research Data Use Agreement, and the data must not be redistributed.
* **GENIE BPC CRC v2.0-public:** download from Synapse after accepting the AACR GENIE terms of access. Do not redistribute.

The `.gitignore` blocks every `*.csv` and the GENIE folder, so patient-level data cannot be committed by accident.

## Reproduce
```bash
pip install -r requirements-notebooks.txt
# put Survival_Data.csv next to the notebook, then run all cells of notebooks/CRC_SEER_Survival_Prediction.ipynb
# -> notebooks/artifacts/ (crc_model.joblib, crc_calibrator.joblib, model_meta.json, core + Set A models)
```
The random seed is 42. Every number reported in the paper comes from a full top-to-bottom run.

## Run the web app
Copy `crc_model.joblib`, `crc_calibrator.joblib` and `model_meta.json` (from `notebooks/artifacts/` or `models/`) into `app/artifacts/`, then run:
```bash
python app/app.py                                  # http://127.0.0.1:5000
docker build -t crc-survival .
docker run -p 7860:7860 crc-survival               # http://127.0.0.1:7860
```
Trained model files are git-ignored by default. Check the SEER data-use terms before publishing them.

## MLOps (scripts, DVC, MLflow Model Registry, CI)
```bash
pip install -r requirements-mlops.txt
dvc repro              # prepare -> train -> evaluate; registers crc-5y-survival in MLflow
dvc metrics show
pytest -q              # synthetic-data test, same as CI
```
MLflow logs to `MLFLOW_TRACKING_URI` when it is set (for example a private DagsHub repo) and to a local `mlflow.db` otherwise. The DVC remote is a local folder, so patient-level data never leaves the machine. The raw SEER file must be placed at `data/raw/Survival_Data.csv` before `dvc repro`.

## Reference studies
* Susič et al. (2023), *Computer Methods and Programs in Biomedicine*: machine-learning pipeline for colorectal cancer survival on a hospital cohort.
* Osman et al. (2022): SEER-trained colorectal cancer survival model with external validation.

## Citation
Cite SEER using the suggested citation from SEER*Stat. For GENIE BPC, follow the AACR Project GENIE publication policy.
