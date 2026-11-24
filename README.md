# Network Intrusion Detection Prototype

A reproducible machine learning prototype that classifies network connection records as **Normal (`0`)** or **Suspicious / Attack (`1`)** using `scikit-learn` on the **NSL-KDD** benchmark dataset, paired with an interactive **Streamlit** triage interface for threshold exploration and batch CSV scoring.

> **Disclaimer:** This is a classroom research and evaluation prototype trained on the historical **1998-era NSL-KDD** simulated benchmark dataset. It is **not** a production security product (IDS/IPS) and must **never** be used to automatically block network traffic, penalize individuals, or make autonomous security decisions without human analyst review.

---

## Overview

Small IT security teams face high volumes of network connection logs with limited analyst headcount. This project delivers an end-to-end, leak-free supervised ML workflow for CUSIT *Programming for Artificial Intelligence (Fall 2026)*:

- **Stage 1 — Data Audit & EDA (Muhammad Huzaifa Sohail):** Compared NSL-KDD against KDD Cup 99, audited schema/duplicates/class balance/unseen attacks (`125,973` train / `22,544` test), implemented `src/config.py`, `src/load_data.py`, and `src/preprocess.py`, generated a 100-row sample CSV, and produced 5 EDA visualizations.
- **Stage 2 — Leak-Free Modeling & Evaluation (Muhammad Arbaz):** Ran Experiments `E1`–`E4` inside `scikit-learn` `Pipeline`s on an 80/20 stratified `KDDTrain+` validation split, selected a bounded Tuned Random Forest (`max_depth=20`, `n_estimators=200`, `3.58 MB` `models/model.joblib`), evaluated once on official `KDDTest+`, and quantified the unseen-attack gap and threshold trade-off.
- **Stage 3 — Inference, Streamlit UI, QA & Docs (Muhammad Umar Siddiqi):** Built `src/predict.py`, the interactive Streamlit triage app (`app/app.py`), an automated `pytest` edge-case and reliability suite (`tests/test_edge_cases.py`), the engineering error log (`docs/error_log.md`), and the team contribution matrix (`docs/contribution_matrix.md`).

---

## Project Structure

```text
network-intrusion-detection-prototype/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # Raw NSL-KDD KDDTrain+.TXT & KDDTest+.TXT (excluded from Git)
│   ├── processed/           # Cleaned feature summary tables
│   └── sample/              # 100-row sample CSV (41 features) for demo and tests
├── notebooks/
│   ├── 01_audit.ipynb       # NSL-KDD vs. KDD Cup 99 dataset audit
│   ├── 02_eda.ipynb         # Data cleaning justifications & 5 EDA charts
│   └── 03_models.ipynb      # Experiments E1–E4, KDDTest+ evaluation & error analysis
├── src/
│   ├── config.py            # Verified 41-feature schema, paths, random seed (42)
│   ├── load_data.py         # Strict file, schema, and numeric validation
│   ├── preprocess.py        # Leak-free ColumnTransformer (imputation, OHE, optional scaling)
│   ├── train.py             # Reproducible E1–E4 training & artifact export (`python -m src.train`)
│   ├── evaluate.py          # Metrics, subgroup error analysis & diagnostic plots
│   └── predict.py           # Validated single-record, DataFrame & CSV inference module
├── models/
│   └── model.joblib         # Compressed E4 Tuned Random Forest pipeline (3.58 MB)
├── app/
│   └── app.py               # Interactive Streamlit triage prototype
├── tests/
│   └── test_edge_cases.py   # Automated pytest edge-case & reliability test suite
├── reports/
│   ├── figures/             # Saved EDA & Stage 2 evaluation figures (01–10)
│   └── results/             # Machine-readable metrics CSVs & model_metadata.json
└── docs/
    ├── proposal.md
    ├── dataset_audit.md
    ├── data_dictionary.md
    ├── experiments.md
    ├── error_log.md
    ├── project_log.md
    └── contribution_matrix.md
```

---

## Setup & Reproducible Usage

### 1. Clone and Create a Virtual Environment

```bash
git clone https://github.com/Mehar-ArbaZ/network-intrusion-detection-prototype.git
cd network-intrusion-detection-prototype

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Automated Edge-Case & Reliability Tests

Uses the pre-trained `models/model.joblib` and `data/sample/sample_connections.csv` (no raw dataset required; automatically also verifies `KDDTest+.TXT` if present in `data/raw/`):

```bash
pytest -v
```

### 3. Launch the Interactive Streamlit Triage Prototype

```bash
streamlit run app/app.py
```

### 4. Re-Run Full Training & Evaluation Pipeline (Optional)

Place `KDDTrain+.TXT` and `KDDTest+.TXT` in `data/raw/` (from the NSL-KDD mirror documented in `docs/dataset_audit.md`), then run:

```bash
python -m src.train
```

---

## Model Results Summary (Stage 2)

All validation comparisons (`E1`–`E4`) were performed strictly on an **80/20 stratified split of `KDDTrain+`** (`100,778` train / `25,195` validation, `random_state=42`). The official **`KDDTest+`** split (`22,544` rows) was held out completely and evaluated **once** at the end.

### Validation Comparison (`KDDTrain+` 20% Stratified Split)

| ID | Model | Features | Val Accuracy | Val Precision | Val Attack Recall | Val Macro F1 | Val FPR | Val ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **E1** | `LogisticRegression(max_iter=1000)` | All 41 | 0.9719 | 0.9764 | 0.9629 | 0.9717 | 0.0203 | 0.9964 |
| **E2** | `LogisticRegression(max_iter=1000)` | Pruned 33 | 0.9699 | 0.9746 | 0.9604 | 0.9697 | 0.0218 | 0.9960 |
| **E3** | `RandomForestClassifier(n_estimators=100)` | All 41 | 0.9990 | 0.9995 | 0.9983 | 0.9990 | 0.0004 | 1.0000 |
| **E4** *(Selected)* | `RandomForest GridSearchCV (max_depth=20, n_estimators=200)` | All 41 | **0.9990** | **0.9995** | **0.9983** | **0.9990** | **0.0004** | **1.0000** |

### Held-Out Official `KDDTest+` Evaluation (`n = 22,544`, Default Threshold `0.50`)

| Model | Test Accuracy | Test Precision | Test Attack Recall | Test Attack F1 | Test Macro F1 | Test FPR | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E1 (Baseline Logistic Regression)** | 0.7540 | 0.9172 | 0.6242 | 0.7428 | 0.7535 | 0.0745 | 0.7975 |
| **E4 (Selected Tuned Random Forest)** | **0.7628** | **0.9672** | 0.6038 | **0.7435** | **0.7615** | **0.0271** | **0.9652** |

### Honest Unseen-Attack Gap & Threshold Trade-Off

- **Why test recall drops below validation recall:** `KDDTest+` introduces **17 unseen attack types** absent from `KDDTrain+` (`3,750` rows, `29.22%` of all test attacks) and nearly triples stealthy R2L volume (`995 → 2,885`). At the default `0.50` cutoff, `E4` achieves **97.29%** normal specificity and **76.89%** recall on seen attacks, but only **20.40%** recall on unseen attacks (overall test recall `0.6038`, below the `0.80` proposal target at default `0.50`).
- **Threshold tuning trade-off:** Because `E4` retains strong probability ranking (`KDDTest+` **ROC-AUC = 0.9652** vs. `0.7975` for E1), lowering the decision threshold in the Streamlit sidebar substantially recovers recall:
  - Threshold **`0.50`** (default): `60.41%` attack recall, `2.71%` false-alarm rate, `0.7616` macro F1.
  - Threshold **`0.30`**: `71.17%` attack recall, `3.03%` false-alarm rate, `0.8228` macro F1.
  - Threshold **`0.10`**: **`84.60%` attack recall**, `9.84%` false-alarm rate, **`0.8688` macro F1**.

See [docs/experiments.md](docs/experiments.md) and `reports/results/` for full tables and figures.

---

## Live Streamlit Demo & Cloud Deployment

- **Live Demo URL:** `[Pending Streamlit Community Cloud URL]` *(update this link immediately after deploying below)*

### Deploy `app/app.py` on Streamlit Community Cloud (3 Steps)

1. Sign in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account and click **Create app** → **Deploy a public app from GitHub**.
2. Select repository **`Mehar-ArbaZ/network-intrusion-detection-prototype`**, branch **`main`**, and set **Main file path** to **`app/app.py`**.
3. Click **Deploy** — Streamlit installs dependencies from `requirements.txt` and loads the `3.58 MB` pre-trained `models/model.joblib` automatically. Once live, copy the URL into the **Live Demo URL** line above.

---

## Limitations & Ethical Boundaries

- **Historical 1998-Era Benchmark:** NSL-KDD derives from simulated 1998 DARPA network traffic and lacks modern encrypted protocols (TLS 1.3, QUIC), cloud/SaaS traffic, and contemporary intrusion techniques. Benchmark scores do **not** validate production readiness.
- **Unseen (Zero-Day) Attack Sensitivity:** Supervised classifiers struggle on novel attack signatures at default thresholds; stealthy R2L/U2R attacks mimic normal authenticated traffic.
- **Per-Connection Scope:** Predictions inspect individual connection summaries independently without multi-step sequence, session, or temporal correlation across hosts.
- **Privacy & Fairness:** NSL-KDD contains no IP addresses, user IDs, or payloads, so it holds no PII; however, applying a similar pipeline to live enterprise logs would expose sensitive metadata and requires formal privacy review, data minimization, and drift monitoring.
- **Strict Human-in-the-Loop Policy:** This tool is strictly an analyst triage aid. It must **never** be wired to automated firewall blocking, account suspension, disciplinary action, or any unreviewed security decision.

---

## AI-Assistance Disclosure

In accordance with course academic integrity guidelines, AI coding assistants were used to assist in drafting boilerplate code structure, Streamlit UI layout, `pytest` test cases, and Markdown documentation templates. All dataset audits, model training runs (`python -m src.train`), evaluation metrics, diagnostic figures, edge-case test executions (`pytest -v`), and documentation were executed on the team's own implementation, reviewed line-by-line, verified against real outputs, and are fully understood by all team members for the project defence and individual viva.

---

## Dataset Citation

- M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, *"A Detailed Analysis of the KDD CUP 99 Data Set,"* Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.
- Canadian Institute for Cybersecurity, University of New Brunswick — NSL-KDD Dataset.
