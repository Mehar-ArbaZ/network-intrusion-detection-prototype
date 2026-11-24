# Dataset Comparison & Quality Audit

**Selected Dataset:** NSL-KDD (`KDDTrain+.TXT` and `KDDTest+.TXT`)  
**Candidate Compared & Rejected:** KDD Cup 99  
**Mirror Source URL:** `https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/` (`KDDTrain+.txt`, `KDDTest+.txt`, verified against `KDDTrain+.arff`)  
**Access Date:** `2026-11-23`  
**Licence & Citation Terms:** Free to redistribute and mirror for research and evaluation; all uses must cite the NSL-KDD dataset (Canadian Institute for Cybersecurity, UNB) and Tavallaee et al. (2009), *"A Detailed Analysis of the KDD CUP 99 Data Set"*, IEEE CISDA. *(Note: The original UNB direct download link is no longer hosted on UNB's page, so files were acquired from the public GitHub mirror listed above and verified against the standard ARFF schema.)*

---

## 1. Candidate Dataset Comparison: NSL-KDD vs. KDD Cup 99

| Criterion | NSL-KDD (Selected) | KDD Cup 99 (Rejected) |
| --- | --- | --- |
| **Train Records** | **125,973** (`KDDTrain+.TXT`, verified via `df.shape`) | 4,898,431 total; reduced to 1,074,992 distinct records (UNB / Tavallaee et al., 2009) |
| **Test Records** | **22,544** (`KDDTest+.TXT`, verified via `df.shape`) | 311,027 total; reduced to 77,289 distinct records (UNB / Tavallaee et al., 2009) |
| **Duplicate Rate** | **0** exact full-row duplicates within train or test | ~78% train / ~75% test duplicates in full dataset (**70.53%** duplicates empirically verified locally on the 494,021-row 10% subset via `sklearn.datasets.fetch_kddcup99`: only 145,586 distinct rows) |
| **Class / Difficulty Balance** | Moderately balanced train split (46.54% attack); stratified difficulty sampling | Heavily dominated by repetitive DoS traffic, inflating naive classifier accuracy |
| **Iterative Feasibility** | Full official splits run quickly on standard hardware without downsampling | Requires aggressive downsampling for rapid experimentation |
| **Licence & Provenance** | Free to redistribute/mirror with citation to UNB & Tavallaee et al. (2009) | Public DARPA 1998 / UCI benchmark |
| **Decision** | **Selected** | **Rejected** due to severe redundancy and scale |

---

## 2. Schema, Dimensions & Data Types

Both `KDDTrain+.TXT` and `KDDTest+.TXT` are comma-separated files **without a header row**. Column names were confirmed directly against the official `KDDTrain+.arff` attribute list (41 features + `attack_type` + `difficulty` = **43 raw columns**). After loading via `src/load_data.py`, two derived columns (`label` and `attack_family`) are appended for analysis (`45` columns total in memory).

| File | Rows | Raw Columns | Feature Columns | Missing Values |
| --- | --- | --- | --- | --- |
| `KDDTrain+.TXT` | **125,973** | 43 | 41 (3 categorical, 38 numeric: 23 `int64`, 15 `float64`) | **0** |
| `KDDTest+.TXT` | **22,544** | 43 | 41 (3 categorical, 38 numeric: 23 `int64`, 15 `float64`) | **0** |

* **Derived binary target (`label`):** `0` for `normal`, `1` for any attack type.
* **Excluded metadata:** `attack_type` (source of target) and `difficulty` (integer `0..21` reflecting how many of 21 benchmark classifiers correctly predicted the record — not an observable network connection attribute).

---

## 3. Missing Values & Duplicate Audit

| Duplicate Check | `KDDTrain+` | `KDDTest+` |
| --- | --- | --- |
| Exact full-row duplicates (all 43 raw columns) | **0** | **0** |
| Feature + `attack_type` duplicates (42 columns) | **0** | **0** |
| Feature-only duplicates (41 predictive features) | **16** (0.01%) | **57** (0.25%) |

**Cross-File Train/Test Overlap:**
* Unique 41-feature vectors appearing in both `KDDTrain+` and `KDDTest+`: **654** (matching **664** rows in `KDDTest+`, or 2.95%).
* Test rows matching both all 41 features and `attack_type` in `KDDTrain+`: **610** rows (2.71% of `KDDTest+`).
* **Decision:** Retain the official NSL-KDD splits without dropping rows so metrics remain comparable to published NSL-KDD baselines, while transparently noting the 2.71% cross-file overlap.

---

## 4. Class Balance, Attack Families & Unseen Attack Types

### Binary Class Distribution

| Split | Normal (`0`) | Attack (`1`) | Attack Share |
| --- | --- | --- | --- |
| `KDDTrain+` (`n = 125,973`) | 67,343 | 58,630 | **46.54%** |
| `KDDTest+` (`n = 22,544`) | 9,711 | 12,833 | **56.92%** |

### Intrusion Family Distribution

| Attack Family | `KDDTrain+` | `KDDTest+` |
| --- | --- | --- |
| Normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,460 |
| Probe | 11,656 | 2,421 |
| R2L (Remote-to-Local) | 995 | 2,885 |
| U2R (User-to-Root) | 52 | 67 |

### Unseen Attack Types in `KDDTest+`

* Unique attack labels in `KDDTrain+`: **22** (2 attack labels appear only in train: `spy`, `warezclient`).
* Unique attack labels in `KDDTest+`: **37**.
* **Unseen attack labels present ONLY in `KDDTest+`:** **17 attack types**, accounting for **3,750 test rows** (**16.63%** of all test records and **29.22%** of all test attacks):
  * `mscan` (996), `apache2` (737), `processtable` (685), `snmpguess` (331), `saint` (319), `mailbomb` (293), `snmpgetattack` (178), `httptunnel` (133), `named` (17), `ps` (15), `sendmail` (14), `xterm` (13), `xlock` (9), `xsnoop` (4), `worm` (2), `sqlattack` (2), `udpstorm` (2).
* **Modeling implication:** Nearly 30% of test attacks represent zero-day/unseen attack classes relative to `KDDTrain+`, and R2L volume nearly triples in `KDDTest+` (995 → 2,885). Consequently, **test recall is expected to drop below validation recall**, and models must be evaluated honestly on `KDDTest+` without tuning on test scores.

---

## 5. Categorical Coverage, Constant & Correlated Features

### Categorical Feature Overlap (`Train` vs. `Test`)

| Column | Train Unique | Test Unique | Train-Only Categories | Test-Only Categories |
| --- | --- | --- | --- | --- |
| `protocol_type` | 3 | 3 | None | None |
| `service` | 70 | 64 | 6 (`aol`, `harvest`, `http_2784`, `http_8001`, `red_i`, `urh_i`) | None |
| `flag` | 11 | 11 | None | None |

Although `KDDTest+` contains no unseen `service` values relative to full `KDDTrain+`, rare services can be absent from an 80% training fold or appear newly in external CSV uploads. Therefore, `OneHotEncoder(handle_unknown="ignore")` is mandatory inside the preprocessing pipeline.

### Constant & Highly Correlated Features (Evaluated on 80% Train Split Only)

* **Constant column (zero variance in both train and test):** `num_outbound_cmds` (always `0`).
* **Highly correlated numeric pairs (`|r| > 0.90` on 80% train split, 13 pairs):**
  * `num_compromised` ↔ `num_root` (`r = 0.9992`)
  * `serror_rate` ↔ `srv_serror_rate` (`0.9934`), `dst_host_serror_rate` (`0.9793`), `dst_host_srv_serror_rate` (`0.9810`)
  * `srv_serror_rate` ↔ `dst_host_serror_rate` (`0.9779`), `dst_host_srv_serror_rate` (`0.9862`)
  * `dst_host_serror_rate` ↔ `dst_host_srv_serror_rate` (`0.9852`)
  * `rerror_rate` ↔ `srv_rerror_rate` (`0.9894`), `dst_host_rerror_rate` (`0.9272`), `dst_host_srv_rerror_rate` (`0.9652`)
  * `srv_rerror_rate` ↔ `dst_host_rerror_rate` (`0.9188`), `dst_host_srv_rerror_rate` (`0.9709`)
  * `dst_host_rerror_rate` ↔ `dst_host_srv_rerror_rate` (`0.9249`)
* **Redundant columns flagged for removal in Stage 2 Experiment E2 (8 total):** `num_outbound_cmds`, `num_root`, `srv_serror_rate`, `srv_rerror_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate` (leaving **33 features**).

---

## 6. Privacy, Bias & Cleaning Summary

* **Privacy:** NSL-KDD contains no IP addresses, hostnames, MAC addresses, user IDs, payloads, or personally identifiable information (PII). However, deploying a similar classifier on live organizational traffic *would* involve sensitive identifiers and requires formal privacy review and data minimization.
* **Historical & Simulation Bias:** As critiqued by McHugh (2000), the underlying DARPA 1998 traffic was synthetically generated on a simulated Air Force base testbed. Modern networks use TLS/QUIC encryption, cloud/SaaS services, and attack vectors absent from 1998 traffic; strong NSL-KDD metrics do **not** validate production readiness.
* **Data Quirks Noted:** `su_attempted` takes values `{0, 1, 2}` rather than strict binary `{0, 1}` (a known NSL-KDD artifact preserved as numeric). Volume features (`src_bytes` skew = `190.67`, `dst_bytes` skew = `290.05`, `duration` skew = `11.88`) are heavily right-skewed, justifying `StandardScaler` inside the Logistic Regression pipeline while keeping raw thresholds available for Random Forest models.
* **Reproducible Audit Code:** See `notebooks/01_audit.ipynb`, `notebooks/02_eda.ipynb`, and `data/processed/feature_summary.csv`.
