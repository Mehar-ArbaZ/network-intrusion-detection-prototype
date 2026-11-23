# Project Proposal: Network Intrusion Detection Prototype

## 1. Stakeholder & Problem Statement

**Intended Stakeholder:** A small IT security operations team responsible for monitoring network connection logs with limited analyst headcount.

**Problem:** Network firewalls, routers, and sensors generate thousands of connection records per hour. Reviewing logs manually is slow and error-prone, while static signature rules struggle when attackers vary traffic patterns or probe unfamiliar services. Small security teams need a lightweight, transparent triage tool that flags suspicious connections early so human analysts can prioritize investigation.

**Intended Use & Boundaries:** This prototype is designed strictly as a **decision-support triage aid** that scores network connection records and ranks alerts using an adjustable probability threshold. It is **not** a production Intrusion Prevention System (IPS) and must **never** be used to automatically block network traffic, penalize users, or execute security actions without human analyst verification.

## 2. Background & Related Work

Modern intrusion detection research builds heavily on benchmark connection-level datasets while recognizing their methodological pitfalls:

1. **McHugh (2000)** — *Testing Intrusion Detection Systems: A Critique of the 1998 and 1999 DARPA Intrusion Detection System Evaluations as Performed by Lincoln Laboratory*, ACM TISSEC 3(4), 262–294. Demonstrated that synthetic testbed traffic lacks the complexity of live enterprise networks and warned against naive ROC/false-alarm interpretations when attack base rates are low.
2. **Tavallaee et al. (2009)** — *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA. Quantified extreme redundancy in KDD Cup 99 (~78% duplicate training records and ~75% duplicate test records) and introduced **NSL-KDD** to remove duplicates and balance record difficulty levels.
3. **Canadian Institute for Cybersecurity (UNB), NSL-KDD Dataset Documentation** — Documents the record counts of KDD Cup 99 (4,898,431 train / 311,027 test reduced to 1,074,992 / 77,289 distinct records), NSL-KDD file structure, redistribution terms, and citation requirements.
4. **Latah & Toker (2018)** — *Towards an Efficient Anomaly-Based Intrusion Detection for Software-Defined Networks* (arXiv:1803.06762). Benchmarked classical supervised ML classifiers on intrusion detection workloads, comparing detection recall, precision, false-alarm rate, and computational overhead.
5. **Ngueajio et al. (2022)** — *Intrusion Detection Systems Using Support Vector Machines on the KDDCUP'99 and NSL-KDD Datasets: A Comprehensive Survey* (arXiv:2209.05579). Synthesized SVM-based IDS studies across both benchmarks, showing how preprocessing choices, feature scaling, and test-split design strongly influence reported metrics.
6. **Maseer et al. (2023)** — *Meta-Analysis and Systematic Review for Anomaly Network Intrusion Detection Systems: Detection Methods, Dataset, Validation Methodology, and Challenges* (arXiv:2308.02805). Highlighted persistent evaluation gaps in anomaly NIDS research, including data leakage from improper preprocessing, over-reliance on raw accuracy, and weak generalization to unseen attack classes.

**Gap Addressed by This Prototype:** Rather than optimizing headline accuracy on leaked or duplicated splits, this project enforces a strict leak-free `scikit-learn` pipeline, evaluates on the untouched official `KDDTest+` split containing unseen attack types, prioritizes attack recall alongside false-alarm cost, and exposes an interactive threshold control in Streamlit so non-technical stakeholders can directly inspect the recall vs. false-alarm trade-off.

## 3. Target Variable & Proposed Features

* **Target Variable (`label`):** Binary classification derived from NSL-KDD's `attack_type` field:
  * `0` = Normal connection (`attack_type == "normal"`)
  * `1` = Suspicious / Attack connection (any DoS, Probe, R2L, or U2R attack label)
* **Input Features (41 Attributes):**
  * **Basic connection features (9):** `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`
  * **Content features (13):** `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login`
  * **Time-based traffic features (9):** `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate`
  * **Host-based traffic features (10):** `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`
* **Excluded Metadata Columns:** `attack_type` (target source) and `difficulty` (post-hoc benchmark difficulty score, which would leak label hardness if included as a feature).

## 4. Measurable Success Criteria

1. **Primary Security Criterion (Recall):** Achieve an **attack recall of at least 0.80** on the held-out official `KDDTest+` benchmark while reporting the corresponding false-alarm rate transparently.
2. **Model Improvement Criterion (Macro F1):** Deliver an improved non-linear model (Random Forest) whose **macro F1-score improves by at least +0.02** over the linear Logistic Regression baseline.

*(If either criterion is not met on `KDDTest+` due to unseen attack types, the gap will be reported transparently and analyzed by attack family.)*

## 5. Planned Workflow & Team Ownership

| Stage | Scope | Owner |
| --- | --- | --- |
| **Stage 1: Data Audit & EDA** | Dataset comparison (NSL-KDD vs. KDD Cup 99), data dictionary, schema loader, preprocessing pipeline, dataset audit, and EDA charts | Muhammad Huzaifa Sohail (Data Engineer / Analyst) |
| **Stage 2: Modeling & Evaluation** | Leak-free training pipeline, experiments E1–E4 (Logistic Regression baseline, feature pruning, Random Forest, GridSearchCV tuning), evaluation metrics, and error analysis | Muhammad Arbaz (Project Lead & ML Developer) |
| **Stage 3: Deployment, QA & Docs** | Edge-case `pytest` suite, error log, Streamlit triage prototype with threshold slider and disclaimer, cloud deployment, and documentation | Umar (UI, Deployment & QA Lead) |
