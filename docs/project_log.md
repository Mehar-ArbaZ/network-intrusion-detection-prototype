# Project Log

Engineering progress log tracking architecture decisions, stage milestones, and team ownership across the Network Intrusion Detection Prototype.

## Team Roles

| Member | Role | Primary Ownership |
| --- | --- | --- |
| Muhammad Arbaz | Project Lead & ML Developer | Repository setup, modeling pipeline (`src/train.py`, `src/evaluate.py`), experiments E1–E4, integration |
| Muhammad Huzaifa Sohail | Data Engineer / Analyst | Dataset comparison & audit, data dictionary, `src/config.py`, `src/load_data.py`, `src/preprocess.py`, EDA notebooks & charts |
| Umar | UI, Deployment & QA Lead | `src/predict.py`, `app/app.py` (Streamlit prototype), `tests/test_edge_cases.py`, deployment & QA documentation |

## Milestone Log

| Date | Stage | Milestone | Status | Notes / Artifacts |
| --- | --- | --- | --- | --- |
| 2026-11-23 | Setup | Repository scaffolding & remote setup | Completed | Initialized public GitHub repository with folder layout, `.gitignore`, `requirements.txt`, and `README.md` (`3fb681d`). |
| 2026-11-23 | Stage 1 | Problem proposal & literature review | Completed | Documented stakeholder need, 6 cited references, target/features, and two measurable success criteria in `docs/proposal.md`. |
| 2026-11-23 | Stage 1 | Dataset acquisition, comparison & audit | Completed | Acquired NSL-KDD (`KDDTrain+.TXT`, `KDDTest+.TXT`), verified 41 feature names against official ARFF header, compared against KDD Cup 99, completed `docs/dataset_audit.md`, `docs/data_dictionary.md`, and `notebooks/01_audit.ipynb`. |
| 2026-11-23 | Stage 1 | Core data modules & EDA | Completed | Implemented `src/config.py`, `src/load_data.py`, `src/preprocess.py`, generated 100-row sample CSV, executed `notebooks/02_eda.ipynb`, and saved 5 EDA figures in `reports/figures/`. |
| Pending | Stage 2 | Baseline, improved models & experiments E1–E4 | Planned | Train Logistic Regression & Random Forest models in `src/train.py`, `src/evaluate.py`, and `notebooks/03_models.ipynb`; log runs in `docs/experiments.md` and `reports/results/experiments.csv`. |
| Pending | Stage 3 | Edge-case tests, Streamlit app & deployment | Planned | Implement `src/predict.py`, `app/app.py`, `tests/test_edge_cases.py`, `docs/error_log.md`, and deploy live demo. |
