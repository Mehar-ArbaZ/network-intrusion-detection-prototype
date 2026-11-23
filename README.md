# Network Intrusion Detection Prototype

A machine learning prototype that classifies network connection records as **normal** or **suspicious (attack)** using `scikit-learn` on the NSL-KDD benchmark dataset, paired with an interactive Streamlit interface for triage and threshold exploration.

> **Disclaimer:** This is a research and evaluation prototype trained on the historical NSL-KDD (1998-era simulated) benchmark dataset. It is not a production security product and must not be used to automatically block network traffic or make autonomous security decisions without human analyst review.

## Overview

Security teams often face high volumes of network connection logs. This project builds an end-to-end, reproducible supervised ML workflow that:

- Audits and preprocesses the NSL-KDD network connection dataset without data leakage.
- Compares linear (Logistic Regression) and non-linear ensemble (Random Forest) baselines and tuned models.
- Prioritizes **attack recall** while monitoring the false-alarm rate and macro F1-score.
- Provides a lightweight Streamlit web prototype to score single connection records or uploaded CSV files with an adjustable decision threshold.

## Project Structure

```text
network-intrusion-detection-prototype/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # Raw NSL-KDD files (excluded from Git)
│   ├── processed/           # Cleaned datasets
│   └── sample/              # Small sample records for demo and tests
├── notebooks/               # Dataset audit, EDA, and modeling notebooks
├── src/                     # Modular data loading, preprocessing, training, evaluation, and inference code
├── models/                  # Serialized trained model artifact
├── app/                     # Streamlit prototype application
├── tests/                   # Automated pytest edge-case tests
├── reports/
│   ├── figures/             # Generated visualizations
│   └── results/             # Evaluation metrics and experiment logs
└── docs/                    # Architecture notes, data dictionary, audit, experiments, and error log
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Mehar-ArbaZ/network-intrusion-detection-prototype.git
   cd network-intrusion-detection-prototype
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place `KDDTrain+.TXT` and `KDDTest+.TXT` (NSL-KDD dataset) inside `data/raw/`.

## Usage

*(Commands will be populated as modules are implemented.)*

- **Run tests:**
  ```bash
  pytest
  ```
- **Launch the Streamlit prototype locally:**
  ```bash
  streamlit run app/app.py
  ```

## Limitations

- **Historical benchmark:** NSL-KDD is derived from 1998-era simulated traffic; strong benchmark performance does not guarantee generalization to modern production networks.
- **Unseen attack types:** The official test split contains attack types absent from training, making zero-day/unseen attack recall a known challenge.
- **Per-connection scope:** Predictions are made on individual connection records without multi-step sequence or temporal context.
- **Human-in-the-loop required:** Any real-world deployment would require retraining on contemporary organizational traffic, privacy review, drift monitoring, and analyst oversight.
