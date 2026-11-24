"""Central configuration: schema definitions, file paths, and reproducibility constants.

All 41 NSL-KDD feature names were verified directly against the official
KDDTrain+.arff attribute header and the 43-column KDDTrain+.TXT / KDDTest+.TXT layout.
"""

from pathlib import Path

# Project paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_DIR: Path = REPORTS_DIR / "results"
DOCS_DIR: Path = ROOT_DIR / "docs"

TRAIN_FILE: Path = DATA_RAW_DIR / "KDDTrain+.TXT"
TEST_FILE: Path = DATA_RAW_DIR / "KDDTest+.TXT"
SAMPLE_FILE: Path = DATA_SAMPLE_DIR / "sample_connections.csv"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "model.joblib"

# Reproducibility and split settings
RANDOM_SEED: int = 42
VALIDATION_SIZE: float = 0.20
DEFAULT_DECISION_THRESHOLD: float = 0.50

# Verified NSL-KDD schema (41 features + attack_type + difficulty = 43 raw columns)
BASIC_FEATURES: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
]

CONTENT_FEATURES: list[str] = [
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
]

TIME_TRAFFIC_FEATURES: list[str] = [
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
]

HOST_TRAFFIC_FEATURES: list[str] = [
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

FEATURE_COLUMNS: list[str] = (
    BASIC_FEATURES
    + CONTENT_FEATURES
    + TIME_TRAFFIC_FEATURES
    + HOST_TRAFFIC_FEATURES
)

ATTACK_TYPE_COLUMN: str = "attack_type"
DIFFICULTY_COLUMN: str = "difficulty"
TARGET_COLUMN: str = "label"

RAW_COLUMNS: list[str] = FEATURE_COLUMNS + [ATTACK_TYPE_COLUMN, DIFFICULTY_COLUMN]

CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS: list[str] = [
    col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS
]

# Reference mapping of NSL-KDD attack labels to standard intrusion families
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "Normal",
    # DoS
    "apache2": "DoS",
    "back": "DoS",
    "land": "DoS",
    "mailbomb": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "processtable": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "mscan": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "saint": "Probe",
    "satan": "Probe",
    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "httptunnel": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "named": "R2L",
    "phf": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "ps": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}
