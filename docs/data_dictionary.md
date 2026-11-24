# Data Dictionary — NSL-KDD Network Connection Dataset

All 41 predictive feature names and their ordering were verified directly against the official `KDDTrain+.arff` attribute specification and checked programmatically against `KDDTrain+.TXT` (`125,973 x 43`) and `KDDTest+.TXT` (`22,544 x 43`). Programmatic summary statistics are also stored in `data/processed/feature_summary.csv`.

---

## 1. Basic Connection Features (9 Columns)

| # | Column Name | Observed Type | Train Unique | Test Unique | Observed Range / Top Values (`KDDTrain+`) | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `duration` | `int64` | 2,981 | 624 | `0` – `42,908` (median `0`) | Length of the connection in seconds |
| 2 | `protocol_type` | `str` (categorical) | 3 | 3 | `tcp`, `udp`, `icmp` | Transport-layer protocol type |
| 3 | `service` | `str` (categorical) | 70 | 64 | `http`, `private`, `domain_u`, … | Destination network service (6 train-only services) |
| 4 | `flag` | `str` (categorical) | 11 | 11 | `SF`, `S0`, `REJ`, `RSTR`, `RSTO`, … | Normal or error status of the TCP connection |
| 5 | `src_bytes` | `int64` | 3,341 | 1,149 | `0` – `1,379,963,888` (median `44`) | Number of data bytes sent from source to destination |
| 6 | `dst_bytes` | `int64` | 9,326 | 3,650 | `0` – `1,309,937,401` (median `0`) | Number of data bytes sent from destination to source |
| 7 | `land` | `int64` (binary) | 2 | 2 | `0`, `1` | `1` if connection is from/to the same host/port; `0` otherwise |
| 8 | `wrong_fragment` | `int64` | 3 | 3 | `0` – `3` | Number of malformed ("wrong") IP fragments |
| 9 | `urgent` | `int64` | 4 | 4 | `0` – `3` | Number of urgent TCP packets |

---

## 2. Content Features (13 Columns)

| # | Column Name | Observed Type | Train Unique | Test Unique | Observed Range (`KDDTrain+`) | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | `hot` | `int64` | 28 | 16 | `0` – `77` | Number of "hot" indicators (e.g., entering system directories, executing programs) |
| 11 | `num_failed_logins` | `int64` | 6 | 5 | `0` – `5` | Number of failed login attempts |
| 12 | `logged_in` | `int64` (binary) | 2 | 2 | `0`, `1` | `1` if successfully logged in; `0` otherwise |
| 13 | `num_compromised` | `int64` | 88 | 23 | `0` – `7,479` | Number of "compromised" conditions |
| 14 | `root_shell` | `int64` (binary) | 2 | 2 | `0`, `1` | `1` if a root shell was obtained; `0` otherwise |
| 15 | `su_attempted` | `int64` | 3 | 3 | `0` – `2` | Indicates `su root` command attempted (`2` is a known legacy NSL-KDD quirk) |
| 16 | `num_root` | `int64` | 82 | 20 | `0` – `7,468` | Number of root accesses (`r = 0.9992` with `num_compromised`) |
| 17 | `num_file_creations` | `int64` | 35 | 9 | `0` – `43` | Number of file creation operations |
| 18 | `num_shells` | `int64` | 3 | 4 | `0` – `2` | Number of shell prompts invoked |
| 19 | `num_access_files` | `int64` | 10 | 5 | `0` – `9` | Number of operations on access control files |
| 20 | `num_outbound_cmds` | `int64` (constant) | 1 | 1 | `0` – `0` | Number of outbound commands in an FTP session (**constant `0` in train & test**) |
| 21 | `is_host_login` | `int64` (binary) | 2 | 2 | `0`, `1` | `1` if login belongs to the "hot" host list; `0` otherwise |
| 22 | `is_guest_login` | `int64` (binary) | 2 | 2 | `0`, `1` | `1` if login is a guest login; `0` otherwise |

---

## 3. Time-Based Traffic Features — 2-Second Window (9 Columns)

| # | Column Name | Observed Type | Train Unique | Test Unique | Observed Range (`KDDTrain+`) | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 23 | `count` | `int64` | 512 | 495 | `0` – `511` (median `14`) | Number of connections to the same destination host in past 2 seconds |
| 24 | `srv_count` | `int64` | 509 | 457 | `0` – `511` (median `8`) | Number of connections to the same service in past 2 seconds |
| 25 | `serror_rate` | `float64` | 89 | 88 | `0.0` – `1.0` | Fraction of same-host connections with SYN errors (`S0`, `S1`, `S2`, `S3`) |
| 26 | `srv_serror_rate` | `float64` | 86 | 82 | `0.0` – `1.0` | Fraction of same-service connections with SYN errors |
| 27 | `rerror_rate` | `float64` | 82 | 90 | `0.0` – `1.0` | Fraction of same-host connections with REJ errors |
| 28 | `srv_rerror_rate` | `float64` | 62 | 93 | `0.0` – `1.0` | Fraction of same-service connections with REJ errors |
| 29 | `same_srv_rate` | `float64` | 101 | 75 | `0.0` – `1.0` | Fraction of same-host connections to the same service |
| 30 | `diff_srv_rate` | `float64` | 95 | 99 | `0.0` – `1.0` | Fraction of same-host connections to different services |
| 31 | `srv_diff_host_rate` | `float64` | 60 | 84 | `0.0` – `1.0` | Fraction of same-service connections to different hosts |

---

## 4. Host-Based Traffic Features — Last 100 Connections (10 Columns)

| # | Column Name | Observed Type | Train Unique | Test Unique | Observed Range (`KDDTrain+`) | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | `dst_host_count` | `int64` | 256 | 256 | `0` – `255` (median `255`) | Count of connections having the same destination host IP |
| 33 | `dst_host_srv_count` | `int64` | 256 | 256 | `0` – `255` (median `63`) | Count of connections having the same destination host and service |
| 34 | `dst_host_same_srv_rate` | `float64` | 101 | 101 | `0.0` – `1.0` | Fraction of same-host connections to the same service |
| 35 | `dst_host_diff_srv_rate` | `float64` | 101 | 101 | `0.0` – `1.0` | Fraction of same-host connections to different services |
| 36 | `dst_host_same_src_port_rate` | `float64` | 101 | 101 | `0.0` – `1.0` | Fraction of same-host connections sharing the same source port |
| 37 | `dst_host_srv_diff_host_rate` | `float64` | 75 | 58 | `0.0` – `1.0` | Fraction of same-host/service connections coming from different source hosts |
| 38 | `dst_host_serror_rate` | `float64` | 101 | 99 | `0.0` – `1.0` | Fraction of same-host connections with SYN errors |
| 39 | `dst_host_srv_serror_rate` | `float64` | 100 | 101 | `0.0` – `1.0` | Fraction of same-host/service connections with SYN errors |
| 40 | `dst_host_rerror_rate` | `float64` | 101 | 101 | `0.0` – `1.0` | Fraction of same-host connections with REJ errors |
| 41 | `dst_host_srv_rerror_rate` | `float64` | 101 | 100 | `0.0` – `1.0` | Fraction of same-host/service connections with REJ errors |

---

## 5. Target & Excluded Metadata Columns

| # | Column Name | Observed Type | Role in Modeling | Description |
| --- | --- | --- | --- | --- |
| 42 | `attack_type` | `str` | **Excluded from `X`** (source of target) | Raw NSL-KDD label (`normal` or specific attack name; 23 values in train, 38 in test) |
| 43 | `difficulty` | `int64` | **Strictly dropped** | Benchmark score (`0`–`21`) recording how many of 21 reference models classified the row correctly; excluded to prevent label leakage |
| — | `label` *(derived)* | `int64` | **Binary target `y`** | `0` = `normal`, `1` = any attack |
| — | `attack_family` *(derived)* | `str` | **Analysis only** | Groups `attack_type` into `Normal`, `DoS`, `Probe`, `R2L`, or `U2R` for error analysis |
