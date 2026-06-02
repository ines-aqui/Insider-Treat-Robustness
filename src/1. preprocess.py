"""
preprocess_unified_raw_v4.py  (v3 → v4 수정사항)
=================================================
[FIX 1] 델타 피처 데이터 누수 제거
  - v3: groupby("user").transform("mean") → 미래 포함 전역 평균
  - v4: expanding().mean().shift(1)       → 과거 누적 평균만 사용

[FIX 2] user-day fallback 라벨 오염 경고 주석 추가
  - 동일 날 정상 세션도 악성으로 묶이는 구조임을 명시

나머지 로직은 v3과 동일.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def normalize_user_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def normalize_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def diagnose_label_overlap(unified_df: pd.DataFrame, label_df: pd.DataFrame) -> None:
    uu = unified_df[["user_key", "session_date_key"]].drop_duplicates().copy()
    ll = label_df[["user_key", "session_date_key"]].drop_duplicates().copy()
    user_overlap = len(set(uu["user_key"]) & set(ll["user_key"]))
    key_overlap  = len(set(map(tuple, uu.values)) & set(map(tuple, ll.values)))
    print(f"  [DIAG] unified unique users      : {uu['user_key'].nunique():,}")
    print(f"  [DIAG] label unique users        : {ll['user_key'].nunique():,}")
    print(f"  [DIAG] overlapping users         : {user_overlap:,}")
    print(f"  [DIAG] overlapping user-date keys: {key_overlap:,}")
    if user_overlap == 0:
        print("  [DIAG][WARN] user key intersection = 0.")
    elif key_overlap == 0:
        print("  [DIAG][WARN] user-date key intersection = 0.")

# ── 0. 설정 ───────────────────────────────────────────────────────────────────
DATA_DIR     = Path("./r4.2")
ANSWERS_DIR  = Path("./answers")
OUTPUT_DIR   = Path("./output/unified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUSINESS_START = 8
BUSINESS_END   = 18

print("=" * 60)
print("CERT r4.2 Unified Feature Extraction v4 (delta leak fixed)")
print("=" * 60)

# ── 1. 원본 데이터 로드 ───────────────────────────────────────────────────────
def load_raw_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    normal_files = {
        "logon": "logon.csv", "device": "device.csv",
        "email": "email.csv", "file":   "file.csv",
    }
    for key, fname in normal_files.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [WARN] {fname} not found"); continue
        df = pd.read_csv(fpath, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["source"] = key
        if "user" in df.columns:
            df["user"] = normalize_user_id(df["user"])
        dfs[key] = df
        print(f"  Loaded {fname}: {len(df):,} rows")

    http_path = data_dir / "http.csv"
    if http_path.exists():
        header  = pd.read_csv(http_path, nrows=0).columns.tolist()
        usecols = [c for c in ["date", "user", "pc", "url", "activity"] if c in header]
        chunks  = pd.read_csv(http_path, usecols=usecols, chunksize=500_000, low_memory=False)
        http_df = pd.concat(chunks, ignore_index=True)
        http_df["date"]   = pd.to_datetime(http_df["date"], errors="coerce")
        http_df["source"] = "http"
        if "user" in http_df.columns:
            http_df["user"] = normalize_user_id(http_df["user"])
        dfs["http"] = http_df
        print(f"  Loaded http.csv: {len(http_df):,} rows")
    return dfs

def load_malicious_events() -> pd.DataFrame:
    scenario_dirs = [
        ANSWERS_DIR / "r4.2-1",
        ANSWERS_DIR / "r4.2-2",
        ANSWERS_DIR / "r4.2-3",
    ]
    frames = []
    for sdir in scenario_dirs:
        if not sdir.exists(): continue
        for fpath in sorted(sdir.glob("*.csv")):
            expected_user = str(fpath.stem.split("-")[-1]).strip().upper()
            df = pd.read_csv(fpath, header=None, on_bad_lines="skip", low_memory=False)
            df["date"]         = pd.to_datetime(df[2], errors="coerce")
            df["user"]         = df[3].astype(str).str.strip().str.upper()
            df["session_date"] = df["date"].dt.date
            df = df[df["user"] == expected_user].copy()
            df = df.dropna(subset=["date", "user"])
            if len(df) == 0: continue
            df["is_malicious"] = 1
            frames.append(df[["user", "session_date", "is_malicious"]])
    if not frames:
        print("  [WARN] No answer files found")
        return pd.DataFrame(columns=["user", "session_date", "is_malicious"])
    result = pd.concat(frames, ignore_index=True).drop_duplicates()
    result["session_date_str"] = result["session_date"].astype(str)
    print(f"  Malicious users     : {result['user'].nunique()}")
    print(f"  Malicious user-days : {result['session_date'].notna().sum()}")
    return result

def build_label_userday(malicious_events: pd.DataFrame) -> pd.DataFrame:
    if len(malicious_events) == 0:
        return pd.DataFrame(columns=["user", "session_date", "session_date_str", "is_malicious"])
    return (malicious_events[["user", "session_date", "session_date_str", "is_malicious"]]
            .drop_duplicates())

print("\n[1] Loading raw data...")
dfs              = load_raw_data(DATA_DIR)
malicious_events = load_malicious_events()
label_df         = build_label_userday(malicious_events)

# ── 2. 세션 ID 부여 ───────────────────────────────────────────────────────────
print("\n[2] Assigning session IDs...")

def build_session_map(logon_df: pd.DataFrame) -> pd.DataFrame:
    df = logon_df.dropna(subset=["user", "date"]).copy()
    df["user"] = normalize_user_id(df["user"])
    df = df.sort_values(["user", "date"]).copy()
    act      = df["activity"].fillna("").str.lower()
    is_logon = ~act.str.contains("off|after", na=False)
    df["_session_num"]  = is_logon.astype(int).groupby(df["user"], sort=False).cumsum()
    df["session_id_raw"] = df["user"].astype(str) + "_s" + df["_session_num"].astype(str)
    df["session_date"]   = df["date"].dt.date
    id_map = {s: i for i, s in enumerate(df["session_id_raw"].unique())}
    df["session_id"] = df["session_id_raw"].map(id_map)
    return df[["user", "date", "session_id", "session_date"]].copy()

session_map = build_session_map(dfs["logon"])
print(f"  Total sessions: {session_map['session_id'].nunique():,}")

def assign_session(df: pd.DataFrame, session_map: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["user", "date"]).copy()
    work["_orig_order"] = np.arange(len(work))
    sm   = session_map.dropna(subset=["user", "date"]).copy()
    work = work.sort_values(["date", "user"], kind="mergesort").reset_index(drop=True)
    sm   = sm.sort_values(["date", "user"],   kind="mergesort").reset_index(drop=True)
    merged = pd.merge_asof(
        work, sm[["user", "date", "session_id", "session_date"]],
        on="date", by="user", direction="backward", allow_exact_matches=True,
    )
    return merged.sort_values("_orig_order", kind="mergesort").drop(columns=["_orig_order"])

session_event_labels = pd.DataFrame(columns=["session_id", "event_label"])
print("  [INFO] Using user-day fallback labels only")
print("  [WARN] user-day fallback: 동일 날짜의 정상 세션도 악성으로 라벨링됨 (노이즈 존재)")

# ── 3. 소스별 피처 추출 ───────────────────────────────────────────────────────
print("\n[3] Extracting features per source...")

def extract_logon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_session(df, session_map)
    df["hour"]          = df["date"].dt.hour
    df["is_after"]      = (~df["date"].dt.hour.between(BUSINESS_START, BUSINESS_END-1)).astype(int)
    df["is_weekend"]    = (df["date"].dt.dayofweek >= 5).astype(int)
    df["act_lower"]     = df["activity"].fillna("").str.lower()
    df["is_logon"]      = (~df["act_lower"].str.contains("off|after")).astype(int)
    df["is_logoff"]     = df["act_lower"].str.contains("off").astype(int)
    df["is_after_logon"]= df["act_lower"].str.contains("after").astype(int)
    g    = df.groupby("session_id")
    feat = g.agg(
        n_logon        =("is_logon",      "sum"),
        n_logoff       =("is_logoff",     "sum"),
        n_after_logon  =("is_after_logon","sum"),
        n_after_hours  =("is_after",      "sum"),
        n_weekend_ev   =("is_weekend",    "sum"),
        logon_hour_mean=("hour","mean"), logon_hour_std=("hour","std"),
        logon_hour_min =("hour","min"),  logon_hour_max=("hour","max"),
        n_logon_events =("date","count"),
    ).fillna(0)
    def interval_feat(dates: pd.Series) -> pd.Series:
        d = dates.sort_values()
        if len(d) < 2:
            return pd.Series({"logon_interval_mean":0.,"logon_interval_std":0.,
                              "logon_interval_max":0.,"logon_interval_skew":0.})
        delta = d.diff().dt.total_seconds().dropna()
        return pd.Series({
            "logon_interval_mean": float(delta.mean()),
            "logon_interval_std":  float(delta.std()) if len(delta)>1 else 0.,
            "logon_interval_max":  float(delta.max()),
            "logon_interval_skew": float(stats.skew(delta)) if len(delta)>2 else 0.,
        })
    interval = g["date"].apply(interval_feat).unstack().fillna(0)
    return pd.concat([feat, interval], axis=1)

print("  Logon features...")
logon_feat = extract_logon_features(dfs["logon"])

def extract_device_features(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_session(df, session_map).dropna(subset=["session_id"])
    df["act_lower"]    = df["activity"].fillna("").str.lower()
    df["is_connect"]   = df["act_lower"].str.contains("connect") & ~df["act_lower"].str.contains("disconnect")
    df["is_disconnect"]= df["act_lower"].str.contains("disconnect")
    df["hour"]         = df["date"].dt.hour
    df["is_after"]     = (~df["date"].dt.hour.between(BUSINESS_START, BUSINESS_END-1)).astype(int)
    g    = df.groupby("session_id")
    feat = g.agg(
        n_device_conn  =("is_connect",   "sum"),
        n_device_disc  =("is_disconnect","sum"),
        n_device_after =("is_after",     "sum"),
        device_hour_mean=("hour","mean"), device_hour_std=("hour","std"),
        n_device_events=("date","count"),
    ).fillna(0)
    feat["device_conn_ratio"] = feat["n_device_conn"] / (feat["n_device_events"] + 1e-9)
    return feat

print("  Device features...")
device_feat = extract_device_features(dfs["device"])

def extract_email_features(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_session(df, session_map).dropna(subset=["session_id"])
    df["hour"]         = df["date"].dt.hour
    df["is_after"]     = (~df["date"].dt.hour.between(BUSINESS_START, BUSINESS_END-1)).astype(int)
    df["size"]         = pd.to_numeric(df.get("size", 0), errors="coerce").fillna(0)
    df["attach"]       = pd.to_numeric(df.get("attachments", 0), errors="coerce").fillna(0)
    df["from"]         = df.get("from", pd.Series("", index=df.index)).fillna("")
    df["to"]           = df.get("to",   pd.Series("", index=df.index)).fillna("")
    df["is_external"]  = ~df["from"].str.lower().str.contains("dtaa\\.com", na=False)
    df["n_recipients"] = df["to"].str.count(";") + 1
    g    = df.groupby("session_id")
    feat = g.agg(
        n_email          =("date","count"),
        n_email_external =("is_external","sum"),
        n_email_attach   =("attach", lambda x: (x>0).sum()),
        email_size_sum   =("size","sum"), email_size_max=("size","max"),
        email_size_mean  =("size","mean"), email_size_std=("size","std"),
        n_after_email    =("is_after","sum"),
        n_recipients_sum =("n_recipients","sum"), n_recipients_max=("n_recipients","max"),
        email_hour_mean  =("hour","mean"), email_hour_std=("hour","std"),
        n_unique_from    =("from","nunique"),
    ).fillna(0)
    feat["email_external_ratio"] = feat["n_email_external"] / (feat["n_email"] + 1e-9)
    feat["email_attach_ratio"]   = feat["n_email_attach"]   / (feat["n_email"] + 1e-9)
    return feat

print("  Email features...")
email_feat = extract_email_features(dfs["email"])

def extract_file_features(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_session(df, session_map).dropna(subset=["session_id"])
    df["hour"]          = df["date"].dt.hour
    df["is_after"]      = (~df["date"].dt.hour.between(BUSINESS_START, BUSINESS_END-1)).astype(int)
    df["fname"]         = df.get("filename", pd.Series("", index=df.index)).fillna("").str.lower()
    df["is_exe"]        = df["fname"].str.contains(r"\.exe|\.bat|\.ps1|\.cmd", regex=True)
    df["is_doc"]        = df["fname"].str.contains(r"\.doc|\.pdf|\.xls|\.ppt", regex=True)
    df["is_compressed"] = df["fname"].str.contains(r"\.zip|\.rar|\.7z|\.gz",   regex=True)
    df["is_image"]      = df["fname"].str.contains(r"\.jpg|\.png|\.bmp|\.gif", regex=True)
    df["path_depth"]    = df["fname"].str.count(r"[/\\]")
    g    = df.groupby("session_id")
    feat = g.agg(
        n_file           =("date","count"),
        n_file_exe       =("is_exe","sum"),       n_file_doc=("is_doc","sum"),
        n_file_compressed=("is_compressed","sum"), n_file_image=("is_image","sum"),
        n_after_file     =("is_after","sum"),
        file_hour_mean   =("hour","mean"),         file_hour_std=("hour","std"),
        path_depth_mean  =("path_depth","mean"),   path_depth_max=("path_depth","max"),
        n_unique_files   =("fname","nunique"),
    ).fillna(0)
    feat["file_exe_ratio"]   = feat["n_file_exe"]   / (feat["n_file"] + 1e-9)
    feat["file_doc_ratio"]   = feat["n_file_doc"]   / (feat["n_file"] + 1e-9)
    feat["file_after_ratio"] = feat["n_after_file"] / (feat["n_file"] + 1e-9)
    return feat

print("  File features...")
file_feat = extract_file_features(dfs["file"])

def extract_http_features(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_session(df, session_map).dropna(subset=["session_id"])
    df["hour"]       = df["date"].dt.hour
    df["is_after"]   = (~df["date"].dt.hour.between(BUSINESS_START, BUSINESS_END-1)).astype(int)
    df["url"]        = df.get("url", pd.Series("", index=df.index)).fillna("").str.lower()
    df["is_upload"]  = df["url"].str.contains("upload|drive|dropbox|mega|box\\.com|onedrive", na=False)
    df["is_job_site"]= df["url"].str.contains("linkedin|monster|indeed|glassdoor|career|job|recruit", na=False)
    df["is_cloud"]   = df["url"].str.contains("dropbox|drive|onedrive|box\\.com|mega|wetransfer", na=False)

    def extract_domain(url: str) -> str:
        try:
            d = urlparse(url).netloc
            return d if d else str(url).split("/")[0]
        except Exception:
            return ""

    df["domain"]     = df["url"].apply(extract_domain)
    df["url_len"]    = df["url"].str.len()

    g    = df.groupby("session_id")
    feat = g.agg(
        n_http          =("date","count"),
        n_upload        =("is_upload","sum"),
        n_after_http    =("is_after","sum"),
        n_job_site      =("is_job_site","sum"),
        n_cloud         =("is_cloud","sum"),
        http_hour_mean  =("hour","mean"),    http_hour_std=("hour","std"),
        n_unique_domain =("domain","nunique"),
        url_len_mean    =("url_len","mean"), url_len_max=("url_len","max"),
    ).fillna(0)

    # 도메인 엔트로피 (다양성)
    def domain_entropy(domains: pd.Series) -> float:
        vc = domains.value_counts(normalize=True)
        return float(-(vc * np.log2(vc + 1e-9)).sum())

    feat["domain_entropy"] = g["domain"].apply(domain_entropy)

    feat["upload_ratio"]   = feat["n_upload"]   / (feat["n_http"] + 1e-9)
    feat["job_site_ratio"] = feat["n_job_site"] / (feat["n_http"] + 1e-9)
    feat["cloud_ratio"]    = feat["n_cloud"]    / (feat["n_http"] + 1e-9)
    return feat

print("  HTTP features...")
http_feat = extract_http_features(dfs["http"])

# ── 4. Cross-source 전환 피처 ─────────────────────────────────────────────────
print("\n[4] Extracting cross-source transition features...")

def extract_transition_features(dfs, session_map):
    frames = []
    for key, df in dfs.items():
        sub = assign_session(df, session_map).dropna(subset=["session_id"])
        frames.append(sub[["session_id", "date", "source"]].copy())
    log = pd.concat(frames, ignore_index=True).sort_values(["session_id", "date"]).reset_index(drop=True)
    log["prev_source"] = log.groupby("session_id")["source"].shift(1)
    log["transition"]  = log["prev_source"].fillna("") + "_to_" + log["source"].astype(str)
    key_transitions = [
        "device_to_file", "file_to_http", "logon_to_device",
        "email_to_http",  "device_to_email",
    ]
    g      = log.groupby("session_id")["transition"]
    result = {f"n_trans_{t}": g.apply(lambda x, t=t: (x == t).sum()) for t in key_transitions}
    return pd.DataFrame(result).fillna(0)

trans_feat = extract_transition_features(dfs, session_map)
print(f"  Transition features: {trans_feat.shape[1]} cols")

# ── 5. 세션 메타 피처 ─────────────────────────────────────────────────────────
print("\n[5] Extracting session meta features...")

session_meta = session_map.groupby("session_id").agg(
    user        =("user", "first"),
    session_date=("session_date", "first"),
    session_dur =("date", lambda x: (x.max()-x.min()).total_seconds()/3600 if len(x) else 0.),
    day_of_week =("date", lambda x: int(x.iloc[0].dayofweek)),
    is_weekend  =("date", lambda x: int(x.iloc[0].dayofweek >= 5)),
).reset_index()

# ── 6. 피처 합치기 ────────────────────────────────────────────────────────────
print("\n[6] Merging all features...")

unified = session_meta.set_index("session_id")
for name, feat in [
    ("logon", logon_feat), ("device", device_feat), ("email", email_feat),
    ("file",  file_feat),  ("http",   http_feat),   ("trans", trans_feat),
]:
    unified = unified.join(feat, how="left")
    print(f"  + {name}: {feat.shape[1]} features")

unified = unified.fillna(0).reset_index()
print(f"  Total features: {unified.shape[1]}")

# ── 7. 델타 피처 (누수 수정) ──────────────────────────────────────────────────
print("\n[7] Adding delta features (leak-free: expanding past mean)...")

DELTA_COLS = [
    "n_device_conn", "n_file", "n_email_external", "n_upload",
    "email_size_sum", "n_after_hours", "n_job_site", "n_cloud",
    "n_unique_domain", "n_trans_device_to_file", "n_trans_file_to_http",
]
DELTA_COLS = [c for c in DELTA_COLS if c in unified.columns]

unified = unified.sort_values(["user", "session_date"]).reset_index(drop=True)

# [FIX] expanding mean with shift(1): 현재 세션 제외, 과거 세션 평균만 사용
#       → val fold에 미래 정보 유입 없음
user_expanding = (
    unified.groupby("user")[DELTA_COLS]
    .transform(lambda x: x.expanding().mean().shift(1))
    .fillna(0)
)
for col in DELTA_COLS:
    unified[f"delta_{col}"] = unified[col] - user_expanding[col]

print(f"  Delta features added: {len(DELTA_COLS)} (expanding past mean, no leakage)")

# ── 8. 라벨 병합 ──────────────────────────────────────────────────────────────
print("\n[8] Merging labels...")

unified = unified.merge(session_event_labels, on="session_id", how="left")
unified["event_label"] = unified["event_label"].fillna(0).astype(int)

unified["user_key"]         = normalize_user_id(unified["user"])
unified["session_date_key"] = normalize_date_series(unified["session_date"])

if len(label_df) > 0:
    label_df = label_df.copy()
    label_df["user_key"]         = normalize_user_id(label_df["user"])
    label_df["session_date_key"] = normalize_date_series(label_df["session_date"])
    diagnose_label_overlap(unified, label_df)
    label_day = (
        label_df[["user_key", "session_date_key", "is_malicious"]]
        .dropna(subset=["user_key", "session_date_key"])
        .drop_duplicates()
        .rename(columns={"is_malicious": "day_label"})
    )
    unified = unified.merge(label_day, on=["user_key", "session_date_key"], how="left")
else:
    unified["day_label"] = 0

unified["day_label"]   = unified["day_label"].fillna(0).astype(int)
unified["is_malicious"]= np.maximum(unified["event_label"], unified["day_label"]).astype(int)

n_mal = int(unified["is_malicious"].sum())
print(f"  Event-labeled : {int(unified['event_label'].sum()):,}")
print(f"  Day-fallback  : {int(unified['day_label'].sum()):,}")
print(f"  Final malicious: {n_mal:,} / {len(unified):,} ({n_mal/len(unified)*100:.3f}%)")

if n_mal == 0 and len(label_df) > 0:
    raise RuntimeError("Malicious count = 0. Check answer file format.")

# ── 9. 샘플 가중치 ────────────────────────────────────────────────────────────
n_neg = int((unified["is_malicious"] == 0).sum())
n_pos = max(n_mal, 1)
w_pos = float(n_neg) / float(n_pos)
unified["sample_weight"] = unified["is_malicious"].map({0: 1.0, 1: w_pos})
print(f"  Sample weight pos: {w_pos:.1f}")

# ── 10. 스키마 저장 ───────────────────────────────────────────────────────────
print("\n[9] Building feature schema...")

EXCLUDE = {
    "session_id","user","session_date","session_date_str",
    "is_malicious","sample_weight","user_key","session_date_key",
    "event_label","day_label",
}
FEATURE_COLS = [c for c in unified.columns if c not in EXCLUDE]
NUM_COLS     = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(unified[c])]
CAT_COLS     = [c for c in FEATURE_COLS if c not in NUM_COLS]

def infer_feature_rule(col: str) -> dict:
    if col.endswith("_ratio"):      return {"type":"ratio","clip_min":0.,"clip_max":1.,"round":False}
    if col == "is_weekend":         return {"type":"binary","clip_min":0,"clip_max":1,"round":True}
    if col == "day_of_week":        return {"type":"calendar_int","clip_min":0,"clip_max":6,"round":True}
    if "_hour_mean" in col or "_hour_min" in col or "_hour_max" in col:
        return {"type":"hour","clip_min":0.,"clip_max":23.,"round":False}
    if "_hour_std" in col:          return {"type":"hour_std","clip_min":0.,"clip_max":24.,"round":False}
    if "interval_" in col or col == "session_dur": return {"type":"duration","clip_min":0.,"round":False}
    if col.startswith("n_") or col.startswith("delta_n_"):
        return {"type":"delta_count","round":False} if col.startswith("delta_") else {"type":"count","clip_min":0.,"round":True}
    if col.startswith("path_depth") or col.startswith("n_unique_"): return {"type":"count_like","clip_min":0.,"round":True}
    if col.startswith("delta_"):    return {"type":"delta","round":False}
    return {"type":"continuous","round":False}

feature_schema = {
    "version": 2,
    "notes": {
        "delta_fix": "expanding past mean (shift=1), no future leakage",
        "train_input": "raw scale for CatBoost",
    },
    "feature_columns":     FEATURE_COLS,
    "numeric_columns":     NUM_COLS,
    "categorical_columns": CAT_COLS,
    "rules": {col: infer_feature_rule(col) for col in FEATURE_COLS},
}

# ── 11. 저장 ──────────────────────────────────────────────────────────────────
print("\n[10] Saving...")

session_key = unified[["session_id","user","session_date","is_malicious","sample_weight"]].copy()
session_key.to_csv(OUTPUT_DIR / "session_key.csv", index=False)
unified.to_csv(OUTPUT_DIR / "unified_features_raw.csv", index=False)
with open(OUTPUT_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
    json.dump(feature_schema, f, ensure_ascii=False, indent=2)

print(f"  unified_features_raw.csv : {unified.shape}")
print(f"  session_key.csv          : {len(session_key):,} rows")
print(f"  feature_schema.json      : {len(FEATURE_COLS)} features")
print("\n" + "=" * 60)
print("Done! (v4 — delta leak fixed)")
print("=" * 60)
