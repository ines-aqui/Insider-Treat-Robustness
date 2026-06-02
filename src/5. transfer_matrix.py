"""
transfer_matrix_full_v2.py
===========================
Transfer Matrix 확장 버전

변경 사항:
  - 공격 방법: v6 8종 + v7 6종 = 14종으로 확장
    (shap_linesearch, global_linesearch, local_only_linesearch,
     shap_multistart, global_only_multistart, local_only_multistart)
  - Arm: A~G 기존 7개 + I(shap_linesearch 재학습) 추가 = 8개

공격 파일 위치:
  v6: output/attack/attacked_samples_{method}_B{budget}.csv
  v7: output/attack_six_methods/attacked_samples_{method}_B{budget}.csv

Arm 모델 위치:
  A:   output/unified/catboost/fold_k/catboost_model.cbm
  B~D: output/retraining/fold_k/{arm}/model.cbm
  E:   output/retraining_top2mix/fold_k/E_top2mix_sqrtasr/model.cbm
  F:   output/retraining_top3mix/fold_k/E_top3mix_sqrtasr/model.cbm
  G:   output/retraining_hardness/fold_k/G_hardness_mix/model.cbm
  I:   output/retraining_v9/fold_k/I_shap_linesearch/model.cbm
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# GPU 0번 고정 (RTX 5090 두 장 중 더 여유 있는 GPU 0 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

CFG = {
    "data_dir":        Path("./output/unified"),
    "attack_dir_v6":   Path("./output/attack"),
    "attack_dir_v7":   Path("./output/attack_six_methods"),
    "output_dir":      Path("./output/transfer_full_v2"),
    "tau":             0.5,
    "n_folds":         5,
    "budgets":         [1, 2, 3, 5],
}

CFG["output_dir"].mkdir(parents=True, exist_ok=True)

# ── 공격 방법 정의 ────────────────────────────────────────────────────────────
# (method_name, source_dir, 분류)
ATTACK_METHODS = [
    # v6 8종
    ("shap_guided",           "v6", "SHAP"),
    ("global_only",           "v6", "SHAP"),
    ("local_only",            "v6", "SHAP"),
    ("global_only_random",    "v6", "SHAP"),
    ("local_only_random",     "v6", "SHAP"),
    ("direction_random",      "v6", "SHAP"),
    ("random",                "v6", "nonSHAP"),
    ("pure_random",           "v6", "nonSHAP"),
    # v7 6종
    ("shap_linesearch",       "v7", "SHAP"),
    ("global_linesearch",     "v7", "SHAP"),
    ("local_only_linesearch", "v7", "SHAP"),
    ("shap_multistart",       "v7", "SHAP"),
    ("global_only_multistart","v7", "SHAP"),
    ("local_only_multistart", "v7", "SHAP"),
]

SHAP_METHODS   = [m for m, _, c in ATTACK_METHODS if c == "SHAP"]
NOSHAP_METHODS = [m for m, _, c in ATTACK_METHODS if c == "nonSHAP"]
ALL_METHODS    = [m for m, _, _ in ATTACK_METHODS]

# ── Arm 정의 ─────────────────────────────────────────────────────────────────
RETRAIN_ARMS = {
    "A_clean_baseline": {
        "model_dirs": [Path(f"./output/unified/catboost/fold_{i}") for i in range(5)],
        "model_name": "catboost_model.cbm",
        "desc": "Clean baseline",
        "src_asr": "-",
    },
    "B_global_only": {
        "model_dirs": [Path(f"./output/retraining/fold_{i}/B_global_only") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "global_only 재학습 (ASR=0.606)",
        "src_asr": "0.606",
    },
    "C_global_only_random": {
        "model_dirs": [Path(f"./output/retraining/fold_{i}/C_global_only_random") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "global_only_random 재학습 (ASR=0.758)",
        "src_asr": "0.758",
    },
    "D_direction_random": {
        "model_dirs": [Path(f"./output/retraining/fold_{i}/D_direction_random") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "direction_random 재학습 (ASR=0.767)",
        "src_asr": "0.767",
    },
    "E_top2mix": {
        "model_dirs": [Path(f"./output/retraining_top2mix/fold_{i}/E_top2mix_sqrtasr") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "top2mix sqrt-ASR 재학습",
        "src_asr": "mix2",
    },
    "F_top3mix": {
        "model_dirs": [Path(f"./output/retraining_top3mix/fold_{i}/E_top3mix_sqrtasr") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "top3mix sqrt-ASR 재학습",
        "src_asr": "mix3",
    },
    "G_hardness_mix": {
        "model_dirs": [Path(f"./output/retraining_hardness/fold_{i}/G_hardness_mix") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "hardness-weighted mix 재학습",
        "src_asr": "hardness",
    },
    "I_shap_linesearch": {
        "model_dirs": [Path(f"./output/retraining_v9/fold_{i}/I_shap_linesearch") for i in range(5)],
        "model_name": "model.cbm",
        "desc": "shap_linesearch 재학습 (ASR=0.863)",
        "src_asr": "0.863",
    },
}

print("=" * 72)
print("Transfer Matrix Full v2")
print(f"  Arms: {len(RETRAIN_ARMS)}  Methods: {len(ALL_METHODS)} (v6 8종 + v7 6종)  Budgets: {CFG['budgets']}")
print("=" * 72)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
unified = pd.read_csv(CFG["data_dir"] / "unified_features_raw.csv", low_memory=False)
DROP_COLS = {
    "session_id", "user", "session_date", "session_date_str",
    "is_malicious", "sample_weight",
    "user_key", "session_date_key", "event_label", "day_label",
}
feature_cols = [c for c in unified.columns if c not in DROP_COLS]
X     = unified[feature_cols].values.astype(np.float64)
y     = unified["is_malicious"].values.astype(int)
users = unified["user"].values
tau   = CFG["tau"]

gkf   = GroupKFold(n_splits=CFG["n_folds"])
folds = list(gkf.split(np.arange(len(y)), y, groups=users))
print(f"  samples={len(X):,}  features={len(feature_cols)}  malicious={y.sum():,}")

# ── 공격 샘플 로드 ────────────────────────────────────────────────────────────
print("\n[2] Loading attack samples...")
attack_samples = {}
for method, src, _ in ATTACK_METHODS:
    attack_samples[method] = {}
    attack_dir = CFG["attack_dir_v6"] if src == "v6" else CFG["attack_dir_v7"]
    for budget in CFG["budgets"]:
        path = attack_dir / f"attacked_samples_{method}_B{budget}.csv"
        if path.exists():
            attack_samples[method][budget] = pd.read_csv(path, low_memory=False)
        else:
            print(f"  [WARN] {method} B={budget} not found")
            attack_samples[method][budget] = pd.DataFrame()

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_arm_models(arm_cfg):
    models = []
    for fi, model_dir in enumerate(arm_cfg["model_dirs"]):
        model_path = model_dir / arm_cfg["model_name"]
        m = CatBoostClassifier()
        if model_path.exists():
            m.load_model(str(model_path))
        else:
            print(f"  [WARN] {model_path} not found → fallback to init")
            m.load_model(str(Path(f"./output/unified/catboost/fold_{fi}/catboost_model.cbm")))
        models.append(m)
    return models

# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def evaluate(models, method, budget):
    adv_df = attack_samples[method][budget]
    if len(adv_df) == 0:
        return None

    fold_asrs, fold_auprc, fold_tpr1 = [], [], []

    for fold_idx, (_, val_idx) in enumerate(folds):
        model = models[fold_idx]
        X_va  = X[val_idx]
        y_va  = y[val_idx]

        probs = model.predict_proba(X_va)[:, 1]
        auprc = average_precision_score(y_va, probs) if y_va.sum() > 0 else 0.
        fpr_a, tpr_a, _ = roc_curve(y_va, probs)
        valid = np.where(fpr_a <= 0.01)[0]
        tpr1  = float(tpr_a[valid[-1]]) if len(valid) > 0 else 0.
        fold_auprc.append(auprc)
        fold_tpr1.append(tpr1)

        # val fold 공격 샘플
        val_sids = set()
        if "session_id" in unified.columns:
            val_sids = set(unified.iloc[val_idx]["session_id"].tolist())
        if "session_id" in adv_df.columns and val_sids:
            adv_filt = adv_df[adv_df["session_id"].isin(val_sids)]
        elif "fold" in adv_df.columns:
            adv_filt = adv_df[adv_df["fold"] == fold_idx]
        else:
            adv_filt = adv_df

        if len(adv_filt) == 0:
            fold_asrs.append(np.nan); continue

        adv_X    = adv_filt[feature_cols].values.astype(np.float64)
        adv_y    = adv_filt["is_malicious"].values.astype(int)
        mal_mask = adv_y == 1
        if mal_mask.sum() == 0:
            fold_asrs.append(np.nan); continue

        ap  = model.predict_proba(adv_X)[:, 1]
        asr = float((ap[mal_mask] < tau).mean())
        fold_asrs.append(asr)

    valid_asrs = [a for a in fold_asrs if not np.isnan(a)]
    return {
        "def_asr_mean":  round(float(np.mean(valid_asrs)),   4) if valid_asrs else None,
        "def_asr_worst": round(float(np.max(valid_asrs)),    4) if valid_asrs else None,
        "clean_auprc":   round(float(np.mean(fold_auprc)),   4),
        "tpr_at_fpr1":   round(float(np.mean(fold_tpr1)),    4),
    }

# ── 메인 실험 ─────────────────────────────────────────────────────────────────
print("\n[3] Running evaluation...")
all_results = {}

for arm_name, arm_cfg in RETRAIN_ARMS.items():
    print(f"\n  ── {arm_name} ({arm_cfg['desc']}) ──")
    models = load_arm_models(arm_cfg)
    arm_results = {}
    for method in ALL_METHODS:
        arm_results[method] = {}
        for budget in CFG["budgets"]:
            res = evaluate(models, method, budget)
            arm_results[method][budget] = res
            if res:
                print(f"    {method:<26} B={budget}  "
                      f"def_ASR={res['def_asr_mean']:.4f}  "
                      f"AUPRC={res['clean_auprc']:.4f}")
    all_results[arm_name] = arm_results

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
print("\n[4] Saving results...")

# 전체 ASR 행렬
rows_asr = []
for arm_name in RETRAIN_ARMS:
    row = {"arm": arm_name}
    for method in ALL_METHODS:
        for budget in CFG["budgets"]:
            res = all_results[arm_name][method][budget]
            row[f"{method}_B{budget}"] = res["def_asr_mean"] if res else None
    rows_asr.append(row)
pd.DataFrame(rows_asr).to_csv(CFG["output_dir"] / "transfer_matrix_asr.csv", index=False)

# 방법별 분류 집계
V6_SHAP    = ["shap_guided","global_only","local_only",
              "global_only_random","local_only_random","direction_random"]
V7_SHAP    = ["shap_linesearch","global_linesearch","local_only_linesearch",
              "shap_multistart","global_only_multistart","local_only_multistart"]
NOSHAP     = ["random","pure_random"]

rows_summary = []
for arm_name in RETRAIN_ARMS:
    ref = all_results[arm_name]["shap_guided"][3]
    row = {
        "arm":         arm_name,
        "desc":        RETRAIN_ARMS[arm_name]["desc"],
        "src_asr":     RETRAIN_ARMS[arm_name]["src_asr"],
        "clean_auprc": ref["clean_auprc"] if ref else None,
        "tpr_at_fpr1": ref["tpr_at_fpr1"] if ref else None,
    }
    for label, methods in [
        ("v6shap",   V6_SHAP),
        ("v7shap",   V7_SHAP),
        ("noshap",   NOSHAP),
        ("overall",  ALL_METHODS),
    ]:
        for B in [3, 5]:
            vals = [all_results[arm_name][m][B]["def_asr_mean"]
                    for m in methods
                    if all_results[arm_name][m][B] is not None
                    and all_results[arm_name][m][B]["def_asr_mean"] is not None]
            row[f"{label}_avg_B{B}"] = round(float(np.mean(vals)), 4) if vals else None
    rows_summary.append(row)

df_summary = pd.DataFrame(rows_summary)
df_summary.to_csv(CFG["output_dir"] / "transfer_matrix_summary.csv", index=False)

# ── 최종 출력 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Transfer Matrix Full v2 Done!")
print("=" * 80)
print(f"\n  {'Arm':<24} {'AUPRC':>7} {'v6SHAP B3':>10} {'v7SHAP B3':>10} "
      f"{'nonSHAP B3':>11} {'overall B3':>11} {'overall B5':>11}")
print("  " + "-" * 86)

base_overall = None
for _, row in df_summary.iterrows():
    if row["arm"] == "A_clean_baseline":
        base_overall = row["overall_avg_B3"]
    marker = ""
    if base_overall and row["arm"] != "A_clean_baseline":
        delta = (row["overall_avg_B3"] or 0.) - base_overall
        marker = f"  ({delta:+.3f})"
    print(f"  {row['arm']:<24} "
          f"{row['clean_auprc'] or 0.:>7.4f} "
          f"{row['v6shap_avg_B3'] or 0.:>10.4f} "
          f"{row['v7shap_avg_B3'] or 0.:>10.4f} "
          f"{row['noshap_avg_B3'] or 0.:>11.4f} "
          f"{row['overall_avg_B3'] or 0.:>11.4f} "
          f"{row['overall_avg_B5'] or 0.:>11.4f}"
          f"{marker}")

print(f"\n[핵심 질문: 일반화된 방어인가?]")
base = df_summary[df_summary["arm"] == "A_clean_baseline"].iloc[0]
for _, row in df_summary.iterrows():
    if row["arm"] == "A_clean_baseline": continue
    dv6   = (row["v6shap_avg_B3"] or 0.) - (base["v6shap_avg_B3"] or 0.)
    dv7   = (row["v7shap_avg_B3"] or 0.) - (base["v7shap_avg_B3"] or 0.)
    dns   = (row["noshap_avg_B3"] or 0.) - (base["noshap_avg_B3"] or 0.)
    gen   = abs(dns) > 0.05
    print(f"  {row['arm']:<24}: "
          f"v6SHAP {dv6:+.3f}  v7SHAP {dv7:+.3f}  nonSHAP {dns:+.3f}  "
          f"→ {'일반화된 방어' if gen else '부분 방어'}")

print(f"\n  결과 저장: {CFG['output_dir']}")
print("=" * 80)
