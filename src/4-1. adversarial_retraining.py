"""
adversarial_retraining_v9.py
==============================
4순위 실험(방향 탐색 개선) 결과 기반 재학습 재구성

핵심 인사이트:
  - step_ratio 고정이 병목이었음
  - linesearch(±방향 × step grid)가 모든 v6 공격을 압도
  - shap_linesearch(0.863) ≈ global_linesearch(0.861)
    → 피처 선택보다 방향 탐색이 더 중요

5-Arm 구성:
  A: clean baseline
  D: direction_random B=3        (v6 기존 최강, 비교 기준)
  I: shap_linesearch B=3         (v7 신규 최강 SHAP 기반)
  J: global_linesearch B=3       (v7 신규 최강 global 기반)
  K: shap+global linesearch mix  (sqrt-ASR weighting)

공격 파일 위치:
  D: output/attack/attacked_samples_direction_random_B3.csv
  I,J,K: output/attack_six_methods/attacked_samples_{method}_B3.csv
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings("ignore")

CFG = {
    "data_dir":          Path("./output/unified"),
    "model_dir":         Path("./output/unified/catboost"),
    "attack_dir_v6":     Path("./output/attack"),             # direction_random
    "attack_dir_v7":     Path("./output/attack_six_methods"), # linesearch
    "output_dir":        Path("./output/retraining_v9"),

    "adv_inject_ratio":  0.05,
    "attack_budget":     3,

    # K arm mix 가중치 (sqrt-ASR)
    "mix_attacks": {
        "shap_linesearch":   0.8626,
        "global_linesearch": 0.8613,
    },

    "iterations":            2000,
    "learning_rate":         0.05,
    "depth":                 8,
    "early_stopping_rounds": 100,
    "eval_metric":           "Logloss",
    "loss_function":         "Logloss",
    "task_type":             "GPU",
    "devices":               "0",
    "tau":                   0.5,
    "n_folds":               5,
    "random_seed":           42,
}

CFG["output_dir"].mkdir(parents=True, exist_ok=True)
np.random.seed(CFG["random_seed"])

print("=" * 65)
print("Adversarial Retraining v9 (linesearch-based)")
print("=" * 65)
print("  A: clean baseline")
print("  D: direction_random     (v6 기존 최강, 비교)")
print("  I: shap_linesearch      (v7 신규, ASR=0.863)")
print("  J: global_linesearch    (v7 신규, ASR=0.861)")
print("  K: shap+global mix      (sqrt-ASR weighting)")

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
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
w     = unified["sample_weight"].values.astype(np.float64)
users = unified["user"].values
tau   = CFG["tau"]

with open(CFG["model_dir"] / "catboost_meta.pkl", "rb") as f:
    meta = pickle.load(f)
print(f"  Fold AUPRCs: {[f'{v:.4f}' for v in meta['fold_auprcs']]}")

# ── 2. 공격 샘플 로드 ─────────────────────────────────────────────────────────
print("\n[2] Loading attack samples...")

B = CFG["attack_budget"]

def load_adv(path):
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        print(f"  {path.name:<40} {len(df):,}개")
        return df
    print(f"  [WARN] {path} not found")
    return pd.DataFrame()

adv_D = load_adv(CFG["attack_dir_v6"] / f"attacked_samples_direction_random_B{B}.csv")
adv_I = load_adv(CFG["attack_dir_v7"] / f"attacked_samples_shap_linesearch_B{B}.csv")
adv_J = load_adv(CFG["attack_dir_v7"] / f"attacked_samples_global_linesearch_B{B}.csv")

# K: mix 가중치 계산
sqrt_sum   = sum(np.sqrt(v) for v in CFG["mix_attacks"].values())
mix_weights = {k: float(np.sqrt(v) / sqrt_sum) for k, v in CFG["mix_attacks"].items()}
print(f"\n  K mix weights:")
for k, v in mix_weights.items():
    print(f"    {k:<24} {v:.4f}")

# ── 3. 재학습 데이터 구성 ─────────────────────────────────────────────────────
def build_single(X_tr, y_tr, w_tr, adv_df, exclude_fold):
    if len(adv_df) == 0:
        return X_tr, y_tr, w_tr, 0
    adv_clean = adv_df[adv_df["fold"] != exclude_fold].copy() \
        if "fold" in adv_df.columns else adv_df.copy()
    if len(adv_clean) == 0:
        return X_tr, y_tr, w_tr, 0
    n_inject   = int(len(X_tr) * CFG["adv_inject_ratio"])
    adv_X      = adv_clean[feature_cols].values.astype(np.float64)
    mal_idx    = np.where(y_tr == 1)[0]
    w_mal_mean = float(w_tr[mal_idx].mean()) if len(mal_idx) > 0 else 1.0
    idx        = np.random.choice(len(adv_X), min(n_inject, len(adv_X)), replace=True)
    X_ret = np.vstack([X_tr, adv_X[idx]])
    y_ret = np.concatenate([y_tr, np.ones(len(idx), dtype=int)])
    w_ret = np.concatenate([w_tr, np.full(len(idx), w_mal_mean)])
    perm  = np.random.permutation(len(X_ret))
    return X_ret[perm], y_ret[perm], w_ret[perm], len(idx)

def build_mix(X_tr, y_tr, w_tr, adv_pool, weights, exclude_fold):
    """sqrt-ASR 비율로 여러 공격 혼합 주입"""
    n_inject_total = int(len(X_tr) * CFG["adv_inject_ratio"])
    mal_idx        = np.where(y_tr == 1)[0]
    w_mal_mean     = float(w_tr[mal_idx].mean()) if len(mal_idx) > 0 else 1.0

    methods  = list(weights.keys())
    alloc    = {}
    remaining = n_inject_total
    for i, m in enumerate(methods):
        if i < len(methods) - 1:
            k = int(round(n_inject_total * weights[m]))
            alloc[m] = k
            remaining -= k
        else:
            alloc[m] = remaining

    parts_X, parts_y, parts_w = [X_tr], [y_tr], [w_tr]
    total_injected = 0
    for method, n_inj in alloc.items():
        adv_df = adv_pool.get(method, pd.DataFrame())
        if len(adv_df) == 0 or n_inj <= 0:
            continue
        adv_clean = adv_df[adv_df["fold"] != exclude_fold].copy() \
            if "fold" in adv_df.columns else adv_df.copy()
        if len(adv_clean) == 0:
            continue
        adv_X = adv_clean[feature_cols].values.astype(np.float64)
        idx   = np.random.choice(len(adv_X), min(n_inj, len(adv_X)), replace=True)
        parts_X.append(adv_X[idx])
        parts_y.append(np.ones(len(idx), dtype=int))
        parts_w.append(np.full(len(idx), w_mal_mean))
        total_injected += len(idx)

    X_ret = np.vstack(parts_X)
    y_ret = np.concatenate(parts_y)
    w_ret = np.concatenate(parts_w)
    perm  = np.random.permutation(len(X_ret))
    return X_ret[perm], y_ret[perm], w_ret[perm], total_injected

# ── 4. 평가 함수 ──────────────────────────────────────────────────────────────
def compute_metrics(model, X_test, y_test, adv_X=None, adv_y=None):
    probs = model.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, probs) if y_test.sum() > 0 else 0.
    auc   = roc_auc_score(y_test, probs)           if y_test.sum() > 0 else 0.
    fpr_a, tpr_a, _ = roc_curve(y_test, probs)
    valid = np.where(fpr_a <= 0.01)[0]
    tpr1  = float(tpr_a[valid[-1]]) if len(valid) > 0 else 0.
    bfpr  = float((probs[y_test==0] >= tau).mean()) if (y_test==0).sum() > 0 else 0.
    result = {"clean_auprc": round(auprc,4), "clean_auc": round(auc,4),
              "tpr_at_fpr1": round(tpr1,4),  "benign_fpr": round(bfpr,4)}
    if adv_X is not None and len(adv_X) > 0:
        ap = model.predict_proba(adv_X)[:, 1]
        mm = (adv_y==1) if adv_y is not None else np.ones(len(adv_X), bool)
        if mm.sum() > 0:
            result.update({
                "asr":           round(float((ap[mm] < tau).mean()), 4),
                "robust_recall": round(float((ap[mm] >= tau).mean()), 4),
            })
    return result

# ── 5. Arm 정의 ───────────────────────────────────────────────────────────────
ARM_CONFIGS = {
    "A_clean_baseline": {"type": "none",   "src_asr": "-",     "desc": "Clean baseline"},
    "D_direction_random":{"type": "single", "adv": adv_D,      "src_asr": "0.767", "desc": "v6 최강 (비교)"},
    "I_shap_linesearch": {"type": "single", "adv": adv_I,      "src_asr": "0.863", "desc": "v7 shap_linesearch"},
    "J_global_linesearch":{"type": "single","adv": adv_J,      "src_asr": "0.861", "desc": "v7 global_linesearch"},
    "K_linesearch_mix":  {"type": "mix",
                          "pool": {"shap_linesearch": adv_I, "global_linesearch": adv_J},
                          "weights": mix_weights,
                          "src_asr": "mix", "desc": "shap+global linesearch mix"},
}

# ── 6. 재학습 ─────────────────────────────────────────────────────────────────
print("\n[3] Running 5-Arm retraining (5-fold)...")

gkf   = GroupKFold(n_splits=CFG["n_folds"])
folds = list(gkf.split(np.arange(len(y)), y, groups=users))
all_fold_results = {arm: [] for arm in ARM_CONFIGS}

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\n  ── Fold {fold_idx+1}/{CFG['n_folds']} "
          f"(train={len(train_idx):,}  val={len(val_idx):,}  "
          f"val_pos={y[val_idx].sum()}) ──")

    X_tr, y_tr, w_tr = X[train_idx], y[train_idx], w[train_idx]
    X_va, y_va        = X[val_idx],   y[val_idx]
    val_sids = set(unified.iloc[val_idx]["session_id"].tolist()) \
        if "session_id" in unified.columns else set()

    for arm_name, arm_cfg in ARM_CONFIGS.items():
        arm_dir = CFG["output_dir"] / f"fold_{fold_idx}" / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 구성
        if arm_cfg["type"] == "none":
            X_ret, y_ret, w_ret, n_inj = X_tr, y_tr, w_tr, 0
        elif arm_cfg["type"] == "single":
            X_ret, y_ret, w_ret, n_inj = build_single(
                X_tr, y_tr, w_tr, arm_cfg["adv"], fold_idx)
        else:  # mix
            X_ret, y_ret, w_ret, n_inj = build_mix(
                X_tr, y_tr, w_tr, arm_cfg["pool"], arm_cfg["weights"], fold_idx)

        if fold_idx == 0:
            print(f"    [{arm_name}] inject={n_inj}")

        train_pool = Pool(X_ret, y_ret, weight=w_ret, feature_names=feature_cols)
        val_pool   = Pool(X_va,  y_va,                feature_names=feature_cols)

        model_ret = CatBoostClassifier(
            iterations            = CFG["iterations"],
            learning_rate         = CFG["learning_rate"],
            depth                 = CFG["depth"],
            early_stopping_rounds = CFG["early_stopping_rounds"],
            eval_metric           = CFG["eval_metric"],
            loss_function         = CFG["loss_function"],
            task_type             = CFG["task_type"],
            devices               = CFG["devices"],
            random_seed           = CFG["random_seed"],
            use_best_model        = True,
            verbose               = False,
        )
        model_ret.fit(train_pool, eval_set=val_pool)
        model_ret.save_model(str(arm_dir / "model.cbm"))

        # 평가용 공격 샘플 (I 기준: shap_linesearch val fold)
        adv_eval = adv_I
        if len(adv_eval) > 0 and val_sids:
            adv_filt = adv_eval
            if "session_id" in adv_filt.columns:
                adv_filt = adv_filt[adv_filt["session_id"].isin(val_sids)]
            adv_X_val = adv_filt[feature_cols].values.astype(np.float64) if len(adv_filt) > 0 else None
            adv_y_val = adv_filt["is_malicious"].values.astype(int)       if len(adv_filt) > 0 else None
        else:
            adv_X_val = adv_y_val = None

        metrics = compute_metrics(model_ret, X_va, y_va, adv_X_val, adv_y_val)
        metrics.update({"arm": arm_name, "fold": fold_idx+1,
                        "best_iter": model_ret.best_iteration_})
        all_fold_results[arm_name].append(metrics)

        print(f"    {arm_name:<24} "
              f"AUPRC={metrics['clean_auprc']:.4f}  "
              f"TPR@1%={metrics['tpr_at_fpr1']:.4f}  "
              f"ASR={metrics.get('asr',0.):.4f}  "
              f"bFPR={metrics['benign_fpr']:.4f}")

# ── 7. 집계 & 보고 ────────────────────────────────────────────────────────────
print("\n[4] Aggregating results...")

all_rows = []
for arm_name, fold_results in all_fold_results.items():
    df  = pd.DataFrame(fold_results)
    row = {"arm": arm_name, "src_asr": ARM_CONFIGS[arm_name]["src_asr"],
           "desc": ARM_CONFIGS[arm_name]["desc"]}
    for col in ["clean_auprc", "clean_auc", "tpr_at_fpr1", "benign_fpr", "asr", "robust_recall"]:
        if col in df.columns and df[col].notna().sum() > 0:
            vals = df[col].dropna().values
            row[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{col}_std"]  = round(float(np.std(vals)),  4)
        else:
            row[f"{col}_mean"] = row[f"{col}_std"] = None
    all_rows.append(row)
    df.to_csv(CFG["output_dir"] / f"fold_results_{arm_name}.csv", index=False)

results_df = pd.DataFrame(all_rows)
results_df.to_csv(CFG["output_dir"] / "retraining_v9_results.csv", index=False)

base_auprc = float(results_df[results_df["arm"]=="A_clean_baseline"]["clean_auprc_mean"].values[0])

print("\n" + "=" * 65)
print("Retraining v9 Done!")
print("=" * 65)
print(f"\n  {'Arm':<24} {'src_ASR':>8} {'AUPRC':>7} {'ΔAUPRC':>8} "
      f"{'TPR@1%':>7} {'def_ASR':>8} {'bFPR':>7}")
print("  " + "-" * 68)
for _, row in results_df.iterrows():
    auprc  = row.get("clean_auprc_mean") or 0.
    delta  = auprc - base_auprc
    marker = " ★" if row["arm"] != "A_clean_baseline" and delta >= -0.003 else ""
    print(f"  {row['arm']:<24} "
          f"{row['src_asr']:>8} "
          f"{auprc:>7.4f} "
          f"{delta:>+8.4f} "
          f"{row.get('tpr_at_fpr1_mean') or 0.:>7.4f} "
          f"{row.get('asr_mean') or 0.:>8.4f} "
          f"{row.get('benign_fpr_mean') or 0.:>7.4f}"
          f"{marker}")

print(f"\n  ★ = AUPRC 하락 -0.003 이내")
print(f"\n  [D vs I/J/K 비교]")
for arm in ["D_direction_random", "I_shap_linesearch", "J_global_linesearch", "K_linesearch_mix"]:
    row = results_df[results_df["arm"]==arm]
    if len(row):
        r = row.iloc[0]
        print(f"  {arm:<24} AUPRC={r.get('clean_auprc_mean',0.):.4f}  "
              f"ASR={r.get('asr_mean',0.):.4f}")
print("=" * 65)
