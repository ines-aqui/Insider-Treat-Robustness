"""
train_catboost_v2.py  (unified → v2 수정사항)
=============================================
[FIX 1] eval_metric "AUC" → "Logloss"
  - GPU에서 AUC 미지원으로 early stopping이 실질적으로 동작하지 않았음
  - Logloss로 변경하면 early stopping 정상 작동

[FIX 2] SHAP을 마지막 fold 모델 하나가 아니라 전체 fold 앙상블 평균으로 계산
  - 특정 fold에 편향된 SHAP이 아닌 더 안정적인 피처 중요도 산출

나머지 로직은 기존과 동일.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# ── 0. 설정 ───────────────────────────────────────────────────────────────────
CFG = {
    "data_dir":   Path("./output/unified"),
    "output_dir": Path("./output/unified/catboost"),

    "iterations":            3000,
    "learning_rate":         0.05,
    "depth":                 8,
    "l2_leaf_reg":           3.0,
    "bagging_temperature":   1.0,
    "random_strength":       1.0,
    "border_count":          254,
    "early_stopping_rounds": 100,
    "eval_metric":           "Logloss",   # [FIX] AUC → Logloss (GPU 호환)
    "loss_function":         "Logloss",

    "task_type":  "GPU",
    "devices":    "0",
    "n_folds":    5,
    "split_mode": "group",
    "random_seed": 42,
}

CFG["output_dir"].mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CatBoost Unified Training v2")
print("=" * 60)

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
unified     = pd.read_csv(CFG["data_dir"] / "unified_features_raw.csv", low_memory=False)
session_key = pd.read_csv(CFG["data_dir"] / "session_key.csv")

DROP_COLS = {
    "session_id", "user", "session_date", "session_date_str",
    "is_malicious", "sample_weight",
    "user_key", "session_date_key",
    "event_label", "day_label",
}
feature_cols = [c for c in unified.columns if c not in DROP_COLS]

X       = unified[feature_cols].values
y       = unified["is_malicious"].values.astype(np.float32)
weights = unified["sample_weight"].values.astype(np.float32)
users   = unified["user"].values
dates   = pd.to_datetime(unified["session_date"]).values

print(f"  X shape  : {X.shape}")
print(f"  Features : {len(feature_cols)}")
print(f"  Pos={y.sum():.0f}  Neg={(y==0).sum():.0f}  ratio={y.mean()*100:.3f}%")

# ── 2. Fold 생성 ──────────────────────────────────────────────────────────────
def make_folds(y, users, dates, n_folds, mode):
    idx = np.arange(len(y))
    if mode == "group":
        gkf = GroupKFold(n_splits=n_folds)
        return list(gkf.split(idx, y, groups=users))
    sorted_idx = np.argsort(dates)
    fold_size  = len(sorted_idx) // n_folds
    folds = []
    for k in range(n_folds):
        vs = k * fold_size
        ve = (k+1)*fold_size if k < n_folds-1 else len(sorted_idx)
        vi = sorted_idx[vs:ve]
        ti = np.concatenate([sorted_idx[:vs], sorted_idx[ve:]])
        folds.append((ti, vi))
    return folds

# ── 3. Cross-Validation ───────────────────────────────────────────────────────
print("\n[2] Starting cross-validation...")
folds     = make_folds(y, users, dates, CFG["n_folds"], CFG["split_mode"])
oof_probs = np.zeros(len(y), dtype=np.float32)
fold_aucs, fold_auprcs = [], []
models = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\n{'─'*50}")
    print(f"  Fold {fold_idx+1}/{CFG['n_folds']}  "
          f"train={len(train_idx):,}  val={len(val_idx):,}  "
          f"val_pos={y[val_idx].sum():.0f}")

    X_tr, y_tr, w_tr = X[train_idx], y[train_idx], weights[train_idx]
    X_va, y_va       = X[val_idx],   y[val_idx]

    train_pool = Pool(X_tr, y_tr, weight=w_tr, feature_names=feature_cols)
    val_pool   = Pool(X_va, y_va,              feature_names=feature_cols)

    model = CatBoostClassifier(
        iterations            = CFG["iterations"],
        learning_rate         = CFG["learning_rate"],
        depth                 = CFG["depth"],
        l2_leaf_reg           = CFG["l2_leaf_reg"],
        bagging_temperature   = CFG["bagging_temperature"],
        random_strength       = CFG["random_strength"],
        border_count          = CFG["border_count"],
        early_stopping_rounds = CFG["early_stopping_rounds"],
        eval_metric           = CFG["eval_metric"],
        loss_function         = CFG["loss_function"],
        task_type             = CFG["task_type"],
        devices               = CFG["devices"],
        random_seed           = CFG["random_seed"],
        use_best_model        = True,
        verbose               = 100,
    )
    model.fit(train_pool, eval_set=val_pool)

    val_probs          = model.predict_proba(val_pool)[:, 1]
    oof_probs[val_idx] = val_probs

    val_auc   = roc_auc_score(y_va, val_probs)
    val_auprc = average_precision_score(y_va, val_probs)
    fold_aucs.append(val_auc)
    fold_auprcs.append(val_auprc)
    print(f"  Best iter : {model.best_iteration_}")
    print(f"  Val AUC   : {val_auc:.4f}")
    print(f"  Val AUPRC : {val_auprc:.4f}")

    fold_dir = CFG["output_dir"] / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(fold_dir / "catboost_model.cbm"))
    models.append(model)

# ── 4. SHAP — 전체 fold 평균 (FIX) ───────────────────────────────────────────
print("\n[3] Computing TreeSHAP (averaged over all folds)...")

shap_n   = min(5000, len(X))
shap_idx = np.random.choice(len(X), shap_n, replace=False)
shap_X   = X[shap_idx]
shap_y   = y[shap_idx]

shap_sum = np.zeros((shap_n, len(feature_cols)))
for m in models:
    pool   = Pool(shap_X, shap_y, feature_names=feature_cols)
    sv     = m.get_feature_importance(pool, type="ShapValues")[:, :-1]
    shap_sum += sv
shap_mean_vals = shap_sum / len(models)

mean_abs_shap = np.abs(shap_mean_vals).mean(axis=0)
shap_df = pd.DataFrame({
    "feature":       feature_cols,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_df["shap_rank"]    = shap_df.index + 1
shap_df["cumsum_ratio"] = (shap_df["mean_abs_shap"].cumsum() /
                            shap_df["mean_abs_shap"].sum())
top10_n = max(1, int(np.ceil(len(feature_cols) * 0.10)))
shap_df["is_top10pct"] = shap_df["shap_rank"] <= top10_n
shap_df.to_csv(CFG["output_dir"] / "catboost_shap_importance.csv", index=False)

print(f"  SHAP on {shap_n} samples  |  Top-10% = {top10_n} features")
print("\n  Top-10% 피처:")
print(shap_df[shap_df["is_top10pct"]][["feature","mean_abs_shap"]].to_string(index=False))

# ── 5. OOF 저장 ───────────────────────────────────────────────────────────────
print("\n[4] Saving OOF predictions...")
oof_df = unified[["session_id","user","session_date","is_malicious"]].copy()
oof_df["catboost_oof_prob"] = oof_probs
oof_df.to_csv(CFG["output_dir"] / "catboost_oof.csv", index=False)

overall_auc   = roc_auc_score(y, oof_probs)
overall_auprc = average_precision_score(y, oof_probs)

with open(CFG["output_dir"] / "catboost_meta.pkl", "wb") as f:
    pickle.dump({"cfg": CFG, "feature_cols": feature_cols,
                 "fold_aucs": fold_aucs, "fold_auprcs": fold_auprcs}, f)

print("\n" + "=" * 60)
print("CatBoost Unified Training v2 Done!")
print("=" * 60)
print(f"  OOF AUC    : {overall_auc:.4f}")
print(f"  OOF AUPRC  : {overall_auprc:.4f}")
print(f"  Fold AUCs  : {[f'{a:.4f}' for a in fold_aucs]}")
print(f"  Fold AUPRCs: {[f'{a:.4f}' for a in fold_auprcs]}")
print(f"  Output     : {CFG['output_dir']}")
print("=" * 60)
