"""
adversarial_retraining_curriculum_linesearch.py
================================================
진짜 Curriculum: 이전 모델을 이어받아 순차 학습 (linesearch 기반)

GPU에서 init_model 미지원 → CPU 사용
속도 보완: round당 iterations=1000 (처음부터 학습 시 2000 대비 단축)

Curriculum 순서 (약 → 강):
  Round 1: global_linesearch B=3  (ASR=0.861)  → 모델 A
  Round 2: shap_linesearch   B=3  (ASR=0.863)  → 모델 A에서 이어받아 → 모델 B
  Round 3: shap_linesearch   B=5  (ASR=0.921)  → 모델 B에서 이어받아 → 모델 C

비교용:
  I: shap_linesearch B=3 단일 (GPU, 처음부터)  ← v9 결과 재사용
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
    "attack_dir_v7":     Path("./output/attack_six_methods"),
    "output_dir":        Path("./output/retraining_curriculum_ls_budget"),

    "adv_inject_ratio":  0.05,

    # curriculum 순서 (같은 방법, budget만 증가)
    "curriculum_steps": [
        {"method": "shap_linesearch", "budget": 1, "src_asr": 0.570, "label": "R1_shap_ls_B1"},
        {"method": "shap_linesearch", "budget": 2, "src_asr": 0.782, "label": "R2_shap_ls_B2"},
        {"method": "shap_linesearch", "budget": 3, "src_asr": 0.863, "label": "R3_shap_ls_B3"},
    ],

    # CPU 사용 (init_model GPU 미지원)
    # round당 iterations 단축 (처음부터 학습이 아니라 이어받으므로 적어도 됨)
    "curriculum_iterations":  1000,
    "learning_rate":          0.05,
    "depth":                  8,
    "early_stopping_rounds":  50,
    "eval_metric":            "Logloss",
    "loss_function":          "Logloss",
    "task_type":              "CPU",   # GPU init_model 미지원
    "tau":                    0.5,
    "n_folds":                5,
    "random_seed":            42,
}

CFG["output_dir"].mkdir(parents=True, exist_ok=True)
np.random.seed(CFG["random_seed"])

print("=" * 65)
print("Curriculum Retraining (shap_linesearch B=1→2→3, CPU init_model)")
print("=" * 65)
print("  방식: 같은 공격(shap_linesearch), budget만 점진적으로 증가")
print("  B=1(약) → B=2(중) → B=3(강)")
print()
for i, step in enumerate(CFG["curriculum_steps"]):
    print(f"  Round {i+1}: {step['method']:<24} B={step['budget']}  "
          f"ASR={step['src_asr']}")

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
adv_pool = {}
for step in CFG["curriculum_steps"]:
    key  = f"{step['method']}_B{step['budget']}"
    path = CFG["attack_dir_v7"] / f"attacked_samples_{step['method']}_B{step['budget']}.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        adv_pool[key] = df
        print(f"  {key:<30} {len(df):,}개")
    else:
        print(f"  [WARN] {path} not found")
        adv_pool[key] = pd.DataFrame()

# ── 3. 재학습 데이터 구성 ─────────────────────────────────────────────────────
def build_retrain_data(X_tr, y_tr, w_tr, adv_df, exclude_fold):
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

# ── 5. Curriculum 재학습 ──────────────────────────────────────────────────────
print("\n[3] Running Curriculum retraining (5-fold × 3 rounds, CPU)...")
print("  ※ CPU 학습이라 시간이 걸릴 수 있어요.\n")

gkf   = GroupKFold(n_splits=CFG["n_folds"])
folds = list(gkf.split(np.arange(len(y)), y, groups=users))

fold_final_results = []
round_history      = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\n  ── Fold {fold_idx+1}/{CFG['n_folds']} "
          f"(train={len(train_idx):,}  val={len(val_idx):,}  "
          f"val_pos={y[val_idx].sum()}) ──")

    X_tr, y_tr, w_tr = X[train_idx], y[train_idx], w[train_idx]
    X_va, y_va        = X[val_idx],   y[val_idx]
    val_sids = set(unified.iloc[val_idx]["session_id"].tolist()) \
        if "session_id" in unified.columns else set()

    # Round 1은 CPU로 처음부터 학습 (GPU 모델과 class params 불일치로 init_model 불가)
    current_model = None

    for step_idx, step in enumerate(CFG["curriculum_steps"]):
        key    = f"{step['method']}_B{step['budget']}"
        label  = step["label"]
        adv_df = adv_pool.get(key, pd.DataFrame())

        X_ret, y_ret, w_ret, n_inj = build_retrain_data(
            X_tr, y_tr, w_tr, adv_df, exclude_fold=fold_idx
        )

        print(f"    [{label}] inject={n_inj}  ", end="", flush=True)

        train_pool = Pool(X_ret, y_ret, weight=w_ret, feature_names=feature_cols)
        val_pool   = Pool(X_va,  y_va,                feature_names=feature_cols)

        new_model = CatBoostClassifier(
            iterations            = CFG["curriculum_iterations"],
            learning_rate         = CFG["learning_rate"],
            depth                 = CFG["depth"],
            early_stopping_rounds = CFG["early_stopping_rounds"],
            eval_metric           = CFG["eval_metric"],
            loss_function         = CFG["loss_function"],
            task_type             = CFG["task_type"],   # CPU
            random_seed           = CFG["random_seed"],
            use_best_model        = True,
            verbose               = False,
        )

        # Round 1: 처음부터 CPU 학습 / Round 2,3: 이전 round 모델 이어받음
        fit_kwargs = {"eval_set": val_pool}
        if current_model is not None:
            fit_kwargs["init_model"] = current_model
        new_model.fit(train_pool, **fit_kwargs)

        # 평가
        adv_filt = adv_df
        if "session_id" in adv_filt.columns and val_sids:
            adv_filt = adv_filt[adv_filt["session_id"].isin(val_sids)]
        adv_X_val = adv_filt[feature_cols].values.astype(np.float64) if len(adv_filt) > 0 else None
        adv_y_val = adv_filt["is_malicious"].values.astype(int)       if len(adv_filt) > 0 else None

        metrics = compute_metrics(new_model, X_va, y_va, adv_X_val, adv_y_val)
        print(f"AUPRC={metrics['clean_auprc']:.4f}  "
              f"ASR={metrics.get('asr',0.):.4f}  "
              f"iter={new_model.best_iteration_}")

        round_history.append({
            "fold": fold_idx+1, "round": step_idx+1,
            "method": step["method"], "budget": step["budget"],
            "label": label, **metrics,
        })

        # 모델 저장
        step_dir = CFG["output_dir"] / f"fold_{fold_idx}" / label
        step_dir.mkdir(parents=True, exist_ok=True)
        new_model.save_model(str(step_dir / "model.cbm"))

        current_model = new_model  # 다음 round 시작점

    # 최종 모델 저장
    final_dir = CFG["output_dir"] / f"fold_{fold_idx}" / "L_curriculum_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    current_model.save_model(str(final_dir / "model.cbm"))

    # 최종 평가: shap_linesearch B=3 기준
    eval_key  = "shap_linesearch_B3"
    adv_final = adv_pool.get(eval_key, pd.DataFrame())
    if "session_id" in adv_final.columns and val_sids:
        adv_final = adv_final[adv_final["session_id"].isin(val_sids)]
    adv_X_f = adv_final[feature_cols].values.astype(np.float64) if len(adv_final) > 0 else None
    adv_y_f = adv_final["is_malicious"].values.astype(int)       if len(adv_final) > 0 else None

    final_m = compute_metrics(current_model, X_va, y_va, adv_X_f, adv_y_f)
    final_m.update({"arm": "L_curriculum_ls", "fold": fold_idx+1})
    fold_final_results.append(final_m)

# ── 6. 저장 & 보고 ────────────────────────────────────────────────────────────
print("\n[4] Saving results...")
pd.DataFrame(round_history).to_csv(
    CFG["output_dir"] / "curriculum_ls_round_history.csv", index=False)
df_final = pd.DataFrame(fold_final_results)
df_final.to_csv(CFG["output_dir"] / "fold_results_L_curriculum_ls.csv", index=False)

cols = ["clean_auprc", "tpr_at_fpr1", "benign_fpr", "asr"]
summary = {col: round(float(df_final[col].mean()), 4)
           for col in cols if col in df_final.columns}

print("\n" + "=" * 65)
print("Curriculum Linesearch Done!")
print("=" * 65)
print(f"\n  L_curriculum_ls 최종 결과:")
print(f"  clean AUPRC : {summary.get('clean_auprc',0.):.4f}")
print(f"  TPR@FPR1%   : {summary.get('tpr_at_fpr1',0.):.4f}")
print(f"  def_ASR     : {summary.get('asr',0.):.4f}")
print(f"  benign FPR  : {summary.get('benign_fpr',0.):.4f}")

print(f"\n  Round별 평균:")
rh = pd.DataFrame(round_history)
for rnd in sorted(rh["round"].unique()):
    sub = rh[rh["round"] == rnd]
    print(f"  Round {rnd} ({sub['label'].iloc[0]:<22})  "
          f"AUPRC={sub['clean_auprc'].mean():.4f}  "
          f"ASR={sub['asr'].mean():.4f}")

print(f"\n  [비교]")
print(f"  D  (direction_random 단일):  AUPRC=0.5624  ASR=0.4429")
print(f"  I  (shap_linesearch 단일):   AUPRC=0.5702  ASR=0.4341")
print(f"  L  (curriculum linesearch):  AUPRC={summary.get('clean_auprc',0.):.4f}  "
      f"ASR={summary.get('asr',0.):.4f}")
print("=" * 65)
