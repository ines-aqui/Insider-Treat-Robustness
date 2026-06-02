"""
attack_generation_v6.py
========================
[수정 1] beam_width=1 (greedy) 로 전부 통일
[수정 2] score_greedy 제거
[수정 3] global_only_random, local_only_random 추가
         (피처는 SHAP, 방향은 무작위)

공격 유형 8개:
  SHAP 기반:
    shap_guided        피처: union+rank  방향: SHAP
    global_only        피처: global top  방향: SHAP
    local_only         피처: local top   방향: SHAP
    global_only_random 피처: global top  방향: 무작위
    local_only_random  피처: local top   방향: 무작위
    direction_random   피처: union+rank  방향: 무작위
  비-SHAP 기반:
    random             피처: 무작위      방향: SHAP
    pure_random        피처: 무작위      방향: 무작위
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

CFG = {
    "data_dir":      Path("./output/unified"),
    "model_dir":     Path("./output/unified/catboost"),
    "output_dir":    Path("./output/attack"),

    "budgets":       [1, 2, 3, 5],
    "global_top_k":  10,
    "local_top_m":   20,
    "beam_width":    1,       # [수정] greedy로 통일
    "step_ratio":    0.3,
    "tau":           0.5,
    "n_folds":       5,
    "random_seed":   42,
}

CFG["output_dir"].mkdir(parents=True, exist_ok=True)
np.random.seed(CFG["random_seed"])

print("=" * 60)
print("SHAP-guided Adversarial Attack Generation v6")
print("(greedy, 8 methods)")
print("=" * 60)

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
print("\n[1] Loading data and models...")

unified   = pd.read_csv(CFG["data_dir"] / "unified_features_raw.csv", low_memory=False)
DROP_COLS = {
    "session_id", "user", "session_date", "session_date_str",
    "is_malicious", "sample_weight",
    "user_key", "session_date_key", "event_label", "day_label",
}
feature_cols = [c for c in unified.columns if c not in DROP_COLS]

X     = unified[feature_cols].values.astype(np.float64)
y     = unified["is_malicious"].values.astype(int)
users = unified["user"].values

with open(CFG["model_dir"] / "catboost_meta.pkl", "rb") as f:
    meta = pickle.load(f)

models = []
for fold_i in range(CFG["n_folds"]):
    m = CatBoostClassifier()
    m.load_model(str(CFG["model_dir"] / f"fold_{fold_i}" / "catboost_model.cbm"))
    models.append(m)
print(f"  Loaded {len(models)} fold models")
print(f"  Fold AUPRCs: {[f'{v:.4f}' for v in meta['fold_auprcs']]}")

# ── 2. 피처 제약 ──────────────────────────────────────────────────────────────
print("\n[2] Building feature constraints...")

feat_lower = np.percentile(X[y == 0], 1,  axis=0)
feat_upper = np.percentile(X[y == 0], 99, axis=0)

LINKED_GROUPS = [
    {
        "counts": ["n_device_conn", "n_device_disc"],
        "total":  "n_device_events",
        "ratios": {"device_conn_ratio": "n_device_conn"},
    },
    {
        "counts": ["n_email_external", "n_email_attach"],
        "total":  "n_email",
        "ratios": {
            "email_external_ratio": "n_email_external",
            "email_attach_ratio":   "n_email_attach",
        },
    },
    {
        "counts": ["n_file_exe", "n_file_doc", "n_file_compressed",
                   "n_file_image", "n_after_file"],
        "total":  "n_file",
        "ratios": {
            "file_exe_ratio":   "n_file_exe",
            "file_doc_ratio":   "n_file_doc",
            "file_after_ratio": "n_after_file",
        },
    },
    {
        "counts": ["n_upload", "n_job_site", "n_cloud", "n_after_http"],
        "total":  "n_http",
        "ratios": {
            "upload_ratio":   "n_upload",
            "job_site_ratio": "n_job_site",
            "cloud_ratio":    "n_cloud",
        },
    },
]

feat_idx = {f: i for i, f in enumerate(feature_cols)}

def classify_feature(fname):
    if fname.startswith("delta_"):
        return "immutable"
    if any(kw in fname for kw in ["is_weekend", "day_of_week", "session_dur"]):
        return "immutable"
    for g in LINKED_GROUPS:
        if fname in g["counts"] or fname == g["total"] or fname in g["ratios"]:
            return "linked"
    return "mutable_bounded"

feat_types    = {f: classify_feature(f) for f in feature_cols}
mutable_feats = [f for f, t in feat_types.items() if t == "mutable_bounded"]
mutable_idx   = [feat_idx[f] for f in mutable_feats]
mutable_set   = set(mutable_idx)

int_feats   = set(feat_idx[f] for f in feature_cols
                  if f.startswith("n_") and f in feat_idx)
ratio_feats = set(feat_idx[f] for f in feature_cols
                  if f.endswith("_ratio") and f in feat_idx)

linked_group_cache = []
for g in LINKED_GROUPS:
    total_fi = feat_idx.get(g["total"])
    if total_fi is None:
        continue
    count_fis = [feat_idx[f] for f in g["counts"] if f in feat_idx]
    ratio_map  = {feat_idx[r]: feat_idx[c]
                  for r, c in g["ratios"].items()
                  if r in feat_idx and c in feat_idx}
    linked_group_cache.append({
        "total_fi": total_fi, "count_fis": count_fis, "ratio_map": ratio_map,
    })

print(f"  Mutable   : {len(mutable_feats)}")
print(f"  Immutable : {sum(1 for t in feat_types.values() if t=='immutable')}")
print(f"  Linked    : {sum(1 for t in feat_types.values() if t=='linked')}")

# ── 3. global top ─────────────────────────────────────────────────────────────
print("\n[3] Global top features (mutable only)...")

shap_df = pd.read_csv(CFG["model_dir"] / "catboost_shap_importance.csv")
mutable_shap = (
    shap_df[shap_df["feature"].isin(mutable_feats)]
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
global_top_features = set(mutable_shap.head(CFG["global_top_k"])["feature"].tolist())
global_top_idx      = set(feat_idx[f] for f in global_top_features if f in feat_idx)
global_rank_map     = {
    feat_idx[row["feature"]]: rank
    for rank, (_, row) in enumerate(mutable_shap.iterrows())
    if row["feature"] in feat_idx
}
print(f"  top-{CFG['global_top_k']}: {list(global_top_features)[:5]}...")

# ── 4. τ ──────────────────────────────────────────────────────────────────────
tau = CFG["tau"]
print(f"\n[4] Fixed τ = {tau}")
oof_df    = pd.read_csv(CFG["model_dir"] / "catboost_oof.csv")
oof_probs = oof_df["catboost_oof_prob"].values
oof_y     = oof_df["is_malicious"].values
tpr_ref   = float((oof_probs[oof_y==1] >= tau).mean())
fpr_ref   = float((oof_probs[oof_y==0] >= tau).mean())
print(f"  OOF @τ=0.5 → TPR={tpr_ref:.4f}  FPR={fpr_ref:.4f}")

with open(CFG["output_dir"] / "threshold_tau.json", "w") as f:
    json.dump({"tau": tau, "method": "fixed_0.5",
               "oof_tpr": round(tpr_ref, 4),
               "oof_fpr": round(fpr_ref, 4)}, f, indent=2)

# ── 5. 공격 대상 ──────────────────────────────────────────────────────────────
print("\n[5] Selecting attack targets...")
gkf   = GroupKFold(n_splits=CFG["n_folds"])
folds = list(gkf.split(np.arange(len(y)), y, groups=users))

fold_attack_map = []
total_mal = 0
for fold_i, (_, val_idx) in enumerate(folds):
    attack_idx = val_idx[y[val_idx] == 1]
    fold_attack_map.append((fold_i, models[fold_i], attack_idx))
    total_mal += len(attack_idx)
    print(f"  Fold {fold_i+1}: {len(attack_idx)}개")
print(f"  전체: {total_mal}개")

# ── 6. 공통 유틸 ──────────────────────────────────────────────────────────────
def project_sample(x):
    x = np.clip(x.copy(), feat_lower, feat_upper)
    for fi in int_feats:
        x[fi] = max(0., round(x[fi]))
    for fi in ratio_feats:
        x[fi] = float(np.clip(x[fi], 0., 1.))
    for g in linked_group_cache:
        total_val = max(x[g["total_fi"]], 1e-9)
        for cfi in g["count_fis"]:
            x[cfi] = float(np.clip(round(x[cfi]), 0, total_val))
        for rfi, cfi in g["ratio_map"].items():
            x[rfi] = float(x[cfi] / total_val)
    return x

def perturb(x, fi, d):
    x_new = x.copy()
    rng   = feat_upper[fi] - feat_lower[fi]
    if rng < 1e-9:
        return x_new
    x_new[fi] = x_new[fi] + d * CFG["step_ratio"] * rng
    return project_sample(x_new)

def get_candidates(local_shap_i, method):
    local_abs = np.abs(local_shap_i)
    local_top = set(np.argsort(local_abs)[::-1][:CFG["local_top_m"]].tolist())

    if method in ("shap_guided", "direction_random"):
        union = list((local_top | global_top_idx) & mutable_set)
        scored = [(fi, 0.5*float(local_abs[fi]) + 0.5/(1+global_rank_map.get(fi,999)))
                  for fi in union]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [fi for fi, _ in scored[:CFG["global_top_k"]]]

    elif method in ("global_only", "global_only_random"):
        return [i for i in global_top_idx if i in mutable_set]

    elif method in ("local_only", "local_only_random"):
        return [i for i in local_top if i in mutable_set]

    elif method in ("random", "pure_random"):
        return list(np.random.choice(
            mutable_idx, size=min(CFG["global_top_k"], len(mutable_idx)), replace=False
        ))
    return []

def get_direction(method, fi, local_shap_i):
    if method in ("direction_random", "global_only_random",
                  "local_only_random", "pure_random"):
        return float(np.random.choice([-1., 1.]))
    d = -np.sign(local_shap_i[fi])
    return d if d != 0 else -1.

def greedy_attack(x_orig, local_shap_i, candidates, method, model, budget):
    """순수 greedy (beam_width=1)"""
    def score(x):
        return model.predict_proba(x.reshape(1, -1))[0, 1]

    if score(x_orig) < tau:
        return x_orig

    x_cur = x_orig.copy()
    used  = set()

    for _ in range(budget):
        best_x, best_s, best_fi = x_cur.copy(), score(x_cur), None
        for fi in candidates:
            if fi in used:
                continue
            d     = get_direction(method, fi, local_shap_i)
            x_try = perturb(x_cur, fi, d)
            s_try = score(x_try)
            if s_try < best_s:
                best_x, best_s, best_fi = x_try, s_try, fi

        if best_fi is None:
            break
        x_cur = best_x
        used.add(best_fi)

    return x_cur

# ── 7. fold별 공격 ────────────────────────────────────────────────────────────
def run_fold(fold_i, model, attack_idx, method, budget):
    X_attack = X[attack_idx].copy()
    pool      = Pool(X_attack, feature_names=feature_cols)
    sv        = model.get_feature_importance(pool, type="ShapValues")
    local_shap = sv[:, :-1]

    evaded = 0
    sb, sa = [], []
    X_adv  = X_attack.copy()

    for i in range(len(X_attack)):
        x_orig   = X_attack[i].copy()
        s_before = model.predict_proba(x_orig.reshape(1, -1))[0, 1]
        sb.append(s_before)

        if s_before < tau:
            sa.append(s_before)
            continue

        cands = get_candidates(local_shap[i], method)
        if not cands:
            sa.append(s_before)
            continue

        x_adv   = greedy_attack(x_orig, local_shap[i], cands, method, model, budget)
        s_after = model.predict_proba(x_adv.reshape(1, -1))[0, 1]
        sa.append(s_after)
        X_adv[i] = x_adv
        if s_after < tau:
            evaded += 1

    return X_adv, sb, sa, evaded

# ── 8. 전체 실행 ──────────────────────────────────────────────────────────────
METHODS = [
    "shap_guided",
    "global_only",
    "local_only",
    "global_only_random",
    "local_only_random",
    "direction_random",
    "random",
    "pure_random",
]

all_results = []

for method in METHODS:
    for budget in CFG["budgets"]:
        print(f"\n  [{method}]  B={budget}")

        all_sb, all_sa = [], []
        all_evaded = 0
        adv_dfs    = []

        for fold_i, model, attack_idx in fold_attack_map:
            X_adv, sb, sa, evaded = run_fold(fold_i, model, attack_idx, method, budget)
            all_sb.extend(sb); all_sa.extend(sa)
            all_evaded += evaded

            fold_df = unified.iloc[attack_idx].copy().reset_index(drop=True)
            for j, fc in enumerate(feature_cols):
                fold_df[fc] = X_adv[:, j]
            fold_df["score_before"] = sb
            fold_df["score_after"]  = sa
            fold_df["evaded"]       = [int(s < tau) for s in sa]
            fold_df["fold"]         = fold_i
            adv_dfs.append(fold_df)

        n_targeted = sum(1 for s in all_sb if s >= tau)
        asr        = all_evaded / n_targeted if n_targeted > 0 else 0.
        mean_delta = float(np.mean(np.array(all_sb) - np.array(all_sa)))
        print(f"    ASR={asr:.4f}  Δscore={mean_delta:.4f}  "
              f"evaded={all_evaded}/{n_targeted}")

        adv_df = pd.concat(adv_dfs, ignore_index=True)
        adv_df["method"] = method
        adv_df["budget"] = budget
        adv_df.to_csv(
            CFG["output_dir"] / f"attacked_samples_{method}_B{budget}.csv",
            index=False
        )

        all_results.append({
            "method":            method,
            "budget":            budget,
            "n_targeted":        n_targeted,
            "n_evaded":          all_evaded,
            "asr":               round(asr, 4),
            "mean_delta_score":  round(mean_delta, 4),
            "mean_score_before": round(float(np.mean(all_sb)), 4),
            "mean_score_after":  round(float(np.mean(all_sa)), 4),
        })

# ── 9. 결과 저장 & 출력 ───────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(CFG["output_dir"] / "attack_results_summary.csv", index=False)

print("\n" + "=" * 60)
print("Attack Generation v6 Done!")
print("=" * 60)

order = ["shap_guided", "global_only", "local_only",
         "global_only_random", "local_only_random",
         "direction_random", "random", "pure_random"]
pivot = results_df.pivot_table(
    index="method", columns="budget", values="asr"
).round(4).reindex([m for m in order if m in results_df["method"].values])
print(pivot.to_string())

print("\n  --- SHAP 기반 ---")
for m in ["shap_guided","global_only","local_only",
          "global_only_random","local_only_random","direction_random"]:
    row = results_df[(results_df.method==m)&(results_df.budget==3)]
    if len(row): print(f"  {m:<22} B=3 ASR={row.iloc[0].asr:.4f}")

print("\n  --- 비-SHAP 기반 ---")
for m in ["random","pure_random"]:
    row = results_df[(results_df.method==m)&(results_df.budget==3)]
    if len(row): print(f"  {m:<22} B=3 ASR={row.iloc[0].asr:.4f}")
print("=" * 60)
