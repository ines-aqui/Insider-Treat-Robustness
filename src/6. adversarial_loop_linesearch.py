"""
adversarial_loop_linesearch.py
================================
adversarial_loop_v7style 기반
공격 방법: shap_linesearch B=3 (v7 최강 SHAP 기반)

변경 사항:
  - attack_method: direction_random → shap_linesearch
  - 공격 함수: random direction → linesearch (±방향 × step grid)
  - loop_dir: output/loop_linesearch
  - 나머지 (재학습, 조기중단, SHAP 재계산) v7style 동일
"""

import json
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
    "data_dir":      Path("./output/unified"),
    "model_dir":     Path("./output/unified/catboost"),
    "loop_dir":      Path("./output/loop_linesearch"),

    # 공격 설정 (shap_linesearch)
    "attack_budget":         3,
    "global_top_k":          10,
    "local_top_m":           20,
    "line_search_steps":     [0.1, 0.2, 0.3, 0.5],   # linesearch step grid
    "tau":                   0.5,

    # 재학습 설정 (v7style 동일)
    "adv_inject_ratio":      0.05,
    "adv_weight_scale":      1.0,
    "retrain_iterations":    2000,
    "learning_rate":         0.05,
    "depth":                 8,
    "early_stopping_rounds": 100,
    "eval_metric":           "Logloss",
    "loss_function":         "Logloss",
    "task_type":             "GPU",
    "devices":               "0",

    # 조기 중단 (v7style 동일)
    "max_rounds":            5,
    "patience_robust":       4,
    "clean_auprc_drop_tol":  0.10,

    "n_folds":     5,
    "random_seed": 42,
}

CFG["loop_dir"].mkdir(parents=True, exist_ok=True)
np.random.seed(CFG["random_seed"])

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
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

gkf   = GroupKFold(n_splits=CFG["n_folds"])
folds = list(gkf.split(np.arange(len(y)), y, groups=users))

# ── 피처 제약 ─────────────────────────────────────────────────────────────────
feat_lower = np.percentile(X[y == 0], 1,  axis=0)
feat_upper = np.percentile(X[y == 0], 99, axis=0)

LINKED_GROUPS = [
    {"counts": ["n_device_conn", "n_device_disc"], "total": "n_device_events",
     "ratios": {"device_conn_ratio": "n_device_conn"}},
    {"counts": ["n_email_external", "n_email_attach"], "total": "n_email",
     "ratios": {"email_external_ratio": "n_email_external", "email_attach_ratio": "n_email_attach"}},
    {"counts": ["n_file_exe","n_file_doc","n_file_compressed","n_file_image","n_after_file"],
     "total": "n_file",
     "ratios": {"file_exe_ratio":"n_file_exe","file_doc_ratio":"n_file_doc","file_after_ratio":"n_after_file"}},
    {"counts": ["n_upload","n_job_site","n_cloud","n_after_http"], "total": "n_http",
     "ratios": {"upload_ratio":"n_upload","job_site_ratio":"n_job_site","cloud_ratio":"n_cloud"}},
]
feat_idx = {f: i for i, f in enumerate(feature_cols)}

def classify_feature(fname):
    if fname.startswith("delta_"): return "immutable"
    if any(kw in fname for kw in ["is_weekend","day_of_week","session_dur"]): return "immutable"
    for g in LINKED_GROUPS:
        if fname in g["counts"] or fname == g["total"] or fname in g["ratios"]: return "linked"
    return "mutable_bounded"

feat_types    = {f: classify_feature(f) for f in feature_cols}
mutable_feats = [f for f, t in feat_types.items() if t == "mutable_bounded"]
mutable_idx   = [feat_idx[f] for f in mutable_feats]
mutable_set   = set(mutable_idx)
int_feats     = set(feat_idx[f] for f in feature_cols if f.startswith("n_") and f in feat_idx)
ratio_feats   = set(feat_idx[f] for f in feature_cols if f.endswith("_ratio") and f in feat_idx)

linked_cache = []
for g in LINKED_GROUPS:
    tfi = feat_idx.get(g["total"])
    if tfi is None: continue
    linked_cache.append({
        "total_fi":  tfi,
        "count_fis": [feat_idx[f] for f in g["counts"] if f in feat_idx],
        "ratio_map": {feat_idx[r]: feat_idx[c] for r, c in g["ratios"].items()
                      if r in feat_idx and c in feat_idx},
    })

def project(x):
    x = np.clip(x.copy(), feat_lower, feat_upper)
    for fi in int_feats:   x[fi] = max(0., round(x[fi]))
    for fi in ratio_feats: x[fi] = float(np.clip(x[fi], 0., 1.))
    for g in linked_cache:
        tv = max(x[g["total_fi"]], 1e-9)
        for cfi in g["count_fis"]: x[cfi] = float(np.clip(round(x[cfi]), 0, tv))
        for rfi, cfi in g["ratio_map"].items(): x[rfi] = float(x[cfi] / tv)
    return x

def perturb_ratio(x, fi, d, step_ratio):
    xn = x.copy()
    rng = feat_upper[fi] - feat_lower[fi]
    if rng < 1e-9: return xn
    xn[fi] = xn[fi] + d * step_ratio * rng
    return project(xn)

# ── 공격 함수 (shap_linesearch) ───────────────────────────────────────────────
def get_global_top(shap_csv_path):
    shap_df = pd.read_csv(shap_csv_path)
    mutable_shap = (shap_df[shap_df["feature"].isin(mutable_feats)]
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True))
    global_top     = set(mutable_shap.head(CFG["global_top_k"])["feature"].tolist())
    global_top_idx = set(feat_idx[f] for f in global_top if f in feat_idx)
    rank_map = {feat_idx[row["feature"]]: rank
                for rank, (_, row) in enumerate(mutable_shap.iterrows())
                if row["feature"] in feat_idx}
    return global_top_idx, rank_map

def attack_fold_linesearch(model, attack_idx, global_top_idx, rank_map):
    """shap_linesearch: union+rank 피처 선택 + ±방향 × step grid 탐색"""
    X_atk = X[attack_idx].copy()
    pool  = Pool(X_atk, feature_names=feature_cols)
    sv    = model.get_feature_importance(pool, type="ShapValues")
    lshap = sv[:, :-1]

    evaded = 0
    sb, sa = [], []
    X_adv  = X_atk.copy()

    for i in range(len(X_atk)):
        x0 = X_atk[i].copy()
        s0 = model.predict_proba(x0.reshape(1,-1))[0,1]
        sb.append(s0)
        if s0 < tau:
            sa.append(s0); continue

        # union+rank 피처 선택 (shap_linesearch)
        labs = np.abs(lshap[i])
        ltop = set(np.argsort(labs)[::-1][:CFG["local_top_m"]].tolist())
        union = list((ltop | global_top_idx) & mutable_set)
        scored = [(fi, 0.5*float(labs[fi]) + 0.5/(1+rank_map.get(fi,999)))
                  for fi in union]
        scored.sort(key=lambda x: x[1], reverse=True)
        cands = [fi for fi, _ in scored[:CFG["global_top_k"]]]

        if not cands:
            sa.append(s0); continue

        # linesearch greedy
        xc   = x0.copy()
        used = set()
        for _ in range(CFG["attack_budget"]):
            bx, bs, bfi = xc.copy(), model.predict_proba(xc.reshape(1,-1))[0,1], None
            for fi in cands:
                if fi in used: continue
                # ±방향 × step grid 전체 탐색
                for d in (-1., 1.):
                    for sr in CFG["line_search_steps"]:
                        xt = perturb_ratio(xc, fi, d, sr)
                        st = model.predict_proba(xt.reshape(1,-1))[0,1]
                        if st < bs:
                            bx, bs, bfi = xt, st, fi
            if bfi is None: break
            xc = bx; used.add(bfi)

        sf = model.predict_proba(xc.reshape(1,-1))[0,1]
        sa.append(sf)
        X_adv[i] = xc
        if sf < tau: evaded += 1

    return X_adv, sb, sa, evaded

# ── 평가 함수 ─────────────────────────────────────────────────────────────────
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
        ap = model.predict_proba(adv_X)[:,1]
        mm = (adv_y==1) if adv_y is not None else np.ones(len(adv_X), bool)
        if mm.sum() > 0:
            asr = float((ap[mm] < tau).mean())
            rr  = float((ap[mm] >= tau).mean())
        else:
            asr = rr = 0.
        result.update({"asr": round(asr,4), "robust_recall": round(rr,4)})
    return result

# ── 재학습 함수 (v7style 동일) ────────────────────────────────────────────────
def retrain_fold(fold_idx, X_tr, y_tr, w_tr, adv_df, X_va, y_va):
    adv_clean = adv_df[adv_df["fold"] != fold_idx].copy() \
        if "fold" in adv_df.columns else adv_df.copy()
    n_inject   = int(len(X_tr) * CFG["adv_inject_ratio"])
    mal_idx    = np.where(y_tr == 1)[0]
    w_mal_mean = float(w_tr[mal_idx].mean()) if len(mal_idx) > 0 else 1.0
    w_adv      = w_mal_mean * CFG["adv_weight_scale"]

    if len(adv_clean) > 0 and n_inject > 0:
        adv_X = adv_clean[feature_cols].values.astype(np.float64)
        idx   = np.random.choice(len(adv_X), min(n_inject, len(adv_X)), replace=True)
        X_ret = np.vstack([X_tr, adv_X[idx]])
        y_ret = np.concatenate([y_tr, np.ones(len(idx), dtype=int)])
        w_ret = np.concatenate([w_tr, np.full(len(idx), w_adv)])
    else:
        X_ret, y_ret, w_ret = X_tr, y_tr, w_tr

    perm = np.random.permutation(len(X_ret))
    X_ret, y_ret, w_ret = X_ret[perm], y_ret[perm], w_ret[perm]

    train_pool = Pool(X_ret, y_ret, weight=w_ret, feature_names=feature_cols)
    val_pool   = Pool(X_va,  y_va,                feature_names=feature_cols)

    model = CatBoostClassifier(
        iterations            = CFG["retrain_iterations"],
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
    model.fit(train_pool, eval_set=val_pool)
    return model

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Adversarial Loop (shap_linesearch B=3)")
print(f"  max_rounds={CFG['max_rounds']}  "
      f"patience={CFG['patience_robust']}  "
      f"drop_tol={CFG['clean_auprc_drop_tol']:.0%}")
print("=" * 65)

init_models = []
for fi in range(CFG["n_folds"]):
    m = CatBoostClassifier()
    m.load_model(str(CFG["model_dir"] / f"fold_{fi}" / "catboost_model.cbm"))
    init_models.append(m)

with open(CFG["model_dir"] / "catboost_meta.pkl", "rb") as f:
    meta = pickle.load(f)

# Round 0 평가
print("\n[Round 0] Initial model evaluation...")
r0_auprcs, r0_asrs = [], []
init_shap_path = CFG["model_dir"] / "catboost_shap_importance.csv"

for fold_idx, (_, val_idx) in enumerate(folds):
    model       = init_models[fold_idx]
    atk_idx     = val_idx[y[val_idx] == 1]
    g_top, rmap = get_global_top(init_shap_path)
    _, sb, sa, evaded = attack_fold_linesearch(model, atk_idx, g_top, rmap)
    n_tgt = sum(1 for s in sb if s >= tau)
    asr   = evaded / n_tgt if n_tgt > 0 else 0.
    m     = compute_metrics(model, X[val_idx], y[val_idx])
    r0_auprcs.append(m["clean_auprc"])
    r0_asrs.append(asr)

init_clean_auprc    = float(np.mean(r0_auprcs))
init_asr            = float(np.mean(r0_asrs))
init_robust_recall  = float(max(0., 1. - init_asr))

print(f"  clean AUPRC   : {init_clean_auprc:.4f}")
print(f"  robust recall : {init_robust_recall:.4f}")
print(f"  mean ASR      : {init_asr:.4f}")

history = [{"round": 0, "clean_auprc": init_clean_auprc,
            "robust_recall": init_robust_recall, "asr": init_asr, "stop_reason": None}]

current_models    = init_models
current_shap_path = init_shap_path
no_improve_count  = 0
prev_clean_auprc  = init_clean_auprc
prev_robust_recall = init_robust_recall
stop_reason       = None

for round_idx in range(1, CFG["max_rounds"] + 1):
    print(f"\n{'='*65}")
    print(f"[Round {round_idx}]")
    round_dir = CFG["loop_dir"] / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 공격 생성 (shap_linesearch)
    print(f"  Step 1: Attack (shap_linesearch B={CFG['attack_budget']})")
    g_top, rmap = get_global_top(current_shap_path)
    all_adv_dfs = []
    all_sb, all_sa, all_evaded = [], [], 0

    for fold_idx, (_, val_idx) in enumerate(folds):
        atk_idx = val_idx[y[val_idx] == 1]
        X_adv, sb, sa, evaded = attack_fold_linesearch(
            current_models[fold_idx], atk_idx, g_top, rmap)
        all_sb.extend(sb); all_sa.extend(sa); all_evaded += evaded

        fold_df = unified.iloc[atk_idx].copy().reset_index(drop=True)
        for j, fc in enumerate(feature_cols): fold_df[fc] = X_adv[:, j]
        fold_df["score_before"] = sb
        fold_df["score_after"]  = sa
        fold_df["evaded"]       = [int(s < tau) for s in sa]
        fold_df["fold"]         = fold_idx
        all_adv_dfs.append(fold_df)

    adv_df = pd.concat(all_adv_dfs, ignore_index=True)
    adv_df.to_csv(round_dir / "attacked_samples.csv", index=False)

    n_tgt     = sum(1 for s in all_sb if s >= tau)
    round_asr = all_evaded / n_tgt if n_tgt > 0 else 0.
    print(f"    ASR={round_asr:.4f}  evaded={all_evaded}/{n_tgt}  "
          f"evade_rate={adv_df['evaded'].mean()*100:.1f}%")

    # Step 2: 재학습
    print(f"  Step 2: Retraining (iterations={CFG['retrain_iterations']})")
    new_models = []
    fold_clean_auprcs   = []
    fold_robust_recalls = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr, w_tr = X[train_idx], y[train_idx], w[train_idx]
        X_va, y_va        = X[val_idx],   y[val_idx]

        new_model = retrain_fold(fold_idx, X_tr, y_tr, w_tr, adv_df, X_va, y_va)

        val_sids  = set(unified.iloc[val_idx]["session_id"].tolist()) \
                    if "session_id" in unified.columns else set()
        adv_filt  = adv_df[adv_df["session_id"].isin(val_sids)] \
                    if "session_id" in adv_df.columns and val_sids else adv_df.iloc[0:0]
        adv_X_val = adv_filt[feature_cols].values.astype(np.float64) if len(adv_filt) > 0 else None
        adv_y_val = adv_filt["is_malicious"].values.astype(int)       if len(adv_filt) > 0 else None

        m = compute_metrics(new_model, X_va, y_va, adv_X_val, adv_y_val)
        fold_clean_auprcs.append(m["clean_auprc"])
        fold_robust_recalls.append(m.get("robust_recall", max(0., 1. - m.get("asr", 1.))))

        fold_model_dir = round_dir / f"fold_{fold_idx}"
        fold_model_dir.mkdir(exist_ok=True)
        new_model.save_model(str(fold_model_dir / "catboost_model.cbm"))
        new_models.append(new_model)

        print(f"    Fold {fold_idx+1}: "
              f"AUPRC={m['clean_auprc']:.4f}  "
              f"ASR={m.get('asr',0.):.4f}  "
              f"bFPR={m['benign_fpr']:.4f}")

    round_clean_auprc   = float(np.mean(fold_clean_auprcs))
    round_robust_recall = float(np.mean(fold_robust_recalls))

    print(f"\n  Round {round_idx} 요약:")
    print(f"    clean AUPRC  : {prev_clean_auprc:.4f} → {round_clean_auprc:.4f} "
          f"({'↑' if round_clean_auprc > prev_clean_auprc else '↓'}"
          f"{abs(round_clean_auprc-prev_clean_auprc):.4f})")
    print(f"    robust recall: {prev_robust_recall:.4f} → {round_robust_recall:.4f}")
    print(f"    ASR          : {round_asr:.4f}")

    # SHAP 재계산
    shap_n    = min(5000, len(X))
    shap_samp = np.random.choice(len(X), shap_n, replace=False)
    shap_pool = Pool(X[shap_samp], y[shap_samp], feature_names=feature_cols)
    sv_all    = np.zeros((shap_n, len(feature_cols)))
    for m_tmp in new_models:
        sv_all += m_tmp.get_feature_importance(shap_pool, type="ShapValues")[:, :-1]
    sv_all /= len(new_models)
    shap_new = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(sv_all).mean(axis=0)}) \
                 .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_new["shap_rank"] = shap_new.index + 1
    new_shap_path = round_dir / "catboost_shap_importance.csv"
    shap_new.to_csv(new_shap_path, index=False)

    # 조기 중단 확인
    round_stop = None
    if round_clean_auprc < prev_clean_auprc - CFG["clean_auprc_drop_tol"]:
        round_stop = f"clean AUPRC 하락 ({prev_clean_auprc:.4f}→{round_clean_auprc:.4f})"
    if round_robust_recall <= prev_robust_recall:
        no_improve_count += 1
        if no_improve_count >= CFG["patience_robust"]:
            round_stop = f"robust recall {CFG['patience_robust']}연속 미개선"
    else:
        no_improve_count = 0

    history.append({
        "round":          round_idx,
        "clean_auprc":    round(round_clean_auprc,   4),
        "robust_recall":  round(round_robust_recall,  4),
        "asr":            round(round_asr, 4),
        "no_improve":     no_improve_count,
        "stop_reason":    round_stop,
    })

    with open(CFG["loop_dir"] / "loop_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if round_stop:
        stop_reason = round_stop
        print(f"\n  *** 조기 중단: {round_stop} ***")
        break

    prev_clean_auprc   = round_clean_auprc
    prev_robust_recall = round_robust_recall
    current_models     = new_models
    current_shap_path  = new_shap_path

# ── 최종 보고 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Loop (shap_linesearch) Done!")
print("=" * 65)
print(f"\n  {'Round':>6}  {'clean_AUPRC':>12}  {'robust_recall':>13}  {'ASR':>7}  {'no_imp':>6}")
print("  " + "-" * 55)
for h in history:
    print(f"  {h['round']:>6}  {h['clean_auprc']:>12.4f}  "
          f"{h['robust_recall']:>13.4f}  {h['asr']:>7.4f}  "
          f"{h.get('no_improve',0):>6}")

print(f"\n  총 라운드: {len(history)-1}")
print(f"  중단 이유: {stop_reason or '최대 라운드 도달'}")
print(f"  결과 저장: {CFG['loop_dir']}")
print("=" * 65)
