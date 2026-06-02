# Robustness of Insider Threat Detection via SHAP-Guided Adversarial Attacks and Retraining

Official implementation of:

> **A Study on the Robustness of Insider Threat Detection via SHAP-Guided Adversarial Attacks and Retraining**

---

## Overview

This repository contains the full pipeline for evaluating and improving the robustness of insider threat detection models against adversarial attacks.

We propose a SHAP-guided adversarial attack framework that identifies and perturbs the most influential features of a CatBoost classifier, then investigates whether adversarial retraining (including curriculum-based strategies) can restore model robustness.

**Key contributions:**
- SHAP-guided adversarial attack strategies (multistart & line search variants)
- Systematic comparison of 6 attack methods across 4 perturbation budgets
- Adversarial retraining experiments (single-source, mixed-source, curriculum)
- Transfer matrix analysis evaluating generalization of defenses

---

## Pipeline

```
1__preprocess.py
      ↓
2__train_catboost.py
      ↓
3-1__attack_generation.py  /  3-2__attack_generation.py
      ↓
4-1__adversarial_retraining.py  /  4-2__adversarial_retraining_curriculum.py
      ↓
5__transfer_matrix.py
      ↓
6__adversarial_loop_linesearch.py
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `1__preprocess.py` | Feature extraction from CERT r4.2 dataset. Builds session-level features across logon, device, email, file, and HTTP logs. Includes leak-free delta features via expanding past mean. |
| `2__train_catboost.py` | 5-fold GroupKFold cross-validation training of CatBoost classifier. Computes ensemble-averaged SHAP importances across all folds. |
| `3-1__attack_generation.py` | attack generation with 8 attack methods. |
| `3-2__attack_generation.py` | attack generation with 6 attack methods. |
| `4-1__adversarial_retraining.py` | 5-arm adversarial retraining experiment (clean baseline + 4 retraining strategies including mixed-source). |
| `4-2__adversarial_retraining_curriculum.py` | Curriculum retraining: sequentially trains with increasing attack budget (B=1 → B=2 → B=3) using `init_model` on CPU. |
| `5__transfer_matrix.py` | Full transfer matrix evaluation: 8 defense arms × 14 attack methods × 4 budgets. Assesses generalization of learned defenses. |
| `6__adversarial_loop_linesearch.py` | Iterative adversarial loop using line search attack. |

---

## Attack Methods

| Method | Feature Selection | Search Strategy |
|--------|-------------------|-----------------|
| `shap_guided` | Union of global + local (rank-weighted) | Greedy / SHAP sign direction |
| `global_only` | Global SHAP top-K | Greedy / SHAP sign direction |
| `local_only` | Local SHAP top-M | Greedy / SHAP sign direction |
| `global_only_random` | Global SHAP top-K | Greedy / Random ± direction |
| `local_only_random` | Local SHAP top-M | Greedy / Random ± direction |
| `direction_random` | Union of global + local (rank-weighted) | Greedy / Random ± direction |
| `random` | Random (mutable feats) | Greedy / SHAP sign direction |
| `pure_random` | Random (mutable feats) | Greedy / Random ± direction |
| `global_only_multistart` | Global SHAP top-K | Random multi-start (5 trials) |
| `local_only_multistart` | Local SHAP top-M | Random multi-start (5 trials) |
| `shap_multistart` | Union of global + local (rank-weighted) | Random multi-start (5 trials) |
| `global_linesearch` | Global SHAP top-K | ± direction × step grid [0.1, 0.2, 0.3, 0.5] |
| `local_only_linesearch` | Local SHAP top-M | ± direction × step grid [0.1, 0.2, 0.3, 0.5] |
| `shap_linesearch` | Union of global + local (rank-weighted) | ± direction × step grid [0.1, 0.2, 0.3, 0.5] |

---

## Dataset

This code uses the **CERT Insider Threat Dataset r4.2**.

> Glasser, J., & Lindauer, B. (2013). Bridging the gap: A pragmatic approach to generating insider threat data. *IEEE Security and Privacy Workshops*.

The dataset is available upon request from the [CERT Division](https://www.sei.cmu.edu/our-work/projects/display.cfm?customel_datapageid_4050=21274).

Expected directory structure:
```
r4.2/
├── logon.csv
├── device.csv
├── email.csv
├── file.csv
└── http.csv

answers/
├── r4.2-1/
├── r4.2-2/
└── r4.2-3/
```

---

## Requirements

```bash
pip install catboost scikit-learn pandas numpy scipy
```

Tested with:
- Python 3.9+
- CatBoost 1.2+
- scikit-learn 1.3+
- CUDA-compatible GPU (recommended for training; CPU fallback available)

---

## Usage

```bash
# 1. Preprocess
python 1__preprocess.py

# 2. Train CatBoost
python 2__train_catboost.py

# 3. Generate adversarial examples
python 3-2__attack_generation.py

# 4. Adversarial retraining
python 4-1__adversarial_retraining.py

# 4-alt. Curriculum retraining
python 4-2__adversarial_retraining_curriculum.py

# 5. Transfer matrix evaluation
python 5__transfer_matrix.py
```

Outputs are saved under `./output/`.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{...,
  title   = {A Study on the Robustness of Insider Threat Detection via SHAP-Guided Adversarial Attacks and Retraining},
  author  = {...},
  journal = {...},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
