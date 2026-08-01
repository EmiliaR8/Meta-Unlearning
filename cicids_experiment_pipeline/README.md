# CICIDS Experiment Pipeline

Continual-learning intrusion detection over a pooled, chronologically-ordered
10-task split of the CICIDS2017 dataset, with an adversarial red-team agent
that adaptively poisons each task's data. This repo holds the final,
paper-ready version of the pipeline: four continual-learning strategies
sharing the same task/poisoning setup, so results are directly comparable
across all of them.

## Strategies

- **Naive** (`naive_cl_pipeline.py`) -- sequential fine-tuning only, no
  replay. Baseline lower bound.
- **Joint** (`joint_cl_pipeline.py`) -- fresh full retrain on the pooled
  data at every task boundary. Baseline upper bound (not a realistic
  continual-learning setup, but a useful ceiling).
- **MADAR** (`madar_cl_pipeline.py`) -- Experience Replay + Knowledge
  Distillation + Synaptic Intelligence, with a bounded replay buffer
  selected via IsolationForest over the classifier's latent embedding
  space.
- **MADAR+Unlearning** (`madar_unlearning_cl_pipeline.py`) -- MADAR plus a
  per-task forgetting mechanism: a small "perturbation_classifier" flags
  samples it believes were adversarially perturbed, and those are actively
  unlearned (pushed toward uniform/max-entropy prediction) rather than
  simply retained.

## The adversarial mechanism (this version)

Each task (after task 0's warm start) trains **four** red-team agents
against the classifier as it stood at the end of the previous task:

- `red_train_pert_agent` / `red_test_pert_agent` -- perturb malicious
  train/test samples toward being misclassified as benign (evasion).
- `red_train_benign_pert_agent` / `red_test_benign_pert_agent` -- the
  mirror-image objective: perturb *benign* train/test samples toward being
  misclassified as malicious.

All four agents per task register in a shared, recency-weighted
contrastive bank, which penalizes similarity to every previously-trained
agent (same task and all earlier tasks) so each one converges on a
distinct perturbation strategy rather than four agents re-discovering the
same attack.

Successfully-evasive perturbations (only) become poison candidates and are
substituted into that task's train/test data at a flat rate per side
(`POISON_FRACTION_FLAT`, default 0.30) -- ground truth labels are left
unchanged on substitution, only the features are altered. This produces
four sample categories, tracked purely for diagnostics: `benign`,
`malicious_clean`, `malicious_perturbed`, `benign_perturbed`.

In MADAR+Unlearning, `perturbation_classifier` is a 4-class
RandomForestClassifier trained each task on a small oracle-seeded sample
per class (`PERTURBATION_CLASSIFIER_N`, default 50/class) to distinguish
the four categories above. Its predictions -- never oracle ground truth
directly -- determine the forget set unlearned each task, and (via
`CLEAN_REPLAY_BUFFER_OF_PERTURBED`) which buffer entries get cleaned out
and refilled.

See the module docstrings at the top of `madar_cl_pipeline.py` and
`madar_unlearning_cl_pipeline.py` for the full mechanics and design
rationale.

## Running an experiment

```bash
pip install -r requirements.txt

python madar_cl_pipeline.py --seed 42 --log_name madar_run1 \
    --h5-path /path/to/subsampled_dataset.h5

python madar_unlearning_cl_pipeline.py --seed 42 --log_name madar_u_run1 \
    --h5-path /path/to/subsampled_dataset.h5

python naive_cl_pipeline.py --seed 42 --log_name naive_run1 \
    --h5-path /path/to/subsampled_dataset.h5

python joint_cl_pipeline.py --seed 42 --log_name joint_run1 \
    --h5-path /path/to/subsampled_dataset.h5
```

Each run writes a `runs/<strategy>/<log_name>/` directory containing:
- `<log_name>.json` -- full per-task results (evaluation metrics, red-agent
  reports, replay buffer composition, poisoned sample ids, and, for
  MADAR+Unlearning, unlearning/perturbation-classifier diagnostics).
- `plots/` -- per-task diagnostic plots (accuracy curves, red-agent
  prototype heatmaps, episode embedding clouds, clean-vs-perturbed sample
  scatter plots).
- `logs/` -- per-agent episode timing and prototype cosine-similarity logs.

Use `compare_cl_runs.py` to compare metrics across two or more result
JSONs.

## File overview

| File | Purpose |
|---|---|
| `naive_cl_pipeline.py` | Naive baseline strategy |
| `joint_cl_pipeline.py` | Joint baseline strategy |
| `madar_cl_pipeline.py` | MADAR (ER+KD+SI) strategy |
| `madar_unlearning_cl_pipeline.py` | MADAR+Unlearning strategy |
| `adversary_env.py` | Red-agent RL environment (`NetworkAttackEnv`) and contrastive prototype bank |
| `h5_data_loader.py` | Loads and pools the chronological CICIDS2017 task splits from the `.h5` dataset |
| `model.py` | `XGBoostIDSWrapper` (used by Naive/Joint) |
| `train_classifer.py` | Classifier training helper used by `model.py` |
| `compare_cl_runs.py` | Utility for comparing metrics across run result JSONs |

## Data

These pipelines expect a pre-processed, `[0, 1]`-scaled CICIDS2017 HDF5
file (default path `/mnt/processed_data/subsampled_dataset.h5`, override
with `--h5-path`). The dataset itself is not included in this repository.
