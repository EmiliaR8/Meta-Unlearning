# Machine Unlearning for Continual Malware and Intrusion Classification

Two related but distinct lines of work. Both build on the same continual-learning
substrate — MADAR-style replay (Experience Replay + Knowledge Distillation +
Synaptic Intelligence) with naive and joint-retraining baselines — and both add a
per-task *unlearning* step. They differ in what that step is for, and they live in
separate directories with no shared code.

| | Part A — Class-IL malware families | Part B — Adversarial intrusion detection |
|---|---|---|
| Data | EMBER 2018, EMBER 2024, LAMDA | CICIDS2017 |
| Task structure | class-incremental: new malware families arrive each task | chronological 10-task split, fixed label space |
| What is unlearned | a fraction of the current task, or of the replay buffer | samples a detector flags as adversarially perturbed |
| Research question | **which** samples are worth forgetting — hand-crafted rules vs. a learned policy | whether unlearning suspected poison restores robustness |

The word "unlearning" therefore means something different in each half. In Part A
the forget set is chosen by a *selection policy* and the object of study is that
policy. In Part B the forget set is chosen by a *perturbation detector* and the
object of study is the adversarial dynamic. Read the two halves independently.

> **Note.** `model.py` exists in both directories with entirely different contents
> (`EmberNN` in Part A, `XGBoostIDSWrapper` in Part B). They must not be merged
> into one directory.

---

# Part A — Class-Incremental Malware Family Classification

## The question

Under a bounded replay budget, a continual learner must decide what to keep. This
work asks the complementary question: given that some samples will be actively
*unlearned* each task, which ones should they be? Three answers are compared —
a random baseline, hand-crafted heuristics motivated by the replay literature, and
a small policy (**NN-1**) meta-trained by Evolution Strategies to maximise
end-of-sequence accuracy.

The same protocol runs on three corpora so that any result can be checked for
corpus-independence, and two secondary studies vary classifier capacity and
sequence length.

## Corpora and protocols

| | EMBER 2018 | EMBER 2024 | LAMDA |
|---|---|---|---|
| Feature dim | 2381 | 2568 | 4561 (binary) |
| Schedule | 50 + 5×10 = 100 families | 50 + 5×10 = 100 families | 30 + 5×10 = 80 families |
| Family eligibility | ≥200 train samples, top-N by frequency | ≥200, top-N | ≥200, top-N |
| Feature scaling | `StandardScaler` fit on task 0 only, clipped to ±10 | same | **none** (features are binary) |
| Task-0 training | 30 epochs, Adam 1e-3 | 30 epochs | 30 epochs |
| CL budget | 2000 iterations/task | 8000 | elbow-derived; 1000 and 4000 both reported |
| Replay buffer | 5000 | 10000 | 2500 |
| SI strength | 1.0 | 100.0 | 100.0 |
| Per-family cap | none | 10000 (`--family-cap`) | none |

Scaling is fit on task 0 alone in every corpus: fitting on the full training set
would leak the statistics of unseen families into the continual protocol.

**LAMDA requires a build step.** The IQSeC-Lab `LAMDA-class-il` curation defines
which samples and families are in the study but ships no feature vectors; the main
`LAMDA` repo ships 4561-d Baseline features but not the curation.
`build_lamda_cache.py` joins them by sha256 into a **raw feature store** — all 145
curated families, both splits, no family selection, no remapping, no scaling.
Selection is deliberately left downstream, because the primary (random-split,
80-family) and secondary (temporal-split, 63-family) experiments select different
family sets. The build verifies feature dimensionality, checks that the `feat_N`
column indices are contiguous, and asserts that every joined row carries
`label == 1` — the curation is malware-only, so a benign hit would indicate hash
misalignment that is otherwise undetectable across 4561 anonymous binary columns.

## Stage 1 — the four conditions

Every corpus runs the same four conditions. They share the network, the family
selection, the scaler, the task schedule, the masked-logit evaluation over
`[:active_count]`, and an identical determinism block; only the training rule
differs.

| Condition | Training rule |
|---|---|
| **Naive** | fine-tune on the current task only. Lower bound. |
| **Joint** | discard the model and retrain from scratch on all data seen so far. Oracle upper bound; not a realistic continual setup. |
| **MADAR** | replay from a bounded buffer selected by IsolationForest in the classifier's latent space, plus knowledge distillation against the previous task's model and a Synaptic Intelligence penalty. |
| **MADAR + Unlearning** | MADAR, plus a per-task forget phase: select a fraction of the current task, drop it from the buffer rebuild, and train its predictions toward uniform. |

```
                         EMBER 2018                    EMBER 2024                      LAMDA
Naive       clmd_ember18_naive_cd.py      clmd_ember24_naive_cd.py    clmd_lamda_classil_naive_cd.py
Joint       clmd_ember18_joint_cd.py      clmd_ember24_joint_cd.py    clmd_lamda_classil_joint_cd.py
MADAR       clmd_ember18_mk9_cd.py        clmd_ember24_madar_cd.py    clmd_lamda_classil_mk9_cd.py
MADAR+U     clmdu_ember18_mk6B_cd.py      clmdu_ember24_madaru_cd.py  clmdu_lamda_classil_mk6B_cd.py
```

### The unlearning step

The forget phase minimises

```
L = α · KL(uniform ‖ p)  +  (1 − α) · (½ CE + ½ KD)  +  c · SI
```

over the selected samples. **α = 0 is the control**: the forget set is still
selected and still removed from the buffer, and the number of gradient steps is
unchanged — only the push toward uniform is absent. The difference between α = 0.2
and α = 0 therefore separates the value of *curation* (which samples are dropped)
from the value of the *forget objective* itself. Both arms are run; the paper
reports whichever improves more over MADAR, and the compilers record which arm was
chosen and by how much.

`--split_option` selects the hand-crafted rule: `b` (the default) ranks the task's
samples by IsolationForest score in latent space and forgets a slab from the
*middle*, keeping both the most anomalous and the most prototypical — the same
prior MADAR uses to *choose* buffer contents, applied to deletion. `a` fits the
forest in raw feature space and forgets everything the buffer budget did not
select.

## Meta — learned forget-set selection

This is the core contribution. It replaces the hand-crafted rule with a policy
trained to maximise final accuracy.

**NN-1** (`nn1_scorer.py`) scores every candidate sample from per-sample features,
z-scored within the task so scales are comparable across tasks and corpora. It is
deliberately tiny — linear by default, 8 or 10 parameters — so that Evolution
Strategies can train it from scalar rewards, and so that each learned weight is
directly readable as "does this property make a sample worth forgetting?".

Two selection problems, with different feature sets:

- **Mode A** (7 features) — choose among the *current task's* samples. Features:
  IsolationForest score in latent space, the same in raw space, cross-entropy loss,
  prediction entropy, top-1-minus-top-2 margin, latent distance to the family
  centroid, log family size.
- **Mode B** (9 features) — choose among *replay-buffer* points, scored against the
  incoming task. Adds gradient conflict with the new task and a density ratio under
  the new distribution.

Selectors compared, by mode:

| Mode A | Mode B |
|---|---|
| `donut` (hand-crafted), `random`, `nn1` | `none` (no unlearning), `random`, `gradconflict`, `densityratio`, `nn1` |

`gradconflict` and `densityratio` are the interference-based criteria the replay
literature would suggest; `none` is MADAR untouched.

### Pipeline

| File | Role |
|---|---|
| `mu_inner_loop.py` | one continual-learning run under one selector — the unit of work. Handles all three corpora; writes a reward JSON. |
| `es_meta_train.py` | Evolution Strategies over NN-1's weights. Mirrored (antithetic) sampling, rank-shaped fitness, and common random numbers within a generation so reward differences reflect the policy rather than run-to-run noise. Evaluates candidates by spawning `mu_inner_loop.py` across GPUs. |
| `compare_selectors.py` | multi-seed comparison of selectors at one fixed configuration. Produces the paired per-seed differences that the ES log cannot: ES evaluates its baselines at a single seed while θ moves. |
| `nn1_scorer.py` | the scorer itself: scoring, top-`ratio` selection, save/load. |

Each `mu_inner_loop.py` run records `per_task` (micro accuracy over all seen
families), `per_task_macro` (mean per-family recall), `per_task_new_fam` (accuracy
on the families just added) and `per_task_task0` (retention), plus a `config` block
naming the corpus, mode, and every protocol parameter. The last two make the
acquisition/retention decomposition possible: `per_task` averages two opposing
quantities, so a selector that trades one for the other is invisible in the
aggregate.

**Feature ablation.** `--zero_features` (accepted by all three scripts) zeroes named
feature columns so they cannot contribute to the score. The ablation is stored
inside the scorer `.npz`, because ES still perturbs the weight of a zeroed feature
— evaluating such a scorer *without* the same ablation would silently activate a
weight the policy was never trained against.

### Reading the paired-difference panels

Every meta figure has two panels. The left shows accuracy by task; the right shows,
for each selector, its accuracy **minus a reference selector's accuracy on the same
seed**. The subtraction is what makes sub-percentage-point effects legible: run-to-
run variation is around a full point and is *shared* across selectors, because
within a workdir every selector uses the same seed and the same task-0 checkpoint.
Differencing cancels it. Units are percentage points (pp) — absolute differences,
not relative.

Pairing is only valid *within* a workdir. Comparing absolute levels across workdirs
built on different task-0 checkpoints is not a paired comparison, and the compilers
say so where it occurs.

## Secondary studies (EMBER 2018)

### Capacity — `arch_*.py`, `compile_arch_study.py`

Does the unlearning gain depend on classifier capacity? The four stage-1 conditions
are re-run with the network width selectable via `--arch`, drawing `EmberNN` from
`model.py`. Presets span roughly 10× in parameters:

| preset | widths | params |
|---|---|---|
| `full` | 1024, 512, 256, 128 | 3.14M |
| `trunk` | 512, 256, 128, **128** | 1.42M |
| `half` | 512, 256, 128, 64 | 1.40M |
| `quarter` | 256, 128, 64, 32 | 0.66M |
| `tiny` | 128, 64, 32, 32 | 0.32M |

`trunk` is not a capacity point. It matches `half` in parameter count to within
1.1% but holds the latent dimension at 128 — and that latent is the space in which
MADAR's buffer selection, the donut rule, and three of NN-1's features all operate.
Shrinking it would confound capacity with selection geometry, so `trunk` vs `half`
isolates the two.

**The headline statistic is not the raw gap.** A smaller network forgets more, so
the MADAR baseline falls, headroom opens, and *any* intervention gains more
percentage points; a growing raw gap across widths is the expected null, not a
finding. The capacity-comparable quantity is the fraction of available headroom
recovered:

```
d(arch, seed) = MU(seed) − MADAR(seed)                 [paired: shared task 0]
recovery(arch, seed) = d(arch, seed) / (mean_joint − mean_madar)
```

which is why joint retraining must be run at every architecture — it is the
denominator, not decoration. The denominator uses cell means deliberately: joint is
stable and expensive, so it runs at fewer seeds, and pairing it per-seed would
inject noise into every recovery value for no gain. Within an architecture MADAR and
MU share the task-0 model at a given seed, so `d` is paired and tested one-sample;
*across* architectures no pairing exists (different layer shapes consume the RNG
stream differently), so the interaction test is two-sample.

`compile_arch_study.py` also reports `d0 = MU(α=0) − MADAR`, the curation-only gain,
and checks the SI penalty magnitude per architecture — `si_loss` is a raw sum over
parameters, so its scale varies with width and it must be confirmed inert rather
than assumed.

### Sequence length — `t20_*.py`, `compile_t20.py`

Does the gain hold over a longer sequence? The schedule extends to 50 + 5×20 = 150
families at the same ≥200-sample threshold, so task-0 size and increment size are
unchanged and sequence length is the only variable. Because the eligible family list
is frequency-sorted then sliced, the first 100 families are exactly the 100-family
protocol's families in the same order — tasks 1–10 are therefore directly comparable
between the two protocols, and tasks 11–20 are the extension.

One caveat is inherent: the replay buffer is shared across all seen families, so a
longer sequence is also a thinner per-family buffer (5000/100 = 50 vs 5000/150 = 33).
`--mem_size 7500` holds per-family depth constant if that needs isolating.

## Compiling results

Each corpus has one compiler. All three delegate rendering to `paper_figs.py`, so
figures and tables cannot drift between datasets; only discovery is corpus-specific.

```bash
python compile_all_2018.py  --root ~/Thesis --out-dir paper
python compile_all_2024.py  --root ~/Thesis --out-dir paper
python compile_all_lamda.py --root ~/Thesis --out-dir paper
```

Each writes, named `{dataset}_{category}_{content}`:

| Output | Contents |
|---|---|
| `{ds}_stage1_accuracy.png` + `.csv` | naive / joint / MADAR / MADAR+Unlearning by task |
| `{ds}_stage1_accuracy_zoom.png` | the same with naive omitted, to separate the rest |
| `{ds}_stage1_decomposition.png` + `.csv` | accuracy on the families just added, and on task 0 |
| `{ds}_meta_accuracy.png` + `.csv` | every selector, plus paired per-seed differences |
| `{ds}_meta_decomposition.png` + `.csv` | acquisition and retention, by selector |
| `{ds}_results.md` | every figure with its table |

Every table carries mean ± std of per-run task-averaged accuracy (the std measures
run-to-run reproducibility, matching the shaded bands in the figures), final
accuracy, and final task-0 accuracy.

**Corpus-specific discovery.** EMBER 2018 uses fixed filename patterns. EMBER 2024
tags every history file with `cap`/`arch`/`iters`/`si` and several operating points
coexist on disk, so `compile_all_2024.py` filters to one before averaging and prints
how many files it excluded. LAMDA ran stage 1 at more than one CL_ITERS budget and
the paper reports both, so `compile_all_lamda.py` emits a complete set per budget
(`lamda_it{N}_*`); naive and joint are epoch-based, carry no budget tag, and are
reused across budgets rather than duplicated.

`compile_arch_study.py` and `compile_t20.py` handle the secondary studies, and
`compile_meta_curves.py` / `compile_meta_decomp.py` render either view for an
arbitrary meta workdir.

## Datasets — Part A

None of the corpora are included. Expected layout:

```
./Datasets/ember2018/            EMBER 2018 vectorised features (built by the `ember` package on first run)
./Datasets/embersim-databank/    avclass family labels (data/raw/ember_original_metadata.csv)
./Datasets/ember2024/            EMBER 2024 features
./Datasets/lamda-class-il/       IQSeC-Lab LAMDA-class-il curation (CSV/JSON)
./Datasets/lamda/                IQSeC-Lab LAMDA Baseline parquets
./Datasets/lamda_class_il_cache.npz   built by build_lamda_cache.py
```

## Reproduction order

```bash
# 0. LAMDA only: build the feature cache (EMBER reads its own vectorised features)
python build_lamda_cache.py --preview-only     # inspect the schedule first
python build_lamda_cache.py

# 1. Stage 1 — four conditions per corpus, multiple seeds
python clmd_ember18_naive_cd.py  --seed 1
python clmd_ember18_joint_cd.py  --seed 1
python clmd_ember18_mk9_cd.py    --seed 1
python clmdu_ember18_mk6B_cd.py  --seed 1 --alpha 0.2
python clmdu_ember18_mk6B_cd.py  --seed 1 --alpha 0.0      # the control

# 2. Meta — build the shared task-0 checkpoint, meta-train, then compare
python mu_inner_loop.py --dataset ember2018 --make_checkpoint
python es_meta_train.py --dataset ember2018 --mode b --gpus 0 1 2 --tag full_b
python compare_selectors.py --dataset ember2018 --mode b --gpus 0 1 2 \
    --scorer_weights es_work_full_b/theta_final.npz --seeds 100 101 102 ... 109

# 3. Compile
python compile_all_2018.py --root . --out-dir paper
```

Seeds: stage 1 retrains task 0 per seed; meta runs share one task-0 checkpoint and
vary only the CL-phase seed. The two are therefore **not** seed-paired with each
other and are never differenced across that boundary.

## File overview — Part A

| File | Purpose |
|---|---|
| `clmd_ember18_naive_cd.py`, `clmd_ember18_joint_cd.py`, `clmd_ember18_mk9_cd.py`, `clmdu_ember18_mk6B_cd.py` | EMBER 2018 stage 1 |
| `clmd_ember24_naive_cd.py`, `clmd_ember24_joint_cd.py`, `clmd_ember24_madar_cd.py`, `clmdu_ember24_madaru_cd.py` | EMBER 2024 stage 1 |
| `clmd_lamda_classil_naive_cd.py`, `clmd_lamda_classil_joint_cd.py`, `clmd_lamda_classil_mk9_cd.py`, `clmdu_lamda_classil_mk6B_cd.py` | LAMDA stage 1 |
| `build_lamda_cache.py` | joins the LAMDA curation to the Baseline features by sha256 |
| `lamda_data.py` | shared LAMDA family selection and task schedule (`build_arrays_lamda`, `select_families`, `make_task_families`) — single source of truth |
| `nn1_scorer.py` | the learned per-sample scorer |
| `mu_inner_loop.py` | one CL run under one selector; all three corpora |
| `es_meta_train.py` | Evolution Strategies meta-training for NN-1 |
| `compare_selectors.py` | multi-seed selector comparison with paired differences |
| `model.py` | `EmberNN` with selectable widths (`get_dims`, `count_params`) |
| `arch_naive.py`, `arch_joint.py`, `arch_madar.py`, `arch_mu.py` | stage 1 with `--arch`, for the capacity study |
| `t20_madar.py`, `t20_mu.py`, `t20_mu_inner_loop.py` | the 20-task sequence-length study |
| `paper_figs.py` | shared figure/table engine for the three paper compilers |
| `compile_all_2018.py`, `compile_all_2024.py`, `compile_all_lamda.py` | per-corpus paper compilers |
| `compile_arch_study.py`, `compile_t20.py` | secondary-study compilers |
| `compile_meta_curves.py`, `compile_meta_decomp.py` | selector and decomposition views of any meta workdir |

## Known issues and caveats — Part A

- **`arch_joint.py` does not currently run.** A stray `.` on its own line (around
  line 180, left over from a comment edit) raises `IndentationError`. Delete that
  line. Existing `results_arch/joint_*.json` were produced beforehand and remain
  valid.
- **EMBER 2018 stage-1 logs have no families-just-added field.** `New_Fam_Acc` was
  added for EMBER 2024 and LAMDA but not retrofitted to the 2018 scripts, so the
  2018 acquisition panel needs those runs repeated after adding the evaluation. It
  cannot be back-filled: no model checkpoints are retained.
- **The original EMBER 2018 meta workdirs predate the decomposition fields.**
  `compare_a_full` and `compare_b_fullES` carry only `per_task`, so the 2018 meta
  decomposition figure and table are omitted rather than emitted blank.
- **Meta workdirs do not all share (α, unlearn_epochs).** Mode A and mode B sweeps
  were run at different operating points on some corpora. The compilers detect this
  and emit a note; it belongs in the figure caption, not in silence.
- **Choosing the better α arm is a researcher degree of freedom.** The compilers
  print both arms' scores and record which was selected, so the choice is visible.
- **LAMDA mode A has no `nn1` run**, so only `donut` and `random` appear for that
  mode on that corpus.
- **`compare_selectors.py` is resumable** — it skips any (selector, seed) whose JSON
  already exists. Re-running into an existing workdir therefore does nothing.
  Earlier versions wrote `{selector}_s{seed}.json` with no corpus, mode or task
  count in the name, which allowed an EMBER result to be silently reused for a LAMDA
  run; filenames now carry the configuration and cached files are validated against
  the requested run.

---

# Part B — Adversarial Continual Learning on CICIDS2017

Continual-learning intrusion detection over a pooled, chronologically ordered
10-task split of CICIDS2017, with an adversarial red-team agent that adaptively
poisons each task's data. Four continual-learning strategies share the same task and
poisoning setup, so results are directly comparable across them.

## Strategies

- **Naive** (`naive_cl_pipeline.py`) — sequential fine-tuning only, no replay.
  Baseline lower bound.
- **Joint** (`joint_cl_pipeline.py`) — fresh full retrain on the pooled data at
  every task boundary. Upper bound; not a realistic continual setup.
- **MADAR** (`madar_cl_pipeline.py`) — Experience Replay + Knowledge Distillation +
  Synaptic Intelligence, with a bounded replay buffer selected via IsolationForest
  over the classifier's latent embedding space.
- **MADAR + Unlearning** (`madar_unlearning_cl_pipeline.py`) — MADAR plus a per-task
  forgetting mechanism: a small perturbation classifier flags samples it believes
  were adversarially perturbed, and those are actively unlearned (pushed toward
  uniform prediction) rather than simply retained.

## The adversarial mechanism

Each task after task 0's warm start trains **four** red-team agents against the
classifier as it stood at the end of the previous task:

- `red_train_pert_agent` / `red_test_pert_agent` — perturb malicious train/test
  samples toward being misclassified as benign (evasion).
- `red_train_benign_pert_agent` / `red_test_benign_pert_agent` — the mirror-image
  objective: perturb *benign* samples toward being misclassified as malicious.

All four register in a shared, recency-weighted contrastive bank that penalises
similarity to every previously trained agent, in the same task and all earlier ones,
so each converges on a distinct perturbation strategy rather than four agents
rediscovering the same attack.

Successfully evasive perturbations — and only those — become poison candidates, and
are substituted into that task's train/test data at a flat rate per side
(`POISON_FRACTION_FLAT`, default 0.30). Ground-truth labels are left unchanged on
substitution; only features are altered. This yields four sample categories, tracked
purely for diagnostics: `benign`, `malicious_clean`, `malicious_perturbed`,
`benign_perturbed`.

In MADAR + Unlearning, `perturbation_classifier` is a 4-class RandomForest trained
each task on a small oracle-seeded sample per class
(`PERTURBATION_CLASSIFIER_N`, default 50/class) to distinguish those four
categories. **Its predictions — never oracle ground truth — determine the forget
set**, and (via `CLEAN_REPLAY_BUFFER_OF_PERTURBED`) which buffer entries are cleaned
out and refilled. This is the key methodological difference from Part A: the forget
set comes from a fallible detector, not from a selection policy, and the detector's
error rate is part of what is being measured.

See the module docstrings at the top of `madar_cl_pipeline.py` and
`madar_unlearning_cl_pipeline.py` for the full mechanics and design rationale.

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

Each run writes `runs/<strategy>/<log_name>/`:

- `<log_name>.json` — full per-task results: evaluation metrics, red-agent reports,
  replay-buffer composition, poisoned sample ids, and for MADAR + Unlearning the
  unlearning and perturbation-classifier diagnostics.
- `plots/` — per-task diagnostics: accuracy curves, red-agent prototype heatmaps,
  episode embedding clouds, clean-vs-perturbed scatter plots.
- `logs/` — per-agent episode timing and prototype cosine-similarity logs.

`compare_cl_runs.py` compares metrics across two or more result JSONs.

## File overview — Part B

| File | Purpose |
|---|---|
| `naive_cl_pipeline.py` | Naive baseline strategy |
| `joint_cl_pipeline.py` | Joint baseline strategy |
| `madar_cl_pipeline.py` | MADAR (ER + KD + SI) strategy |
| `madar_unlearning_cl_pipeline.py` | MADAR + Unlearning strategy |
| `adversary_env.py` | Red-agent RL environment (`NetworkAttackEnv`) and contrastive prototype bank |
| `h5_data_loader.py` | Loads and pools the chronological CICIDS2017 task splits from the `.h5` dataset |
| `model.py` | `XGBoostIDSWrapper` (used by Naive and Joint) |
| `train_classifer.py` | Classifier training helper used by `model.py` |
| `compare_cl_runs.py` | Compares metrics across run result JSONs |

## Data

These pipelines expect a pre-processed, `[0, 1]`-scaled CICIDS2017 HDF5 file
(default `/mnt/processed_data/subsampled_dataset.h5`, override with `--h5-path`).
The dataset is not included in this repository.
