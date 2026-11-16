"""Configuration: defaults, per-dataset protocol, CLI overrides.

Precedence, lowest to highest:

    BASE_HYPERPARAMS  ->  DATASET_PROTOCOL[dataset]  ->  --config file  ->  CLI flag

A dataset's entry holds the parameters that are PROTOCOL for that corpus (its
replay budget, SI strength, iteration budget), so `--dataset ember2018` selects a
protocol rather than requiring six flags to be remembered correctly. An explicit
flag always wins, and whatever wins is recorded in the run's JSON -- there is no
value in effect that the log does not name.
"""

from __future__ import annotations

import json
from pathlib import Path

# Parameters shared by every corpus.
BASE_HYPERPARAMS = {
    # task 0 / epoch-based training
    "task0_epochs": 30,
    "task0_lr": 1e-3,
    "batch_size": 256,
    "eval_batch_size": 512,
    # continual phase
    "cl_iters": 2000,
    "cl_lr": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-6,
    "grad_clip": 1.0,
    "kd_temp": 2.0,
    "rnt_mode": "inverse",
    "rnt_floor": 0.25,
    "rnt_value": 0.5,
    # Retain the 1/(t+1) scalar in ablations that removed the KD term it was
    # weighting against. Off by default there -- see train_continual.
    "keep_rnt": False,
    "freeze_bn": True,
    # replay
    "mem_size": 5000,
    "contamination": 0.1,
    # 'raw' fits the Isolation Forest on input features (MADAR as published);
    # 'latent' fits it on the classifier's representation (MADAR-theta).
    "buffer_space": "latent",
    # Which part of each family's Isolation Forest ranking the buffer keeps.
    #   both           -- MADAR as published: half most anomalous, half most
    #                     prototypical, interleaved. The default.
    #   anomalous      -- the most anomalous n_select only, so the
    #                     prototypical end never enters replay at all.
    #   representative -- the most prototypical n_select only.
    #   middle         -- the band MADAR discards.
    # `both` refills from whatever survives the forget set, so removing a
    # band from the TASK cannot keep it out of the BUFFER -- the next rung
    # is simply promoted. This knob can.
    "buffer_fill": "both",
    # Which policy FILLS the memory. 'madar' is the diversity-aware Isolation
    # Forest selection; 'reservoir' is uniform sampling over the stream, which
    # is what A-GEM specifies. Methods that bring their own policy use it by
    # default so a baseline is not silently reported as a hybrid.
    "buffer_policy": "madar",
    # synaptic intelligence
    "si_c": 1.0,
    "si_eps": 0.1,
    # unlearning
    "alpha": 0.2,
    "forget_ratio": 0.10,
    # Which samples the mode-A forget set may be drawn from.
    #   all        -- the whole arriving task. The selector runs BEFORE the
    #                 buffer is built, so its choices also change what MADAR
    #                 gets to keep: an arm's accuracy then reflects curation
    #                 and the forget objective together.
    #   complement -- only what MADAR was going to discard anyway. The buffer
    #                 is then identical across arms by construction, so a
    #                 difference between arms is attributable to the forget set
    #                 alone. Costs some freedom, buys a cleaner comparison.
    # Arms are only comparable to each other WITHIN one setting of this.
    "forget_pool": "all",
    # --- gnb: the non-forgetting-classifier control ----------------------
    # Gaussian naive Bayes keeps per-class sufficient statistics, so a family's
    # parameters cannot be changed by anything another family contributes.
    # Buffer exclusion therefore cannot implement forgetting there, and the
    # arms separate exclusion from removal:
    #   A      -- no forget set at all.
    #   B      -- the forget set is excluded from the buffer rebuild and
    #             nothing else. It was already folded into the statistics
    #             during the task, so its influence is untouched. This is the
    #             analogue of madar_unlearn at alpha = 0.
    #   C      -- the same forget set is subtracted from the statistics,
    #             exactly rather than approximately.
    #   C-rand -- a random forget set with the SAME PER-FAMILY COUNTS, removed
    #             the same way. Without it, "removal helped" cannot be told
    #             apart from "any removal helps" or "less data helps".
    "gnb_arm": "A",
    # Fold the replay buffer into the fit as well. Off is the clean form: the
    # statistics are already cumulative, so replay is redundant and arm B is
    # then IDENTICAL to arm A by construction -- which is the claim, stated as
    # an identity. On makes B-vs-A empirical instead, at the cost of
    # double-counting replayed samples. Both are worth running.
    "gnb_replay": False,
    # Path to the madar_unlearn record whose forget sets are replayed. `donut`
    # fits an Isolation Forest on the NETWORK's latents, so the selection
    # cannot be re-derived from the data and seed alone -- it has to come from
    # that run. Required for B/C/C-rand; arm A selects nothing and ignores it.
    # The loader refuses a record whose dataset, task_setup, seed, protocol or
    # forget_ratio disagrees with this run.
    "forget_from": "",
    "unlearn_epochs": 3,
    "unlearn_lr": 1e-4,
    "unlearn_grad_clip": 0.5,
    # Matched-memory control for mode B. 'none' lets the buffer shrink by
    # whatever the selector removed (the method as such); 'family' resamples
    # each family back to its pre-removal slot count so arms differ only in
    # WHICH points they kept. See ReplayBuffer.refill_to.
    "refill_policy": "none",
    # Depth cap for mode B: the most of ANY ONE family's slots a single task's
    # removal may take, as a fraction. 0 = uncapped. Volume is preserved -- the
    # ranked order is walked and a point is admitted only while its family is
    # under cap -- so capping equalises DEPTH across selectors without changing
    # how many points each removes. See meta_unlearn.select_forget_set.
    "forget_per_family_cap": 0.0,
    # Validation slice carved from TRAIN, per family. 0 = off (the whole train
    # split is used and the ES reward reads the test split, which is what the
    # meta driver did before this existed). `val_seed` is fixed independently
    # of the run seed so every condition shares one split.
    "val_fraction": 0.0,
    "val_seed": 4242,
    # data
    "min_family_samples": 200,
    # --- DER family (xder, star_xder, deduce_star_xder) ------------------
    # The X-DER paper's grid lives in supplemental material not on hand.
    # alpha/beta sit inside the range the STAR paper reports for DER++
    # (Table 8: alpha 0.15-0.4, beta 0.1-0.4); gamma and margin are stated in
    # X-DER's main text. lambda, eta and the temperature are OURS and untuned.
    # --- GCR -------------------------------------------------------------
    # alpha/beta mirror the DER settings above on purpose: GCR and X-DER should
    # differ in HOW THE BUFFER IS CHOSEN, not in how hard each replay term is
    # pulled. The paper's own grid (lr 0.03, alpha 0.1-0.5, beta 0.1-1.0,
    # gamma 0.01-0.1) was tuned on CIFAR/TinyImageNet with a ResNet-18 and does
    # not transfer to a malware MLP. gamma, lambda and the temperature are OURS
    # and untuned; the paper states no temperature at all.
    "gcr_alpha": 0.3,            # weighted distillation, Eq. (5b)
    "gcr_beta": 0.2,             # weighted label loss, Eq. (5c)
    "gcr_gamma": 0.1,            # weighted supervised contrastive, Eq. (5d)
    "gcr_temp": 0.1,             # supcon temperature; OURS
    "gcr_lambda": 1.0,           # l2 on the coreset weights, Eq. (4)
    "gcr_task0": "shared",       # 'shared' (comparable task 0) or 'full'
    "xder_alpha": 0.3,           # weight on logit replay, Eq. (5)
    "xder_beta": 0.2,            # weight on the memory cross-entropy, Eq. (13)
    "xder_lambda": 0.05,         # weight on future preparation, Eq. (9); 0 = off
    "xder_eta": 0.1,             # weight on the past/future constraint, Eq. (11)
    "xder_gamma": 0.75,          # implant attenuation, Eq. (7) -- paper's value
    "xder_margin": 0.3,          # m in Eq. (11) -- paper's value
    "xder_temp": 0.1,            # contrastive temperature in Eq. (8)
    "xder_implant": True,        # false reproduces "X-DER w/o memory update"
    "xder_task0": "shared",      # 'shared' | 'full' -- see experiments/xder.py
    "xder_aug": "both",          # both | noise | dropout | none
    "xder_aug_sigma": 0.1,       # noise scale, in batch standard deviations
    "xder_aug_drop": 0.1,        # fraction of features zeroed
    "logit_buffer_policy": "balanced",   # balanced | reservoir
    # --- STAR (Eskandar et al., ICLR 2025) -------------------------------
    # Paper Table 8 for X-DER-RPC + STAR: gamma 0.001-0.05, lambda 0.01-0.05.
    "star_lambda": 0.01,         # weight on the stability term, Eq. (8)
    "star_gamma": 0.001,         # perturbation ratio ||delta_l|| / ||theta_l||
    "star_eps": 1e-3,            # scale of the delta_0 noise, Eq. (9)
    "star_steps": 1,             # ascent steps for delta; the paper uses 1
    # --- DEDUCE (Wang et al., ICLR 2026) ---------------------------------
    "deduce_detect": "gradient",  # gradient | bound -- OUR(G) / OUR(B)
    "deduce_epsilon": 0.0,       # tolerance in Eq. (7); paper uses 0.0
    "deduce_detect_every": 1,    # batches between detections; 1 is the paper
    "deduce_delta": 0.001,       # local unlearning rate, paper's value
    "deduce_phi": 1e-5,          # global unlearning rate, paper's value
    "deduce_alpha": 1.0,         # proximal weight in Eq. (9)
    "deduce_beta": 0.1,          # FIM regulariser weight in Eq. (11)
    "deduce_eta": 0.99,          # contribution decay in Eq. (12)
    "deduce_maturity": 100,      # m, steps before a neuron may be reset again
    "deduce_k": 10,              # theta_t^k anchor: batches into the task
    "deduce_fim_batches": 32,    # batches used to estimate the diagonal FIM
    "deduce_fim_damping": 1e-3,  # added to F before inversion in Eq. (10)
    "deduce_bound_c": 1.0,       # scaling factor c in Eq. (4)
    # --- meta / NN-1 (meta_unlearn) --------------------------------------
    # Empty means "no scorer supplied", which every selector but `nn1` is fine
    # with. Kept as strings so they travel through the generic --<name> CLI
    # plumbing and land in the run record, where an ablation is then legible
    # from the result file rather than only from whoever ran it.
    "scorer_weights": "",        # path to a saved NN-1 .npz
    "zero_features": "",         # space/comma separated feature names to zero
    # MalCL only
    "malcl_epochs": 50,
    "malcl_k": 3,
    "malcl_z_dim": 62,
    "family_cap": 0,
    "cap_seed": 12345,
    "feature_clip": 10.0,
}

# Per-corpus protocol. Only the values that genuinely differ.
DATASET_PROTOCOL = {
    "ember2018": {"cl_iters": 2000, "mem_size": 5000, "si_c": 1.0,
                  "family_cap": 0, "min_family_samples": 200},
    # 8000 iters keeps epochs-per-sample comparable given 2024's larger tasks;
    # the cap compresses a ~94:1 family imbalance.
    "ember2024": {"cl_iters": 8000, "mem_size": 10000, "si_c": 100.0,
                  "family_cap": 10000, "min_family_samples": 200},
    # mem 2500 is 1.85% of task 0, matching the EMBER ratio. si_c 100 for
    # cross-corpus consistency (measured near-inert on this corpus). No cap: any
    # cap tight enough to dent 128:1 discards most of the corpus.
    "lamda_classil": {"cl_iters": 4000, "mem_size": 2500, "si_c": 100.0,
                      "family_cap": 0, "min_family_samples": 200},
    # Tiny ImageNet: 200 classes x 500 images, all classes exactly balanced, so
    # min_family_samples and family_cap have nothing to do -- both are set to
    # their inert values rather than left at the EMBER defaults, where
    # min_family_samples=200 would be a filter that happens to pass everything
    # and would quietly start dropping classes if the corpus were ever subset.
    # mem_size 2000 is the buffer the X-DER and STAR papers report for Tiny
    # ImageNet, which is what makes those numbers comparable to these; it is 2%
    # of the training set, close to EMBER 2018's 2.5%.
    #
    # cl_iters 3000 at batch 32 is 96,000 samples per increment. Against the
    # 100+10x10 schedule (5,000 images per increment) that is 19 epochs; against
    # 20+20x9 (10,000 images) it is 10. Both are SHORT of the ~50 epochs per
    # task the X-DER and STAR papers train for on this benchmark, and the gap
    # matters when reading their tables next to these: matching them needs
    # roughly --cl-iters 7800 on 100+10x10 or 15600 on 20+20x9, which is a five-
    # fold increase in a sweep that already runs 3 seeds x 11 methods. 3000 is
    # chosen as the point where the methods separate at a cost that fits the
    # budget; it is a protocol choice, not a reproduction of theirs.
    "tinyimagenet": {"cl_iters": 3000, "mem_size": 2000, "si_c": 1.0,
                     "family_cap": 0, "min_family_samples": 1,
                     "batch_size": 32, "cl_lr": 0.03, "task0_epochs": 50,
                     "task0_lr": 0.03, "feature_clip": 0.0,
                     "buffer_space": "latent"},
    "synthetic_image": {"cl_iters": 40, "mem_size": 200, "si_c": 1.0,
                        "family_cap": 0, "min_family_samples": 1,
                        "batch_size": 32, "task0_epochs": 2,
                        "unlearn_epochs": 1, "feature_clip": 0.0,
                        "buffer_space": "latent"},
    "synthetic": {"cl_iters": 60, "mem_size": 300, "si_c": 1.0,
                  "task0_epochs": 3, "unlearn_epochs": 1,
                  "min_family_samples": 20, "batch_size": 64},
}

# Per-METHOD defaults, applied after the dataset protocol and before any CLI
# flag. Only for parameters a method defines for itself -- a baseline that
# brings its own memory policy should use it without the caller remembering to
# ask, or the row silently becomes a hybrid.
# A corpus whose examples are images cannot run the MLP presets, so the runner's
# `--model full` default has to give way. This is the ONLY place the default
# model varies by dataset; an explicit --model still wins.
DATASET_MODEL = {"tinyimagenet": "resnet18",
                 "synthetic_image": "resnet18_quarter"}


METHOD_DEFAULTS = {
    "agem": {"buffer_policy": "reservoir"},
    # The DER family keeps its own memory: (x, y, logits), class-balanced at the
    # same mem_size. Naming the policy here rather than in the experiment keeps
    # it in the run's recorded hyperparameters, where the aggregator can see it.
    "xder": {"buffer_policy": "logit"},
    "star_xder": {"buffer_policy": "logit"},
    "deduce_star_xder": {"buffer_policy": "logit"},
    # GCR keeps a LogitBuffer whose entries also carry a coreset weight, and
    # re-selects the whole thing by gradient matching at each boundary. Named
    # distinctly from "logit" because the difference between GCR and the DER
    # family IS the selection rule, and a record that called both "logit" would
    # hide the one thing the comparison is about.
    "gcr": {"buffer_policy": "coreset"},
}

HYPERPARAM_TYPES = {k: type(v) for k, v in BASE_HYPERPARAMS.items()}

PROTOCOLS = ("stage1", "meta")


def default_hyperparams(dataset: str, method: str | None = None) -> dict:
    """The values a run gets with no overrides at all."""
    hp = dict(BASE_HYPERPARAMS)
    hp.update(DATASET_PROTOCOL.get(dataset, {}))
    if method:
        hp.update(METHOD_DEFAULTS.get(method, {}))
    return hp


def overrides_of(dataset: str, resolved: dict, method: str | None = None) -> dict:
    """Which hyperparameters actually differ from this dataset's defaults.

    The run id hashes THIS rather than the full parameter set, so that adding a
    new knob with a default cannot change the identity of runs that predate it.
    Passing a value explicitly that happens to equal the default is not an
    override, so `--cl-iters 2000` and omitting it produce the same run.
    """
    defaults = default_hyperparams(dataset, method)
    return {k: v for k, v in resolved.items()
            if k not in defaults or v != defaults[k]}


def hyperparams_for(dataset: str, overrides: dict | None = None,
                    method: str | None = None) -> dict:
    hp = default_hyperparams(dataset, method)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in hp:
            raise SystemExit(
                f"unknown hyperparameter {k!r}. Known: {', '.join(sorted(hp))}")
        hp[k] = v
    return hp


def load_config_file(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--config {p} not found")
    with p.open() as f:
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise SystemExit("PyYAML is required to read a .yaml config")
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"--config {p}: expected a mapping at the top level")
    return data


def split_config(raw: dict) -> tuple[dict, dict]:
    """Separate run-identity keys from hyperparameters."""
    hp = {k: v for k, v in raw.items() if k in BASE_HYPERPARAMS}
    rest = {k: v for k, v in raw.items() if k not in BASE_HYPERPARAMS}
    return rest, hp


def describe(config: dict, hyperparams: dict) -> str:
    lines = ["configuration:"]
    for k in ("dataset", "task_setup", "method", "model", "selector", "protocol",
              "seed", "perm_seed", "group", "n_tasks"):
        if config.get(k) is not None:
            lines.append(f"  {k:14s} {config[k]}")
    lines.append("hyperparameters:")
    for k in sorted(hyperparams):
        lines.append(f"  {k:20s} {hyperparams[k]}")
    return "\n".join(lines)