# Experiment runner

```
iclr/
  iclr                  launcher: ./iclr {run|sweep|aggregate|plots|paths}
  paths.json            committed path defaults
  paths.local.json      per-machine overrides (gitignored) -- the only file that
                        changes when storage moves to the mount
  code/
    runner/             what to run, and what to do with the results
      paths.py          storage resolution
      config.py         defaults, per-dataset protocol, overrides
      run.py            ONE run -> one JSON
      sweep.py          a matrix of runs, over GPUs
      aggregate.py      JSONs -> tidy CSVs + summary tables
      plots.py          JSONs -> figures
    experiments/        how each method differs. One file per method.
      base.py           the shared loop every method inherits
      naive.py joint.py madar.py madar_unlearn.py
    core/               the engine every method shares
      data/             corpus loaders + family selection + scaling
      tasks.py          task setups (50+5x10 and friends)
      models.py         EmberNN, width presets, parameter counts
      buffer.py         MADAR diversity-aware replay buffer
      training.py       task-0 fitting, the continual step, SI, evaluation
      selectors.py      forget-set selectors
      metrics.py        confusion matrix -> accuracy / precision / recall / F1
      records.py        the run-record schema
      seeding.py        determinism
```

## Running

```bash
./iclr paths --show                     # where is storage? (check this first)
./iclr migrate --group <g>              # after an upgrade: bring old records forward

./iclr run --dataset ember2018 --task-setup ember \
           --experiment madar --model full --seed 1 --group demo

./iclr sweep --dataset ember2018 --group demo \
             --experiment naive joint madar madar_unlearn \
             --seed 1 2 3 4 5 --gpus 0 1 2

./iclr aggregate --group demo           # -> Logs/demo/tables/
./iclr plots     --group demo           # -> Logs/demo/figures/
```

Figures show curves; for sub-point gaps read the tables, not the curves (see
`plots.py` on the paired-difference gap).

`--dry-run` on `run` or `sweep` resolves everything and prints it without
training. It is the fastest check that paths and hyperparameters are what you
think they are.

### The selectable axes

| flag | what it picks |
|---|---|
| `--dataset` | corpus: `ember2018`, `ember2024`, `lamda_classil`, `tinyimagenet`, `synthetic`, `synthetic_image` |
| `--task-setup` | schedule: `ember` (50+5x10), `lamda` (30+5x10), `ember_t20`, or any `<t0>+<step>x<n>` |
| `--experiment` | method: `naive`, `joint`, `madar`, `madar_unlearn`, `er_only`, `si_only`, `malcl`, `agem`, `xder`, `star_xder`, `deduce_star_xder` |
| `--model` | MLP width: `full`, `trunk`, `half`, `quarter`, `tiny`, or four widths. Convolutional: `resnet18`, `resnet18_half`, `resnet18_quarter`, `resnet10`. Defaults to the corpus's own. |
| `--selector` | forget set: `donut`, `random`, `leftover` (madar_unlearn only) |
| `--seed` | run seed |
| `--protocol` | `stage1` (task 0 retrained per seed) or `meta` (shared checkpoint) |
| `--group` | which `Logs/<group>/` the run lands in |
| any hyperparameter | `--cl-iters`, `--mem-size`, `--si-c`, `--alpha`, `--forget-ratio`, ... |

### Tiny ImageNet

The one non-malware corpus, and the only one whose examples are images rather
than feature vectors. 200 classes x 500 training images, 64x64 RGB; the
distributed test split is unlabelled, so the 10,000-image validation split is
used as the test set, as it is in every published result on this benchmark.

```bash
cd $ICLR_DATA_ROOT
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
```

The first run decodes 110,000 JPEGs and writes a uint8 cache under the cache
root; later runs memory-map it. Budget a few minutes for that first load and
about 1.3 GB of cache.

Two things behave differently here, both deliberate:

- **The backbone.** `--model` defaults to `resnet18` -- the 3x3-stem, no-maxpool
  variant the continual-learning literature uses for 64x64 inputs, not
  torchvision's ImageNet model. Asking for an MLP preset on this corpus is
  refused rather than reshaped, and so is asking for a convolutional preset on a
  malware corpus.
- **Two schedules.** `100+10x10` matches the shape of the malware protocols, so
  task-0 retention means the same thing across corpora. `20+20x9` is the ten
  equal tasks the X-DER and STAR papers report, so those numbers are comparable
  to these. Neither is the default for the other's purpose; pick with
  `--task-setup`.

`malcl` is refused on an image corpus -- its generator is a 1-D convolutional
network over a feature vector, and running it on flattened pixels would be a
different method wearing the same name. `--buffer-space raw` is refused for the
same class of reason: an Isolation Forest fitted on 12,288 raw pixel values is
not MADAR transplanted to images.

`synthetic_image` is to this path what `synthetic` is to the tabular one -- a
seconds-long smoke corpus that exercises the ResNet, the uint8 storage, the GPU
augmentation and DEDUCE's convolutional neuron reinitialisation without needing
the real data on disk.

`--group` is the unit the aggregator averages over: put everything you intend to
compare in one group, and things you do not in another.

## What gets logged

One JSON per run in `Logs/<group>/jsons/`, self-describing, named from its own
configuration. Each carries:

- **configuration** -- task setup, dataset, model (with width and parameter
  count), every hyperparameter in effect, seed, protocol
- **environment** -- python/torch/sklearn versions, GPU, git commit and whether
  the tree was dirty
- **per task** -- confusion matrix (sparse COO), and for each of three
  evaluation scopes: accuracy, macro/weighted precision, recall, F1, plus
  per-class support
- **curves** -- accuracy per task, accuracy on the most recent task, accuracy on
  task 0, hoisted out of the nested form so a plot need not walk it
- **summary** -- the scalars a table quotes

The three scopes are `seen` (all families so far -- the headline), `recent`
(the families just added -- acquisition), and `task0` (retention). `seen`
averages the other two, so a method trading one for the other is invisible in
it alone.

## Adding things

- **a method**: one file in `experiments/`, an `@register("name")` subclass of
  `Experiment` or `ContinualExperiment`, one import in `experiments/__init__.py`.
  Implement `train_task`; everything else is inherited.
- **a dataset**: one module in `core/data/` with `load(paths, **kw) -> Corpus`,
  one line in `core/data/__init__.py:REGISTRY`.
- **a selector**: one `@register("name")` function in `core/selectors.py`.
- **a task setup**: nothing -- pass `--task-setup 50+5x20`.
- **a metric**: add it in `core/metrics.py`; it flows to the logs, tables and
  (if listed in `plots.TITLES`) the figures.

Nothing downstream is method- or dataset-aware, so none of these require edits
in more than the places listed.

## Moving storage to the mount

Storage is resolved in `runner/paths.py` and nowhere else. When the mount
exists:

```bash
cp paths.local.json.example paths.local.json   # then edit the three roots
./iclr paths --show                            # confirm, and check they exist
```

Or, without editing a file:

```bash
export ICLR_DATA_ROOT=/mount/iclr_m/Datasets
export ICLR_LOG_ROOT=/mount/iclr_m/Logs
export ICLR_CACHE_ROOT=/mount/iclr_m/Cache
```

`data_root` holds corpora, `log_root` holds run records, and `cache_root` holds
DERIVED artifacts (joined feature caches, task-0 checkpoints). Cache is separate
because it is regenerable -- it can be wiped without touching a corpus, and it
is the directory most likely to need the mount's space first. Moving existing
data is a plain `mv`; nothing records an absolute path except the run JSONs,
which record where they were written as provenance, not as a dependency.

## Two things this enforces rather than documents

**Protocols are never differenced.** `stage1` retrains task 0 per seed; `meta`
shares one task-0 checkpoint. Their accuracies are not comparable. The
aggregator tabulates them as separate blocks and `plots.py` never draws them on
shared axes.

**A crashed run is not a result.** `sweep.py` records a non-zero child as failed
and exits non-zero. It is never scored, defaulted, or filled in -- a missing
number that silently becomes NaN and then sorts to the top of a ranking is a
real failure mode, and the sweep boundary is where to refuse it.
