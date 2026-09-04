"""Path resolution for datasets, logs and derived caches.

The mount that will eventually hold Datasets/ and Logs/ does not exist yet and
its directory names are not known. Every path in the codebase therefore comes
from here, so relocating storage is a one-file edit rather than a grep-and-hope
across the tree.

Resolution order, highest priority first:

    1. explicit argument   (CLI --data-root / --log-root / --cache-root)
    2. environment         ICLR_DATA_ROOT / ICLR_LOG_ROOT / ICLR_CACHE_ROOT
    3. paths.local.json    per-machine, gitignored
    4. paths.json          committed defaults
    5. built-in fallback   <repo>/Datasets, <repo>/Logs, <repo>/Cache

Relative values resolve against the repo root, so the committed defaults keep
working in the current in-djarin layout with no configuration at all.

`cache_root` holds DERIVED artifacts (feature caches, task-0 checkpoints). It is
separate from data_root because it is regenerable: it can be wiped without
touching a corpus, and it is the directory most likely to need the mount's space
before the corpora themselves move.

Inspect the resolved configuration with:

    python -m runner.paths --show
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT_KEYS = ("data_root", "log_root", "cache_root")
ENV_VARS = {
    "data_root": "ICLR_DATA_ROOT",
    "log_root": "ICLR_LOG_ROOT",
    "cache_root": "ICLR_CACHE_ROOT",
}
FALLBACKS = {"data_root": "Datasets", "log_root": "Logs", "cache_root": "Cache"}


def repo_root() -> Path:
    """The repository root: two levels above this file (code/runner/paths.py)."""
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"{path}: could not be read as JSON ({e})")
    return {k: v for k, v in data.items() if k in ROOT_KEYS}


def _absolute(value: str, root: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p)


@dataclass(frozen=True)
class Paths:
    """Resolved storage roots, plus a record of where each value came from."""

    repo_root: Path
    data_root: Path
    log_root: Path
    cache_root: Path
    sources: dict

    # -- derived locations -------------------------------------------------
    def dataset_dir(self, name: str) -> Path:
        return self.data_root / name

    def group_dir(self, group: str) -> Path:
        return self.log_root / group

    def json_dir(self, group: str) -> Path:
        """Per-run records. Matches the Logs/<group>/jsons/ layout."""
        return self.group_dir(group) / "jsons"

    def figure_dir(self, group: str) -> Path:
        return self.group_dir(group) / "figures"

    def table_dir(self, group: str) -> Path:
        return self.group_dir(group) / "tables"

    def checkpoint_dir(self) -> Path:
        return self.cache_root / "checkpoints"

    def feature_cache_dir(self) -> Path:
        return self.cache_root / "features"

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: str(v) for k, v in d.items() if k != "sources"}


def resolve(data_root=None, log_root=None, cache_root=None,
            config_file: Path | None = None) -> Paths:
    """Resolve the three storage roots. Nothing is created on disk."""
    root = repo_root()
    committed = _load_json(config_file or (root / "paths.json"))
    local = _load_json(root / "paths.local.json")
    explicit = {"data_root": data_root, "log_root": log_root, "cache_root": cache_root}

    values, sources = {}, {}
    for key in ROOT_KEYS:
        if explicit[key]:
            raw, src = explicit[key], "cli"
        elif os.environ.get(ENV_VARS[key]):
            raw, src = os.environ[ENV_VARS[key]], f"env:{ENV_VARS[key]}"
        elif key in local:
            raw, src = local[key], "paths.local.json"
        elif key in committed:
            raw, src = committed[key], "paths.json"
        else:
            raw, src = FALLBACKS[key], "built-in default"
        values[key] = _absolute(str(raw), root)
        sources[key] = src

    return Paths(repo_root=root, sources=sources, **values)


def ensure_writable(paths: Paths, group: str | None = None) -> None:
    """Create the log/cache directories a run needs. Never creates data_root:
    a missing corpus directory is a real error, not something to paper over
    with an empty directory that then fails deeper in with a stranger message.
    """
    targets = [paths.log_root, paths.cache_root,
               paths.checkpoint_dir(), paths.feature_cache_dir()]
    if group:
        targets += [paths.json_dir(group), paths.figure_dir(group),
                    paths.table_dir(group)]
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)


def describe(paths: Paths) -> str:
    lines = [f"repo_root   {paths.repo_root}"]
    for key in ROOT_KEYS:
        value = getattr(paths, key)
        mark = "exists" if value.exists() else "MISSING"
        lines.append(f"{key:11s} {value}\n{'':12s}[{mark}] from {paths.sources[key]}")
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Show resolved storage paths")
    ap.add_argument("--show", action="store_true", help="print resolved paths (default)")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any root is missing")
    ap.add_argument("--create", action="store_true",
                    help="create log_root and cache_root (never data_root)")
    ap.add_argument("--data-root"); ap.add_argument("--log-root")
    ap.add_argument("--cache-root")
    a = ap.parse_args()

    p = resolve(a.data_root, a.log_root, a.cache_root)
    print(describe(p))
    if a.create:
        ensure_writable(p)
        print("\ncreated log_root and cache_root (data_root untouched by design)")
    if a.check:
        missing = [k for k in ROOT_KEYS if not getattr(p, k).exists()]
        if missing:
            raise SystemExit(f"\nmissing: {', '.join(missing)}")
        print("\nall roots present")


if __name__ == "__main__":
    _main()
