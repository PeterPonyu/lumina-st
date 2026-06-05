"""Unified experiment driver for LuminaST (issue #179).

The reproducibility harness already ships its pieces separately: ``set_seed``
(``lumina_st.repro``) pins every RNG, and :class:`RunManifest`
(``lumina_st.experiment.run_manifest``) is the self-describing snapshot a run
emits next to its outputs. What was missing is the *one* driver that ties them
together so every experiment / sweep / fit-path goes through the same seeded,
manifest-emitting entry point.

:func:`run_experiment` is that driver. It is a context manager that, on entry:

1. Applies the seed via :func:`set_seed` (``config.seed`` wins, else ``seed``)
   and records the applied integer seed.
2. Resolves and creates a ``run_dir`` (default ``config.output_dir/<run_id>``).
3. Builds a :class:`RunManifest` capturing config / seed / sweep_params /
   eval_cadence / dataset_id / logging_config and writes ``run_manifest.json``
   into the run_dir immediately, so even a crashed run is self-describing.
4. Yields an :class:`ExperimentRun` handle (``run_dir``, ``seed``, ``manifest``)
   with ``record_metrics`` / ``finalize`` helpers.

Metrics + wall-clock are recorded **without touching the frozen RunManifest
schema** (``run_manifest.py`` is consume-only): they are written to a
``metrics.json`` sidecar beside the manifest. Wall-clock is measured with
``time.perf_counter`` only — never a wall-clock date — so it does not perturb
determinism. Manifest/metrics emission is best-effort: a write failure warns
but never raises, so the experiment body's own result is never masked.

Usage::

    from lumina_st.experiment import run_experiment

    with run_experiment(
        config=cfg,
        sweep_params={"lr": 1e-4},
        eval_cadence={"every_n_epochs": 1},
    ) as run:
        metrics = train(run.run_dir, seed=run.seed)
        run.finalize(metrics=metrics)
"""

from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from ..repro import set_seed
from .run_manifest import RunManifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config.lumina_config import LuminaSTConfig

__all__ = ["ExperimentRun", "run_experiment"]

#: Filename of the manifest written into every run_dir.
MANIFEST_FILENAME = "run_manifest.json"
#: Filename of the metrics/wall-clock sidecar (RunManifest schema is frozen).
METRICS_FILENAME = "metrics.json"


@dataclass
class ExperimentRun:
    """Handle yielded by :func:`run_experiment` for the duration of a run.

    Attributes:
        run_dir: Directory holding ``run_manifest.json`` and ``metrics.json``.
        seed: The integer seed actually applied via :func:`set_seed`.
        manifest: The :class:`RunManifest` written on entry (consume-only).
    """

    run_dir: Path
    seed: int
    manifest: RunManifest
    _metrics: dict[str, Any] = field(default_factory=dict, repr=False)
    _start: float = field(default_factory=time.perf_counter, repr=False)
    _finalized: bool = field(default=False, repr=False)

    @property
    def manifest_path(self) -> Path:
        """Path to the emitted ``run_manifest.json``."""
        return self.run_dir / MANIFEST_FILENAME

    @property
    def metrics_path(self) -> Path:
        """Path to the ``metrics.json`` sidecar (written on finalize)."""
        return self.run_dir / METRICS_FILENAME

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge ``metrics`` into the pending metrics record (no I/O here).

        Call repeatedly during a run; the accumulated metrics are flushed to
        the ``metrics.json`` sidecar by :meth:`finalize` (or on context exit).
        """
        self._metrics.update(metrics)

    def finalize(self, *, metrics: Optional[dict[str, Any]] = None) -> Path:
        """Merge any ``metrics`` and write the ``metrics.json`` sidecar.

        Captures wall-clock seconds via ``time.perf_counter``. Best-effort:
        a write failure warns and returns the intended path rather than
        raising, so it never masks the experiment body's own outcome. Safe to
        call once explicitly; the context manager calls it on exit if you did
        not, so an early-finalized run is not written twice.
        """
        if metrics:
            self._metrics.update(metrics)
        self._finalized = True
        payload = {
            "run_id": self.manifest.run_id,
            "seed": self.seed,
            "wallclock_s": time.perf_counter() - self._start,
            "metrics": dict(self._metrics),
        }
        path = self.metrics_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            tmp.replace(path)
        except Exception as exc:  # pragma: no cover - best-effort emission
            warnings.warn(f"failed to write {path}: {exc}", RuntimeWarning, stacklevel=2)
        return path


@contextmanager
def run_experiment(
    *,
    config: Optional["LuminaSTConfig"] = None,
    seed: Optional[int] = None,
    run_id: Optional[str] = None,
    run_dir: Optional[Union[str, Path]] = None,
    sweep_params: Optional[dict[str, Any]] = None,
    logging_config: Optional[dict[str, Any]] = None,
    eval_cadence: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    resume_from: Optional[Union[str, Path]] = None,
    dataset_id: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Iterator[ExperimentRun]:
    """Seeded, manifest-emitting context manager for a single experiment.

    On entry: applies the seed (``config.seed`` if set, else ``seed``, else
    the config/seed default), resolves+creates ``run_dir``, and writes
    ``run_manifest.json`` so a crashed run is still self-describing. On exit
    (normal or exceptional) it finalizes the ``metrics.json`` sidecar if the
    body did not already call :meth:`ExperimentRun.finalize`.

    Args:
        config: The :class:`LuminaSTConfig` for the run; snapshotted into the
            manifest. May be ``None`` for config-free smoke runs.
        seed: Seed to apply when ``config.seed`` is absent. Falls back to 42.
        run_id: Manifest run identifier; defaults to the config's
            ``experiment_name`` (or ``"run"``).
        run_dir: Where outputs/manifest live. Defaults to
            ``config.output_dir/<run_id>`` (or ``results/<run_id>``).
        sweep_params: Sweep coordinate for this run (recorded, not executed).
        logging_config: Logging backend config recorded into the manifest.
        eval_cadence: Eval-cadence spec, e.g. ``{"every_n_epochs": 1}``;
            recorded into the manifest (this driver does not run a loop).
        checkpoint_path: Checkpoint this run writes/resolves.
        resume_from: Checkpoint this run resumed from, if any.
        dataset_id: Stable dataset identifier (optional, ``None`` pre-data).
        dataset_hash: Best-effort dataset content hash (optional).
        timestamp: Injected ISO-8601 timestamp for deterministic tests; when
            ``None`` the manifest reads the wall clock once at creation.

    Yields:
        An :class:`ExperimentRun` handle.
    """
    # 1) Resolve + apply the seed (config wins), record what was applied.
    resolved_seed: int
    if config is not None and getattr(config, "seed", None) is not None:
        resolved_seed = int(config.seed)
    elif seed is not None:
        resolved_seed = int(seed)
    else:
        resolved_seed = 42
    applied_seed = set_seed(resolved_seed)

    # 2) Resolve + create the run_dir.
    resolved_run_id = run_id or (
        getattr(config, "experiment_name", None) if config is not None else None
    ) or "run"
    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
    else:
        base = Path(getattr(config, "output_dir", "results")) if config is not None else Path(
            "results"
        )
        resolved_run_dir = base / resolved_run_id
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    # 3) Build + emit the manifest immediately (best-effort, never fatal).
    manifest = RunManifest.create(
        run_id=resolved_run_id,
        config=config,
        seed=applied_seed,
        timestamp=timestamp,
        sweep_params=sweep_params,
        logging_config=logging_config,
        eval_cadence=eval_cadence,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )
    try:
        manifest.to_json(resolved_run_dir / MANIFEST_FILENAME)
    except Exception as exc:  # pragma: no cover - best-effort emission
        warnings.warn(
            f"failed to write {resolved_run_dir / MANIFEST_FILENAME}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # 4) Yield the handle; finalize on exit if the body did not.
    run = ExperimentRun(run_dir=resolved_run_dir, seed=applied_seed, manifest=manifest)
    try:
        yield run
    finally:
        if not run._finalized:
            run.finalize()
