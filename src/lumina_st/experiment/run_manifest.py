"""Run manifest contract for the unified experiment harness (#179 / #181).

A :class:`RunManifest` is the self-describing snapshot every LuminaST run
emits next to its outputs/checkpoint so the run can be reproduced and audited:
seeds, the exact :class:`LuminaSTConfig` used, the environment (torch / numpy /
python versions + device), git provenance, intended checkpoint/eval/sweep
behaviour, and a best-effort dataset identity.

Design notes
------------
* **Plain dataclass.** The rest of the package config is pydantic, but a
  manifest is a flat provenance record (no validation/coercion needed) and
  must serialize trivially, so a ``@dataclass`` matches the lighter contract
  modules (``repro.py``, ``results_contract.py``) rather than duplicating the
  pydantic config machinery.
* **No import-time clocks/RNGs.** The timestamp is *injected* by the caller
  (``RunManifest.create(..., timestamp=...)``). When omitted, ``create`` reads
  the wall clock exactly once at call time — never at import — so tests can pin
  a deterministic value.
* **Reuses existing helpers.** Git provenance comes from the read-only
  ``results_contract.git_sha`` (which resolves the package repo from its own
  ``__file__``, not the cwd) and seeds come from ``repro.set_seed`` callers;
  this module does not re-implement either responsibility.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config.lumina_config import LuminaSTConfig


def _utc_now_iso() -> str:
    """Wall-clock UTC timestamp in ISO-8601, read once at call time.

    Kept as a function (not a module-level constant) so importing this module
    never freezes a timestamp and callers can inject their own value instead.
    """
    return datetime.now(timezone.utc).isoformat()


def _best_effort_git_sha() -> str:
    """Resolve the package repo's short git SHA, tolerating non-git checkouts.

    Delegates to the read-only ``results_contract.git_sha`` helper, which
    resolves the repo from this package's ``__file__`` and already returns
    ``"unknown"`` when git is unavailable. Any unexpected error degrades to
    ``"unknown"`` rather than failing manifest creation.
    """
    try:
        from ..results_contract import git_sha

        return git_sha()
    except Exception:
        return "unknown"


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return "absent"


def _numpy_version() -> str:
    try:
        import numpy

        return str(numpy.__version__)
    except Exception:
        return "absent"


def _device_string() -> str:
    """Best-effort hardware string (CUDA device name when available)."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:
        return "cpu"


@dataclass
class RunManifest:
    """Reproducible-run snapshot emitted by training and inference paths.

    All fields are JSON-serializable. Construct via :meth:`create` so the
    environment fields (versions, device, git SHA) are captured consistently;
    the bare constructor stays available for tests/round-trips.
    """

    # --- run identity ----------------------------------------------------
    run_id: str
    timestamp: str
    git_commit: str

    # --- reproducibility / environment -----------------------------------
    seed: Optional[int] = None
    seeds: dict[str, int] = field(default_factory=dict)
    python_version: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=_torch_version)
    numpy_version: str = field(default_factory=_numpy_version)
    device: str = field(default_factory=_device_string)

    # --- config snapshot -------------------------------------------------
    config: dict[str, Any] = field(default_factory=dict)

    # --- sweep / logging / checkpoint / eval intent ----------------------
    sweep_params: dict[str, Any] = field(default_factory=dict)
    logging_config: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    resume_from: Optional[str] = None
    eval_cadence: dict[str, Any] = field(default_factory=dict)

    # --- dataset identity (best-effort, may be None pre-data) ------------
    dataset_id: Optional[str] = None
    dataset_hash: Optional[str] = None

    # schema version of the manifest contract itself
    manifest_version: str = "1.0.0"

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        config: Optional["LuminaSTConfig"] = None,
        seed: Optional[int] = None,
        timestamp: Optional[str] = None,
        git_commit: Optional[str] = None,
        sweep_params: Optional[dict[str, Any]] = None,
        logging_config: Optional[dict[str, Any]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from: Optional[Union[str, Path]] = None,
        eval_cadence: Optional[dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
        dataset_hash: Optional[str] = None,
    ) -> "RunManifest":
        """Build a manifest, capturing the environment best-effort.

        Args:
            run_id: Caller-chosen unique identifier for this run.
            config: The :class:`LuminaSTConfig` used; serialized via its
                ``model_dump_for_checkpoint`` (JSON-safe). May be ``None``.
            seed: The applied seed. Defaults to ``config.seed`` when a config
                is given and ``seed`` is omitted.
            timestamp: Injected ISO-8601 timestamp. When ``None`` the wall
                clock is read once here (never at import) so production runs
                are stamped but tests can pin a deterministic value.
            git_commit: Override the resolved git SHA (best-effort otherwise).
            sweep_params: Sweep coordinate for this run (grid/ablation point).
            logging_config: Logging backend config (e.g. wandb project/flag).
            checkpoint_path: Where this run writes/resolves its checkpoint.
            resume_from: Checkpoint this run resumed from, if any.
            eval_cadence: How often eval runs (e.g. ``{"every_n_epochs": 1}``).
            dataset_id: Stable dataset identifier; ``None`` pre-data is fine.
            dataset_hash: Best-effort content hash of the dataset; optional.
        """
        cfg_snapshot: dict[str, Any] = {}
        resolved_seed = seed
        resolved_logging = dict(logging_config or {})
        if config is not None:
            cfg_snapshot = config.model_dump_for_checkpoint()
            if resolved_seed is None:
                resolved_seed = getattr(config, "seed", None)
            # Surface the logging-relevant config knobs by default without
            # overriding anything the caller passed explicitly.
            resolved_logging.setdefault("use_wandb", getattr(config, "use_wandb", None))
            resolved_logging.setdefault(
                "wandb_project", getattr(config, "wandb_project", None)
            )

        seeds: dict[str, int] = {}
        if resolved_seed is not None:
            seeds["global"] = int(resolved_seed)

        return cls(
            run_id=run_id,
            timestamp=timestamp if timestamp is not None else _utc_now_iso(),
            git_commit=git_commit if git_commit is not None else _best_effort_git_sha(),
            seed=int(resolved_seed) if resolved_seed is not None else None,
            seeds=seeds,
            config=cfg_snapshot,
            sweep_params=dict(sweep_params or {}),
            logging_config=resolved_logging,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            resume_from=str(resume_from) if resume_from is not None else None,
            eval_cadence=dict(eval_cadence or {}),
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
        )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dict of all fields."""
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> Path:
        """Atomically write the manifest to ``path`` as pretty JSON.

        The parent directory is created if needed. Returns the written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
        return path

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RunManifest":
        """Reconstruct a :class:`RunManifest` from a JSON file written by
        :meth:`to_json`, round-tripping every field.
        """
        data = json.loads(Path(path).read_text())
        return cls(**data)


def dataset_hash_for(path: Union[str, Path], *, max_bytes: Optional[int] = None) -> Optional[str]:
    """Best-effort SHA-256 of a dataset file; ``None`` if unreadable.

    Optional helper for callers that already hold a dataset path on disk. Never
    raises — returns ``None`` when the path is missing or unreadable so manifest
    emission stays non-fatal pre-data.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        read = 0
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
                read += len(chunk)
                if max_bytes is not None and read >= max_bytes:
                    break
        return h.hexdigest()
    except Exception:
        return None
