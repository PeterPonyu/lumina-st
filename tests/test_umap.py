"""Regression test for lumina-st #98.

``latent_umap`` previously wrapped the UMAP path in
``except (ImportError, Exception)`` — which is just ``except Exception``.
Any failure (a real UMAP bug, a bad input, an OOM) was silently caught
and the function returned the PCA fallback with no warning/log. Callers
could not tell "UMAP not installed" apart from "UMAP crashed", and a
genuine UMAP regression looked like a clean PCA fallback.

The fix splits the catch:
  * ``ImportError`` → silent PCA fallback (umap-learn missing is the
    documented "not available" path).
  * Any other exception from the UMAP path → log a ``WARNING`` with
    ``type(exc).__name__`` + message, and record the same string in the
    returned dict's new ``fallback_reason`` field so provenance is
    preserved (callers persisting this in ``adata.uns["latent_umap_method"]``
    or similar see why PCA was used).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lumina_st.benchmarks.latent_embed import latent_umap


def _force_umap_to_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``umap.UMAP`` with a class that always raises at fit time.

    Using monkeypatch so the test does not depend on whether umap-learn is
    installed in CI: if it isn't, we synthesize a fake module with the
    same name so the ``import umap`` inside ``latent_umap`` succeeds and
    we exercise the runtime-error branch.
    """

    class _BrokenUMAP:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fit_transform(self, _x):  # noqa: ANN001
            raise RuntimeError("synthetic UMAP failure for regression test")

    fake_module = type("_FakeUmap", (), {"UMAP": _BrokenUMAP})()
    monkeypatch.setitem(__import__("sys").modules, "umap", fake_module)


def test_pca_fallback_not_silent(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real UMAP runtime failure must NOT degrade into a silent fallback.

    The contract:
      * the function still returns (deterministic PCA replaces the broken
        UMAP — callers should not blow up on a visualization helper),
      * but the warning is logged with the exception type/message,
      * AND the returned dict's ``fallback_reason`` is set so any caller
        recording provenance (e.g. ``.uns["latent_umap_method"]``) can
        see WHY the PCA path ran.
    """
    _force_umap_to_fail(monkeypatch)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4))

    with caplog.at_level(logging.WARNING, logger="lumina_st.benchmarks.latent_embed"):
        out = latent_umap(x, n_components=2, n_neighbors=5, random_state=0)

    # PCA path ran — function did NOT raise.
    assert out["method"] == "pca-fallback", out
    assert out["embedding"].shape == (20, 2)

    # Provenance: returned dict carries the reason.
    assert "fallback_reason" in out, out
    assert out["fallback_reason"] is not None, out
    assert "RuntimeError" in out["fallback_reason"], out["fallback_reason"]
    assert "synthetic UMAP failure" in out["fallback_reason"], out["fallback_reason"]

    # Logging: at least one WARNING from the latent_embed logger mentions
    # the exception type/message.
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "lumina_st.benchmarks.latent_embed"
    ]
    assert warning_records, [r.getMessage() for r in caplog.records]
    msg = " ".join(r.getMessage() for r in warning_records)
    assert "RuntimeError" in msg, msg
    assert "synthetic UMAP failure" in msg, msg


def test_umap_success_path_has_no_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When UMAP succeeds, ``fallback_reason`` must be ``None`` and
    ``method`` must be ``"umap"`` — the new field cannot leak a stale
    error string into the happy path.
    """

    class _IdentityUMAP:
        def __init__(self, n_components: int = 2, **_kw) -> None:
            self.n_components = n_components

        def fit_transform(self, x):  # noqa: ANN001
            x = np.asarray(x, dtype=np.float64)
            return x[:, : self.n_components]

    fake_module = type("_FakeUmap", (), {"UMAP": _IdentityUMAP})()
    monkeypatch.setitem(__import__("sys").modules, "umap", fake_module)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(15, 5))
    out = latent_umap(x, n_components=2, n_neighbors=5, random_state=0)

    assert out["method"] == "umap", out
    assert out["fallback_reason"] is None, out
