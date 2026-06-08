"""Tests for the multi-seed + negative-control uplift wrapper
(scripts/uplift_transferred_labels_multiseed.py).

Mirrors tests/test_uplift_transferred.py: exercises the *harness* on a tiny
synthetic CPU problem — that mean/std aggregation is well-formed, that the
negative-control (permuted-label) block is emitted, and that the circularity
caveat is preserved. Metric VALUES are not asserted (untrained tiny problem).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import anndata as ad
import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SCRIPT = _SCRIPTS / "uplift_transferred_labels_multiseed.py"


def _load() -> ModuleType:
    # the wrapper does `from uplift_transferred_labels import transfer_labels`,
    # so the sibling single-seed script must be importable.
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref(n=120, g=40, seed=0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    labels = np.array(["A", "B", "C"])[rng.integers(0, 3, size=n)]
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs["cell_type"] = labels
    return a


def _enhanced(n=60, g=40, latent=8, seed=1) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.poisson(2.0, size=(n, g)).astype(np.float32))
    a.var_names = [f"g{i}" for i in range(g)]
    a.obsm["latent_observed"] = rng.normal(size=(n, latent)).astype(np.float32)
    a.obsm["latent_enhanced"] = rng.normal(size=(n, latent)).astype(np.float32)
    return a


def test_mean_std_well_formed():
    mod = _load()
    out = mod._mean_std([1.0, 2.0, 3.0])
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(2.0)
    assert out["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_main_writes_meanstd_and_negative_control(tmp_path):
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    mod = _load()
    ref_p = tmp_path / "ref.h5ad"
    enh_p = tmp_path / "enh.h5ad"
    out_p = tmp_path / "uplift_ms.json"
    _ref().write(ref_p)
    _enhanced().write(enh_p)
    ns = __import__("argparse").Namespace(
        enhanced=str(enh_p), reference=str(ref_p), label_key="cell_type",
        card_id="test_pair", seeds="0,1", out=str(out_p),
    )
    mod.main(ns)
    payload = json.loads(out_p.read_text())
    assert payload["gt_is_native"] is False
    assert payload["seeds"] == [0, 1]
    caveat = payload["circularity_caveat"].lower()
    assert "do not promote" in caveat and "#336" in caveat
    # mean/std present for every metric in both the real run and the neg control
    for m in ("ari", "ami", "nmi", "homo"):
        key = f"uplift_{m}_transferred_delta"
        for block in (payload["uplift_mean_std"], payload["negative_control_permuted_labels"]["uplift_mean_std"]):
            assert key in block
            assert {"mean", "std", "n"} <= block[key].keys()
            assert block[key]["n"] == 2
    assert len(payload["per_seed_raw"]) == 2


def test_missing_latent_raises(tmp_path):
    mod = _load()
    enh = _enhanced()
    del enh.obsm["latent_enhanced"]
    enh_p = tmp_path / "enh.h5ad"
    ref_p = tmp_path / "ref.h5ad"
    enh.write(enh_p)
    _ref().write(ref_p)
    ns = __import__("argparse").Namespace(
        enhanced=str(enh_p), reference=str(ref_p), label_key="cell_type",
        card_id="x", seeds="0", out=str(tmp_path / "o.json"),
    )
    with pytest.raises(AssertionError):
        mod.main(ns)
