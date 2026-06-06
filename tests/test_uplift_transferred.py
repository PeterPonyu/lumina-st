"""Tests for the transferred-label uplift harness (scripts/uplift_transferred_labels.py).

The COAD Visium target has NO native GT, so uplift is scored against labels
transferred from the matched COAD scRNA reference. These tests exercise the
*harness* on a tiny synthetic problem on CPU: that label-transfer returns one
label per target cell drawn from the reference's label set, and that the
end-to-end ``main`` writes a schema-valid JSON carrying the circularity caveat.
The metric VALUES are not asserted (untrained tiny problem).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import anndata as ad
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "uplift_transferred_labels.py"


def _load() -> ModuleType:
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


def test_transfer_labels_one_per_cell():
    mod = _load()
    ref, tgt = _ref(), _enhanced()
    labels = mod.transfer_labels(ref, tgt, "cell_type", seed=0)
    assert len(labels) == tgt.n_obs
    assert set(map(str, labels)).issubset({"A", "B", "C"})


def test_main_writes_caveated_json(tmp_path):
    mod = _load()
    ref_p = tmp_path / "ref.h5ad"
    enh_p = tmp_path / "enh.h5ad"
    out_p = tmp_path / "uplift.json"
    _ref().write(ref_p)
    _enhanced().write(enh_p)
    ns = __import__("argparse").Namespace(
        enhanced=str(enh_p), reference=str(ref_p), label_key="cell_type",
        card_id="test_pair", seed=0, out=str(out_p),
    )
    mod.main(ns)
    payload = json.loads(out_p.read_text())
    assert payload["gt_is_native"] is False
    caveat = payload["circularity_caveat"].lower()
    assert "do not promote" in caveat and "#336" in caveat
    for m in ("ari", "ami", "nmi", "homo"):
        assert f"uplift_{m}_transferred_delta" in payload


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
        card_id="x", seed=0, out=str(tmp_path / "o.json"),
    )
    with pytest.raises(AssertionError):
        mod.main(ns)
