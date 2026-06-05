"""Foundation-model provenance guard (issue #310).

Foundation-model numbers (scGPT, Nicheformer, Geneformer, UCE) are CITED from
their respective papers; LuminaST never re-runs them locally. This module
provides the canonical set of foundation backends plus helpers to stamp, query,
and guard provenance on benchmark rows so that a cited number can never be
recorded as a claimed local re-run.

Public API
----------
FOUNDATION_BACKENDS
    Frozenset of backend name strings that are paper-cited, not locally run.
mark_reference_reported(result, *, source)
    Stamp an AdapterResult (or Provenance) as reference-reported with a
    citation source string.
is_reference_reported(result_or_provenance)
    Return True iff the row carries the reference_reported flag.
require_reference_reported(backend)
    Raise FoundationRerunError if *backend* is a foundation backend whose
    result lacks the reference_reported stamp — i.e., it is about to be
    recorded as a claimed local re-run.
assert_not_claimed_rerun(method_or_backend)
    Alias for require_reference_reported; preferred name in guard call-sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from lumina_st.benchmarks.contract import AdapterResult, Provenance

__all__ = [
    "FOUNDATION_BACKENDS",
    "FoundationRerunError",
    "mark_reference_reported",
    "is_reference_reported",
    "require_reference_reported",
    "assert_not_claimed_rerun",
]

# ---------------------------------------------------------------------------
# Canonical set of foundation-model backends
# ---------------------------------------------------------------------------

FOUNDATION_BACKENDS: frozenset[str] = frozenset({"scgpt", "nicheformer", "geneformer", "uce"})
"""Backend names whose benchmark numbers are cited from papers, never locally run."""


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class FoundationRerunError(ValueError):
    """Raised when a foundation backend's result would be recorded as a claimed re-run.

    A foundation-model row MUST carry reference_reported=True because its
    numbers come from the published paper, not a local execution.  If code
    attempts to record such a row without the flag, this error is raised
    (fail-closed) rather than silently emitting a misleading provenance.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mark_reference_reported(
    result_or_provenance: Union["AdapterResult", "Provenance"],
    *,
    source: str,
) -> Union["AdapterResult", "Provenance"]:
    """Stamp *result_or_provenance* as reference-reported and record *source*.

    Mutates the object in-place (dataclass fields) and returns it for
    convenience in chained calls.

    Args:
        result_or_provenance: An :class:`~lumina_st.benchmarks.contract.AdapterResult`
            or :class:`~lumina_st.benchmarks.contract.Provenance` dataclass instance.
        source: Human-readable citation string, e.g. ``"scGPT paper Table 2"``.

    Returns:
        The same object, mutated.
    """
    # Both AdapterResult and Provenance carry these fields (added in contract.py).
    result_or_provenance.reference_reported = True
    result_or_provenance.reference_source = source
    return result_or_provenance


def is_reference_reported(result_or_provenance: Union["AdapterResult", "Provenance"]) -> bool:
    """Return True iff the row carries the ``reference_reported`` flag.

    Safe to call on any object; returns False if the attribute is absent
    (backward-compatible with rows that pre-date the field).
    """
    return bool(getattr(result_or_provenance, "reference_reported", False))


def require_reference_reported(backend: str) -> None:
    """Raise :class:`FoundationRerunError` if *backend* is a foundation model.

    Call this before recording a result row to prove the caller has explicitly
    acknowledged that foundation-model numbers come from papers, not local runs.
    Non-foundation backends are unaffected.

    Args:
        backend: The method/backend string, e.g. ``"scgpt"`` or ``"mean"``.

    Raises:
        FoundationRerunError: If *backend* is in :data:`FOUNDATION_BACKENDS`.
    """
    if backend.lower() in FOUNDATION_BACKENDS:
        raise FoundationRerunError(
            f"Backend {backend!r} is a foundation model (one of "
            f"{sorted(FOUNDATION_BACKENDS)}) whose benchmark numbers are "
            "paper-cited, not locally re-run. "
            "Call mark_reference_reported(result, source=...) before recording "
            "this row, then pass the stamped result through instead of calling "
            "require_reference_reported (#310)."
        )


def assert_not_claimed_rerun(method_or_backend: str) -> None:
    """Alias for :func:`require_reference_reported`; preferred at guard call-sites.

    Same semantics: raises :class:`FoundationRerunError` if
    *method_or_backend* is a foundation backend that has not been stamped as
    reference-reported.
    """
    require_reference_reported(method_or_backend)
