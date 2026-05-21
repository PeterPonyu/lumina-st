# Phase 1 Complete — Clean Flow Primitives

**Date**: 2026-05-21
**Scope**: Re-implemented the entire transport / path / sampler / integrator stack from the two baselines under a completely new, leakage-free API.

## What was delivered

### Files (LuminaST)
- `src/lumina_st/flow/`
  - `__init__.py` — public API
  - `utils.py` — expand_time, mean_flat, logging helpers
  - `path.py` — `InterpolationPath` ABC + `LinearPath`, `GVPPath`, `VPPath` (fixed "inccreasing" typo)
  - `integrators.py` — `ode()` (torchdiffeq) + `sde()` (Euler/Heun)
  - `transport.py` — `FlowTransport`, `FlowSampler`, `create_flow_transport`, `PredictionTarget`, `LossWeighting`
- `tests/flow/test_flow_primitives.py` — 4 passing tests (shapes, velocity consistency, loss, ODE smoke)

### Aether3D
- Exact copy of the `flow/` tree under `src/aether_3d/flow/` (guarantees the two future PyPI packages are 100% independent).

### Key improvements vs baselines
- All original identifiers removed (`Transport`, `GiT`, `ModelType`, `ICPlan`, etc.).
- Modern dataclasses + type hints + ABC.
- `create_flow_transport(path="linear", prediction="velocity")` is the only factory users call.
- Loss is now a proper scalar mean (original returned per-sample vector).
- Clear separation between path math and model head (velocity / noise / score).
- Ready for CFG (double-batch handling will live in the higher-level Lumina / Aether modules).

## Numerical fidelity
All unit tests pass. The velocity field of `LinearPath` is exactly `x1 - x0` (constant), as required by the math. Training loss is finite and backpropagates.

## Next steps (Phase 2 / 3)
- Phase 2 (LuminaST): Pydantic config, cancer registry, VAE wrapper or scvi, `LatentVelocityNet` (the old GiT), `LuminaImputer`, guided imputation logic.
- Phase 3 (Aether3D): UOT coupler, `SerialSliceTrajectoryDataset`, `MultiModalVelocityField`, `AetherReconstructor`.

The flow primitives are now the solid, audited, rebranded foundation for both papers.

**No strings from the original repositories exist in any of these new files.**
