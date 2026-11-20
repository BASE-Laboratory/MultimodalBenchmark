# Agent & Contributor Guidelines (`MultimodalBenchmark`)

Read [ECOSYSTEM.md](./ECOSYSTEM.md) before adding a module, dependency, file reader, or dataset constant. The master copy of `ECOSYSTEM.md` lives in `BASE-Laboratory/basecoat`.

## Ecosystem Layer

- **Repository:** `BASE-Laboratory/MultimodalBenchmark`
- **Layer:** **Layer 4 — Workflows and benchmarks**
- **Allowed ecosystem dependencies:** `basecoat` (Layer 1) and, via explicit benchmark/evaluation extras or CLI adapters, downstream tools and solvers (`openimpala`, `OpenLSR-X`, `X-FUSE`).
- **Reverse-dependency rule:** No Layer 1, Layer 2, or Layer 3 package may import `multimodal-benchmark`.

## Ownership Boundaries

- **Owns:** Zn-13X manifest and Zenodo errata (`configs/manifest.yaml`), frozen 7-DOF I13-2 similarity transforms (`configs/i13_benchmark_transforms.json`), ruby-sphere metrology and Stage 0 audits (`registration/`), DIAD `-47.5 deg` motor-zero correction and dataset-specific scan pairing, leakage-free splits (`splits/`), benchmark task specs and evaluator (`src/multimodal_benchmark/evaluator.py`), adapters invoking downstream tools (`baselines/integrations/`), and derived Zenodo / Croissant release packaging (`release_packaging/`).
- **Delegates to `basecoat` (`v0.1`):** Calibrated `Volume` / `AxisSpec` containers, OME-Zarr v0.4 multiscale writing and reading (`basecoat.write_ome_zarr`, `basecoat.read_ome_zarr`), and cell-centred 3-D `SimilarityTransform3D` / `AffineTransform3D` pullback resampling (`basecoat.resample_volume`).
- **Staged for `basecoat v0.2+` (two-consumer rule):** Savu `(X, Z, Y)` HDF5 reader, DIAD NeXus sinogram/peak-map reader, Rigaku `.asc` powder reader, and generic conservative polar ring-artefact filter algorithm remain here until a second repository migrates onto them.
