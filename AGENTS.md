# Agent & Contributor Guidelines (`MultimodalBenchmark`)

Read [ECOSYSTEM.md](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before adding a module, dependency, file reader, or dataset constant. The master copy of `ECOSYSTEM.md` lives in `BASE-Laboratory/basecoat`.

## Ecosystem Layer

- **Repository:** `BASE-Laboratory/MultimodalBenchmark`
- **Layer:** **Layer 4 — Workflows and benchmarks**
- **Allowed ecosystem dependencies:** `basecoat` (Layer 1) and, via explicit benchmark/evaluation extras or CLI adapters, downstream tools and solvers (`openimpala`, `sms`, `OpenLSR-X`, `X-FUSE`).
- **Banned imports (enforced by `[tool.basecoat.layers]`):** `zarr`, `numcodecs` (all OME-Zarr v0.4 and volume I/O goes through `basecoat`).
- **Reverse-dependency rule:** No Layer 1, Layer 2, or Layer 3 package may import `multimodal-benchmark`.

## Ownership Boundaries

- **Owns:** Zn-13X manifest and Zenodo errata (`configs/manifest.yaml`), 7-DOF I13-2 similarity transforms (`configs/i13_benchmark_transforms.json`, `configs/i13_certified_regions.yaml`), ruby-sphere metrology and Stage 0 audits (`registration/`), DIAD `-47.5 deg` motor-zero correction and dataset-specific scan pairing, leakage-free splits (`splits/`), benchmark task specs and joint `OpenImpala` + `SMS` / Willot-FFT evaluator (`src/multimodal_benchmark/evaluator.py`), adapters and baseline input staging invoking downstream tools (`baselines/integrations/`, `preprocessing/prepare_baseline_inputs.py`), and derived Zenodo / Croissant release packaging (`release_packaging/`).
- **Delegates to `basecoat` (`v0.3.0`):** Calibrated `Volume` / `VolumeSet` (`labels` + `uncertainties`) / `AxisSpec` / `LazyVolume` containers, OME-Zarr v0.4 multiscale writing and reading (`basecoat.write_ome_zarr`, `basecoat.read_ome_zarr`), out-of-core Savu `(X, Z, Y)` HDF5 reading (`basecoat.open_savu_h5`, `basecoat.read_savu_subvolume`), generic TIFF/NPY/RAW I/O (`basecoat.read_volume`, `basecoat.write_volume`), cell-centred 3-D `SimilarityTransform3D` / `AffineTransform3D` pullback resampling (`basecoat.resample_volume`), conservative polar ring removal and radial cupping detrending (`basecoat.remove_rings_polar_conservative`, `basecoat.detrend_radial_cupping`), and declarative AST layer linting (`python -m basecoat.layers`).
- **Retained in `MultimodalBenchmark` (single-consumer rule):** DIAD NeXus XRD-CT sinogram/peak-map readers (`load_xrdct_sinogram`, `load_xrdct_peak_maps`) and Rigaku `.asc` powder reader (`load_ascii_xrd`) remain here until a second repository migrates onto them.
