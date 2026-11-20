# Datasheet for `Zn-13X-Bench`

*Structured following Gebru et al. (2021), "Datasheets for Datasets," for the NeurIPS Datasets & Benchmarks track.*

---

## 1. Motivation

- **For what purpose was the dataset created?**
  `Zn-13X-Bench` provides verified multi-resolution synchrotron pairs, leakage-free splits, and physics-based [`OpenImpala`](https://github.com/BASE-Laboratory/OpenImpala) pore-transport and reaction-front scoring on top of the open synchrotron tomography and diffraction-tomography archive of Green *et al.*, *Scientific Data* **12**, 329 (2025) ([doi:10.1038/s41597-025-04605-9](https://doi.org/10.1038/s41597-025-04605-9)). Most tomographic super-resolution and multimodal fusion methods are trained and scored on synthetic blur-downsample degradations using image-only metrics (PSNR/SSIM). `Zn-13X-Bench` tests whether conclusions drawn from synthetic degradations and pixel metrics hold on **real multi-objective synchrotron acquisitions** when evaluated on **3D pore connectivity, effective diffusivity ($D_{\text{eff}}/D_0$), tortuosity ($\tau$), specific surface area (SSA), and chemical reaction-front position**.
- **Who created the benchmark and who funded the underlying acquisitions?**
  Created by James Le Houx (University of Greenwich; The Faraday Institution), Calum Green (Imperial College London, work conducted), Sharif Ahmed, Shashidhara Marathe, Liam Perera, and Alberto Leonardi (Diamond Light Source), and Killian Gmyrek and Daniele Dini (Imperial College London). Synchrotron acquisitions were performed at **Diamond Light Source** beamlines **I13-2** (visit `mg32980-2`) and **K11 DIAD** (visit `mg32980-1`), with laboratory powder diffraction at the **ISIS Materials Characterisation Laboratory**.

---

## 2. Composition

- **What do the instances represent?**
  All measurements capture a single controlled porous catalyst specimen: a binderless **Zn-exchanged Zeolite 13X** fragment mounted inside a `~1 mm` borosilicate glass capillary alongside a nominal `500 µm` $\text{Al}_2\text{O}_3$ ruby calibration sphere and `100–400 µm` NIST borosilicate glass beads:
  1. **Task T1 (`Real-Pair XCT Super-Resolution`):** Paired `32-bit float` 3D sub-volumes across four objectives on DLS I13-2 at ruby-sphere-calibrated core magnification ratios of **`1.604×`**, **`2.026×`** (headline), **`2.493×`**, and **`5.051×`**, plus an **exploratory** chained **`8.101×`** track (`169068 → 169065`, where direct overlap has `r = 0.07` and a `2.4%` direct-vs-chained scale gap), split into disjoint axial slabs above the ruby sphere's pink-beam shadow.
  2. **Task T2 (`Chemical Super-Resolution & Fusion`):** Four `25 µm` pencil-beam XRD-CT reference slices (`40 × 40` images = `~2,200` in-pellet pixels total from `40 × 79` unique-angle sinograms at heights `z = 9, 14, 17, 18`: scans `43324, 43322, 43323, 43325`) paired with `50 µm` full-volume XRD-CT (`20 × 39` unique angles, `+47.5°` motor-zero corrected) and co-registered `0.54 µm` DIAD `float32` XCT, evaluated via **4-fold Leave-One-Slice-Out (LOSO) cross-validation**. Because labeled supervision spans only three `40 × 40` slices per fold, T2 favors zero-shot / self-supervised forward-PSF methods (`Guided Filter`, `X-FUSE`) over heavily parameterized supervised decoders.
  3. **Task T3 (`Cross-Beamline Registration`, *Gated Pending Stage 0C*):** Unaligned I13-2 (`169067`, `1.625 µm`) and DIAD (`k11-43334`, `0.54 µm`) `float32` XCT volumes evaluated against the fiducial-verified Stage 0C 7-DOF similarity transform (*conditional until Stage 0A-2 and Stage 0C execute on SCARF/Hamilton*).
  4. **Task T4 (`Sparse-View XRD-CT Reconstruction`, *Optional Extension*):** Angularly subsampled `25 µm` XRD-CT sinograms (`10`, `20`, and `40` views) evaluated against the full `79`-angle reference reconstructions.
- **Are there any errors, noise, or redundancies in the data?**
  Yes—documenting these physical limits is central to the benchmark:
  - **Effective resolution of `25 µm` XRD-CT references:** Spearman–Brown full-dataset corrected odd/even `FRC(0.5)` is **`58.5 µm`** (`1/2-bit`: `57.9 µm`; raw half-map `FRC(0.5)`: `64.3 µm`) vs. **`103.6 µm`** (`1/2-bit`: `111.5 µm`) on `50 µm` inputs (`1.79×–1.93×` true resolution gain). T2 scores primary RMSE and reaction-front residuals band-limited to `k ≤ 0.43 cyc/px` (`~58 µm`).
  - **Residual sub-voxel registration uncertainty:** Fiducial-vs-pellet 7-DOF comparisons bound rigid stability to `1.9–3.0 µm` (`~2.4–3.7` HR voxels at `0.80 µm`). T1 therefore includes a `synthetic_jittered` training control (`~2.5` HR-voxel 3D jitter) to separate misalignment-induced L1/L2 smoothing from physical scintillator/phase-contrast domain shift.
  - **Unretrieved Fresnel phase fringes:** Neither I13-2 (`z_sd = 30 mm`) nor DIAD applied Paganin phase retrieval, and scintillator thickness (`500/100/25 µm`) and exposure (`0.024–1.0 s`) vary across objectives, creating an irreducible high-frequency fringe floor.
  - **Ruby sphere tolerance:** Manufacturing tolerance on the `500 µm` ruby sphere cancels identically in magnification *ratios* (`R_i / R_j`), while propagating (`±δD / 500 µm`) to absolute voxel sizes.
  - **Axial chemistry shift:** Because $\text{Zn}^{2+} \leftrightarrow \text{Na}^{+}$ exchange varies with height (`99% Zn` base $\to$ `10% Zn` top), axial `train / val / test` slabs intentionally test out-of-distribution attenuation generalization.

---

## 3. Collection Process

- **How was the data acquired?**
  Full experimental details are documented in Green *et al.*, *Scientific Data* **12**, 329 (2025). I13-2 multi-magnification scans (`169065–169068`) were acquired in reverse magnification order (`10× → 4× → 2× → 1.25×`) over 2h 19m on `2024-04-15` without sample-stage translation (`Δss1_samplexyz = 0.00 µm`). K11 DIAD dual-beamline XCT (`k11-43334`) and XRD-CT (`k11-43322..43401`) were acquired on `2024-05-01..05`.

---

## 4. Preprocessing, Cleaning, and Documented Tracks

- **Two-Track Release Policy:**
  - **Primary Track (`raw_float32`):** Registered, **uncleaned** `32-bit float` sub-volumes resampled with the 7-DOF transforms in [`configs/i13_benchmark_transforms.json`](configs/i13_benchmark_transforms.json) (*currently provisional from 8-bit previews; refitted directly on `32-bit float` volumes in [`scripts/slurm/01_stage0_audit_and_extract.sh`](scripts/slurm/01_stage0_audit_and_extract.sh) before derived-release freeze*). Used as the primary scoring reference for all tasks.
  - **Cleaned Track (`cleaned_dering_v1`):** Versioned secondary track applying frozen conservative polar de-ringing (`remove_rings_polar_conservative`, calibrated at `98.7–99.2%` true edge sharpness retention on a ground-truth phantom and $|r(\Delta_1, \text{Zn})| \le 0.026$), ruby `z`-slab shadow masking (`clean_z_mask`), and XRD-CT `0°/360°` duplicate-endpoint + conjugate COR fixes, published with exact difference maps. Polynomial radial cupping detrending and axial $z$-flattening are excluded from reference generation because empirical checks show radial/axial polynomial fits correlate with real $\text{Zn}/\text{Na}$ core–shell and axial chemistry ($r = +0.28\text{ to }+0.59$).
  - The `8-bit` cropped sub-volumes (`Zenodo 14245751`) are never used for quantitative scoring.

---

## 5. Uses

- **Intended uses:** Benchmarking 3D tomographic super-resolution (especially synthetic-trained vs. real-trained generalization under `OpenImpala` transport scoring), multimodal XCT + XRD-CT chemical fusion, cross-beamline registration, and sparse-view diffraction tomography reconstruction.
- **Tasks that should NOT use this benchmark:** Standalone de-ringing or standalone heuristic Otsu segmentation competitions (where references would merely reflect algorithmic post-processing rather than independent physical measurements).

---

## 6. Distribution

- **How is the benchmark distributed?**
  - Upstream raw and reconstructed synchrotron archives remain on their existing Zenodo DOIs (listed in [`configs/manifest.yaml`](configs/manifest.yaml), noting that DIAD `0.54 µm` XCT resolves at `10.5281/zenodo.13349402`).
  - Derived registered `float32` sub-volumes (`primary_raw_float32`, `cleaned_dering_v1` + difference maps, OME-Zarr multiscale pyramid, and `lite_4x` preview) are distributed under **CC-BY-4.0** via a Zenodo derived-data record (`IsDerivedFrom`), a Hugging Face mirror, MLCommons Croissant metadata ([`configs/croissant.json`](configs/croissant.json)), and tagged releases of [`BASE-Laboratory/MultimodalBenchmark`](https://github.com/BASE-Laboratory/MultimodalBenchmark).

---

## 7. Maintenance & Pending Verification Gates

- **Maintainers:** BASE Laboratory (`https://github.com/BASE-Laboratory/MultimodalBenchmark`).
- **Conditional Gates Prior to v1.0 Derived-Volume Freeze:**
  1. **32-bit I13-2 Transform Refit (`scripts/slurm/01_stage0_audit_and_extract.sh` Step 5):** Refit the provisional 8-bit preview 7-DOF similarity parameters directly on the `32-bit float` Savu reconstructions and synchronize [`configs/i13_certified_regions.yaml`](configs/i13_certified_regions.yaml) with `OpenLSR-X`.
  2. **Stage 0A-2 (`registration/diad_correlative.py` on SCARF/Hamilton):** Confirm DIAD `AstraReconGpu` `flip_x` chirality, sub-pixel XCT $\leftrightarrow$ XRD-CT offset, and raw `/entry` vertical stage $z$ (including low-margin pure-Zn slice `43324`).
  3. **Stage 0C (`registration/cross_beamline_audit.py` on SCARF/Hamilton):** Execute cross-beamline I13-2 $\leftrightarrow$ DIAD fiducial-vs-pellet slip and octant audit to ungate **Task T3** and XCT-guided **Task T2** baselines.
