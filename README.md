# MultimodalBenchmark: 3D Synchrotron XCT & XRD-CT Benchmark Suite

Official codebase and standardized machine-learning benchmark suite for **Three-dimensional, multimodal synchrotron data for machine learning applications** (*Scientific Data* **12**, 329, 2025; [doi:10.1038/s41597-025-04605-9](https://doi.org/10.1038/s41597-025-04605-9) | [arXiv:2409.07322](https://arxiv.org/abs/2409.07322)).

The dataset captures a binderless **Zn-doped Zeolite 13X** catalyst fragment mounted inside a 1 mm borosilicate glass capillary alongside a 0.5 mm ruby sphere and 0.1–0.4 mm NIST borosilicate calibration beads across two Diamond Light Source (DLS) beamlines and a laboratory diffractometer:

- **DLS I13-2 Multi-Magnification XCT Ladder:** `2.6 µm`, `1.625 µm`, `0.8125 µm`, and `0.325 µm` voxel sizes (`2110 × 2510 × 2510`, 32-bit float + 8-bit cropped sub-volumes).
- **DLS K11 (DIAD) Correlative Dual-Beamline Acquisition:** Co-mounted `0.54 µm` XCT (`2160 × 2560 × 2560`) paired with full-volume `50 µm` XRD-CT (`20 × 20` slices, 40 projections) and high-resolution `25 µm` ROI XRD-CT (`40 × 40` slices, 79–80 projections) resolving the `0.005–0.01 Å⁻¹` peak shift between **Na-13X** (`q ≈ 1.651 Å⁻¹`) and **Zn-13X** (`q ≈ 1.656 Å⁻¹`).
- **ISIS MCL Rigaku SmartLab Powder XRD:** High-resolution Cu-$K\alpha$ reference patterns for pure Na-13X, partially Zn-exchanged 13X, fully Zn-exchanged 13X, and an empty borosilicate capillary.

---

## Repository Structure

```text
MultimodalBenchmark/
├── configs/
│   ├── manifest.yaml              # Verified Zenodo DOIs, MD5s, shapes & HDF5 paths
│   ├── tasks/                     # Formal YAML task definitions (T1, T2, T3, T4)
│   └── baselines/                 # 3-seed configs for OpenLSR-X & X-FUSE
├── registration/                  # Stage 0 audit & 7-DOF correlative registration
│   ├── xrdct_reference.py         # Conjugate COR solver, NNLS unmixing & FRC noise floor
│   ├── diad_correlative.py        # Astra/TomoPy chirality check & 3D PSF XCT<->XRD-CT alignment
│   ├── i13_ladder.py              # Multi-scale 7-DOF I13-2 ladder registration & TRE
│   └── cross_beamline_audit.py    # Fiducial-vs-pellet slip/octant audit & region gating YAML
├── preprocessing/                 # Conservative ("do-no-harm") artifact cleanup & spatial z-gating
│   └── artifact_cleanup.py        # Edge-protected polar de-ringing, 7-term cupping detrend & ruby z-mask
├── tasks/                         # Task loaders, contracts & submission validators
├── splits/                        # Leakage-free 3D train/val/test split generator (guard-banded)
├── baselines/                     # Classical (Bicubic, JBU, Guided-TV, CNMF, Otsu) & learned runners
├── src/multimodal_benchmark/      # Core I/O, manifest verifier & evaluation metrics
├── evaluate.py                    # Single-command CLI evaluator outputting JSON + Markdown rows
└── *.ipynb, DIAD_Savu/, I13_2_Savu/ # Original 2025 Scientific Data companion notebooks & Savu lists
```

---

## Benchmark Tasks

| Task | Input | Reference Target | Evaluation Region | Key Metrics |
|---|---|---|---|---|
| **T1: Multi-Magnification XCT Super-Resolution** | Real lower-res I13-2 `float32` XCT (`~1.604×`: `2.6→1.625 µm`, `~2.026×`: `1.625→0.8125 µm`, `~2.493×`: `0.8125→0.325 µm`, `~5.051×`: `1.625→0.325 µm`, `~8.101×`: `2.6→0.325 µm`) | Paired higher-res I13-2 `float32` XCT (resampled via fitted 7-DOF transforms) | Non-cupping 3D overlap sub-volumes (`float32` required for thresholding & transport) | Cupping-invariant PSNR/SSIM, Edge MTF50, **3D FSC (`0.5`)**, frozen-threshold **OpenImpala** $D_{\text{eff}}, \tau$ |
| **T2: Cross-Beamline XCT Super-Resolution** *(Gated on Stage 0C)* | I13-2 `1.625 µm` `float32` XCT | DIAD `0.54 µm` `float32` XCT (`~3.009×`) | Rigid pellet sub-volumes passing Stage 0C (`< 1.0 µm` octant drift) | Same as T1 (remains gated until Stage 0C runs on SCARF/Hamilton) |
| **T3: Correlative Chemical Super-Resolution / Fusion** | DIAD `0.54 µm` XCT + `+47.5°`-corrected `50 µm` XRD-CT (`20 × 20`) | Sparse-sampled (`40 × 80`) continuum-subtracted `25 µm` Zn/Na **reference** maps (**T3a**, 4 slices: `43322–43325`) + HR physical consistency (**T3b**) | 4 held-out `25 µm` ROI slices (`±100 µm` z-guard) | **Noise-normalized $\chi^2_\nu$** vs odd/even FRC $\sigma^2_{\text{ref}}$, RMSE, Pearson $r$, pore-void leakage |
| **T4: Multimodal Phase Segmentation** | `float32` XCT sub-volume | **T4a:** `Pore / Solid / Bead`<br>**T4b:** XRD-validated `Pore / Na-13X / Zn-13X` | Guard-banded 3D sub-volumes | Per-class Dice, Boundary F1 (`1, 2` vox), phase-fraction error, **OpenImpala** $D_{\text{eff}}, \tau$, SSA |

---

## Installation

### Modern Benchmark Package (`uv` / `pip`, Recommended)

```bash
git clone https://github.com/BASE-Laboratory/MultimodalBenchmark.git
cd MultimodalBenchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[transport,dev]"
```

### Legacy 2025 Paper Notebooks (`conda`)

To run the original 2025 *Scientific Data* companion notebooks (`XRD_Analysis_ascii.ipynb`, `Sinograms.ipynb`, `DiffTomo_Analysis.ipynb`, `I13-2_Alignment.ipynb`):

```bash
conda env create -f multimodal-xrdct.yml
conda activate multimodal-xrdct
jupyter notebook Sinograms.ipynb
```

---

## Quickstart

### 1. Inspect & Verify Zenodo Records (`configs/manifest.yaml`)

> **Important Errata & Quantitative Data Note:**
> 1. In the published *Scientific Data* paper body text, the reconstructed DIAD `0.54 µm` XCT link `10.5281/zenodo.13327988` returns HTTP 404. Use Reference 18's verified DOI **`10.5281/zenodo.13349402`** (`tomo_p4_astra_recon_gpu.h5`), mapped automatically in [`configs/manifest.yaml`](configs/manifest.yaml).
> 2. The `500 MB` `8-bit` cropped sub-volumes (`10.5281/zenodo.14245751`) are suitable for geometric registration previews only; all quantitative pore thresholding and `OpenImpala` transport scoring must use the `32-bit float` reconstructed volumes (`12206815, 13327651, 13327682, 13327692, 13349402`). Do not download the `53–57 GB` HDF5 stacks to a local workstation—run Stage 0A-2 and Stage 0C directly on **SCARF** or **Hamilton** where Diamond's `mg32980-1` / `mg32980-2` archives or Lustre mirrors reside.

```bash
# Download lightweight (<4 MB) powder XRD & reconstructed XRD-CT reference files and verify MD5s
mmbench verify-manifest --manifest configs/manifest.yaml --cache-dir data/cache/zenodo --download-lightweight
```

### 2. Stage 0 Registration & Reference Artefact Audit (SCARF / Hamilton Ready)

```bash
# Stage 0A-1: Audit powder XRD endmembers + 25 um / 50 um XRD-CT reference maps & odd/even FRC artefact floor
python -m registration.xrdct_reference --data-dir data/cache/zenodo --out-dir outputs/stage0_xrdct

# Stage 0A-2 (run on SCARF/Hamilton): Stream-align 57 GB DIAD XCT <-> XRD-CT, check Astra flip_x chirality & raw /entry stage z
python -m registration.diad_correlative \
  --diad-xct-h5 /dls/k11/data/2024/mg32980-1/processing/tomo_p4_astra_recon_gpu.h5 \
  --xrdct-dir data/cache/zenodo \
  --dls-raw-nexus-dir /dls/k11/data/2024/mg32980-1/nexus \
  --out-json outputs/stage0a_diad_correlative.json

# Stage 0C (run on SCARF/Hamilton): Separate fiducial-vs-pellet 7-DOF slip & 2x2x2 octant deformation gate (Task T2)
python -m registration.cross_beamline_audit \
  --i13-h5 /dls/i13/data/2024/mg32980-2/processing/169067_tomo_p4_tomopy_recon.h5 \
  --diad-h5 /dls/k11/data/2024/mg32980-1/processing/tomo_p4_astra_recon_gpu.h5 \
  --out-yaml outputs/stage0c_cross_beamline_regions.yaml
```

### 3. Generate Leakage-Free Splits & Score a Submission

```bash
# Generate deterministic 3D splits with >=32 um spatial & >=100 um axial z-guard bands
python -m splits.generate_splits --out splits/benchmark_splits_v1.json

# Evaluate any baseline or model submission into JSON + Markdown leaderboard rows
python evaluate.py --task t3 --prediction outputs/jbu_pred.npz --reference outputs/t3_ref.npz --markdown
```

---

## Citation

If you use this dataset or benchmark suite, please cite:

```bibtex
@article{green2025multimodal,
  title   = {Three-dimensional, multimodal synchrotron data for machine learning applications},
  author  = {Green, Calum and Ahmed, Sharif and Marathe, Shashidhara and Perera, Liam and Leonardi, Alberto and Gmyrek, Killian and Dini, Daniele and Le Houx, James},
  journal = {Scientific Data},
  volume  = {12},
  number  = {1},
  pages   = {329},
  year    = {2025},
  doi     = {10.1038/s41597-025-04605-9}
}
```

For transport metrics computed via [`OpenImpala`](https://github.com/BASE-Laboratory/OpenImpala), super-resolution baselines via [`OpenLSR-X`](https://github.com/BASE-Laboratory/OpenLSR-X), or correlative fusion baselines via [`X-FUSE`](https://github.com/BASE-Laboratory/X-FUSE), please also cite the respective BASE Laboratory packages.
