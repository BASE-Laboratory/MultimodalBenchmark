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
│   └── cross_beamline_audit.py    # Bead-vs-pellet rigid/octant audit & certified_regions.yaml
├── tasks/                         # Task loaders, contracts & submission validators
├── splits/                        # Leakage-free 3D train/val/test split generator (guard-banded)
├── baselines/                     # Classical (Bicubic, JBU, Guided-TV, CNMF, Otsu) & learned runners
├── src/multimodal_benchmark/      # Core I/O, manifest verifier & evaluation metrics
├── evaluate.py                    # Single-command CLI evaluator outputting JSON + Markdown rows
└── *.ipynb, DIAD_Savu/, I13_2_Savu/ # Original 2025 Scientific Data companion notebooks & Savu lists
```

---

## Benchmark Tasks

| Task | Input | Ground Truth | Evaluation Region | Key Metrics |
|---|---|---|---|---|
| **T1: Multi-Magnification XCT Super-Resolution** | Real lower-res I13-2 XCT (`1.6×`: `2.6→1.625 µm`, `2×`: `1.625→0.8125 µm`, `2.5×`: `0.8125→0.325 µm`, `5×`: `1.625→0.325 µm`, `8×`: `2.6→0.325 µm`) | Paired higher-res I13-2 XCT | Certified non-cupping overlap sub-volumes | Cupping-invariant PSNR/SSIM, Edge MTF50, **3D FSC (`0.5`)**, frozen-threshold **OpenImpala** $D_{\text{eff}}, \tau$ |
| **T2: Cross-Beamline XCT Super-Resolution** | I13-2 `1.625 µm` XCT | DIAD `0.54 µm` XCT (`3.009×`) | Stage-0C certified rigid pellet sub-volumes | Same as T1 (gated on Stage 0C TRE threshold) |
| **T3: Correlative Chemical Super-Resolution / Fusion** | DIAD `0.54 µm` XCT + `+47.5°`-corrected `50 µm` XRD-CT (`20 × 20`) | Measured continuum-subtracted `25 µm` Zn/Na XRD-CT maps (**T3a**, 4 slices: `43322–43325`) + HR physical consistency (**T3b**) | 4 held-out `25 µm` ROI slices (`±100 µm` z-guard) | **Noise-normalized $\chi^2_\nu$** vs odd/even FRC $\sigma^2_{\text{ref}}$, RMSE, Pearson $r$, pore-void leakage |
| **T4: Multimodal Phase Segmentation** | XCT sub-volume | **T4a:** `Pore / Solid / Bead`<br>**T4b:** XRD-validated `Pore / Na-13X / Zn-13X` | Guard-banded 3D sub-volumes | Per-class Dice, Boundary F1 (`1, 2` vox), phase-fraction error, **OpenImpala** $D_{\text{eff}}, \tau$, SSA |

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

> **Important Errata Note:** In the published *Scientific Data* paper body text, the reconstructed DIAD `0.54 µm` XCT link `10.5281/zenodo.13327988` returns HTTP 404. Use Reference 18's verified DOI **`10.5281/zenodo.13349402`** (`tomo_p4_astra_recon_gpu.h5`), mapped automatically in [`configs/manifest.yaml`](configs/manifest.yaml).

```bash
# Download lightweight (<4 MB) powder XRD & reconstructed XRD-CT reference files and verify MD5s
mmbench verify-manifest --manifest configs/manifest.yaml --cache-dir data/cache/zenodo --download-lightweight
```

For the full `>50 GB` 32-bit reconstructed I13-2 XCT volumes, use `curl` with explicit output filenames:
- `169065_recon.h5` — `0.325 µm` I13-2 XCT (`https://zenodo.org/records/13327692/files/tomo_p4_tomopy_recon.h5`)
- `169066_recon.h5` — `0.8125 µm` I13-2 XCT (`https://zenodo.org/records/13327682/files/tomo_p4_tomopy_recon.h5`)
- `169067_recon.h5` — `1.625 µm` I13-2 XCT (`https://zenodo.org/records/13327651/files/tomo_p4_tomopy_recon.h5`)
- `169068_recon.h5` — `2.6 µm` I13-2 XCT (`https://zenodo.org/records/12206815/files/tomo_p4_tomopy_recon.h5`)
- `k11_43334_recon.h5` — `0.54 µm` DIAD XCT (`https://zenodo.org/records/13349402/files/tomo_p4_astra_recon_gpu.h5`)

### 2. Stage 0 Registration & Reference Noise Audit

```bash
# Audit powder XRD endmembers + 25 um / 50 um XRD-CT reference maps & reflection-subset noise floor
python -m registration.xrdct_reference --data-dir data/cache/zenodo --out-dir outputs/stage0_xrdct

# Run Stage 0C bead-vs-pellet rigid & octant deformation audit (I13-2 <-> DIAD)
python -m registration.cross_beamline_audit --help
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
