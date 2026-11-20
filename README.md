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
│   ├── manifest.yaml                  # Verified Zenodo DOIs, MD5s, shapes & HDF5 paths
│   ├── i13_benchmark_transforms.json  # Frozen 7-DOF transforms & ruby-sphere scale ratios
│   ├── croissant.json                 # MLCommons Croissant 1.0 JSON-LD metadata (NeurIPS D&B)
│   ├── tasks/tasks.yaml               # Formal YAML specs for Tasks T1, T2, T3 & optional T4
│   └── baselines/                     # 3-seed Synthetic-vs-Real T1 & 4-fold LOSO T2 configs
├── registration/                      # Metrology, stage-motor & 7-DOF correlative registration
│   ├── xrdct_reference.py             # Conjugate COR, NNLS unmixing & SB-FRC / 1/2-bit floor
│   ├── diad_correlative.py            # Stage 0A-2: Astra flip_x, XCT<->XRD-CT & /entry stage z
│   ├── i13_ladder.py                  # Stage 0B: 7-DOF I13-2 ladder & 500 um ruby metrology
│   └── cross_beamline_audit.py        # Stage 0C: Fiducial-vs-pellet slip & octant gate (Task T3)
├── preprocessing/                     # Documented secondary track (NOT a standalone task)
│   ├── artifact_cleanup.py            # Frozen polar de-ringing, ruby z-mask & 360-closure fix
│   └── extract_benchmark_tracks.py    # SCARF/Hamilton float32 primary/cleaned/lite/NGFF extractor
├── tasks/                             # Task specs & submission schema validators
├── splits/                            # LR-voxel-buffered T1 slabs & 4-fold LOSO T2 CV splits
├── baselines/                         # Tricubic, Guided Filter, FBP, SIRT, OpenLSR-X & X-FUSE
├── src/multimodal_benchmark/          # Core I/O, manifest verifier & OpenImpala/FRC/TRE evaluators
├── DATASHEET.md                       # NeurIPS Datasets & Benchmarks Datasheet (Gebru et al.)
├── evaluate.py                        # Unified CLI evaluator outputting JSON + Markdown rows
└── *.ipynb, DIAD_Savu/, I13_2_Savu/   # Original 2025 Scientific Data companion notebooks
```

---

## `Zn-13X-Bench` Tasks & Two-Track Reference Versioning

- **Primary Reference Track (`raw_float32`, Default):** Registered, **uncleaned** `32-bit float` sub-volumes resampled via the frozen 7-DOF transforms in [`configs/i13_benchmark_transforms.json`](configs/i13_benchmark_transforms.json).
- **Cleaned Track (`cleaned_dering_v1`, Secondary Documented Track):** Same sub-volumes after frozen conservative polar de-ringing (`98.7–99.2%` ground-truth phantom edge retention; $|r(\Delta_1, \text{Zn})| \le 0.026$), ruby `z`-masking, and XRD-CT `360°`/COR fixes, published with exact difference maps (polynomial cupping and axial $z$-detrending excluded to preserve real $\text{Zn}/\text{Na}$ chemistry).
- **Headline Experiment (`T1`):** Compare models (`3D U-Net`, `SwinIR`, `OpenLSR-X`) trained on **synthetic degradations** (blur + downsample, with/without noise) vs. **real lower-resolution rungs** (`2.026×` and `8.101×`, 3 seeds), tested exclusively on real pairs and scored with both image metrics and `OpenImpala` transport.

| Task | Input | Reference (`raw_float32` Primary) | Baselines | Metrics |
|---|---|---|---|---|
| **T1: Real-Pair XCT Super-Resolution** | Lower-res I13-2 `float32` rung at true ratios (`1.604×, 2.026×, 2.493×, 5.051×, 8.101×`) | Higher-res rung resampled via frozen 7-DOF transform (LR-voxel-buffered slabs above ruby shadow) | `Tricubic`, `3D U-Net`, `SwinIR`, `OpenLSR-X` | PSNR (`full`, `fringe_masked`, `bandlimited`), SSIM, 3D FSC; **OpenImpala** $D_{\text{eff}}/D_0, \tau$, SSA, porosity (+ `±5%` threshold sensitivity) |
| **T2: Chemical Super-Resolution & Fusion** | `50 µm` XRD-CT Na/Zn maps (`103.6 µm` SB-FRC) + co-registered `0.54 µm` DIAD XCT | `25 µm` XRD-CT maps at `z = 9, 14, 17, 18` (`58.5 µm` SB-FRC; 4-fold Leave-One-Slice-Out CV) | `Bicubic`, `Guided Filter`, `Dual-Encoder U-Net`, `X-FUSE` | Band-limited (`k ≤ 0.43 cyc/px`) variance-weighted $\chi^2_\nu$ & RMSE; in-pellet Na/Zn correlation; reaction-front position (`µm`) |
| **T3: Cross-Beamline Registration** *(Gated on Stage 0C)* | Unaligned I13-2 (`1.625 µm`) & DIAD (`0.54 µm`) `float32` XCT | Fiducial-verified 7-DOF transform from Stage 0C *(pending SCARF/Hamilton run)* | `3-DOF Rigid`, `7-DOF NMI`, `VoxelMorph` | Disjoint fiducial TRE (`µm`); scale recovery (`%`); `2×2×2` octant residual drift (`µm`) |
| **T4: Sparse-View Reconstruction** *(Optional)* | Subsampled `25 µm` XRD-CT sinograms (`20, 40` views) | Full `40 × 79` continuum-subtracted reconstruction | `FBP`, `SIRT`, `U-Net Post-Processing` | Variance-weighted $\chi^2_\nu$, RMSE, 2D `FRC(0.5)` / `SB-FRC(0.5)` |

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
