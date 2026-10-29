
# 🌙 Project SHADOW-VISION

**Physics-Based Deep Learning for Enhancement and Hazard Mapping of Permanently Shadowed Lunar Crater Regions (Chandrayaan-2/3 OHRC)**

Smart India Hackathon 2026 · Problem Statement **SIH1732** · Theme: Space Technology · Team **Th3 ErR0R 4o4**

SHADOW-VISION enhances ultra-low-light images of Permanently Shadowed Regions (PSRs) near the lunar poles, such as those captured by the Chandrayaan-2/3 OHRC (Orbiter High Resolution Camera, target 0.25 m/px). Raw PSR images are dominated by shot noise, CCD stripe artifacts and thermal read noise. The project combines a Restormer-inspired network with a Lommel-Seeliger/Hapke-based photometric consistency idea, with the goal of revealing terrain detail without inventing fake rocks or craters. On top of the enhanced image it flags landing hazards and water-ice candidates, suggests landing zones, and exports GIS-ready data and a mission report, all through an interactive Streamlit dashboard.



---

## 🎯 Problem & Approach

- **Problem:** PSR floors never receive direct sunlight, so OHRC images of them are near-blank and noisy. Naive contrast stretching or blind denoising can either fail to reveal anything or hallucinate terrain that could mislead landing-hazard products.
- **Approach:** (1) calibrate and destripe, (2) enhance with a Restormer-style network, (3) validate brightness against a physical reflectance model, (4) derive hazard and water-ice products from the validated output.

---

## ✨ Features Implemented in This Repository

| Module | What it does |
|---|---|
| **CCD Destriping** | Removes periodic pushbroom detector line artifacts with FFT-domain filtering |
| **Restormer-style Enhancer** | Lightweight transformer restorer (transposed self-attention + feed-forward blocks) |
| **Hazard & Volatile Detection** | Sobel-gradient slope proxy, Laplacian-based micro-boulder detection, bright-anomaly water-ice tagging |
| **Gigapixel Tile Processor** | Sliding-window processing with Gaussian-weighted overlap blending for large swaths |
| **3D Terrain Viewer** | Interactive Plotly surface with hazard and ice overlays |
| **Landing Site Selector** | Ranks hazard-free landing circles using distance transforms, terrain flatness and proximity to ice |
| **Benchmarking** | PSNR, SSIM, Shannon entropy, and a Lommel-Seeliger photometric residual |
| **GeoTIFF Export** | 3-band raster (enhanced image, hazard mask, ice mask) with a Lunar Polar Stereographic CRS |
| **PDF Mission Report** | Auto-generated landing site assessment briefing (ReportLab) |

### Proposed in the paper, not yet in the code

PI-GAN training, Hapke-based loss during training, SPICE/GDAL geometry, PDS4/ISDA-native export, FastAPI + React + Docker deployment, and ONNX/TensorRT acceleration (expected ~3× speed-up) are part of the project design but are not implemented in this repository yet.

---

## 🧭 Pipeline Overview

```
Raw OHRC PSR image
      │
      ▼
Calibration: CCD destriping (FFT)
      │
      ▼
AI enhancement: ShadowVisionEnhancer (Restormer-inspired)
      │
      ▼
Hazard / micro-boulder / water-ice detection
      │
      ├──► 3D terrain view
      ├──► Landing zone selection
      ├──► Benchmarking & validation
      └──► GeoTIFF + PDF mission report
```

---

## 📁 Project Structure

```
shadowvision/
├── app.py               # Streamlit dashboard (main entry point)
├── main.py              # Command-line demo; saves shadow_vision_output.png
├── pipeline.py          # LunarPSRPipeline: destriping, enhancement, hazard/ice detection
├── model.py             # ShadowVisionEnhancer, MDTA attention, loss module
├── tile_processor.py    # GigapixelTileProcessor (overlap-blended tiling)
├── terrain_3d.py        # Plotly 3D surface renderer
├── landing_site.py      # LandingZoneSelector
├── benchmarking.py      # LunarModelEvaluator (PSNR, SSIM, entropy, L-S residual)
├── exporter.py          # LunarGeoTIFFExporter (rasterio)
├── pdf_generator.py     # MissionReportGenerator (ReportLab)
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- (Optional) CUDA-capable GPU; the pipeline falls back to CPU automatically

### Installation

```bash
git clone https://github.com/<your-username>/shadowvision.git
cd shadowvision
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

> **Note:** `requirements.txt` currently ends with a `//` line containing a run command. Delete that line before running `pip install`, because pip does not support `//` comments.

### Run the dashboard

```bash
streamlit run app.py
```

### Run the command-line demo

```bash
python main.py
```

Generates a synthetic PSR crater image, runs the pipeline and saves a 4-panel figure to `shadow_vision_output.png`.

---

## 🖥️ Using the Dashboard

1. **Choose input:** the built-in simulated OHRC PSR crater, or upload a PNG/JPG/TIF image.
2. **Tune thresholds** in the sidebar: slope hazard angle, micro-boulder sensitivity, and optional tiling (256/512/1024 px) for large images.
3. **Explore results** in three tabs: Pipeline Results (raw, enhanced, hazard map, ice anomalies), 3D Terrain & Hazards, and Model Benchmarking & Validation.
4. **Review landing zones:** top 3 touchdown candidates (30 m radius) with hazard and ice distances.
5. **Export:** download the 3-band GeoTIFF or the PDF mission report.

---

## 📤 Output Products

| GeoTIFF band | Content |
|---|---|
| 1 | Enhanced PSR radiance (Float32) |
| 2 | Steep slope / hazard mask (0 or 255) |
| 3 | Water-ice candidate mask (0 or 255) |

CRS: Moon polar stereographic (sphere radius 1,737,400 m), pixel size 0.25 m.

---

## 📊 Preliminary Results (Prototype)

Measured on a **single simulated low-photon PSR tile** (demonstration build, as reported in the project paper). These are prototype-level diagnostics, not validation against flight data or independent ground truth.

| Metric | Value |
|---|---|
| Dynamic range gain | +14.2 dB |
| Structural similarity (SSIM) | 0.1024 |
| Signal gain (PSNR) | 6.83 dB |
| Lommel-Seeliger photometric loss | 0.1836 |
| Entropy / information gain | +55.0% |
| Hazard area coverage | 0.2% |
| Micro-boulders detected | 1,535 |
| Ice-candidate pixels | 512 |

The low SSIM and PSNR reflect how different the enhanced output is from the near-dark simulated input. They are not accuracy figures.

---

## ⚠️ Current Status & Limitations

- **Untrained model by default.** `LunarPSRPipeline` loads weights only if `model_weights_path` is given, and no training script or weights are included yet.
- **Proxy-based hazard detection.** "Slope" is derived from image intensity gradients, not a DEM or photoclinometry, so hazard maps are indicative only.
- **Photometric loss.** The `PhotometricConsistencyLoss` class in `model.py` currently combines an L1 term with a gradient-smoothness penalty. The Lommel-Seeliger check exists only in `benchmarking.py` as an evaluation metric.
- **Hard-coded dashboard/report values.** The "+14.2 dB" gain and some PDF figures (e.g., the 0.0031 photometric loss and landing offset text) are fixed strings, not computed.
- **Synthetic data only.** Results on real OHRC or LROC data have not been validated.
- **No hallucination guarantee yet.** The "zero hallucination" goal is a design objective; it still needs to be tested against independent ground truth and non-physics baselines.
- **Not for operational use.**

---

## 🗺️ Roadmap

- [ ] Train the enhancer on paired low-light / reference lunar data and ship weights
- [ ] Apply a Lommel-Seeliger/Hapke photometric loss during training
- [ ] Validate on LROC PSR imagery with independent ground truth
- [ ] Benchmark against plain Restormer and CLAHE to measure the physics term's contribution
- [ ] Replace gradient-based slopes with DEM or photoclinometry-based estimation
- [ ] Compute all dashboard and report metrics dynamically
- [ ] PDS4/ISDA-compatible export; ONNX/TensorRT inference; FastAPI + Docker deployment

---

## 👥 Team — Th3 ErR0R 4o4

- **Meysintha M S**
- **Nethra K**
- **Mythili K**

**Mentor:** Ms. Jenny, AP/CSE
Department of Computer Science and Engineering, V.S.B College of Engineering Technical Campus, Kinathukadavu, Coimbatore, Tamil Nadu, India

---

## 📚 References

1. P. J. Bickel et al., "Peering into lunar permanently shadowed regions with deep learning," *Nature Communications*, 2021.
2. S. W. Zamir et al., "Restormer: Efficient transformer for high-resolution image restoration," *CVPR*, 2022.
3. B. Hapke, *Theory of Reflectance and Emittance Spectroscopy*, 2nd ed., Cambridge University Press, 2012.
4. ISRO, Indian Space Science Data Archive (ISDA): Chandrayaan-2 data products and OHRC specifications.

## 🙏 Acknowledgements

ISRO and the Smart India Hackathon (SIH1732), and the faculty mentor and Department of CSE at V.S.B College of Engineering Technical Campus.

