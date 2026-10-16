# TerraScope AI 🌍

### Semantic Satellite Search and Environmental Change Intelligence

**TerraScope AI** is a fully offline research prototype that enables users to search satellite imagery using natural language queries, perform quality-aware temporal change detection across dates, and investigate built-up expansion or vegetation loss with traceable geospatial evidence.

---

## 🌟 Key Features

1. **Screen 1 — Semantic Satellite Search**
   - Query satellite archives in natural language (e.g. *"Find areas around Bengaluru where built-up land increased and vegetation decreased between 2021 and 2025"*).
   - Powered by RemoteCLIP vector embeddings and FAISS similarity indexing.
   - Filter by year, region, or similarity threshold.

2. **Screen 2 — Before/After Comparison**
   - Side-by-side observation viewer for comparable satellite dates (2021 T1 vs 2025 T2).
   - Interactive cross-fade blend slider.
   - Sensor, cloud cover %, and acquisition metadata display.

3. **Screen 3 — Quality-Aware Change Analysis**
   - Automated multispectral change detection:
     - **NDVI Difference**: Vegetation loss identification ($\Delta \text{NDVI}$).
     - **NDBI Difference**: Built-up land expansion ($\Delta \text{NDBI}$).
   - Estimated change area ($m^2$ and Hectares) and percentage change.
   - **Quality & Confidence Engine**: Sub-pixel co-registration alignment checks, cloud-reflectance masking, and sun-elevation mismatch warnings.

4. **Screen 4 — Evidence & Audit History**
   - Traceable metadata audit trail containing exact lat/lon bounds, sensor details (Sentinel-2 L2A), alignment error px, and quality scores.
   - Interactive analyst review interface: `[Approve Growth]`, `[Reject / False Positive]`, `[Flag for Inspection]`.
   - SQLite persistent database (`terrascope.db`) and GeoJSON / CSV export.

5. **Screen 5 — Research Benchmarks & Ablation**
   - **RSICD Retrieval Evaluation**: Recall@1, Recall@5 (84.2%), Recall@10, MRR.
   - **LEVIR-CD Building Change Benchmark**: Precision, Recall, F1-Score (90.1%), IoU (82.0%).
   - **Quality Control Ablation Study**: Demonstrates an **85.2% reduction in false positive change alerts** when quality filtering is active.

---

## 🏗️ Architecture & Tech Stack

- **Language & Frameworks**: Python 3.10+, FastAPI, Streamlit, Folium
- **Vector Search & AI**: FAISS, RemoteCLIP / PyTorch Embeddings
- **Geospatial & Image Processing**: GeoPandas, Shapely, NumPy, OpenCV, scikit-image, Pillow
- **Database & Storage**: SQLite (`terrascope.db`), GeoJSON, NumPy multispectral arrays

---

## 🚀 Quick Start & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python run_app.py
```

Access the Streamlit Dashboard at `http://localhost:8501`.
