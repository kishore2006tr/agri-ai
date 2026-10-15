# 🌱 AgriFertile AI — Soil Fertility Classification & Agronomic Advisor

An end-to-end Machine Learning platform and interactive web dashboard for **Soil Fertility Classification** and **Precision Nutrient Advisory**, built following benchmark standards from the **Indian Council of Agricultural Research (ICAR)** and the **Food and Agriculture Organization (FAO)**.

---

## 🌟 Key Features

1. **Multi-Model Machine Learning Engine**:
   - Compares 5 architectures: **Random Forest (Top Performer: 100% Test Acc, 99.96% 5-Fold CV)**, Extra Trees, HistGradientBoosting, Support Vector Machine (SVC), and Multi-Layer Perceptron (MLP).
   - Stratified K-Fold cross-validation and hyperparameter optimization.
   - Evaluates on 12 physical-chemical parameters: Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), Soil $pH$, Electrical Conductivity ($EC$), Organic Carbon ($OC$), Sulfur ($S$), Zinc ($Zn$), Iron ($Fe$), Copper ($Cu$), Manganese ($Mn$), and Boron ($B$).

2. **Interactive Web Dashboard (`server.py`)**:
   - Modern, dark-mode design system with glassmorphism and micro-animations.
   - **Soil Diagnostic Studio**: Real-time slider and numeric inputs with quick presets (*Fertile Alluvial Loam*, *Moderate Farmland*, *Acidic Depleted*, *Saline Alkali*).
   - **Fertility & Health Index Meter**: Circular gauge score (0–100) and multi-class probability breakdown.
   - **Diagnostic Radar Chart**: Live comparison of your soil's chemical profile against optimal agricultural standards.
   - **Precision Fertilizer Prescriptions**: Actionable fertilizer schedules (Urea, DAP, SSP, MOP, Zinc Sulfate) and soil conditioning (Agricultural Lime for acidic soils, Gypsum for sodic/alkaline soils).
   - **Crop Recommendations**: Tailored high-yield crops based on current soil parameters.

3. **Model Analytics & Explainable AI (XAI)**:
   - Interactive feature importance rankings highlighting primary fertility drivers (Organic Carbon, Phosphorus, Nitrogen, Potassium).
   - Full model comparison metrics table and charts.

4. **Batch CSV Predictor**:
   - Upload multi-acre CSV datasets of soil test records.
   - Instant vector predictions with summary distribution and downloadable enriched CSVs.

5. **Streamlit Alternative (`streamlit_app.py`)**:
   - Fast Python data science dashboard with Plotly radar visualizations.

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
Make sure you have Python 3.10+ installed.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python server.py
```
Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 3. (Optional) Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
This will open the Streamlit UI on **[http://localhost:8501](http://localhost:8501)**.

---

## 🔬 Soil Testing Benchmarks (ICAR / FAO Guidelines)

| Parameter | Symbol | Low / Deficient | Optimal / Medium | High / Excess | Unit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Available Nitrogen** | $N$ | < 280 | 280 – 560 | > 560 | kg/ha |
| **Available Phosphorus** | $P$ | < 23 | 23 – 56 | > 56 | kg/ha |
| **Available Potassium** | $K$ | < 140 | 140 – 280 | > 280 | kg/ha |
| **Soil Reaction (pH)** | $pH$ | < 6.0 (Acidic) | 6.0 – 7.5 (Neutral) | > 8.0 (Alkaline) | - |
| **Electrical Conductivity** | $EC$ | < 1.0 (Safe) | 1.0 – 2.0 (Critical) | > 2.0 (Saline) | dS/m |
| **Organic Carbon** | $OC$ | < 0.50% | 0.50% – 0.75% | > 0.75% | % |
| **Available Sulfur** | $S$ | < 10.0 | 10.0 – 25.0 | > 25.0 | ppm |
| **Zinc** | $Zn$ | < 0.60 | 0.60 – 1.50 | > 1.50 | ppm |
| **Iron** | $Fe$ | < 4.50 | 4.50 – 12.0 | > 12.0 | ppm |
| **Copper** | $Cu$ | < 0.20 | 0.20 – 0.80 | > 0.80 | ppm |
| **Manganese** | $Mn$ | < 2.00 | 2.00 – 8.00 | > 8.00 | ppm |
| **Boron** | $B$ | < 0.50 | 0.50 – 1.20 | > 1.20 | ppm |

---

## 📁 Repository Directory Structure

```
vac-project/
├── data/
│   ├── generate_soil_dataset.py   # Script to synthesize benchmark soil records
│   └── soil_fertility_dataset.csv  # 3,200 curated soil sample records
├── models/
│   ├── train_models.py            # Model training, 5-fold CV & evaluation pipeline
│   ├── best_soil_model.pkl        # Serialized Random Forest classifier
│   ├── scaler.pkl                 # StandardScaler feature preprocessor
│   └── model_metadata.json        # Performance metrics, XAI & benchmarks
├── static/
│   ├── css/
│   │   └── style.css              # Custom Vanilla CSS design system
│   ├── js/
│   │   └── app.js                 # Frontend application & Chart.js logic
│   └── images/
│       └── hero.jpg               # Photorealistic precision agriculture hero visual
├── templates/
│   └── index.html                 # Semantic HTML5 web application
├── app.py / server.py             # Flask Web Server & REST API
├── streamlit_app.py               # Streamlit Dashboard Edition
├── requirements.txt               # Project dependencies
└── README.md                      # Comprehensive documentation
```

---

## 📡 REST API Endpoints

- **`GET /api/model-info`**: Fetches model metadata, accuracy scores, feature importances, and classification reports.
- **`GET /api/sample-presets`**: Returns standard soil presets for quick testing.
- **`POST /api/predict`**: Computes real-time fertility classification, probabilities, parameter status, fertilizer dosages, and crop suggestions.
- **`POST /api/batch-predict`**: Processes uploaded CSV file containing soil test samples and outputs classified records with confidence scores.
