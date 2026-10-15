"""
AgriFertile AI - Streamlit Dashboard Edition
Fast Python data science web interface for Soil Fertility Classification.
Run with: streamlit run streamlit_app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="AgriFertile AI - Soil Fertility Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_model_and_metadata():
    model_path = os.path.join(MODELS_DIR, 'best_soil_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_model_and_metadata()
feature_cols = meta.get('feature_cols', ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'])
class_names = meta.get('class_names', ['Low', 'Moderate', 'High'])

# Header
st.markdown('<div class="main-title">🌱 AgriFertile AI — Soil Fertility Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Precision Machine Learning Model & Agronomic Nutrient Advisory System (ICAR & FAO Standards)</div>', unsafe_allow_html=True)

# Sidebar: Soil Input Controls
st.sidebar.header("🧪 Soil Test Parameters")

# Presets
preset = st.sidebar.selectbox(
    "Load Quick Soil Preset:",
    ["High Fertility (Alluvial Loam)", "Moderate Fertility (Farmland)", "Low Fertility (Acidic Depleted)", "Low Fertility (Saline-Alkali)"]
)

default_vals = {
    "High Fertility (Alluvial Loam)": {"N": 480.0, "P": 58.0, "K": 340.0, "pH": 6.8, "EC": 0.55, "OC": 0.95, "S": 22.0, "Zn": 1.45, "Fe": 11.5, "Cu": 0.75, "Mn": 6.2, "B": 1.05},
    "Moderate Fertility (Farmland)": {"N": 310.0, "P": 32.0, "K": 210.0, "pH": 6.7, "EC": 0.65, "OC": 0.60, "S": 13.5, "Zn": 0.85, "Fe": 7.5, "Cu": 0.45, "Mn": 3.8, "B": 0.65},
    "Low Fertility (Acidic Depleted)": {"N": 150.0, "P": 11.0, "K": 105.0, "pH": 4.8, "EC": 0.35, "OC": 0.32, "S": 6.0, "Zn": 0.35, "Fe": 7.0, "Cu": 0.18, "Mn": 1.5, "B": 0.25},
    "Low Fertility (Saline-Alkali)": {"N": 175.0, "P": 14.0, "K": 185.0, "pH": 8.7, "EC": 2.80, "OC": 0.28, "S": 14.0, "Zn": 0.40, "Fe": 2.6, "Cu": 0.20, "Mn": 1.2, "B": 0.70}
}[preset]

st.sidebar.subheader("🌿 Macronutrients (NPK)")
n_val = st.sidebar.slider("Nitrogen (N) [kg/ha]", 50.0, 680.0, default_vals['N'], step=1.0)
p_val = st.sidebar.slider("Phosphorus (P) [kg/ha]", 4.0, 95.0, default_vals['P'], step=0.5)
k_val = st.sidebar.slider("Potassium (K) [kg/ha]", 40.0, 550.0, default_vals['K'], step=1.0)

st.sidebar.subheader("⚗️ Physical-Chemical Properties")
ph_val = st.sidebar.slider("Soil pH", 3.8, 9.8, default_vals['pH'], step=0.05)
ec_val = st.sidebar.slider("Electrical Conductivity (EC) [dS/m]", 0.1, 4.5, default_vals['EC'], step=0.05)
oc_val = st.sidebar.slider("Organic Carbon (OC) [%]", 0.1, 1.8, default_vals['OC'], step=0.02)

st.sidebar.subheader("🔬 Micronutrients (ppm)")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    s_val = st.number_input("Sulfur (S)", 0.0, 50.0, default_vals['S'], step=0.5)
    fe_val = st.number_input("Iron (Fe)", 0.0, 30.0, default_vals['Fe'], step=0.5)
    mn_val = st.number_input("Manganese (Mn)", 0.0, 20.0, default_vals['Mn'], step=0.5)
with col_s2:
    zn_val = st.number_input("Zinc (Zn)", 0.0, 5.0, default_vals['Zn'], step=0.05)
    cu_val = st.number_input("Copper (Cu)", 0.0, 4.0, default_vals['Cu'], step=0.05)
    b_val = st.number_input("Boron (B)", 0.0, 3.0, default_vals['B'], step=0.05)

# Tabs
tab1, tab2, tab3 = st.tabs(["🌱 Live Soil Diagnostic", "📊 Model Performance & Explainability", "📁 Batch Sample Testing"])

input_features = [n_val, p_val, k_val, ph_val, ec_val, oc_val, s_val, zn_val, fe_val, cu_val, mn_val, b_val]
input_array = np.array([input_features])
if meta.get('best_is_scaled', False):
    input_array = scaler.transform(input_array)

pred_idx = model.predict(input_array)[0]
pred_class = class_names[pred_idx]
probs = model.predict_proba(input_array)[0]
conf = float(np.max(probs)) * 100

with tab1:
    col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 1.6])
    
    with col_res1:
        color = "#10b981" if pred_class == "High" else ("#f59e0b" if pred_class == "Moderate" else "#f43f5e")
        st.markdown(f"""
        <div style="background:{color}18; border:2px solid {color}; border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:0.85rem; font-weight:700; color:{color}; text-transform:uppercase;">Predicted Fertility Class</div>
            <div style="font-size:2.2rem; font-weight:800; color:#fff; margin:6px 0;">{pred_class} Fertility</div>
            <div style="font-size:1rem; color:#cbd5e1;">Model Confidence: <strong>{conf:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": [round(p * 100, 1) for p in probs]
        })
        fig_prob = px.bar(prob_df, x="Class", y="Probability (%)", color="Class",
                          color_discrete_map={"High": "#10b981", "Moderate": "#f59e0b", "Low": "#f43f5e"})
        fig_prob.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_res3:
        # Radar Chart of normalized values vs benchmark
        optimal_benchmarks = {'N': 420, 'P': 45, 'K': 220, 'pH': 6.8, 'EC': 0.6, 'OC': 0.85, 'S': 18, 'Zn': 1.2, 'Fe': 9.0, 'Cu': 0.6, 'Mn': 5.0, 'B': 0.85}
        norm_vals = [min(180, max(20, (v / optimal_benchmarks[k]) * 100)) for k, v in zip(feature_cols, input_features)]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=feature_cols,
            fill='toself',
            name='Current Sample (% Opt)',
            line_color='#10b981'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[100]*len(feature_cols),
            theta=feature_cols,
            name='Optimal Standard (100%)',
            line=dict(color='gray', dash='dash')
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 180])), height=240, margin=dict(l=30, r=30, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    
    # Agronomic Recommendations
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.subheader("🌾 Targeted Fertilizer & Soil Conditioners")
        if ph_val < 6.0:
            st.warning(f"⚠️ Acidic Soil (pH {ph_val}): Apply Agricultural Limestone (CaCO₃) @ 2.5 - 3.5 tons/ha to neutralize acidity.")
        elif ph_val > 8.0:
            st.warning(f"⚠️ Alkaline/Sodic Soil (pH {ph_val}): Apply Agricultural Gypsum (CaSO₄·2H₂O) @ 2.5 - 4.0 tons/ha with leaching.")
            
        if oc_val < 0.50:
            st.info(f"🍂 Low Organic Matter (OC {oc_val}%): Incorporate 8 - 10 tons/ha Well-decomposed Farmyard Manure or Vermicompost.")
            
        if n_val < 280:
            st.write("• **Urea (46% N)**: 120 - 130 kg/ha in 2-3 split applications (Basal, Vegetative, Panicle).")
        if p_val < 23:
            st.write("• **Single Super Phosphate (SSP)**: 120 - 150 kg/ha basal placement.")
        if k_val < 140:
            st.write("• **Muriate of Potash (MOP)**: 60 - 80 kg/ha for disease & drought tolerance.")
        if zn_val < 0.6:
            st.write("• **Zinc Sulfate (ZnSO₄)**: 20 - 25 kg/ha soil application or 0.5% foliar spray.")

    with col_rec2:
        st.subheader("🌱 Recommended Crop Varieties")
        if pred_class == "High":
            st.success("✅ **High-Productivity Crops**: Wheat, Hybrid Maize, Sugarcane, Cotton, High-value Vegetables.")
        elif pred_class == "Moderate":
            st.info("ℹ️ **Medium Demand Crops**: Soybeans, Chickpeas, Mustard, Barley, Groundnut, Millets.")
        else:
            st.error("⚠️ **Hardy / Tolerant Crops**: Pearl Millet, Sorghum, Cowpea, Lentils, Tea/Coffee (if acidic).")

with tab2:
    st.subheader("🏆 Model Benchmarks & Cross-Validation")
    comp = meta.get('model_comparison', {})
    comp_df = pd.DataFrame([
        {
            "Model": k,
            "5-Fold CV Acc (%)": v["cv_accuracy_mean"],
            "Test Accuracy (%)": v["test_accuracy"],
            "Precision (%)": v["precision_weighted"],
            "F1-Score (%)": v["f1_weighted"]
        } for k, v in comp.items()
    ])
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("🔍 Feature Importance Ranking (Explainable AI)")
    fi = meta.get('feature_importances', {})
    fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Relative Importance": list(fi.values())})
    fig_fi = px.bar(fi_df, x="Relative Importance", y="Feature", orientation='h', color="Relative Importance", color_continuous_scale="Viridis")
    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

with tab3:
    st.subheader("📁 Batch Soil Testing via CSV")
    uploaded_file = st.file_uploader("Upload Soil Test Samples CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            X_batch = batch_df[feature_cols].values
            if meta.get('best_is_scaled', False):
                X_batch = scaler.transform(X_batch)
            batch_preds = model.predict(X_batch)
            batch_df['Predicted_Fertility'] = [class_names[i] for i in batch_preds]
            
            st.success(f"Successfully processed {len(batch_df)} samples!")
            st.dataframe(batch_df.head(50), use_container_width=True)
            
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Enriched Predictions CSV", csv, "soil_predictions.csv", "text/csv")
