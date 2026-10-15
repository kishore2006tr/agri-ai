"""
Soil Fertility Classification - Flask Web Server & Agronomic Recommendation API
Serves modern web application and provides real-time ML inference endpoints.
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load Model, Scaler, and Metadata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model_path = os.path.join(MODELS_DIR, 'best_soil_model.pkl')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')

best_model = None
scaler = None
metadata = {}

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {metadata.get('best_model_name')} model successfully.")
else:
    print("Warning: Model artifacts not found. Please run train_models.py first.")

FEATURE_COLS = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']
CLASS_NAMES = ['High', 'Low', 'Moderate'] if not metadata else metadata.get('class_names', ['Low', 'Moderate', 'High'])

# Agronomic Benchmarks (ICAR & FAO Standards)
BENCHMARKS = {
    "N": {"name": "Nitrogen (N)", "unit": "kg/ha", "opt_min": 280, "opt_max": 560, "low": 280, "high": 560},
    "P": {"name": "Phosphorus (P)", "unit": "kg/ha", "opt_min": 23, "opt_max": 56, "low": 23, "high": 56},
    "K": {"name": "Potassium (K)", "unit": "kg/ha", "opt_min": 140, "opt_max": 280, "low": 140, "high": 280},
    "pH": {"name": "Soil pH", "unit": "pH", "opt_min": 6.0, "opt_max": 7.5, "low": 6.0, "high": 7.5},
    "EC": {"name": "Salinity (EC)", "unit": "dS/m", "opt_min": 0.1, "opt_max": 1.0, "low": 0.0, "high": 1.5},
    "OC": {"name": "Organic Carbon (OC)", "unit": "%", "opt_min": 0.50, "opt_max": 0.75, "low": 0.50, "high": 0.75},
    "S": {"name": "Sulfur (S)", "unit": "ppm", "opt_min": 10.0, "opt_max": 25.0, "low": 10.0, "high": 25.0},
    "Zn": {"name": "Zinc (Zn)", "unit": "ppm", "opt_min": 0.60, "opt_max": 1.50, "low": 0.60, "high": 1.50},
    "Fe": {"name": "Iron (Fe)", "unit": "ppm", "opt_min": 4.50, "opt_max": 12.0, "low": 4.50, "high": 12.0},
    "Cu": {"name": "Copper (Cu)", "unit": "ppm", "opt_min": 0.20, "opt_max": 0.80, "low": 0.20, "high": 0.80},
    "Mn": {"name": "Manganese (Mn)", "unit": "ppm", "opt_min": 2.00, "opt_max": 8.00, "low": 2.00, "high": 8.00},
    "B": {"name": "Boron (B)", "unit": "ppm", "opt_min": 0.50, "opt_max": 1.20, "low": 0.50, "high": 1.20}
}

SAMPLE_PRESETS = [
    {
        "id": "fertile_alluvial",
        "name": "High Fertility - Alluvial Loam",
        "description": "Rich in organic matter and balanced primary/secondary nutrients. Ideal prime farmland.",
        "values": {"N": 480.0, "P": 58.0, "K": 340.0, "pH": 6.8, "EC": 0.55, "OC": 0.95, "S": 22.0, "Zn": 1.45, "Fe": 11.5, "Cu": 0.75, "Mn": 6.2, "B": 1.05}
    },
    {
        "id": "moderate_field",
        "name": "Moderate Fertility - Typical Farmland",
        "description": "Good overall baseline with mild nitrogen & organic carbon requirements for optimal yield.",
        "values": {"N": 310.0, "P": 32.0, "K": 210.0, "pH": 6.7, "EC": 0.65, "OC": 0.60, "S": 13.5, "Zn": 0.85, "Fe": 7.5, "Cu": 0.45, "Mn": 3.8, "B": 0.65}
    },
    {
        "id": "acidic_degraded",
        "name": "Low Fertility - Acidic & Exhausted",
        "description": "Strongly acidic soil (pH 4.8) with depleted nitrogen, phosphorus, zinc and low organic carbon.",
        "values": {"N": 150.0, "P": 11.0, "K": 105.0, "pH": 4.8, "EC": 0.35, "OC": 0.32, "S": 6.0, "Zn": 0.35, "Fe": 7.0, "Cu": 0.18, "Mn": 1.5, "B": 0.25}
    },
    {
        "id": "saline_alkali",
        "name": "Low Fertility - Saline-Alkali Soil",
        "description": "High pH (8.7) and excessive electrical conductivity (salts) causing micronutrient lockup.",
        "values": {"N": 175.0, "P": 14.0, "K": 185.0, "pH": 8.7, "EC": 2.80, "OC": 0.28, "S": 14.0, "Zn": 0.40, "Fe": 2.6, "Cu": 0.20, "Mn": 1.2, "B": 0.70}
    }
]

def analyze_soil_parameters(values: dict):
    evaluations = {}
    deficiencies = []
    excesses = []
    optimal_count = 0

    for key in FEATURE_COLS:
        val = float(values.get(key, 0))
        bench = BENCHMARKS[key]
        
        if key == 'pH':
            if val < 5.5:
                status = "Strongly Acidic"
                color = "danger"
                deficiencies.append("Severe Soil Acidity")
            elif val < 6.0:
                status = "Moderately Acidic"
                color = "warning"
                deficiencies.append("Mild Soil Acidity")
            elif val <= 7.5:
                status = "Optimal / Neutral"
                color = "success"
                optimal_count += 1
            elif val <= 8.5:
                status = "Moderately Alkaline"
                color = "warning"
                excesses.append("Alkalinity")
            else:
                status = "Strongly Alkaline / Sodic"
                color = "danger"
                excesses.append("Severe Alkalinity")
        elif key == 'EC':
            if val <= 1.0:
                status = "Non-Saline (Safe)"
                color = "success"
                optimal_count += 1
            elif val <= 2.0:
                status = "Slight Salinity"
                color = "warning"
                excesses.append("Moderate Salinity (EC)")
            else:
                status = "High Salinity (Injurious)"
                color = "danger"
                excesses.append("High Salinity (EC)")
        else:
            if val < bench['low']:
                status = "Deficient"
                color = "danger"
                deficiencies.append(f"{bench['name']} Deficient")
            elif val <= bench['high']:
                status = "Sufficient / Optimal"
                color = "success"
                optimal_count += 1
            else:
                status = "High / Surfeit"
                color = "info"

        evaluations[key] = {
            "value": val,
            "unit": bench['unit'],
            "name": bench['name'],
            "status": status,
            "badge_color": color,
            "opt_min": bench['opt_min'],
            "opt_max": bench['opt_max']
        }

    # Generate Soil Health Score (0 - 100)
    health_score = int(min(100, max(15, (optimal_count / len(FEATURE_COLS)) * 100 + (values.get('OC', 0.5) * 15))))
    
    return evaluations, deficiencies, excesses, health_score

def generate_recommendations(pred_class: str, values: dict, deficiencies: list, excesses: list):
    recommendations = []
    fertilizers = []
    crops = []

    ph = values.get('pH', 6.5)
    ec = values.get('EC', 0.5)
    oc = values.get('OC', 0.5)
    n = values.get('N', 300)
    p = values.get('P', 30)
    k = values.get('K', 200)
    zn = values.get('Zn', 0.8)
    fe = values.get('Fe', 7.0)

    # 1. Soil Amendment & Conditioning
    if ph < 6.0:
        recommendations.append({
            "category": "Soil Conditioning",
            "title": "Apply Agricultural Lime (CaCO3)",
            "detail": f"Soil pH is acidic ({ph}). Apply 2.0 - 3.5 tons/hectare of finely ground agricultural limestone or dolomite to neutralize acidity and unlock phosphorus fixation."
        })
    elif ph > 8.0:
        recommendations.append({
            "category": "Soil Conditioning",
            "title": "Apply Gypsum (CaSO4·2H2O)",
            "detail": f"Soil pH is alkaline/sodic ({ph}). Incorporate agricultural gypsum (2.5 - 4.0 tons/ha) accompanied by deep leaching to displace exchangeable sodium."
        })

    if ec > 1.5:
        recommendations.append({
            "category": "Salinity Management",
            "title": "Install Leaching & Drainage Channels",
            "detail": f"Electrical conductivity is elevated ({ec} dS/m). Ensure adequate drainage to wash root-zone salts and avoid saline groundwater irrigation."
        })

    # 2. Organic Matter
    if oc < 0.50:
        recommendations.append({
            "category": "Organic Carbon Enhancement",
            "title": "Incorporate Compost / Farmyard Manure (FYM)",
            "detail": f"Organic Carbon is low ({oc}%). Incorporate 8 - 12 tons/ha of well-decomposed FYM, vermicompost, or cultivate green manure crops (Dhaincha/Sunn hemp)."
        })
        fertilizers.append({"name": "Vermicompost / Well-rotted FYM", "dose": "8 - 10 tons/ha"})
    elif oc < 0.75:
        recommendations.append({
            "category": "Organic Carbon Maintenance",
            "title": "Maintain Mulching & Crop Residues",
            "detail": "Retain crop stubble and add 4 - 5 tons/ha compost annually to maintain active microbial biomass."
        })

    # 3. Primary Macronutrient Fertilizer Program
    if n < 280:
        fertilizers.append({"name": "Urea (46% N) or Ammonium Sulfate", "dose": "110 - 130 kg/ha in 2-3 split doses"})
        recommendations.append({
            "category": "Nutrient Rectification",
            "title": "Nitrogen Boost via Split Application",
            "detail": "Soil nitrogen is deficient. Administer split urea applications (at basal, vegetative, and panicle initiation) to prevent leaching losses."
        })
    elif n <= 560:
        fertilizers.append({"name": "Balanced Urea Maintenance", "dose": "70 - 90 kg/ha in split doses"})

    if p < 23:
        fertilizers.append({"name": "Single Super Phosphate (SSP) or DAP", "dose": "120 - 150 kg/ha applied as basal"})
        recommendations.append({
            "category": "Nutrient Rectification",
            "title": "Phosphorus Application (Basal Placement)",
            "detail": "Place phosphorus 5cm below and beside seeds during sowing for vigorous root system development."
        })
    elif p <= 56:
        fertilizers.append({"name": "Di-Ammonium Phosphate (DAP)", "dose": "50 - 65 kg/ha basal"})

    if k < 140:
        fertilizers.append({"name": "Muriate of Potash (MOP - 60% K2O)", "dose": "60 - 80 kg/ha basal"})
        recommendations.append({
            "category": "Nutrient Rectification",
            "title": "Potash Application for Stress Resistance",
            "detail": "Potassium is low. Apply MOP to enhance lodging resistance, drought tolerance, and grain filling."
        })

    # 4. Micronutrients
    if zn < 0.60:
        fertilizers.append({"name": "Zinc Sulfate Heptahydrate (ZnSO4)", "dose": "20 - 25 kg/ha soil application or 0.5% foliar spray"})
    if fe < 4.50:
        fertilizers.append({"name": "Ferrous Sulfate (FeSO4)", "dose": "15 - 20 kg/ha or 1.0% foliar spray with citric acid"})

    # 5. Crop Recommendations
    if pred_class == "High":
        crops = [
            {"crop": "Wheat / Rice", "yield_potential": "Excellent (95-100%)", "suitability": "Ideal nutrient balance ensures peak grain yield."},
            {"crop": "Maize & Hybrid Corn", "yield_potential": "High (90-95%)", "suitability": "Heavy feeder crops flourish in high organic carbon soil."},
            {"crop": "Cotton / Sugarcane", "yield_potential": "Optimal (92-98%)", "suitability": "Deep root network extracts rich nutrient reserves."},
            {"crop": "High-Value Vegetables (Tomatoes, Capsicum)", "yield_potential": "Exceptional", "suitability": "Excellent return on investment with premium quality produce."}
        ]
    elif pred_class == "Moderate":
        crops = [
            {"crop": "Soybeans / Pulses (Chickpea, Lentil)", "yield_potential": "Good (80-85%)", "suitability": "Legumes fix atmospheric nitrogen and improve fertility."},
            {"crop": "Wheat / Barley", "yield_potential": "Good (75-82%)", "suitability": "Performs well with standard N-P-K supplemental fertilization."},
            {"crop": "Mustard / Groundnut (Oilseeds)", "yield_potential": "Moderate to High", "suitability": "Thrives well with recommended sulfur and potash top-dressing."},
            {"crop": "Millets (Pearl Millet, Finger Millet)", "yield_potential": "Very Resilient", "suitability": "Low input cost with reliable harvest stability."}
        ]
    else: # Low fertility
        if ph < 5.8:
            crops = [
                {"crop": "Tea / Coffee / Cardamom", "yield_potential": "Good", "suitability": "Naturally tolerant to acidic soil conditions."},
                {"crop": "Potato", "yield_potential": "Fair (with liming)", "suitability": "Tolerates moderate acidity; scab incidence is lower."},
                {"crop": "Buckwheat / Oats", "yield_potential": "Sturdy", "suitability": "Extracts bound soil phosphorus effectively."}
            ]
        elif ec > 1.8:
            crops = [
                {"crop": "Barley", "yield_potential": "High Tolerance", "suitability": "Most salt-tolerant cereal crop available."},
                {"crop": "Sugar Beet", "yield_potential": "Moderate", "suitability": "High tolerance to saline conditions."},
                {"crop": "Cotton", "yield_potential": "Moderate", "suitability": "Tolerates moderate osmotic soil stress."}
            ]
        else:
            crops = [
                {"crop": "Millets (Sorghum, Pearl Millet)", "yield_potential": "Dependable", "suitability": "Hardy drought and nutrient-stress tolerant crops."},
                {"crop": "Legumes (Cowpea, Green Gram)", "yield_potential": "Recommended", "suitability": "Rehabilitates depleted soil through biological nitrogen fixation."},
                {"crop": "Cassava / Sweet Potato", "yield_potential": "Fair", "suitability": "Yields reliably even in impoverished soils."}
            ]

    return recommendations, fertilizers, crops

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    if not metadata:
        return jsonify({"error": "Model metadata not available"}), 500
    return jsonify({
        "status": "success",
        "model_name": metadata.get("best_model_name", "Random Forest"),
        "test_accuracy": metadata.get("best_test_accuracy", 100.0),
        "total_samples": metadata.get("total_samples", 3200),
        "train_samples": metadata.get("train_samples", 2560),
        "test_samples": metadata.get("test_samples", 640),
        "class_names": metadata.get("class_names", ['Low', 'Moderate', 'High']),
        "model_comparison": metadata.get("model_comparison", {}),
        "feature_importances": metadata.get("feature_importances", {}),
        "benchmarks": BENCHMARKS
    })

@app.route('/api/sample-presets', methods=['GET'])
def get_presets():
    return jsonify({"presets": SAMPLE_PRESETS})

@app.route('/api/predict', methods=['POST'])
def predict_soil():
    if best_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        # Extract features in required order
        features_list = []
        parsed_vals = {}
        for col in FEATURE_COLS:
            val = float(data.get(col, 0.0))
            features_list.append(val)
            parsed_vals[col] = val

        raw_array = np.array([features_list])
        use_scaled = metadata.get('best_is_scaled', False)
        if use_scaled and scaler is not None:
            features_input = scaler.transform(raw_array)
        else:
            features_input = raw_array

        pred_idx = int(best_model.predict(features_input)[0])
        pred_class = CLASS_NAMES[pred_idx]

        # Probabilities
        if hasattr(best_model, 'predict_proba'):
            probs = best_model.predict_proba(features_input)[0]
            prob_dict = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASS_NAMES))}
            confidence = float(np.max(probs)) * 100
        else:
            prob_dict = {pred_class: 100.0}
            confidence = 100.0

        # Agronomic Analysis
        evaluations, deficiencies, excesses, health_score = analyze_soil_parameters(parsed_vals)
        recommendations, fertilizers, crops = generate_recommendations(pred_class, parsed_vals, deficiencies, excesses)

        return jsonify({
            "status": "success",
            "prediction": pred_class,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
            "health_score": health_score,
            "evaluations": evaluations,
            "deficiencies": deficiencies,
            "excesses": excesses,
            "recommendations": recommendations,
            "fertilizers": fertilizers,
            "crops": crops
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    if best_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        # Check required columns
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required columns in CSV: {', '.join(missing)}"}), 400

        X = df[FEATURE_COLS].values
        use_scaled = metadata.get('best_is_scaled', False)
        X_input = scaler.transform(X) if use_scaled and scaler is not None else X

        preds = best_model.predict(X_input)
        df['Predicted_Fertility'] = [CLASS_NAMES[i] for i in preds]

        if hasattr(best_model, 'predict_proba'):
            probs = best_model.predict_proba(X_input)
            df['Confidence_%'] = [round(float(np.max(p)) * 100, 1) for p in probs]

        summary = df['Predicted_Fertility'].value_counts().to_dict()
        total = len(df)
        
        sample_results = df.head(100).to_dict(orient='records')
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        return jsonify({
            "status": "success",
            "total_samples": total,
            "summary": summary,
            "preview": sample_results,
            "csv_content": csv_buffer.getvalue()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Soil Fertility Classification Web App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
