"""
Soil Fertility Machine Learning Model Training Pipeline
Trains, evaluates, compares multiple ML classifiers, and exports the top-performing
model alongside scalers and detailed metadata for the web dashboard.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

def train_and_evaluate_models():
    # 1. Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'soil_fertility_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading soil dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Features and target definition
    feature_cols = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']
    target_col = 'Fertility_Class'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode target: Low -> 0, Moderate -> 1, High -> 2
    label_encoder = LabelEncoder()
    label_encoder.fit(['Low', 'Moderate', 'High'])
    y_encoded = label_encoder.transform(y)
    class_names = list(label_encoder.classes_)

    # 3. Stratified Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # 4. Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Candidates Dictionary
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=4, random_state=42),
            "use_scaled": False
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(n_estimators=200, max_depth=14, random_state=42),
            "use_scaled": False
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42),
            "use_scaled": False
        },
        "Support Vector Machine (RBF)": {
            "model": SVC(kernel='rbf', C=2.0, probability=True, random_state=42),
            "use_scaled": True
        },
        "Neural Network (MLP)": {
            "model": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, alpha=0.001, random_state=42),
            "use_scaled": True
        }
    }

    results = {}
    best_model_name = None
    best_accuracy = -1.0
    best_model_obj = None
    best_is_scaled = False

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*60)
    print("Training and Evaluating Machine Learning Classifiers")
    print("="*60)

    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # 5-fold CV score
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')
        mean_cv_acc = float(np.mean(cv_scores))

        # Train on full training split
        clf.fit(X_tr, y_train)

        # Test evaluation
        y_pred = clf.predict(X_te)
        test_acc = float(accuracy_score(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Class-wise report
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        results[name] = {
            "cv_accuracy_mean": round(mean_cv_acc * 100, 2),
            "cv_accuracy_std": round(float(np.std(cv_scores)) * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
            "precision_weighted": round(float(precision) * 100, 2),
            "recall_weighted": round(float(recall) * 100, 2),
            "f1_weighted": round(float(f1) * 100, 2),
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "use_scaled": use_scaled
        }

        print(f"[{name}]")
        print(f"  5-Fold CV Acc: {mean_cv_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | F1: {f1*100:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_name = name
            best_model_obj = clf
            best_is_scaled = use_scaled

    print("="*60)
    print(f"Top Performing Model: {best_model_name} with Test Accuracy: {best_accuracy*100:.2f}%")
    print("="*60)

    # 6. Feature Importances (from best model or Random Forest)
    rf_model = models["Random Forest"]["model"]
    feature_importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, rf_model.feature_importances_)
    }
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # 7. Agronomic Reference Benchmarks for recommendations & gauge charts
    soil_benchmarks = {
        "N": {"name": "Available Nitrogen", "unit": "kg/ha", "low_max": 280, "opt_min": 280, "opt_max": 560, "high_min": 560, "min_disp": 0, "max_disp": 700},
        "P": {"name": "Available Phosphorus", "unit": "kg/ha", "low_max": 23, "opt_min": 23, "opt_max": 56, "high_min": 56, "min_disp": 0, "max_disp": 110},
        "K": {"name": "Available Potassium", "unit": "kg/ha", "low_max": 140, "opt_min": 140, "opt_max": 280, "high_min": 280, "min_disp": 0, "max_disp": 600},
        "pH": {"name": "Soil pH", "unit": "", "low_max": 6.0, "opt_min": 6.0, "opt_max": 7.5, "high_min": 7.5, "min_disp": 3.0, "max_disp": 10.0},
        "EC": {"name": "Electrical Conductivity", "unit": "dS/m", "opt_min": 0.1, "opt_max": 1.0, "critical": 1.5, "saline": 2.0, "min_disp": 0, "max_disp": 5.0},
        "OC": {"name": "Organic Carbon", "unit": "%", "low_max": 0.50, "opt_min": 0.50, "opt_max": 0.75, "high_min": 0.75, "min_disp": 0, "max_disp": 2.0},
        "S": {"name": "Available Sulfur", "unit": "ppm", "low_max": 10.0, "opt_min": 10.0, "opt_max": 25.0, "high_min": 25.0, "min_disp": 0, "max_disp": 50.0},
        "Zn": {"name": "Zinc (Zn)", "unit": "ppm", "low_max": 0.60, "opt_min": 0.60, "opt_max": 1.50, "high_min": 1.50, "min_disp": 0, "max_disp": 3.5},
        "Fe": {"name": "Iron (Fe)", "unit": "ppm", "low_max": 4.50, "opt_min": 4.50, "opt_max": 12.0, "high_min": 12.0, "min_disp": 0, "max_disp": 30.0},
        "Cu": {"name": "Copper (Cu)", "unit": "ppm", "low_max": 0.20, "opt_min": 0.20, "opt_max": 0.80, "high_min": 0.80, "min_disp": 0, "max_disp": 3.0},
        "Mn": {"name": "Manganese (Mn)", "unit": "ppm", "low_max": 2.00, "opt_min": 2.00, "opt_max": 8.00, "high_min": 8.00, "min_disp": 0, "max_disp": 15.0},
        "B": {"name": "Boron (B)", "unit": "ppm", "low_max": 0.50, "opt_min": 0.50, "opt_max": 1.20, "high_min": 1.20, "min_disp": 0, "max_disp": 2.5}
    }

    # 8. Save artifacts
    model_save_path = os.path.join(models_dir, 'best_soil_model.pkl')
    scaler_save_path = os.path.join(models_dir, 'scaler.pkl')
    metadata_save_path = os.path.join(models_dir, 'model_metadata.json')

    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    metadata = {
        "best_model_name": best_model_name,
        "best_test_accuracy": round(best_accuracy * 100, 2),
        "best_is_scaled": best_is_scaled,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "model_comparison": results,
        "feature_importances": sorted_importances,
        "benchmarks": soil_benchmarks,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts successfully saved:")
    print(f" -> Best Model: {model_save_path}")
    print(f" -> Scaler: {scaler_save_path}")
    print(f" -> Metadata: {metadata_save_path}")

    # 9. Verify test sample inference
    sample_val = np.array([[380.0, 42.0, 260.0, 6.8, 0.5, 0.82, 18.0, 1.2, 9.5, 0.6, 5.0, 0.9]])
    if best_is_scaled:
        sample_val = scaler.transform(sample_val)
    pred_cls_idx = best_model_obj.predict(sample_val)[0]
    pred_probs = best_model_obj.predict_proba(sample_val)[0]
    print(f"\nVerification Prediction: Class = {class_names[pred_cls_idx]} (Probs: {dict(zip(class_names, [round(p, 3) for p in pred_probs]))})")

if __name__ == '__main__':
    train_and_evaluate_models()
