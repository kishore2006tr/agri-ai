"""
Soil Fertility Benchmark Dataset Generator
Generates realistic agricultural soil samples based on ICAR (Indian Council of Agricultural Research),
USDA-NRCS, and FAO soil testing criteria.

Features:
- N: Available Nitrogen (kg/ha) [Range: 50 - 650]
- P: Available Phosphorus (kg/ha) [Range: 5 - 100]
- K: Available Potassium (kg/ha) [Range: 50 - 550]
- pH: Soil pH [Range: 4.2 - 9.8]
- EC: Electrical Conductivity (dS/m) [Range: 0.1 - 4.5]
- OC: Organic Carbon (%) [Range: 0.15 - 1.80]
- S: Available Sulfur (ppm) [Range: 2.0 - 45.0]
- Zn: Zinc (ppm) [Range: 0.1 - 3.5]
- Fe: Iron (ppm) [Range: 1.0 - 25.0]
- Cu: Copper (ppm) [Range: 0.05 - 2.5]
- Mn: Manganese (ppm) [Range: 0.5 - 15.0]
- B: Boron (ppm) [Range: 0.1 - 2.2]

Target:
- Fertility_Class: 'Low', 'Moderate', 'High'
"""

import os
import numpy as np
import pandas as pd

def generate_soil_dataset(n_samples: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    # 1. Generate base characteristics across diverse agro-climatic zones
    # We will simulate 3 core clusters with realistic noise and variations
    # Cluster 0: Degraded / Acidic / Saline / Nutrient-poor soils (~30%)
    # Cluster 1: Average / Moderately fertile farmland (~40%)
    # Cluster 2: Prime fertile / Organic rich / Balanced soils (~30%)

    n_low = int(n_samples * 0.32)
    n_mod = int(n_samples * 0.40)
    n_high = n_samples - n_low - n_mod

    records = []

    # Low fertility generation
    for _ in range(n_low):
        # Could be acidic degraded, saline-alkali, or severely exhausted soil
        sub_type = np.random.choice(['acidic_depleted', 'saline_alkali', 'nutrient_exhausted'])
        
        if sub_type == 'acidic_depleted':
            ph = np.random.normal(5.0, 0.4)
            ec = np.random.normal(0.4, 0.15)
            oc = np.random.normal(0.35, 0.1)
            n_val = np.random.normal(160, 45)
            p_val = np.random.normal(12, 5)
            k_val = np.random.normal(110, 30)
            s_val = np.random.normal(6.5, 2.5)
            zn = np.random.normal(0.38, 0.12)
            fe = np.random.normal(6.5, 3.0)  # Fe often soluble in acid
            cu = np.random.normal(0.2, 0.08)
            mn = np.random.normal(1.8, 0.6)
            b = np.random.normal(0.28, 0.1)
        elif sub_type == 'saline_alkali':
            ph = np.random.normal(8.6, 0.4)
            ec = np.random.normal(2.6, 0.7)  # high salinity
            oc = np.random.normal(0.30, 0.1)
            n_val = np.random.normal(180, 50)
            p_val = np.random.normal(15, 6)
            k_val = np.random.normal(190, 50)
            s_val = np.random.normal(14.0, 5.0)
            zn = np.random.normal(0.42, 0.15)  # Zn deficiency common in high pH
            fe = np.random.normal(2.8, 1.0)   # Fe deficiency common in alkaline
            cu = np.random.normal(0.22, 0.09)
            mn = np.random.normal(1.4, 0.5)
            b = np.random.normal(0.7, 0.3)
        else: # nutrient_exhausted
            ph = np.random.normal(6.6, 0.5)
            ec = np.random.normal(0.5, 0.2)
            oc = np.random.normal(0.28, 0.08)
            n_val = np.random.normal(140, 40)
            p_val = np.random.normal(10, 4)
            k_val = np.random.normal(95, 25)
            s_val = np.random.normal(5.0, 2.0)
            zn = np.random.normal(0.35, 0.1)
            fe = np.random.normal(3.5, 1.2)
            cu = np.random.normal(0.18, 0.07)
            mn = np.random.normal(1.6, 0.5)
            b = np.random.normal(0.25, 0.09)

        records.append({
            'N': np.clip(n_val, 50, 400),
            'P': np.clip(p_val, 4, 35),
            'K': np.clip(k_val, 40, 300),
            'pH': np.clip(ph, 3.8, 9.8),
            'EC': np.clip(ec, 0.1, 4.5),
            'OC': np.clip(oc, 0.1, 0.65),
            'S': np.clip(s_val, 1.5, 30.0),
            'Zn': np.clip(zn, 0.08, 1.5),
            'Fe': np.clip(fe, 0.8, 20.0),
            'Cu': np.clip(cu, 0.05, 1.2),
            'Mn': np.clip(mn, 0.4, 8.0),
            'B': np.clip(b, 0.08, 1.5),
            'Fertility_Class': 'Low'
        })

    # Moderate fertility generation
    for _ in range(n_mod):
        ph = np.random.normal(6.7, 0.45)
        ec = np.random.normal(0.65, 0.25)
        oc = np.random.normal(0.60, 0.12)
        n_val = np.random.normal(310, 65)
        p_val = np.random.normal(32, 8)
        k_val = np.random.normal(210, 45)
        s_val = np.random.normal(13.5, 3.5)
        zn = np.random.normal(0.85, 0.25)
        fe = np.random.normal(7.5, 2.2)
        cu = np.random.normal(0.45, 0.15)
        mn = np.random.normal(3.8, 1.1)
        b = np.random.normal(0.65, 0.18)

        records.append({
            'N': np.clip(n_val, 180, 520),
            'P': np.clip(p_val, 18, 60),
            'K': np.clip(k_val, 130, 360),
            'pH': np.clip(ph, 5.8, 8.0),
            'EC': np.clip(ec, 0.2, 1.4),
            'OC': np.clip(oc, 0.42, 0.85),
            'S': np.clip(s_val, 7.0, 28.0),
            'Zn': np.clip(zn, 0.5, 2.0),
            'Fe': np.clip(fe, 3.8, 16.0),
            'Cu': np.clip(cu, 0.2, 1.2),
            'Mn': np.clip(mn, 1.8, 8.0),
            'B': np.clip(b, 0.35, 1.4),
            'Fertility_Class': 'Moderate'
        })

    # High fertility generation
    for _ in range(n_high):
        ph = np.random.normal(6.8, 0.35)
        ec = np.random.normal(0.55, 0.18)
        oc = np.random.normal(0.95, 0.20)
        n_val = np.random.normal(460, 70)
        p_val = np.random.normal(55, 12)
        k_val = np.random.normal(340, 60)
        s_val = np.random.normal(22.0, 5.0)
        zn = np.random.normal(1.45, 0.35)
        fe = np.random.normal(11.5, 3.0)
        cu = np.random.normal(0.75, 0.22)
        mn = np.random.normal(6.2, 1.5)
        b = np.random.normal(1.05, 0.25)

        records.append({
            'N': np.clip(n_val, 320, 680),
            'P': np.clip(p_val, 32, 100),
            'K': np.clip(k_val, 240, 550),
            'pH': np.clip(ph, 6.2, 7.5),
            'EC': np.clip(ec, 0.15, 1.1),
            'OC': np.clip(oc, 0.72, 1.85),
            'S': np.clip(s_val, 12.0, 45.0),
            'Zn': np.clip(zn, 0.85, 3.2),
            'Fe': np.clip(fe, 6.0, 24.0),
            'Cu': np.clip(cu, 0.38, 2.2),
            'Mn': np.clip(mn, 3.2, 14.0),
            'B': np.clip(b, 0.6, 2.0),
            'Fertility_Class': 'High'
        })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # Round columns to realistic precision
    round_rules = {
        'N': 1, 'P': 1, 'K': 1, 'pH': 2, 'EC': 2, 'OC': 2,
        'S': 1, 'Zn': 2, 'Fe': 2, 'Cu': 2, 'Mn': 2, 'B': 2
    }
    for col, dec in round_rules.items():
        df[col] = df[col].round(dec)

    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'soil_fertility_dataset.csv')
    
    print("Generating realistic agricultural soil dataset...")
    df = generate_soil_dataset(n_samples=3200, random_seed=42)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df['Fertility_Class'].value_counts())
    print("\nSample summary statistics:")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
