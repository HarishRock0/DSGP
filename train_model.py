#!/usr/bin/env python
"""
train_model.py — Fixed SkillDev K-Means training script.

Replaces ML/skilldev.ipynb with a standalone, runnable script that includes:
  FIX 1: Income imputation — all remaining NaN income → 0 (non-employed = zero income)
  FIX 2: Digital_Divide_Flag — (Q60A==2) | (Q61==2) instead of (Q2==1) & (Q60A==2)
  FIX 3: Saves scaler.pkl separately so NLPC outlier detection works correctly

Usage:
    python train_model.py
    python train_model.py --data data/LFS-2023.csv --out model/
"""

import os
import sys
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SkillDev Model Class (fixed)
# ─────────────────────────────────────────────────────────────────────────────

class SkillDev:
    """
    Intervention-based K-Means clustering model for LFS-2023.

    Clusters respondents into 4 welfare intervention groups:
      - Digitally Excluded       → needs tech/digital training
      - Economically Vulnerable  → needs social safety net
      - High Skill Gap           → needs job matching & upskilling
      - Stable Workforce         → needs advanced/leadership skills

    FIXES vs original notebook:
      1. Income imputation: all remaining NaN income set to 0 (non-employed)
      2. Digital_Divide_Flag: (Q60A==2) | (Q61==2) captures true digital exclusion
      3. Scaler saved separately for NLPC outlier detection
    """

    def __init__(self, file_path: str, n_clusters: int = 4):
        print(f"📂 Loading data from {file_path}…")
        self.df = pd.read_csv(file_path)
        self.n_clusters = n_clusters
        self.scaler = RobustScaler()
        self.kmeans = None
        self.features = None
        self.needs_features = None
        self.scaled_data = None
        self.weighted_data = None
        self.cluster_mapping = {}
        print(f"✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")

    # ── Step 1: Income imputation ─────────────────────────────────────────────

    def clean_and_impute_income(self):
        """
        Impute Q45_A_1 (monthly income) using LFS routing logic.

        Fix vs notebook: adds a final catch-all rule that sets ALL remaining
        NaN income to 0 (non-employed people have zero cash income).

        Rules applied in order:
          1. Unpaid family workers (Q16==4)                 → income = 0
          2. Strictly not employed (Q2==2 AND Q3==2)        → income = 0
          3. Employed (Q16 in {1,2,3}) with missing income  → group-median imputation
             Fallback chain: [SECTOR, Q16, EDU] → [SECTOR, Q16] → [Q16] → global median
          4. FIX: all remaining NaN                         → income = 0
        """
        df = self.df

        # Ensure numeric
        for col in ['Q45_A_1', 'Q2', 'Q3', 'Q16', 'EDU', 'SECTOR']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rule 1: Unpaid family workers → zero income
        if 'Q16' in df.columns:
            df.loc[df['Q16'] == 4, 'Q45_A_1'] = 0

        # Rule 2: Strictly not employed → zero income
        if 'Q2' in df.columns and 'Q3' in df.columns:
            df.loc[(df['Q2'] == 2) & (df['Q3'] == 2), 'Q45_A_1'] = 0

        # Rule 3: Employed with missing income → group-median
        if 'Q16' in df.columns and 'Q45_A_1' in df.columns:
            employed_mask = df['Q16'].isin([1, 2, 3])
            still_missing = employed_mask & df['Q45_A_1'].isna()

            if still_missing.any():
                for fallback_cols in [
                    [c for c in ['SECTOR', 'Q16', 'EDU'] if c in df.columns],
                    [c for c in ['SECTOR', 'Q16'] if c in df.columns],
                    ['Q16'],
                ]:
                    if not still_missing.any():
                        break
                    if fallback_cols:
                        gm = df[employed_mask].groupby(fallback_cols)['Q45_A_1'].transform('median')
                        df.loc[still_missing, 'Q45_A_1'] = gm[still_missing].values
                        still_missing = employed_mask & df['Q45_A_1'].isna()

                # Global median fallback
                if still_missing.any():
                    df.loc[still_missing, 'Q45_A_1'] = df['Q45_A_1'].median()

        # ✅ FIX: any remaining NaN income → 0 (non-employed, outside labour force)
        remaining_nan = df['Q45_A_1'].isna().sum() if 'Q45_A_1' in df.columns else 0
        if remaining_nan > 0:
            df['Q45_A_1'] = df['Q45_A_1'].fillna(0)
            print(f"   ✅ FIX applied: {remaining_nan} remaining NaN income → 0 (non-employed)")

        self.df = df
        final_nan = self.df['Q45_A_1'].isna().sum() if 'Q45_A_1' in self.df.columns else 0
        print(f"   ✅ Income imputation complete. Remaining NaNs: {final_nan}")

    # ── Step 2: Feature engineering ──────────────────────────────────────────

    def preprocess_data(self):
        """Engineer needs-based features for intervention-based clustering."""
        print("\n📊 Engineering Needs-Based Features…")

        self.clean_and_impute_income()

        # Convert all to numeric
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Feature 1: Skill Gap Score = education advantage vs income
        if 'EDU' in self.df.columns and 'Q45_A_1' in self.df.columns:
            edu_min, edu_max = self.df['EDU'].min(), self.df['EDU'].max()
            inc_min, inc_max = self.df['Q45_A_1'].min(), self.df['Q45_A_1'].max()
            edu_scaled = (self.df['EDU'] - edu_min) / (edu_max - edu_min + 1e-9)
            inc_scaled = (self.df['Q45_A_1'] - inc_min) / (inc_max - inc_min + 1e-9)
            self.df['Skill_Gap_Score'] = edu_scaled - inc_scaled
            print("   ✅ Skill_Gap_Score engineered")
        else:
            self.df['Skill_Gap_Score'] = 0.0

        # Feature 2: Vulnerability Index (income + disability)
        vulnerability_score = pd.Series(0.0, index=self.df.index)

        if 'Q45_A_1' in self.df.columns:
            income_inv = 1.0 / (self.df['Q45_A_1'] + 1)
            inc_min, inc_max = income_inv.min(), income_inv.max()
            income_scaled = (income_inv - inc_min) / (inc_max - inc_min + 1e-9)
            vulnerability_score += income_scaled * 0.4

        disability_cols = ['P15', 'P16', 'P17', 'P18', 'P19', 'P20']
        d_sum = pd.Series(0.0, index=self.df.index)
        d_cnt = 0
        for col in disability_cols:
            if col in self.df.columns:
                d_sum += (self.df[col].fillna(1) - 1) / 3.0
                d_cnt += 1
        if d_cnt > 0:
            vulnerability_score += (d_sum / d_cnt) * 0.6

        self.df['Vulnerability_Index'] = vulnerability_score
        print("   ✅ Vulnerability_Index engineered")

        # Feature 3: Digital Divide Flag
        # ✅ FIX: original (Q2==1) & (Q60A==2) was near-zero for all clusters.
        #         New: anyone who cannot use a computer OR has not used internet.
        if 'Q60A' in self.df.columns and 'Q61' in self.df.columns:
            self.df['Digital_Divide_Flag'] = (
                (self.df['Q60A'] == 2) | (self.df['Q61'] == 2)
            ).astype(int)
            pct = self.df['Digital_Divide_Flag'].mean() * 100
            print(f"   ✅ Digital_Divide_Flag engineered  ({pct:.1f}% of records flagged)")
        elif 'Q60A' in self.df.columns:
            self.df['Digital_Divide_Flag'] = (self.df['Q60A'] == 2).astype(int)
            print("   ✅ Digital_Divide_Flag engineered (Q60A only)")
        else:
            self.df['Digital_Divide_Flag'] = 0
            print("   ⚠️ Digital_Divide_Flag defaulted to 0 (Q60A/Q61 missing)")

        self.needs_features = ['Skill_Gap_Score', 'Vulnerability_Index', 'Digital_Divide_Flag']

        # Fill any remaining NaN in engineered features with median
        for col in self.needs_features:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        self.df = self.df.dropna(subset=self.needs_features)
        self.features = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if col.upper() not in ['ID', 'INDEX']
        ]

        print(f"✅ Feature engineering complete. Records: {len(self.df):,}")
        print(f"   Clustering features: {self.needs_features}")
        return self.df

    # ── Step 3: Cluster training ──────────────────────────────────────────────

    def train_clusters(self):
        """Train K-Means on needs-based features with survey weights."""
        print(f"\n🤖 Training K-Means (k={self.n_clusters}) on needs features…")

        self.scaled_data = self.scaler.fit_transform(self.df[self.needs_features])
        self.weighted_data = self.scaled_data  # equal feature weights

        # Survey weights (Annual_Factor) for population-representative clusters
        weights = None
        if 'Annual_Factor' in self.df.columns:
            weights = self.df['Annual_Factor'].fillna(1.0).values
            print("   Using Annual_Factor survey weights")

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.kmeans.fit(self.weighted_data, sample_weight=weights)
        self.df['cluster_id'] = self.kmeans.predict(self.weighted_data)

        # Compute centroid distances and store in dataframe
        ids = self.df['cluster_id'].values
        distances = np.linalg.norm(self.weighted_data - self.kmeans.cluster_centers_[ids], axis=1)
        self.df['distance_to_center'] = distances
        print("   ✅ distance_to_center computed and stored")

        self._assign_dynamic_labels()
        self.df['cluster_label'] = self.df['cluster_id'].map(self.cluster_mapping)

        print(f"✅ Clustering complete!")
        for cid, label in sorted(self.cluster_mapping.items()):
            cnt = (self.df['cluster_id'] == cid).sum()
            pct = cnt / len(self.df) * 100
            print(f"   {cid}: {label}  ({cnt:,} members, {pct:.1f}%)")

    def _assign_dynamic_labels(self):
        """
        Assign cluster labels using strict one-to-one elimination.

        Priority:
          1. Digitally Excluded   → highest Digital_Divide_Flag centroid
          2. Economically Vulnerable → highest Vulnerability_Index (remaining)
          3. High Skill Gap        → highest Skill_Gap_Score (remaining)
          4. Stable Workforce      → last remaining cluster
        """
        centers = self.kmeans.cluster_centers_
        available = set(range(self.n_clusters))
        new_mapping = {}

        def feat_idx(name):
            try:
                return self.needs_features.index(name)
            except ValueError:
                return None

        idx_d = feat_idx('Digital_Divide_Flag')
        idx_v = feat_idx('Vulnerability_Index')
        idx_s = feat_idx('Skill_Gap_Score')

        if idx_d is not None:
            cid = max(available, key=lambda i: centers[i][idx_d])
            new_mapping[cid] = 'Digitally Excluded - Needs Tech Training'
            available.discard(cid)

        if idx_v is not None and available:
            cid = max(available, key=lambda i: centers[i][idx_v])
            new_mapping[cid] = 'Economically Vulnerable - Needs Social Safety Net'
            available.discard(cid)

        if idx_s is not None and available:
            cid = max(available, key=lambda i: centers[i][idx_s])
            new_mapping[cid] = 'High Skill Gap - Needs Job Matching'
            available.discard(cid)

        for cid in available:
            new_mapping[cid] = 'Stable Workforce - Needs Leadership/Advanced Skills'

        self.cluster_mapping = new_mapping
        print("   🏷️  Dynamic labels assigned:")
        for cid, label in sorted(new_mapping.items()):
            print(f"      {cid}: {label}")

    # ── Step 4: Summary ───────────────────────────────────────────────────────

    def show_clusters(self):
        print("\n📋 CLUSTER SUMMARY:")
        print(f"   Total records: {len(self.df):,}")
        print()
        for cid in range(self.n_clusters):
            cd = self.df[self.df['cluster_id'] == cid]
            cnt = len(cd)
            pct = cnt / len(self.df) * 100
            sg  = cd['Skill_Gap_Score'].mean()
            vi  = cd['Vulnerability_Index'].mean()
            dd  = cd['Digital_Divide_Flag'].mean() * 100
            print(f"   Cluster {cid}: {self.cluster_mapping[cid]}")
            print(f"     Size: {cnt:,} ({pct:.1f}%)")
            print(f"     Skill Gap: {sg:.3f}  |  Vulnerability: {vi:.3f}  |  Digital Divide: {dd:.1f}%")
        print()
        print("📊 Cluster centres (needs features):")
        for i, center in enumerate(self.kmeans.cluster_centers_):
            print(f"   Cluster {i}: {center}")

    # ── Step 5: Save ─────────────────────────────────────────────────────────

    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✅ Model saved → {model_path}")

    def save_scaler(self, scaler_path: str):
        """Save scaler separately for NLPC outlier detection."""
        os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved → {scaler_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SkillDev K-Means model")
    parser.add_argument('--data', default=None, help="Path to LFS-2023.csv")
    parser.add_argument('--out',  default=None, help="Output directory for model files")
    parser.add_argument('--clusters', type=int, default=4, help="Number of clusters (default 4)")
    args = parser.parse_args()

    # ── Locate data file ────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_data = [
        args.data,
        os.path.join(script_dir, 'data', 'LFS-2023.csv'),
        os.path.join(script_dir, '..', 'data', 'LFS-2023.csv'),
        os.path.join(os.getcwd(), 'data', 'LFS-2023.csv'),
    ]
    csv_file = next((p for p in candidate_data if p and os.path.exists(p)), None)
    if not csv_file:
        print("❌ LFS-2023.csv not found. Tried:")
        for p in candidate_data:
            if p:
                print(f"   {p}")
        print("\nPlease specify the path: python train_model.py --data path/to/LFS-2023.csv")
        sys.exit(1)

    # ── Output paths ─────────────────────────────────────────────────────────
    out_dir = args.out or os.path.join(script_dir, 'model')
    model_path  = os.path.join(out_dir, 'skilldev_model.pkl')
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    csv_out     = os.path.join(script_dir, 'lfs_clustered_data.csv')

    print("\n" + "="*70)
    print("🚀 SkillDev Model Training (Fixed Version)")
    print("="*70)
    print(f"   Data  : {csv_file}")
    print(f"   Output: {out_dir}")
    print(f"   k     : {args.clusters}")
    print("="*70 + "\n")

    # ── Train ────────────────────────────────────────────────────────────────
    model = SkillDev(csv_file, n_clusters=args.clusters)
    model.preprocess_data()
    model.train_clusters()
    model.show_clusters()

    # ── Save ─────────────────────────────────────────────────────────────────
    model.save_model(model_path)
    model.save_scaler(scaler_path)

    model.df.to_csv(csv_out, index=False)
    print(f"✅ Clustered CSV saved → {csv_out} ({len(model.df):,} rows)")
    print("\n" + "="*70)
    print("✅ Training complete! Run NLP/NLP.py to start the agent.")
    print("="*70)


if __name__ == '__main__':
    main()
