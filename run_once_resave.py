# # import sys
# # sys.path.insert(0, ".")
# #
# # from models.poverty_risk_model import PovertyRiskModel
# # import pickle
# #
# # with open("model/poverty_risk_model.pkl", "rb") as f:
# #     model = pickle.load(f)
# #
# # model.encoder = None
# #
# # with open("model/poverty_risk_model.pkl", "wb") as f:
# #     pickle.dump(model, f)
# #
# # print("✅ Done")
#
#
# """
# Run this ONCE from the project root to fix the child case pkl.
#
# Usage:
#     cd /Users/upali/Documents/Yasiru/PythonProject/DSGP
#     source .venv/bin/activate
#     python run_once_resave_child.py
# """
# import sys
# sys.path.insert(0, ".")
#
# from models.child_case_model import (  # noqa: F401
#     ChildCaseRuleEngine,
#     rule_high_cases, rule_low_cases, rule_moderate_risk,
#     rule_top5_critical, rule_by_risk_tier, rule_district_lookup, rule_year_trend
# )
# import pickle
#
# model_path = "model/child_case_nlp.pkl"
#
# with open(model_path, "rb") as f:
#     bundle = pickle.load(f)
#
# print("✅ Loaded old bundle, keys:", list(bundle.keys()))
#
# # Rebuild TEMPLATES with rule functions pointing to models.child_case_rule_engine
# TEMPLATES = [
#     {'key': 'high_cases',    'query': bundle['templates'][0]['query'], 'action': rule_high_cases,    'params': {'n': 10}},
#     {'key': 'low_cases',     'query': bundle['templates'][1]['query'], 'action': rule_low_cases,     'params': {'n': 10}},
#     {'key': 'moderate_risk', 'query': bundle['templates'][2]['query'], 'action': rule_moderate_risk, 'params': {'n': 10}},
#     {'key': 'top5_critical', 'query': bundle['templates'][3]['query'], 'action': rule_top5_critical, 'params': {'n': 5}},
#     {'key': 'high_tier',     'query': bundle['templates'][4]['query'], 'action': rule_by_risk_tier,  'params': {'tier': 'High'}},
#     {'key': 'low_tier',      'query': bundle['templates'][5]['query'], 'action': rule_by_risk_tier,  'params': {'tier': 'Low'}},
#     {'key': 'moderate_tier', 'query': bundle['templates'][6]['query'], 'action': rule_by_risk_tier,  'params': {'tier': 'Moderate'}},
# ]
#
# engine = ChildCaseRuleEngine(
#     templates=TEMPLATES,
#     template_queries=bundle['template_queries'],
#     template_embeddings=bundle['template_embeddings'],
#     df_main=bundle['df_main'],
#     known_districts=bundle['known_districts'],
# )
#
# with open(model_path, "wb") as f:
#     pickle.dump(engine, f)
#
# print("✅ Resaved as ChildCaseRuleEngine")
# print("   Rule functions now point to models.child_case_rule_engine")
# print("   You can delete run_once_resave_child.py after this")

"""
rebuild_child_welfare_pkl.py
────────────────────────────
Run this script LOCALLY (in your NumPy 1.26.4 environment) to regenerate
child_welfare_pipeline.pkl so it is compatible with your project.

Usage:
    cd /path/to/your/DSGP/project/root
    python rebuild_child_welfare_pkl.py

The pkl will be saved to:
    model/resource allocation models/child_welfare_pipeline.pkl
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ── Resolve project root (this script should sit at project root) ────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PKL_DIR      = os.path.join(PROJECT_ROOT, "model", "resource allocation models")
PKL_PATH     = os.path.join(PKL_DIR, "child_welfare_pipeline.pkl")

os.makedirs(PKL_DIR, exist_ok=True)
print(f"Project root : {PROJECT_ROOT}")
print(f"PKL output   : {PKL_PATH}")
print(f"NumPy version: {np.__version__}")
print()

# ── STEP 1 — Raw district data (2010-2024) ───────────────────────────────────
print("STEP 1 — Loading district data...")

data = {
    'District':     ['Ampara','Anuradhapura','Badulla','Batticaloa','Colombo','Galle',
                     'Gampaha','Hambantota','Jaffna','Kalutara','Kandy','Kegalle',
                     'Kilinochchi','Kurunegala','Mannar','Matale','Matara','Monaragala',
                     'Mullaitivu','Nuwara Eliya','Polonnaruwa','Puttalam','Rathnapura',
                     'Trincomalee','Vavuniya'],
    '2010': [70,161,122,31,720,308,524,132,7,260,153,169,2,289,4,87,162,65,2,87,89,150,221,57,19],
    '2011': [113,345,180,35,1037,435,797,187,15,408,245,226,7,442,19,143,244,160,11,97,156,295,341,81,57],
    '2012': [129,406,213,59,1174,618,948,244,33,479,302,252,35,636,21,144,299,170,26,108,204,344,431,88,55],
    '2013': [220,578,286,81,1477,853,1146,362,46,622,417,438,47,785,29,196,440,281,53,220,306,491,668,135,96],
    '2014': [241,568,270,155,1403,682,1169,336,213,600,456,399,144,756,86,226,357,256,160,241,227,452,596,162,160],
    '2015': [246,573,271,158,1522,700,1187,439,198,634,474,404,104,827,65,222,389,241,121,235,302,540,622,130,128],
    '2016': [183,521,271,148,1271,635,917,412,169,558,407,351,116,722,68,192,307,252,119,211,221,463,593,139,115],
    '2017': [208,441,198,179,1302,586,974,310,188,591,382,286,126,681,69,200,316,226,132,184,245,403,518,138,131],
    '2018': [229,434,237,189,1330,617,1066,361,207,540,420,349,104,823,69,210,346,224,98,168,246,440,537,137,131],
    '2019': [201,487,224,166,1167,537,888,335,175,478,390,288,141,726,80,140,301,221,122,177,229,375,471,133,106],
    '2020': [192,404,243,159,1134,454,944,366,174,467,363,254,119,627,67,136,301,226,118,160,223,368,454,120,92],
    '2021': [230,563,250,180,2175,624,1088,392,196,658,461,388,115,789,89,240,416,285,133,199,268,463,693,173,119],
    '2022': [234,597,277,160,1708,703,1027,395,166,585,458,344,87,791,56,239,388,284,104,195,263,435,652,156,81],
    '2023': [237,521,246,178,1174,541,907,18,138,548,424,293,52,760,50,193,315,291,82,184,283,396,598,133,65],
    '2024': [213,477,219,111,870,505,820,296,67,468,366,254,45,600,39,161,281,257,61,179,236,349,466,137,60],
    'Avg_cases': [196.4,471.7,233.8,132.6,1297.6,586.5,960.1,305.7,132.8,526.4,381.2,313.0,
                  82.9,683.6,54.1,181.9,324.1,229.3,89.5,176.3,233.2,397.6,524.1,127.9,94.3],
}

df = pd.DataFrame(data)
df.columns = [str(c) for c in df.columns]
print(f"  Loaded {df.shape[0]} districts × {df.shape[1]} columns")


# ── STEP 2 — Fit scalers ─────────────────────────────────────────────────────
print("\nSTEP 2 — Fitting scalers...")

_input_r1 = df[['Avg_cases']]
_input_r2 = (df['2022'] * 0.30 + df['2023'] * 0.35 + df['2024'] * 0.35).values.reshape(-1, 1)
_input_r3 = ((df['2024'] - df['2010']) / df['2010'].replace(0, 1) * 100).values.reshape(-1, 1)
_input_r4 = (
    (df[['2020','2021','2022','2023','2024']].max(axis=1) - df['Avg_cases'])
    / df['Avg_cases'].replace(0, 1) * 100
).clip(lower=0).values.reshape(-1, 1)
_input_r5 = (
    (df['2021'] - df['2024']) / df['2021'].replace(0, 1) * 100
).clip(lower=0).values.reshape(-1, 1)

scaler = {
    'R1_avg_burden':    MinMaxScaler((0, 40)).fit(_input_r1),
    'R2_recent_trend':  MinMaxScaler((0, 25)).fit(_input_r2),
    'R3_growth_rate':   MinMaxScaler((0, 15)).fit(_input_r3),
    'R4_surge_penalty': MinMaxScaler((0, 10)).fit(_input_r4),
    'R5_recovery_gap':  MinMaxScaler((0, 10)).fit(_input_r5),
}
print(f"  Scalers fit: {list(scaler.keys())}")

feature_cols = {
    'R1_avg_burden':    ['Avg_cases'],
    'R2_recent_trend':  ['2022', '2023', '2024'],
    'R3_growth_rate':   ['2010', '2024'],
    'R4_surge_penalty': ['2020', '2021', '2022', '2023', '2024', 'Avg_cases'],
    'R5_recovery_gap':  ['2021', '2024'],
}

ALLOCATION_RULES = {
    'tier_thresholds': {
        'CRITICAL': {'score_min': 65, 'score_max': 100, 'tier_weight': 4.0},
        'HIGH':     {'score_min': 45, 'score_max':  64, 'tier_weight': 2.5},
        'MODERATE': {'score_min': 25, 'score_max':  44, 'tier_weight': 1.5},
        'LOW':      {'score_min':  0, 'score_max':  24, 'tier_weight': 1.0},
    },
    'allocation_formula':  'Allocated_LKR = (Risk_Score x Tier_Weight / sum) x Budget',
    'recent_year_weights': {'2022': 0.30, '2023': 0.35, '2024': 0.35},
    'min_floor_pct':       1.0,
    'score_range':         (0, 100),
    'n_districts':         25,
}


# ── STEP 3 — Compute risk scores ─────────────────────────────────────────────
print("\nSTEP 3 — Computing risk scores...")

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame()
    scores['District'] = df['District']

    scores['R1_avg_burden'] = scaler['R1_avg_burden'].transform(df[['Avg_cases']]).round(2)

    r2_input = (df['2022'] * 0.30 + df['2023'] * 0.35 + df['2024'] * 0.35).values.reshape(-1, 1)
    scores['R2_recent_trend'] = scaler['R2_recent_trend'].transform(r2_input).round(2)

    r3_input = ((df['2024'] - df['2010']) / df['2010'].replace(0, 1) * 100).values.reshape(-1, 1)
    scores['R3_growth_rate'] = scaler['R3_growth_rate'].transform(r3_input).round(2)

    peak    = df[['2020','2021','2022','2023','2024']].max(axis=1)
    r4_input = ((peak - df['Avg_cases']) / df['Avg_cases'].replace(0, 1) * 100).clip(lower=0).values.reshape(-1, 1)
    scores['R4_surge_penalty'] = scaler['R4_surge_penalty'].transform(r4_input).round(2)

    drop_pct = ((df['2021'] - df['2024']) / df['2021'].replace(0, 1) * 100).clip(lower=0).values.reshape(-1, 1)
    inverted = 10 - scaler['R5_recovery_gap'].transform(drop_pct).flatten()
    scores['R5_recovery_gap'] = inverted.round(2)

    scores['Risk_Score'] = (
        scores['R1_avg_burden'] + scores['R2_recent_trend'] +
        scores['R3_growth_rate'] + scores['R4_surge_penalty'] +
        scores['R5_recovery_gap']
    ).round(2)

    def classify(s):
        if s >= 65:   return 'CRITICAL'
        elif s >= 45: return 'HIGH'
        elif s >= 25: return 'MODERATE'
        else:         return 'LOW'

    scores['Risk_Tier']   = scores['Risk_Score'].apply(classify)
    scores['Tier_Weight'] = scores['Risk_Tier'].map(
        {t: cfg['tier_weight'] for t, cfg in ALLOCATION_RULES['tier_thresholds'].items()}
    )
    return scores

risk_df = compute_risk_scores(df)
print(f"  Tier distribution:\n{risk_df['Risk_Tier'].value_counts().to_string()}")


# ── STEP 4 — Load encoder & encode corpus ────────────────────────────────────
print("\nSTEP 4 — Loading SentenceTransformer (all-MiniLM-L6-v2)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("  Encoder loaded")

DISTRICT_ALIASES = {
    'colombo': 'Colombo', 'gampaha': 'Gampaha', 'kalutara': 'Kalutara',
    'kandy': 'Kandy', 'matale': 'Matale', 'nuwara eliya': 'Nuwara Eliya',
    'galle': 'Galle', 'matara': 'Matara', 'hambantota': 'Hambantota',
    'jaffna': 'Jaffna', 'kilinochchi': 'Kilinochchi', 'mannar': 'Mannar',
    'vavuniya': 'Vavuniya', 'mullaitivu': 'Mullaitivu', 'batticaloa': 'Batticaloa',
    'ampara': 'Ampara', 'trincomalee': 'Trincomalee', 'kurunegala': 'Kurunegala',
    'puttalam': 'Puttalam', 'anuradhapura': 'Anuradhapura',
    'polonnaruwa': 'Polonnaruwa', 'badulla': 'Badulla',
    'monaragala': 'Monaragala', 'rathnapura': 'Rathnapura', 'kegalle': 'Kegalle',
}

INTENT_CORPUS = {
    'allocate_budget': [
        'allocate budget of rupees for districts',
        'distribute funds among districts based on risk',
        'divide money for child welfare programs',
        'split resources across provinces',
        'assign budget to districts',
        'how should I distribute lkr among regions',
        'fund allocation for child protection',
        'resource distribution plan for districts',
        'give money to all 25 districts',
        'budget breakdown for child welfare',
        'allocate rupees based on risk scores',
        'spend money on high risk areas',
    ],
    'query_risk': [
        'which districts are high risk',
        'show critical zones',
        'what is the risk level of a district',
        'risk assessment for districts',
        'most vulnerable districts',
        'danger zones in sri lanka',
        'risk tier classification',
        'show all risk scores',
        'which areas need the most attention',
        'list districts by risk',
    ],
    'compare_districts': [
        'compare two districts',
        'difference between districts',
        'ranking of districts by cases',
        'which district has the most cases',
        'compare risk scores between regions',
        'side by side comparison of districts',
    ],
}

corpus_sentences, corpus_labels = [], []
for intent, sentences in INTENT_CORPUS.items():
    for s in sentences:
        corpus_sentences.append(s)
        corpus_labels.append(intent)

print("  Encoding intent corpus...")
corpus_embeddings = encoder.encode(corpus_sentences, convert_to_numpy=True)
print(f"  Corpus encoded — {len(corpus_sentences)} sentences, shape {corpus_embeddings.shape}")


# ── STEP 5 — Build ChildResourceAllocationModel ──────────────────────────────
print("\nSTEP 5 — Building ChildResourceAllocationModel...")

# Import from your project so the class path in the pkl matches your codebase
sys.path.insert(0, PROJECT_ROOT)
from models.child_protection_allocation_model import ChildResourceAllocationModel

pipeline_model = ChildResourceAllocationModel(
    scaler=scaler,
    feature_cols=feature_cols,
    rules=ALLOCATION_RULES,
    encoder=None,         # encoder excluded from pkl to save memory
)
pipeline_model.attach_nlp(
    corpus_embeddings=corpus_embeddings,
    corpus_labels=corpus_labels,
    corpus_sentences=corpus_sentences,
    district_aliases=DISTRICT_ALIASES,
)
pipeline_model.attach_data(risk_df=risk_df, case_df=df)
pipeline_model.meta = {
    'model_name':  'all-MiniLM-L6-v2',
    'version':     '1.0',
    'year_range':  ('2010', '2024'),
    'n_districts': 25,
}
print("  Model assembled")


# ── STEP 6 — Dump pkl ────────────────────────────────────────────────────────
print(f"\nSTEP 6 — Dumping pkl to: {PKL_PATH}")
with open(PKL_PATH, 'wb') as f:
    pickle.dump(pipeline_model, f, protocol=pickle.HIGHEST_PROTOCOL)

size_mb = os.path.getsize(PKL_PATH) / (1024 * 1024)
print(f"  Saved ({size_mb:.1f} MB)")


# ── STEP 7 — Verify reload ───────────────────────────────────────────────────
print("\nSTEP 7 — Verifying reload...")
with open(PKL_PATH, 'rb') as f:
    loaded = pickle.load(f)

# Scaler sanity: R1 on Colombo avg (1297.6) → should be 40.0
r1_check = loaded.scaler['R1_avg_burden'].transform([[1297.6]])[0][0]
assert abs(r1_check - 40.0) < 0.01, f"R1 scaler check failed: {r1_check}"
print(f"  R1 scaler check: {r1_check:.4f} / 40 ✅")

# Tier distribution
print(f"  Risk tier distribution:\n{loaded.risk_df['Risk_Tier'].value_counts().to_string()}")

# NLP check
inf_enc = SentenceTransformer(loaded.meta['model_name'])
q_emb   = inf_enc.encode(['Allocate LKR 500 million for child welfare'], convert_to_numpy=True)
sims    = cosine_similarity(q_emb, loaded.corpus_embeddings).flatten()
intent  = loaded.corpus_labels[int(np.argmax(sims))]
print(f"  NLP intent check: '{intent}' (conf={sims.max():.3f}) ✅")

print("\n✅ child_welfare_pipeline.pkl rebuilt successfully with NumPy", np.__version__)