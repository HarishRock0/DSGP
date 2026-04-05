import os
import re
import pickle
import pandas as pd
import numpy as np
import warnings
from difflib import get_close_matches
from dotenv import load_dotenv
import sys

warnings.filterwarnings('ignore')

# SkillDev class stub for pickle unpickling
# Must be defined here so pickle can reconstruct the model object
class SkillDev:
    """Minimal stub to support unpickling SkillDev instances."""
    pass


# Custom pickle unpickler to handle pandas compatibility issues
class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles pandas StringDtype and SkillDev compatibility."""

    @staticmethod
    def _make_string_dtype_proxy():
        """Return a StringDtype subclass that silently accepts any __init__ args."""
        try:
            from pandas import StringDtype as _SD
            class _StringDtypeCompat(_SD):
                def __init__(self, *args, **kwargs):
                    # Accept legacy positional args from old pickles; discard extras
                    try:
                        super().__init__()
                    except TypeError:
                        pass
            return _StringDtypeCompat
        except Exception:
            return None

    def find_class(self, module, name):
        # Fix StringDtype incompatibility (handles both old and new pandas locations)
        if name == 'StringDtype' and 'pandas' in module:
            proxy = self._make_string_dtype_proxy()
            if proxy is not None:
                return proxy
        # Handle SkillDev class (from any module)
        if name == 'SkillDev':
            return SkillDev
        return super().find_class(module, name)


def load_model_safely(model_path):
    """Load pickled model with compatibility fixes for pandas versions."""
    sys.modules['__main__'].SkillDev = SkillDev  # register for pickle

    with open(model_path, 'rb') as f:
        try:
            return CompatibleUnpickler(f).load()
        except Exception as e1:
            print(f"  Custom unpickler failed: {e1}")
            f.seek(0)
            try:
                return pickle.load(f)
            except Exception as e2:
                print(f"  Standard pickle also failed: {e2}")
                raise RuntimeError(
                    f"Cannot load model from {model_path}.\n"
                    f"Errors: {e1} | {e2}\n"
                    f"Run: python train_model.py  to regenerate the model."
                ) from e2


# Load environment variables from .env file
# Explicitly specify path to handle different working directories
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Groq — direct LLM only (PandasQueryEngine removed — it forces code generation)
try:
    from llama_index.llms.groq import Groq
    _LLAMA_INDEX_IMPORT_ERROR = None
    print(f" Groq imported from: {__file__}")
except ImportError as _groq_err:
    Groq = None
    _LLAMA_INDEX_IMPORT_ERROR = _groq_err
    print(f" Groq import failed: {_groq_err}")

# Settings import (kept for compatibility, no longer used for PandasQueryEngine)
try:
    from llama_index.core import Settings
except Exception:
    Settings = None


from constants import (
    COLUMN_DESCRIPTIONS,
    SECTOR_MAP,
    DISTRICT_MAP,
    EMPLOYMENT_STATUS,
    ETHNICITY_MAP,
    RELIGION_MAP,
    MARITAL_MAP,
)

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))   # NLP/Engines/
_PROJECT_DIR = os.path.dirname(os.path.dirname(_ENGINE_DIR))  # project_root/


class LLMQueryEngine:

    # Education level labels used in scoring & display
    EDU_MAP = {
        0: 'No schooling', 1: 'Grade 1', 2: 'Grade 2', 3: 'Grade 3',
        4: 'Grade 4', 5: 'Grade 5', 6: 'Grade 6', 7: 'Grade 7',
        8: 'Grade 8', 9: 'Grade 9', 10: 'Grade 10', 11: 'O/L',
        12: 'Passed O/L', 13: 'A/L', 14: 'Passed A/L', 15: 'Degree',
        16: 'Postgraduate', 19: 'No schooling',
    }

    def __init__(self, model_path=None, df=None,
                 csv_path=os.path.join(_PROJECT_DIR, "lfs_clustered_data.csv")):
        print(" Initializing LLM Query Engine …")

        # ---- Load data ----
        if df is not None:
            self.df = df.copy()
            self.has_clusters = 'cluster_id' in df.columns
        elif model_path and os.path.exists(model_path):
            print(f" Loading model from {model_path}")
            model = load_model_safely(model_path)
            self.df = model.df.copy() if hasattr(model, 'df') and model.df is not None else None
            self.has_clusters = self.df is not None and 'cluster_id' in self.df.columns

            # Derive cluster_label from cluster_id using model.cluster_mapping
            if self.has_clusters and hasattr(model, 'cluster_mapping'):
                self.df['cluster_label'] = self.df['cluster_id'].map(model.cluster_mapping)

            # Use pre-computed centroid distances from CSV if available (preferred)
            if self.df is not None and 'distance_to_center' in self.df.columns:
                print(" centroid distances loaded from CSV (pre-computed)")
            elif (self.has_clusters
                    and hasattr(model, 'kmeans')
                    and hasattr(model, 'weighted_data')
                    and model.weighted_data is not None):
                # Fallback: compute at runtime if not in CSV
                try:
                    centroids = model.kmeans.cluster_centers_
                    ids = self.df['cluster_id'].values
                    if model.weighted_data.shape[0] == len(self.df):
                        distances = np.linalg.norm(
                            model.weighted_data - centroids[ids], axis=1
                        )
                        self.df['distance_to_center'] = distances
                        print(" centroid distances computed at runtime (consider pre-computing in pipeline)")
                    else:
                        print("  weighted_data row count mismatch — skipping centroid distance computation")
                except Exception as e:
                    print(f"️  Could not compute centroid distances: {e}")
            else:
                print(" distance_to_center not available — nearest-centroid selection disabled")
        elif os.path.exists(csv_path):
            print(f" Loading data from {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.has_clusters = 'cluster_id' in self.df.columns
            print(f" Loaded {len(self.df)} records × {len(self.df.columns)} columns")
        else:
            self.df = None
            self.has_clusters = False
            print(" No data loaded")

        # ---- Pre-compute group-median income lookup (for allocation scoring) ----
        # Built from the ~14% of records that DO have income data.
        # Used in _compute_need_score() to give better estimates for employed people
        # with missing income, before falling back to employment-status proxy.
        self._income_median_lookup: dict = {}
        if self.df is not None and 'Q45_A_1' in self.df.columns:
            try:
                df_tmp = self.df.copy()
                df_tmp['_inc'] = pd.to_numeric(df_tmp['Q45_A_1'], errors='coerce')
                has_inc = df_tmp['_inc'].notna()
                if has_inc.sum() > 50:
                    grp_cols = [c for c in ['SECTOR', 'Q16', 'EDU'] if c in df_tmp.columns]
                    lookup_series = (
                        df_tmp[has_inc]
                        .groupby(grp_cols, dropna=True)['_inc']
                        .median()
                    )
                    for keys, val in lookup_series.items():
                        norm_key = tuple(
                            int(k) for k in (keys if isinstance(keys, tuple) else (keys,))
                        )
                        self._income_median_lookup[norm_key] = float(val)
                    print(f" Income group-median lookup: {len(self._income_median_lookup)} groups")
            except Exception as _e:
                print(f" Income lookup build failed (non-critical): {_e}")

        # ---- LLM setup (direct only — PandasQueryEngine removed) ----
        self.llm = None
        self.query_engine = None   # kept as None; referenced in legacy code paths
        try:
            if Groq is None:
                raise ImportError(
                    "Missing llama-index Groq LLM package. "
                    "Install 'llama-index-llms-groq'."
                ) from _LLAMA_INDEX_IMPORT_ERROR

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("Missing GROQ_API_KEY environment variable.")

            self.llm = Groq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                timeout=120.0,
            )
            if Settings is not None:
                Settings.llm = self.llm

            print(" Direct LLM ready (Groq — all query types)")

        except Exception as e:
            print(f" LLM not available: {e}")
            print("Set GROQ_API_KEY env var.  Get key → https://console.groq.com/keys")

        self.last_question = None
        self.last_answer = None

    # ==================================================================
    #  QUERY-TYPE DETECTION
    # ==================================================================

    def _detect_analysis_type(self, question_lower: str):
        """Return (analysis_type, params) tuple."""

        # ---- Resource allocation ----
        alloc_match = re.search(
            r'(?:give|distribut|allocat|provide|deliver|hand\s*out|send|assign|target)\s+'
            r'(\d+)\s+(\w+)',
            question_lower,
        )
        if alloc_match:
            return "resource_allocation", {
                "num_items": int(alloc_match.group(1)),
                "item_type": alloc_match.group(2),
            }

        # Phrases that imply allocation even without a verb+number
        if any(kw in question_lower for kw in [
            'who should', 'identify beneficiar', 'target population',
            'most in need', 'most vulnerable', 'priority list',
            'neediest', 'poorest', 'allocate resource',
        ]):
            num_match = re.search(r'(\d+)', question_lower)
            return "resource_allocation", {
                "num_items": int(num_match.group(1)) if num_match else 10,
                "item_type": "items",
            }

        return "general", {}

    # ==================================================================
    #  NEED-SCORE COMPUTATION  (core of resource allocation)
    # ==================================================================

    def _compute_need_score(self, row, item_type: str, question_lower: str) -> float:
        score = 0.0
        weights = {
            'income': 0.40,
            'education': 0.15,
            'disability': 0.15,
            'informality': 0.10,
            'sector': 0.10,
            'item_specific': 0.10,
        }

        # 1. Income component (0-100, lower income → higher score)
        income_raw = row.get('Q45_A_1')
        try:
            income = float(income_raw) if pd.notna(income_raw) else None
        except (TypeError, ValueError):
            income = None

        if income is not None:
            # Known income: scale 0 LKR → 100 pts, 100k+ LKR → 0 pts
            score += weights['income'] * max(0.0, 100.0 - (income / 1000.0))
        else:
            # No income on record — try group-median lookup first
            group_score = None
            try:
                sect = row.get('SECTOR')
                q16_v = row.get('Q16')
                edu_v = row.get('EDU')
                if pd.notna(sect) and pd.notna(q16_v) and pd.notna(edu_v):
                    key = (int(float(sect)), int(float(q16_v)), int(float(edu_v)))
                    grp_median = getattr(self, '_income_median_lookup', {}).get(key)
                    if grp_median is not None and pd.notna(grp_median):
                        group_score = max(0.0, 100.0 - (grp_median / 1000.0))
            except Exception:
                pass

            if group_score is not None:
                score += weights['income'] * group_score
            else:
                # Last resort: infer vulnerability from employment status
                q2  = row.get('Q2')
                q16 = row.get('Q16')
                q47 = row.get('Q47')
                if pd.notna(q2) and float(q2) == 2:
                    score += weights['income'] * 90   # Not working → highest need
                elif pd.notna(q16) and float(q16) == 4:
                    score += weights['income'] * 85   # Unpaid family worker
                elif pd.notna(q16) and float(q16) == 3:
                    score += weights['income'] * 75   # Own-account / self-employed
                elif pd.notna(q47) and float(q47) == 2:
                    score += weights['income'] * 70   # Informal employee
                else:
                    score += weights['income'] * 40   # Formal employed, income unreported

        # 2. Education (0-100, lower edu → higher score)
        edu = row.get('EDU')
        if pd.notna(edu):
            edu = float(edu)
            if edu <= 5:
                score += weights['education'] * 100
            elif edu <= 10:
                score += weights['education'] * 70
            elif edu <= 12:
                score += weights['education'] * 40
            elif edu <= 14:
                score += weights['education'] * 20
            else:
                score += weights['education'] * 5
        else:
            score += weights['education'] * 50

        # 3. Disability burden (sum of P15-P20; each 1-4)
        disability_cols = ['P15', 'P16', 'P17', 'P18', 'P19', 'P20']
        d_total = 0
        d_count = 0
        for dc in disability_cols:
            v = row.get(dc)
            if pd.notna(v):
                d_total += (float(v) - 1)  # 0 = no difficulty
                d_count += 1
        if d_count > 0:
            # max possible = 3*6 = 18
            score += weights['disability'] * min(100.0, (d_total / 18.0) * 100.0)

        # 4. Informality
        q47 = row.get('Q47')
        if pd.notna(q47) and float(q47) == 2:
            score += weights['informality'] * 100
        q46_val = row.get('Q46')
        if pd.notna(q46_val) and float(q46_val) == 2:
            score += weights['informality'] * 50  # no EPF/ETF

        # 5. Sector
        sector = row.get('SECTOR')
        if pd.notna(sector):
            sector_scores = {3: 100, 2: 60, 1: 20}
            score += weights['sector'] * sector_scores.get(int(sector), 40)

        # 6. Item-specific bonus
        is_tech = any(kw in question_lower for kw in ['laptop', 'computer', 'tablet', 'phone', 'device'])
        is_transport = any(kw in question_lower for kw in ['taxi', 'vehicle', 'transport', 'bus', 'wheel'])
        is_food = any(kw in question_lower for kw in ['food', 'meal', 'ration', 'nutrition'])

        if is_tech:
            q60a = row.get('Q60A')
            # Prefer people who CAN use computers but can't afford them
            if pd.notna(q60a) and float(q60a) == 1:
                score += weights['item_specific'] * 80
            age = row.get('AGE')
            if pd.notna(age) and 16 <= float(age) <= 45:
                score += weights['item_specific'] * 20
        elif is_transport:
            p17 = row.get('P17')
            if pd.notna(p17) and float(p17) >= 2:
                score += weights['item_specific'] * 100  # mobility difficulty
            age = row.get('AGE')
            if pd.notna(age) and 18 <= float(age) <= 65:
                score += weights['item_specific'] * 20
        elif is_food:
            age = row.get('AGE')
            if pd.notna(age):
                if float(age) < 18 or float(age) > 60:
                    score += weights['item_specific'] * 80  # children / elderly
            rship = row.get('RSHIP')
            if pd.notna(rship) and float(rship) == 1:
                score += weights['item_specific'] * 40  # head of household
        else:
            # Generic: prefer working-age unemployed
            q2 = row.get('Q2')
            if pd.notna(q2) and float(q2) == 2:
                score += weights['item_specific'] * 60  # not working

        return round(score, 2)

    # ==================================================================
    #  CLUSTER PROFILE DECODER  (for enriching LLM cluster selection)
    # ==================================================================

    def _build_cluster_profiles(self) -> str:
        if not self.has_clusters or self.df is None:
            return "No cluster information available."

        lines = []
        cluster_ids = sorted(self.df['cluster_id'].dropna().unique().astype(int))

        for cid in cluster_ids:
            cluster_df = self.df[self.df['cluster_id'] == cid]
            n = len(cluster_df)

            # Label
            label = "Unknown"
            if 'cluster_label' in cluster_df.columns:
                label = cluster_df['cluster_label'].mode().iloc[0] if len(cluster_df) > 0 else "Unknown"

            # Demographics
            avg_age = round(cluster_df['AGE'].mean(), 1) if 'AGE' in cluster_df.columns else 'N/A'

            pct_male = (
                round((cluster_df['SEX'] == 1).sum() / n * 100, 1)
                if 'SEX' in cluster_df.columns else 'N/A'
            )

            sector_mode = (
                SECTOR_MAP.get(int(cluster_df['SECTOR'].mode().iloc[0]), 'N/A')
                if 'SECTOR' in cluster_df.columns and len(cluster_df) > 0 else 'N/A'
            )

            emp_mode = (
                EMPLOYMENT_STATUS.get(int(cluster_df['Q16'].mode().iloc[0]), 'N/A')
                if 'Q16' in cluster_df.columns and cluster_df['Q16'].notna().any() else 'N/A'
            )

            avg_income = 'N/A'
            if 'Q45_A_1' in cluster_df.columns:
                inc = pd.to_numeric(cluster_df['Q45_A_1'], errors='coerce').dropna()
                if len(inc) > 0:
                    avg_income = f"LKR {round(inc.mean()):,}"

            pct_no_computer = (
                round((cluster_df['Q60A'] == 2).sum() / n * 100, 1)
                if 'Q60A' in cluster_df.columns else 'N/A'
            )

            lines.append(
                f"  Cluster {cid} — \"{label}\" ({n:,} people)\n"
                f"    avg_age={avg_age}, male={pct_male}%, sector={sector_mode}\n"
                f"    employment={emp_mode}, avg_income={avg_income}\n"
                f"    no_computer_access={pct_no_computer}%"
            )

        return "\n".join(lines)

    # ==================================================================
    #  RESOURCE ALLOCATION  — fully pre-computed, direct LLM call
    # ==================================================================

    def _handle_resource_allocation(self, question: str, num_items: int, item_type: str) -> str:
        df = self.df
        question_lower = question.lower()

        pool = df.copy()

        if len(pool) == 0:
            return " No data available for need-based resource allocation."

        print(f" Scoring all {len(pool):,} records for need-based allocation …")

        # ---- Score & rank all records ----
        pool['_need_score'] = pool.apply(
            lambda r: self._compute_need_score(r, item_type, question_lower), axis=1
        )
        pool = pool.sort_values('_need_score', ascending=False)

        n = min(num_items, len(pool))
        beneficiaries = pool.head(n).reset_index(drop=False)

        # ---- Outlier Guardrail — Confidence Score ----
        if 'distance_to_center' in beneficiaries.columns:
            avg_dist = round(float(beneficiaries['distance_to_center'].mean()), 2)
        else:
            avg_dist = None

        if avg_dist is None:
            trust_level = 'Unknown'
            trust_guidance = ('The distance_to_center column is unavailable, so '
                              'representativeness cannot be mathematically assessed. '
                              'Apply standard manual verification.')
        elif avg_dist < 2.0:
            trust_level = 'High'
            trust_guidance = ('These individuals are PERFECT ARCHETYPES for this '
                              'intervention — they sit extremely close to the cluster '
                              'centroid and are highly representative of the lifestyle '
                              'and needs profile. You may allocate resources with high '
                              'confidence.')
        elif avg_dist < 3.5:
            trust_level = 'Moderate'
            trust_guidance = ('These are GOOD MATCHES but show some individual variance '
                              'from the cluster centre. Most will benefit strongly from '
                              'the intervention; a small minority may have edge-case '
                              'circumstances worth reviewing.')
        else:
            trust_level = 'Low'
            trust_guidance = ('These individuals are PERIPHERAL OUTLIERS who only '
                              'partially match the cluster profile. Be transparent that '
                              'they fall at the boundary of the target group and may '
                              'require manual case-by-case verification before resources '
                              'are committed.')

        print(f' Outlier Guardrail — avg distance: {avg_dist}, trust: {trust_level}')

        # ---- Build translated beneficiary table ----
        rows = []
        for i, (_, row) in enumerate(beneficiaries.iterrows(), 1):
            r = {'No': i, 'Index': int(row['index'])}
            try:
                r['Income (LKR)'] = f"Rs. {float(row['Q45_A_1']):,.0f}" if pd.notna(row['Q45_A_1']) else 'N/A'
            except (TypeError, ValueError):
                r['Income (LKR)'] = 'N/A'
            r['Need Score'] = round(row['_need_score'], 1)
            _cl = row.get('cluster_label')
            if _cl is not None and pd.notna(_cl):
                r['Group'] = str(_cl)
            if pd.notna(row.get('AGE')):
                r['Age'] = int(row['AGE'])
            if pd.notna(row.get('SEX')):
                r['Sex'] = 'Male' if int(row['SEX']) == 1 else 'Female'
            if pd.notna(row.get('EDU')):
                r['Education'] = self.EDU_MAP.get(int(row['EDU']), f"Code {int(row['EDU'])}")
            if pd.notna(row.get('SECTOR')):
                r['Sector'] = SECTOR_MAP.get(int(row['SECTOR']), str(int(row['SECTOR'])))
            if pd.notna(row.get('DISTRICT')):
                r['District'] = DISTRICT_MAP.get(int(row['DISTRICT']), str(int(row['DISTRICT'])))
            if pd.notna(row.get('Q16')):
                r['Employment'] = EMPLOYMENT_STATUS.get(int(row['Q16']), str(int(row['Q16'])))
            if pd.notna(row.get('Q47')):
                r['Formality'] = 'Formal' if int(row['Q47']) == 1 else 'Informal'
            disability_labels = []
            for dc, dl in [('P15','Vision'), ('P16','Hearing'), ('P17','Mobility'),
                           ('P18','Cognition'), ('P19','Self-care'), ('P20','Communication')]:
                dv = row.get(dc)
                if pd.notna(dv) and int(dv) >= 2:
                    sev = {2: 'Some', 3: 'A lot', 4: 'Cannot do'}.get(int(dv), '')
                    disability_labels.append(f"{dl}:{sev}")
            if disability_labels:
                r['Disabilities'] = '; '.join(disability_labels)
            rows.append(r)

        table_df = pd.DataFrame(rows)
        if len(table_df) > 200:
            table_df = table_df.head(200)
        table_str = table_df.to_markdown(index=False)

        # ---- Summary statistics ----
        incomes = beneficiaries['Q45_A_1'].dropna()
        sector_dist = {}
        if 'SECTOR' in beneficiaries.columns:
            for code, label in SECTOR_MAP.items():
                cnt = int((beneficiaries['SECTOR'] == code).sum())
                if cnt > 0:
                    sector_dist[label] = cnt

        district_dist = {}
        if 'DISTRICT' in beneficiaries.columns:
            for code, cnt in beneficiaries['DISTRICT'].value_counts().head(10).items():
                district_dist[DISTRICT_MAP.get(int(code), str(int(code)))] = int(cnt)

        gender_dist = {}
        if 'SEX' in beneficiaries.columns:
            gender_dist['Male'] = int((beneficiaries['SEX'] == 1).sum())
            gender_dist['Female'] = int((beneficiaries['SEX'] == 2).sum())

        summary = (
            f"\n=== SUMMARY STATISTICS (PRE-COMPUTED — EXACT) ===\n"
            f"Total beneficiaries selected: {n}\n"
            f"Items to distribute: {num_items} {item_type}\n"
            f"Average income: Rs. {incomes.mean():,.0f}\n"
            f"Median income: Rs. {incomes.median():,.0f}\n"
            f"Income range: Rs. {incomes.min():,.0f} – Rs. {incomes.max():,.0f}\n"
            f"Average need score: {beneficiaries['_need_score'].mean():.1f} / 100\n"
            f"Gender: {gender_dist}\n"
            f"Sector distribution: {sector_dist}\n"
            f"Top districts: {district_dist}\n"
        )

        # ---- Overall dataset context ----
        overall_income = df['Q45_A_1'].dropna()
        dataset_context = (
            f"\n=== DATASET CONTEXT ===\n"
            f"Total records: {len(df)}\n"
            f"Records with income data: {len(overall_income)}\n"
            f"Overall income — Mean: Rs. {overall_income.mean():,.0f}, "
            f"Median: Rs. {overall_income.median():,.0f}\n"
            f"Sector: Urban={int((df['SECTOR']==1).sum())}, "
            f"Rural={int((df['SECTOR']==2).sum())}, "
            f"Estate={int((df['SECTOR']==3).sum())}\n"
        )

        # ---- Compact prompt for LLM — aggregates only, NO full table ----
        intro_prompt = f"""You are a resource-allocation analyst for Sri Lanka's Labour Force Survey (LFS-2023).

USER QUESTION: {question}

Python has ALREADY selected the top {n} beneficiaries using a multi-factor need scoring system
(income 40%, education 15%, disability 15%, informality 10%, sector deprivation 10%, item-specific 10%).

PRE-COMPUTED AGGREGATE STATISTICS FOR THE SELECTED {n} PEOPLE (100% accurate, use these only):
{summary}
{dataset_context}
TRUST LEVEL: {trust_level} — {trust_guidance}

Write ONLY a concise allocation report (5-8 sentences):
1. Who was selected and why (cite exact numbers from statistics above).
2. Key demographic patterns (sector split, gender, disability burden).
3. Two targeted policy recommendations for distributing {num_items} {item_type}.

DO NOT list individual beneficiaries or row indices.
DO NOT write any Python code. DO NOT use ``` blocks.
DO NOT fabricate numbers. Use ONLY the statistics above. Be concise."""

        # ---- Call LLM for SHORT intro paragraph only ----
        intro_text = ""
        if self.llm is not None:
            try:
                print(f" Requesting intro paragraph from LLM ({len(intro_prompt)} chars) …")
                response = self.llm.complete(intro_prompt)
                intro_text = str(response).strip()
            except Exception as e:
                print(f" LLM intro call failed: {e}")
                intro_text = (
                    f"Allocation of {num_items} {item_type}: "
                    f"Top {n} beneficiaries selected by need score (Python-computed, exact)."
                )
        else:
            intro_text = (
                f"Top {n} beneficiaries for {num_items} {item_type} — "
                f"selected by Python need-score engine (LLM unavailable)."
            )

        # ---- Stitch full answer locally ----
        final_answer = (
            f"=== RESOURCE ALLOCATION REPORT ===\n"
            f"Query: {question}\n\n"
            f"{intro_text}\n\n"
            f"=== BENEFICIARY LIST (ALL {n} — ranked by need score) ===\n\n"
            f"{table_str}\n\n"
            f"=== SUMMARY STATISTICS ===\n"
            f"{summary.strip()}\n\n"
            f"Data Confidence: {trust_level} — {trust_guidance}"
        )

        self.last_question = question
        self.last_answer = final_answer
        return final_answer

    # ==================================================================
    #  DIRECT ALLOCATION ENTRY POINT
    # ==================================================================

    @staticmethod
    def _derive_employment_label(row) -> str:
        """Derive a human-readable employment label from Q2 / Q16 / Q36."""
        q2 = row.get('Q2')
        q16 = row.get('Q16')
        q36 = row.get('Q36')

        if pd.notna(q2) and int(q2) == 1:
            if pd.notna(q16):
                return {
                    1: 'Employee',
                    2: 'Employer',
                    3: 'Self-employed',
                    4: 'Family Worker',
                }.get(int(q16), 'Employed')
            return 'Employed'

        if pd.notna(q36) and int(q36) == 1:
            return 'Unemployed'

        return 'Inactive'

    def handle_allocation(self, question: str, num_items: int = None, item_type: str = None) -> str:
        """
        Direct entry point for resource allocation.
        Skips re-detection; called by NLP routing layer.
        """
        if self.df is None:
            return " No data loaded."

        question_lower = question.lower()

        if num_items is None:
            num_match = re.search(r'\b(\d+)\b', question_lower)
            num_items = int(num_match.group(1)) if num_match else 10

        if item_type is None:
            alloc_match = re.search(
                r'(?:give|distribut|allocat|provide|deliver|hand\s*out|send|assign|target)'
                r'\s+\d+\s+(\w+)',
                question_lower,
            )
            item_type = alloc_match.group(1) if alloc_match else "items"

        print(f" Direct allocation route: {num_items}x {item_type}")
        return self._handle_resource_allocation(question, num_items, item_type)

    # ==================================================================
    #  PRE-COMPUTED STATISTICS  (injected into every general LLM prompt)
    # ==================================================================

    def _compute_statistics(self, question: str) -> str:
        """
        Pre-compute exact statistics from the DataFrame so the LLM never
        needs to guess, estimate, or generate code to fill data gaps.
        """
        df = self.df
        parts = []
        ql = question.lower()

        # Always-present baseline
        parts.append(f"DATASET: {len(df)} rows × {len(df.columns)} columns")
        parts.append(
            f"\nAge: min={df['AGE'].min()}, max={df['AGE'].max()}, "
            f"mean={df['AGE'].mean():.1f}, median={df['AGE'].median():.0f}"
        )
        parts.append(
            f"Gender: Male(1)={int((df['SEX']==1).sum())}, "
            f"Female(2)={int((df['SEX']==2).sum())}"
        )
        parts.append(
            f"Sector: Urban(1)={int((df['SECTOR']==1).sum())}, "
            f"Rural(2)={int((df['SECTOR']==2).sum())}, "
            f"Estate(3)={int((df['SECTOR']==3).sum())}"
        )

        # ── Income (overall) ──────────────────────────────────────────
        if any(kw in ql for kw in ['income', 'salary', 'poverty', 'poor', 'rich', 'wage',
                                    'earning', 'resource', 'allocat', 'beneficiar',
                                    'give', 'distribut', 'q45', 'average income',
                                    'district', 'region']):
            inc = pd.to_numeric(df['Q45_A_1'], errors='coerce').dropna()
            if len(inc):
                parts.append(
                    f"\nINCOME (Q45_A_1) — {len(inc)} records with data / {len(df)} total:"
                )
                parts.append(f"  Mean: Rs. {inc.mean():,.0f}  | Median: Rs. {inc.median():,.0f}")
                parts.append(f"  Min:  Rs. {inc.min():,.0f}  | Max:    Rs. {inc.max():,.0f}")
                parts.append(
                    f"  25th pct: Rs. {inc.quantile(.25):,.0f} | "
                    f"75th pct: Rs. {inc.quantile(.75):,.0f}"
                )
                parts.append(
                    f"  <20k: {int((inc<20000).sum())} | "
                    f"<10k: {int((inc<10000).sum())} | "
                    f"Zero: {int((inc==0).sum())}"
                )

        # ── Income by district (computed — no guessing needed) ────────
        if any(kw in ql for kw in ['district', 'region', 'average income', 'income by',
                                    'colombo', 'kandy', 'galle', 'jaffna', 'province']):
            df_inc = df.copy()
            df_inc['_inc'] = pd.to_numeric(df_inc['Q45_A_1'], errors='coerce')
            parts.append("\nINCOME BY DISTRICT (mean of Q45_A_1 where data exists):")
            for d in sorted(df['DISTRICT'].dropna().unique()):
                d = int(d)
                ddf = df_inc[df_inc['DISTRICT'] == d]
                inc_d = ddf['_inc'].dropna()
                name = DISTRICT_MAP.get(d, '?')
                total_in_district = len(ddf)
                if len(inc_d):
                    parts.append(
                        f"  {d} ({name}): n={total_in_district}, "
                        f"with_income={len(inc_d)}, "
                        f"mean=Rs. {inc_d.mean():,.0f}, "
                        f"median=Rs. {inc_d.median():,.0f}"
                    )
                else:
                    parts.append(
                        f"  {d} ({name}): n={total_in_district}, no income data recorded"
                    )

        # ── Employment ────────────────────────────────────────────────
        if any(kw in ql for kw in ['employ', 'work', 'job', 'occupation', 'formal',
                                    'informal', 'labor', 'labour', 'q16', 'q2', 'q47']):
            parts.append("\nEMPLOYMENT:")
            q2 = df['Q2'].dropna()
            parts.append(
                f"  Worked last 7d (Q2): "
                f"Yes={int((q2==1).sum())}, No={int((q2==2).sum())}"
            )
            for c, l in EMPLOYMENT_STATUS.items():
                cnt = int((df['Q16']==c).sum())
                if cnt:
                    parts.append(f"  Q16={c} ({l}): {cnt}")
            parts.append(
                f"  Formal(Q47=1): {int((df['Q47']==1).sum())}, "
                f"Informal(Q47=2): {int((df['Q47']==2).sum())}"
            )

        # ── Education ─────────────────────────────────────────────────
        if any(kw in ql for kw in ['education', 'school', 'degree', 'literacy',
                                    'edu', 'literat', 'computer', 'digital']):
            parts.append("\nEDUCATION:")
            edu = df['EDU'].dropna()
            for c in sorted(edu.unique()):
                parts.append(
                    f"  EDU={int(c)} "
                    f"({self.EDU_MAP.get(int(c), f'Grade {int(c)}')}): "
                    f"{int((edu==c).sum())}"
                )
            if 'Q60A' in df.columns:
                parts.append(
                    f"  Computer literate (Q60A=1): {int((df['Q60A']==1).sum())}"
                )
                parts.append(
                    f"  Not computer literate (Q60A=2): {int((df['Q60A']==2).sum())}"
                )

        # ── Disability ────────────────────────────────────────────────
        if any(kw in ql for kw in ['disab', 'difficult', 'vision', 'hear', 'walk',
                                    'mobil', 'self-care', 'communicat',
                                    'p15', 'p16', 'p17', 'p18', 'p19', 'p20']):
            parts.append("\nDISABILITY:")
            for col, lbl in [('P15','Vision'), ('P16','Hearing'), ('P17','Mobility'),
                              ('P18','Cognition'), ('P19','Self-care'), ('P20','Communication')]:
                if col in df.columns:
                    p = df[col].dropna()
                    parts.append(
                        f"  {col} ({lbl}): "
                        f"Some+={int((p>=2).sum())}, "
                        f"A lot+={int((p>=3).sum())}, "
                        f"Cannot={int((p==4).sum())}"
                    )

        # ── District counts ───────────────────────────────────────────
        if any(kw in ql for kw in ['district', 'region', 'province',
                                    'colombo', 'kandy', 'galle', 'jaffna']):
            parts.append("\nDISTRICT POPULATION COUNTS:")
            for c in sorted(df['DISTRICT'].dropna().unique()):
                parts.append(
                    f"  {int(c)} ({DISTRICT_MAP.get(int(c),'?')}): "
                    f"{int((df['DISTRICT']==c).sum())}"
                )

        # ── Cluster stats ─────────────────────────────────────────────
        if any(kw in ql for kw in ['cluster', 'segment', 'group', 'compare']):
            if self.has_clusters:
                parts.append("\nCLUSTER STATISTICS (exact, Python-computed):")
                for cid in sorted(df['cluster_id'].dropna().unique().astype(int)):
                    cdf = df[df['cluster_id'] == cid]
                    n_c = len(cdf)
                    mean_age = round(cdf['AGE'].mean(), 1) if 'AGE' in cdf.columns else 'N/A'
                    inc_c = pd.to_numeric(cdf['Q45_A_1'], errors='coerce').dropna()
                    mean_inc = f"Rs. {inc_c.mean():,.0f}" if len(inc_c) else 'N/A'
                    dom_sector = (
                        SECTOR_MAP.get(int(cdf['SECTOR'].mode().iloc[0]), 'N/A')
                        if 'SECTOR' in cdf.columns else 'N/A'
                    )
                    label = (
                        cdf['cluster_label'].mode().iloc[0]
                        if 'cluster_label' in cdf.columns else f"Cluster {cid}"
                    )
                    parts.append(
                        f"  Cluster {cid} ({label}): "
                        f"n={n_c}, mean_age={mean_age}, "
                        f"mean_income={mean_inc}, dominant_sector={dom_sector}"
                    )

        # ── Ethnicity ─────────────────────────────────────────────────
        if any(kw in ql for kw in ['ethnic', 'sinhala', 'tamil', 'moor', 'malay', 'burgher']):
            parts.append("\nETHNICITY:")
            for c, l in ETHNICITY_MAP.items():
                cnt = int((df['ETH']==c).sum())
                if cnt:
                    parts.append(f"  {c} ({l}): {cnt}")

        # ── Marital ───────────────────────────────────────────────────
        if any(kw in ql for kw in ['marital', 'married', 'widow', 'divorced', 'single']):
            parts.append("\nMARITAL:")
            for c, l in MARITAL_MAP.items():
                cnt = int((df['MARITAL']==c).sum())
                if cnt:
                    parts.append(f"  {c} ({l}): {cnt}")

        return "\n".join(parts)

    def _detect_relevant_columns(self, question_lower: str) -> list:
        """Return columns relevant to the question (used for context)."""
        kw_map = {
            'income':    ['Q45_A_1','Q47','Q2','Q16'],
            'salary':    ['Q45_A_1','Q16'],
            'poverty':   ['Q45_A_1','SECTOR','DISTRICT','EDU'],
            'employ':    ['Q2','Q16','Q20','Q47','Q8'],
            'education': ['EDU','CUEDU','DEGREE'],
            'disab':     ['P15','P16','P17','P18','P19','P20'],
            'difficult': ['P15','P16','P17','P18','P19','P20'],
            'vision':    ['P15'], 'hear':['P16'], 'walk':['P17'], 'mobil':['P17'],
            'computer':  ['Q60A','Q60B'],
            'digital':   ['Q60A','Q60B','Q61','Q64'],
            'internet':  ['Q61','Q64'],
            'gender':    ['SEX'], 'male':['SEX'], 'female':['SEX'],
            'ethnic':    ['ETH'], 'religion':['REL'],
            'district':  ['DISTRICT'], 'sector':['SECTOR'],
            'age':       ['AGE'],
            'literacy':  ['SIN','TAMIL','ENG'],
            'marital':   ['MARITAL'],
        }
        cols = set()
        for kw, cl in kw_map.items():
            if kw in question_lower:
                cols.update(cl)
        return list(cols) or ['AGE','SEX','SECTOR','DISTRICT','EDU','Q45_A_1','Q2','Q16']

    # ==================================================================
    #  INSTRUCTION STRING  (system prompt for direct LLM calls)
    # ==================================================================

    def _build_instruction_str(self) -> str:
        s = (
            "You are a data analyst for Sri Lanka's Labour Force Survey (LFS-2023).\n"
            "Answer questions in clear, human-readable English using ONLY the "
            "pre-computed statistics provided to you.\n\n"
            "DATASET: ~18,937 rows × 128 columns.\n\n"
            "COLUMN MEANINGS:\n"
        )
        for col, desc in COLUMN_DESCRIPTIONS.items():
            s += f"- {col}: {desc}\n"

        s += "\n=== VALUE MAPPINGS ===\n"
        s += "SECTOR: " + ", ".join(f"{k}={v}" for k, v in SECTOR_MAP.items()) + "\n"
        s += "DISTRICT: " + ", ".join(f"{k}={v}" for k, v in DISTRICT_MAP.items()) + "\n"
        s += "Q16: " + ", ".join(f"{k}={v}" for k, v in EMPLOYMENT_STATUS.items()) + "\n"
        s += "ETH: " + ", ".join(f"{k}={v}" for k, v in ETHNICITY_MAP.items()) + "\n"
        s += "REL: " + ", ".join(f"{k}={v}" for k, v in RELIGION_MAP.items()) + "\n"
        s += "MARITAL: " + ", ".join(f"{k}={v}" for k, v in MARITAL_MAP.items()) + "\n"
        s += "P15-P20: 1=None, 2=Some, 3=A lot, 4=Cannot do\n"

        s += """
RULES:
- NEVER write Python code. NEVER suggest code. NEVER use ``` blocks.
- Use ONLY the pre-computed statistics given to you — never invent or estimate numbers.
- If the statistics do not contain enough information to fully answer, say so clearly in plain English.
- Translate ALL numeric codes to human labels (e.g. District 12 → Gampaha, Sector 2 → Rural).
- Write in clear, concise paragraphs. Use a Markdown table only when comparing multiple categories.
- Never fabricate formulas or approximations to fill gaps in the data.
"""
        return s

    # ==================================================================
    #  MAIN ENTRY POINT
    # ==================================================================

    def analyze_data(self, question: str):
        """Route question to resource-allocation or general analysis."""
        print(f"\n Processing: {question}")

        if self.df is None:
            return " No data loaded."

        ql = question.lower()

        # Quick answers — no LLM needed
        if any(kw in ql for kw in ['columns', 'column names', 'fields', 'structure', 'schema']):
            info = [
                f"• {c}: {COLUMN_DESCRIPTIONS.get(c, 'Survey variable')}"
                for c in self.df.columns
            ]
            return f" {len(self.df.columns)} columns:\n" + "\n".join(info)

        if 'shape' in ql or ('how many' in ql and 'row' in ql):
            return f" {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"

        # Detect type
        analysis_type, params = self._detect_analysis_type(ql)

        # ---- Resource allocation ----
        if analysis_type == "resource_allocation":
            return self._handle_resource_allocation(
                question,
                params.get("num_items", 10),
                params.get("item_type", "items"),
            )

        # ---- General query — direct LLM only (PandasQueryEngine removed) ----
        if self.llm is None:
            # No LLM available — return raw statistics
            stats = self._compute_statistics(question)
            return f"Here are the relevant statistics:\n\n{stats}"

        stats = self._compute_statistics(question)
        system_prompt = self._build_instruction_str()

        enhanced = f"""{system_prompt}

USER QUESTION: {question}

=== PRE-COMPUTED STATISTICS (EXACT — calculated directly from the dataset) ===
{stats}

INSTRUCTIONS:
1. Use ONLY the pre-computed statistics above — they are exact.
2. Do NOT write or suggest any Python code whatsoever.
3. Translate ALL numeric codes to human-readable labels.
4. If the statistics above are sufficient to answer, answer directly in plain English.
5. If the statistics do not cover what was asked, say so honestly — do not invent numbers.
6. Where multiple categories are compared, present a clean Markdown table."""

        try:
            response = self.llm.complete(enhanced)
            answer = self._translate_codes_in_response(str(response))
            self.last_question = question
            self.last_answer = answer
            return answer
        except Exception as e:
            print(f" Direct LLM error: {e}")

        # Final fallback — return raw stats
        return f"Here are the relevant statistics:\n\n{stats}"

    # ==================================================================
    #  POST-PROCESSING
    # ==================================================================

    def _translate_codes_in_response(self, text: str) -> str:
        """Translate remaining numeric codes in LLM output."""
        text = re.sub(r'\bSector:\s*1\b', 'Sector: 1 (Urban)', text)
        text = re.sub(r'\bSector:\s*2\b', 'Sector: 2 (Rural)', text)
        text = re.sub(r'\bSector:\s*3\b', 'Sector: 3 (Estate)', text)
        for code, name in DISTRICT_MAP.items():
            text = re.sub(rf'\bDistrict:\s*{code}\b', f'District: {code} ({name})', text)
        text = re.sub(r'\bSex:\s*1\b', 'Sex: 1 (Male)', text)
        text = re.sub(r'\bSex:\s*2\b', 'Sex: 2 (Female)', text)
        return text

    # ==================================================================
    #  CLUSTER & INSIGHT HELPERS
    # ==================================================================

    def ask_about_clusters(self, question: str) -> str:
        if not self.has_clusters:
            return " No cluster information in the dataset."
        return self.analyze_data(f"Regarding the cluster_id column: {question}")

    def compare_clusters(self) -> str:
        """
        Returns a fully Python-computed natural language cluster comparison.
        No LLM involved — guarantees exact numbers and zero code generation.
        """
        if not self.has_clusters or self.df is None:
            return "No cluster information in the dataset."

        df = self.df
        sector_map = {1: 'Urban', 2: 'Rural', 3: 'Estate'}
        rows = []

        for cid in sorted(df['cluster_id'].dropna().unique().astype(int)):
            cdf = df[df['cluster_id'] == cid]
            n = len(cdf)
            mean_age = round(cdf['AGE'].mean(), 1) if 'AGE' in cdf.columns else None
            inc = pd.to_numeric(cdf['Q45_A_1'], errors='coerce').dropna()
            mean_inc = round(inc.mean(), 0) if len(inc) else None
            dom_sector = (
                int(cdf['SECTOR'].mode().iloc[0])
                if 'SECTOR' in cdf.columns and len(cdf) > 0 else None
            )
            sector_label = sector_map.get(dom_sector, 'Unknown')
            label = (
                cdf['cluster_label'].mode().iloc[0]
                if 'cluster_label' in cdf.columns else f"Cluster {cid}"
            )
            rows.append((cid, label, n, mean_age, mean_inc, sector_label))

        total = sum(r[2] for r in rows)
        lines = ["Here's a breakdown of the workforce clusters identified in the dataset:\n"]

        for cid, label, n, age, inc, sector in rows:
            pct = round(n / total * 100, 1)
            inc_str = f"Rs. {int(inc):,}" if inc is not None else "unknown"
            age_str = str(age) if age is not None else "unknown"
            lines.append(
                f"**Cluster {cid} — {label}**\n"
                f"This cluster has {n:,} people ({pct}% of the total), "
                f"mostly from {sector} areas. "
                f"The average age is {age_str} years and the average monthly "
                f"income is {inc_str}.\n"
            )

        lines.append(
            f"In total, {total:,} individuals are covered across {len(rows)} clusters."
        )
        return "\n".join(lines)

    def get_insights(self, topic: str = None) -> str:
        if topic:
            return self.analyze_data(f"Key insights about {topic} in the dataset.")
        return self.analyze_data(
            "5 key insights covering income, employment, education, disability, regions."
        )