import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DELIVERY_COMPLEXITY_MAP = {"Low": 0.9, "Medium": 0.55, "High": 0.2}  #Lookup dict to fill numerical values to categorical columns 
STRATEGIC_FIT_MAP = {"Low": 0.2, "Medium": 0.55, "High": 0.9}
BUDGET_SIGNAL_MAP = {"Confirmed": 1.0, "Likely": 0.66, "Unclear": 0.33, "None": 0.0}


def safe_div(a, b):
    return a / b if b not in (0, None) and not pd.isna(b) else 0.0 #safety mechanism to ensure does not / by b when it's o,None or NaN isntead fills 0.0

#Reading the datasets
def build_features():
    concepts = pd.read_csv(DATA_DIR / "product_concepts.csv")
    demo = pd.read_csv(DATA_DIR / "customer_demo_signals.csv")
    usage = pd.read_csv(DATA_DIR / "sandbox_usage.csv")
    comm = pd.read_csv(DATA_DIR / "commercial_signals.csv")
    text = pd.read_csv(DATA_DIR / "text_feedback.csv")


    numeric_demo_cols = ["feedback_score", "follow_up_requested",
                          "decision_maker_present", "objections_count"]
    numeric_usage_cols = ["trial_sessions", "feature_clicks", "repeat_usage_days",
                           "active_users", "time_spent_minutes", "abandoned_features"]
    numeric_comm_cols = ["pilot_interest", "urgency_score", "willingness_to_pay",
                          "expected_value", "implementation_risk"]

    #Replaceing the NaN and missing values by median for missing values in a column and global median for a column with all missing value
    def impute_per_concept(df, cols):
        df = df.copy()
        for col in cols:
            global_median = df[col].median()
            df[col] = df.groupby("concept_id")[col].transform(
                lambda s: s.fillna(s.median() if s.notna().any() else global_median)
            )
        return df

    demo = impute_per_concept(demo, numeric_demo_cols)
    usage = impute_per_concept(usage, numeric_usage_cols)
    comm = impute_per_concept(comm, numeric_comm_cols)
    comm["budget_signal"] = comm["budget_signal"].fillna("Unclear")
    comm["budget_numeric"] = comm["budget_signal"].map(BUDGET_SIGNAL_MAP)

    text = text.fillna({"customer_comments": "", "pain_point_statements": "",
                         "objection_themes": "", "requested_capabilities": ""}) #filling them with empty strings rather than 0 as they are text fields 

    rows = []
    for _, concept in concepts.iterrows():
        cid = concept["concept_id"]
        d = demo[demo["concept_id"] == cid]
        u = usage[usage["concept_id"] == cid]
        c = comm[comm["concept_id"] == cid]
        t = text[text["concept_id"] == cid]

        n_demos = len(d)
        n_usage = len(u)
        n_comm = len(c)
        n_text = len(t)

        """Choosing values that are more likely to affect the respective feature , purely based on judgement """
        
        # --- Demand intensity: how strongly customers react in demos + commercial signals ---
        avg_feedback = d["feedback_score"].mean() if n_demos else 2.5
        follow_up_rate = d["follow_up_requested"].mean() if n_demos else 0
        pilot_interest_rate = c["pilot_interest"].mean() if n_comm else 0
        avg_urgency = c["urgency_score"].mean() if n_comm else 0
        demand_intensity = (
            0.35 * (avg_feedback / 5)
            + 0.25 * follow_up_rate
            + 0.25 * pilot_interest_rate
            + 0.15 * (avg_urgency / 100)
        )

        # --- Repeatability: sandbox stickiness across multiple customers/sessions ---
        avg_repeat_days = u["repeat_usage_days"].mean() if n_usage else 0
        avg_sessions = u["trial_sessions"].mean() if n_usage else 0
        n_distinct_customers = u["customer_id"].nunique() if n_usage else 0
        repeatability = (
            0.4 * min(avg_repeat_days / 15, 1.0)
            + 0.3 * min(avg_sessions / 10, 1.0)
            + 0.3 * min(n_distinct_customers / 5, 1.0)
        )

        # --- Engagement depth: how deeply customers use the sandbox ---
        avg_clicks = u["feature_clicks"].mean() if n_usage else 0
        avg_time = u["time_spent_minutes"].mean() if n_usage else 0
        avg_abandoned = u["abandoned_features"].mean() if n_usage else 0
        engagement_depth = (
            0.4 * min(avg_clicks / 60, 1.0)
            + 0.4 * min(avg_time / 120, 1.0)
            + 0.2 * (1 - min(avg_abandoned / 5, 1.0))
        )

        # --- Segment similarity: how concentrated demand is across segments
        if n_demos and d["segment"].nunique() > 0:
            seg_counts = d["segment"].value_counts(normalize=True)
            segment_similarity = seg_counts.max()  # concentration in dominant segment
            segment_diversity = d["segment"].nunique() / len(SEGMENTS_ALL)
        else:
            segment_similarity = 0.0
            segment_diversity = 0.0

        # --- Revenue potential ---
        avg_wtp = c["willingness_to_pay"].mean() if n_comm else 0
        avg_expected_value = c["expected_value"].mean() if n_comm else 0
        avg_budget_signal = c["budget_numeric"].mean() if n_comm else 0
        revenue_potential = (
            0.35 * min(avg_wtp / 80000, 1.0)
            + 0.35 * min(avg_expected_value / 400000, 1.0)
            + 0.30 * avg_budget_signal
        )

        # --- Feasibility (delivery complexity, inverted; + implementation risk) ---
        complexity_score = DELIVERY_COMPLEXITY_MAP.get(concept["delivery_complexity"], 0.5)
        avg_impl_risk = c["implementation_risk"].mean() if n_comm else 50
        feasibility = 0.5 * complexity_score + 0.5 * (1 - avg_impl_risk / 100)

        # --- Strategic fit (from stated rating) ---
        strategic_fit = STRATEGIC_FIT_MAP.get(concept["strategic_fit"], 0.5)

        # --- Objection load (negative signal, feeds risk) ---
        avg_objections = d["objections_count"].mean() if n_demos else 0
        objection_theme_count = (
            t["objection_themes"].apply(lambda s: len(s.split(";")) if s else 0).mean()
            if n_text else 0
        )

        # --- Confidence: based on evidence volume / data completeness ---
        evidence_volume = min((n_demos + n_usage + n_comm) / 15, 1.0)
        confidence_score = round(100 * (0.6 * evidence_volume + 0.4 * min(n_distinct_customers / 4, 1.0)), 1)

        rows.append({
            "concept_id": cid,
            "concept_name": concept["concept_name"],
            "industry": concept["industry"],
            "problem_area": concept["problem_area"],
            "n_demo_sessions": n_demos,
            "n_sandbox_trials": n_usage,
            "n_commercial_signals": n_comm,
            "demand_intensity": round(demand_intensity, 4),
            "repeatability": round(repeatability, 4),
            "engagement_depth": round(engagement_depth, 4),
            "segment_similarity": round(segment_similarity, 4),
            "segment_diversity": round(segment_diversity, 4),
            "revenue_potential": round(revenue_potential, 4),
            "feasibility": round(feasibility, 4),
            "strategic_fit_score": round(strategic_fit, 4),
            "avg_objections": round(avg_objections, 2),
            "avg_objection_themes": round(objection_theme_count, 2),
            "confidence_score": confidence_score,
        })

    features = pd.DataFrame(rows)
    features.to_csv(DATA_DIR / "concept_features.csv", index=False)
    print(f"Built feature table: {features.shape[0]} concepts x {features.shape[1]} columns")
    print(features.head())
    return features


SEGMENTS_ALL = ["Enterprise", "Mid-Market", "SMB", "Public Sector"]

if __name__ == "__main__":
    build_features()
