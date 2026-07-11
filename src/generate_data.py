"""
generate_data.py
-----------------
Generates a realistic, noisy mock dataset simulating customer demo, sandbox
usage, commercial, and text feedback signals for a set of early-stage AI
product concepts.

Design principle: each concept has a HIDDEN latent "true readiness" profile
(demand, repeatability, feasibility, strategic fit) that we do NOT expose
directly. Instead, we use it to bias the generation of noisy, partially
missing, sometimes contradictory raw signals across five related tables —
exactly like real messy pre-commercialization data would look. The ML
pipeline later has to re-discover the pattern from these raw signals, which
is the whole point of the exercise.

Outputs (in ../data/):
  - product_concepts.csv
  - customer_demo_signals.csv
  - sandbox_usage.csv
  - commercial_signals.csv
  - text_feedback.csv
  - _ground_truth_debug.csv   (latent scores, NOT to be used as a model
                                feature — kept only for your own sanity
                                checking / writeup, delete before submitting
                                if you want a stricter exercise)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

N_STANDARD_CONCEPTS = 40
N_FLAGSHIP_CONCEPTS = 10   # concepts modeled as having undergone extra internal
                            # refinement/vetting before their first customer
                            # touchpoint -- genuinely higher-quality latent
                            # distribution, but still fully noisy downstream.
                            # Outcomes are NOT hardcoded; they still depend
                            # entirely on the same scoring/rule pipeline as
                            # every other concept.
N_CONCEPTS = N_STANDARD_CONCEPTS + N_FLAGSHIP_CONCEPTS

INDUSTRIES = ["Healthcare", "Retail", "Financial Services", "Manufacturing",
              "Logistics", "Education", "Energy", "Insurance"]
PROBLEM_AREAS = ["Forecasting", "Anomaly Detection", "Process Automation",
                  "Customer Insights", "Document Intelligence",
                  "Quality Control", "Personalization", "Risk Scoring",
                  "Scheduling Optimization", "Fraud Detection"]
TARGET_USERS = ["Ops Manager", "Data Analyst", "Frontline Worker",
                 "Executive Sponsor", "IT Admin", "Compliance Officer",
                 "Store Manager", "Clinician"]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Public Sector"]

CONCEPT_NAME_STEMS = ["Signal", "Pulse", "Vista", "Nexus", "Orbit", "Beacon",
                       "Forge", "Lumen", "Atlas", "Compass", "Prism", "Flux",
                       "Sentry", "Harbor", "Catalyst", "Momentum", "Anchor",
                       "Horizon", "Vantage", "Cascade"]


def make_concept_ids(n):
    return [f"C{str(i).zfill(3)}" for i in range(1, n + 1)]


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=int(rng.integers(0, delta.days + 1)))


def maybe_missing(value, p_missing=0.06):
    """Randomly null out a value to simulate real-world missingness."""
    return np.nan if rng.random() < p_missing else value


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def generate():
    concept_ids = make_concept_ids(N_CONCEPTS)

    # ---- 1. Latent (hidden) readiness drivers per concept ----
    # These are NOT written to product_concepts.csv directly as a single
    # score. They only bias how noisy the downstream tables get generated.
    #
    # Two blocks, concatenated:
    #   - Standard concepts: beta(2,2)/beta(2.5,2) -- centered ~0.5, realistic
    #     spread of early-stage ideas (as before).
    #   - Flagship concepts: beta(5,2) -- genuinely higher-quality latent
    #     distribution (mean ~0.71 vs ~0.5), representing concepts that
    #     underwent additional internal refinement before their first
    #     customer touchpoint. This does NOT guarantee any specific outcome
    #     -- the same noise, same formulas, same rules apply downstream.
    n_std = N_STANDARD_CONCEPTS
    n_flag = N_FLAGSHIP_CONCEPTS
    latent = pd.DataFrame({
        "concept_id": concept_ids,
        "true_demand_intensity": np.concatenate([
            rng.beta(2, 2, n_std), rng.beta(5, 2, n_flag)
        ]),
        "true_repeatability": np.concatenate([
            rng.beta(2, 2, n_std), rng.beta(5, 2, n_flag)
        ]),
        "true_feasibility": np.concatenate([
            rng.beta(2.5, 2, n_std), rng.beta(5, 2, n_flag)
        ]),
        "true_strategic_fit_latent": np.concatenate([
            rng.beta(2, 2, n_std), rng.beta(5, 2, n_flag)
        ]),
        "is_flagship": [False] * n_std + [True] * n_flag,
    })

    # Shuffle so flagship concepts are randomly scattered across the ID
    # range instead of predictably clustered at C041-C050 -- avoids an
    # obvious, easily-noticed pattern where "later IDs = better outcomes".
    shuffle_idx = rng.permutation(len(latent))
    latent = latent.iloc[shuffle_idx].reset_index(drop=True)
    latent["concept_id"] = concept_ids  # reassign sequential IDs post-shuffle
    latent["true_composite"] = (
        0.32 * latent["true_demand_intensity"]
        + 0.28 * latent["true_repeatability"]
        + 0.20 * latent["true_feasibility"]
        + 0.20 * latent["true_strategic_fit_latent"]
    )

    # ---- 2. product_concepts.csv ----
    rows = []
    for i, cid in enumerate(concept_ids):
        stem = CONCEPT_NAME_STEMS[i % len(CONCEPT_NAME_STEMS)]
        suffix = "AI" if rng.random() < 0.5 else "ML"
        name = f"{stem}{suffix}"
        strategic_fit_latent = latent.loc[i, "true_strategic_fit_latent"]
        # strategic_fit as a noisy categorical rating a real exec might give
        fit_score = clip(strategic_fit_latent + rng.normal(0, 0.12), 0, 1)
        strategic_fit = (
            "High" if fit_score > 0.66 else "Medium" if fit_score > 0.33 else "Low"
        )
        feasibility_latent = latent.loc[i, "true_feasibility"]
        complexity_score = clip(1 - feasibility_latent + rng.normal(0, 0.15), 0, 1)
        delivery_complexity = (
            "High" if complexity_score > 0.66 else "Medium" if complexity_score > 0.33 else "Low"
        )
        rows.append({
            "concept_id": cid,
            "concept_name": name,
            "industry": rng.choice(INDUSTRIES),
            "problem_area": rng.choice(PROBLEM_AREAS),
            "target_user": rng.choice(TARGET_USERS),
            "delivery_complexity": delivery_complexity,
            "strategic_fit": strategic_fit,
        })
    product_concepts = pd.DataFrame(rows)

    # ---- 3. customer_demo_signals.csv (multiple demos per concept) ----
    demo_rows = []
    cust_counter = 1
    for i, cid in enumerate(concept_ids):
        demand = latent.loc[i, "true_demand_intensity"]
        n_demos = int(rng.integers(2, 9))  # 2-8 demo sessions per concept
        start_date = datetime(2025, 9, 1)
        end_date = datetime(2026, 6, 30)
        for _ in range(n_demos):
            cust_id = f"CUST{str(cust_counter).zfill(4)}"
            cust_counter += 1
            base_feedback = clip(demand * 5 + rng.normal(0, 1.1), 1, 5)
            demo_rows.append({
                "customer_id": cust_id,
                "concept_id": cid,
                "segment": rng.choice(SEGMENTS),
                "demo_date": random_date(start_date, end_date).strftime("%Y-%m-%d"),
                "feedback_score": maybe_missing(round(base_feedback, 1)),
                "follow_up_requested": maybe_missing(
                    int(rng.random() < clip(demand + rng.normal(0, 0.2), 0, 1))
                ),
                "decision_maker_present": maybe_missing(int(rng.random() < 0.45)),
                "objections_count": maybe_missing(
                    int(clip(rng.poisson(lam=(1 - demand) * 4), 0, 12))
                ),
            })
    customer_demo_signals = pd.DataFrame(demo_rows)

    # ---- 4. sandbox_usage.csv (usage sessions, tied to same customers where relevant) ----
    usage_rows = []
    demo_customers_by_concept = customer_demo_signals.groupby("concept_id")["customer_id"].apply(list)
    for i, cid in enumerate(concept_ids):
        repeatability = latent.loc[i, "true_repeatability"]
        demand = latent.loc[i, "true_demand_intensity"]
        custs = demo_customers_by_concept.get(cid, [])
        # not every customer who saw a demo tries the sandbox (real-world drop-off)
        trial_custs = [c for c in custs if rng.random() < 0.7]
        if not trial_custs:
            trial_custs = custs[:1]
        for cust_id in trial_custs:
            sessions = int(clip(rng.poisson(lam=repeatability * 6 + 1), 1, 20))
            usage_rows.append({
                "customer_id": cust_id,
                "concept_id": cid,
                "trial_sessions": maybe_missing(sessions),
                "feature_clicks": maybe_missing(
                    int(clip(rng.poisson(lam=demand * 40 + 5), 0, 300))
                ),
                "repeat_usage_days": maybe_missing(
                    int(clip(rng.poisson(lam=repeatability * 10), 0, 30))
                ),
                "active_users": maybe_missing(
                    int(clip(rng.poisson(lam=demand * 5 + 1), 1, 40))
                ),
                "time_spent_minutes": maybe_missing(
                    round(clip(rng.normal(repeatability * 90 + 10, 20), 1, 400), 1)
                ),
                "abandoned_features": maybe_missing(
                    int(clip(rng.poisson(lam=(1 - repeatability) * 3), 0, 10))
                ),
            })
    sandbox_usage = pd.DataFrame(usage_rows)

    # ---- 5. commercial_signals.csv ----
    comm_rows = []
    for i, cid in enumerate(concept_ids):
        demand = latent.loc[i, "true_demand_intensity"]
        feasibility = latent.loc[i, "true_feasibility"]
        strategic = latent.loc[i, "true_strategic_fit_latent"]
        custs = demo_customers_by_concept.get(cid, [])
        sample_custs = custs if len(custs) <= 5 else list(rng.choice(custs, 5, replace=False))
        for cust_id in sample_custs:
            urgency = clip(demand + rng.normal(0, 0.15), 0, 1)
            comm_rows.append({
                "customer_id": cust_id,
                "concept_id": cid,
                "pilot_interest": maybe_missing(
                    int(rng.random() < clip(demand * 0.8 + strategic * 0.2, 0, 1))
                ),
                "urgency_score": maybe_missing(round(urgency * 100, 1)),
                "budget_signal": maybe_missing(
                    rng.choice(["Confirmed", "Likely", "Unclear", "None"],
                               p=[0.15, 0.30, 0.35, 0.20])
                ),
                "willingness_to_pay": maybe_missing(
                    round(clip(rng.normal(demand * 50000 + 5000, 12000), 0, 200000), 0)
                ),
                "expected_value": maybe_missing(
                    round(clip(rng.normal((demand * 0.5 + strategic * 0.5) * 300000, 60000), 0, 900000), 0)
                ),
                "implementation_risk": maybe_missing(
                    round(clip(1 - feasibility + rng.normal(0, 0.1), 0, 1) * 100, 1)
                ),
            })
    commercial_signals = pd.DataFrame(comm_rows)

    # ---- 6. text_feedback.csv ----
    pain_points_bank = [
        "manual process takes too long", "data quality issues slow us down",
        "hard to get buy-in from leadership", "current tools don't scale",
        "too many false positives", "lack of visibility into root causes",
        "integration with legacy systems is painful", "team lacks technical bandwidth",
    ]
    objection_bank = [
        "price", "integration effort", "change management", "data privacy",
        "unclear ROI", "vendor lock-in", "accuracy concerns", "timeline",
    ]
    capability_bank = [
        "better reporting dashboard", "mobile access", "API access",
        "role-based permissions", "offline mode", "multi-language support",
        "custom alert thresholds", "audit trail",
    ]
    comment_templates_positive = [
        "The team was impressed with how quickly it surfaced patterns we'd missed.",
        "Several stakeholders asked when they could start a pilot.",
        "This addresses a real bottleneck in our current workflow.",
        "Leadership seemed genuinely excited after the walkthrough.",
    ]
    comment_templates_neutral = [
        "Interesting demo, but we need to see it work on our own data first.",
        "Some good ideas here, though the value wasn't fully clear to everyone.",
        "The team had mixed reactions and wants a follow-up session.",
    ]
    comment_templates_negative = [
        "This overlaps too much with a tool we already have in place.",
        "The team didn't see a strong enough case to prioritize this now.",
        "Concerns about accuracy came up multiple times during the session.",
    ]

    text_rows = []
    for i, cid in enumerate(concept_ids):
        demand = latent.loc[i, "true_demand_intensity"]
        custs = demo_customers_by_concept.get(cid, [])
        for cust_id in custs:
            r = rng.random()
            if r < demand:
                comment = rng.choice(comment_templates_positive)
            elif r < demand + 0.3:
                comment = rng.choice(comment_templates_neutral)
            else:
                comment = rng.choice(comment_templates_negative)
            n_pain = int(rng.integers(0, 3))
            n_obj = int(rng.integers(0, 3))
            n_cap = int(rng.integers(0, 3))
            text_rows.append({
                "customer_id": cust_id,
                "concept_id": cid,
                "customer_comments": maybe_missing(comment, 0.08),
                "pain_point_statements": maybe_missing(
                    "; ".join(rng.choice(pain_points_bank, n_pain, replace=False)) if n_pain else "",
                    0.1
                ),
                "objection_themes": maybe_missing(
                    "; ".join(rng.choice(objection_bank, n_obj, replace=False)) if n_obj else "",
                    0.1
                ),
                "requested_capabilities": maybe_missing(
                    "; ".join(rng.choice(capability_bank, n_cap, replace=False)) if n_cap else "",
                    0.1
                ),
            })
    text_feedback = pd.DataFrame(text_rows)

    # ---- write outputs ----
    product_concepts.to_csv(OUT_DIR / "product_concepts.csv", index=False)
    customer_demo_signals.to_csv(OUT_DIR / "customer_demo_signals.csv", index=False)
    sandbox_usage.to_csv(OUT_DIR / "sandbox_usage.csv", index=False)
    commercial_signals.to_csv(OUT_DIR / "commercial_signals.csv", index=False)
    text_feedback.to_csv(OUT_DIR / "text_feedback.csv", index=False)
    latent.to_csv(OUT_DIR / "_ground_truth_debug.csv", index=False)

    print(f"Generated {N_CONCEPTS} concepts.")
    print(f"  product_concepts:        {len(product_concepts)} rows")
    print(f"  customer_demo_signals:   {len(customer_demo_signals)} rows")
    print(f"  sandbox_usage:           {len(sandbox_usage)} rows")
    print(f"  commercial_signals:      {len(commercial_signals)} rows")
    print(f"  text_feedback:           {len(text_feedback)} rows")
    print(f"Files written to: {OUT_DIR}")


if __name__ == "__main__":
    generate()
