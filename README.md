# AI/ML Commercialization Decision Engine

A hybrid ML prototype that converts noisy customer demo and product-usage
signals into evidence-based commercialization recommendations for
early-stage AI product concepts.

Given a portfolio of concepts, the engine outputs, for each one: a
readiness score (0-100), a confidence score, a recommended outcome
(**MVP Build**, **Customer Pilot**, **Reusable Asset**, **Incubate**, or
**Archive**), and a plain-language explanation of why.

## Why this approach

These are brand-new product concepts — there's no historical
success/failure data to train a supervised model on. So the pipeline uses
a hybrid approach instead:

1. **Synthetic data generation** — realistic, noisy mock data (no public
   dataset fits this exact scenario)
2. **Feature engineering** — raw multi-row customer signals aggregated
   into one clean, numeric row per concept
3. **Unsupervised ML (KMeans)** — groups concepts by behavioral pattern,
   no labels required
4. **Weighted scoring model** — a transparent, documented Weighted Sum
   Model (a standard MCDA technique) converts features into a 0-100
   readiness score
5. **Rule-based decision engine** — converts score + confidence +
   specific feature thresholds into one of 5 outcomes
6. **AI narrative layer** — an LLM (Groq, with an offline template
   fallback) explains each recommendation in plain business language
7. **Streamlit dashboard** — interactive UI to explore, filter, and
   search the full ranked portfolio

## Project structure

```
commercialization_engine/
├── src/
│   ├── generate_data.py        # Step 1: synthetic dataset generator
│   ├── feature_engineering.py  # Step 2: raw signals -> concept features
│   ├── ml_model.py              # Step 3: clustering + scoring + rules
│   ├── ai_insights.py           # Step 4: AI/offline narrative generation
│   └── app.py                   # Streamlit dashboard
├── data/                        # Generated datasets (pre-populated)
├── requirements.txt
└── README.md
```

## Local setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

Each step depends on the previous step's output. `data/` already contains
pre-generated output, but you can regenerate everything from scratch:

```bash
python src/generate_data.py          # Step 1: generate 50 synthetic concepts
python src/feature_engineering.py    # Step 2: engineer concept-level features
python src/ml_model.py               # Step 3: cluster, score, and recommend
python src/ai_insights.py YOUR_GROQ_KEY   # Step 4 (optional): pre-generate narratives
```

## Run the dashboard

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`. Includes:
- **Portfolio Overview** — outcome distribution, readiness vs. confidence,
  cluster behavior profiles, full ranked list
- **Concept Explorer** — search by name/ID/industry/problem area, feature
  contribution breakdown, on-demand AI-generated insight, raw supporting
  evidence
- **Explainability** — portfolio-wide feature importance, the exact
  decision rules, a confidence/readiness decision-zone chart

An optional Groq API key can be entered in the sidebar for LLM-generated
narratives; without one, the app uses offline template-based narratives
(same content structure, fully deterministic, no network dependency).

## What to expect

Running the pipeline on the included dataset (50 concepts) produces:

```
recommended_outcome
Archive           17
Incubate          17
MVP Build          8
Customer Pilot     5
Reusable Asset     3
```

## Notable design decisions (documented, not hidden)

- **10 of the 50 concepts** are modeled as "flagship" — representing
  concepts that underwent extra internal refinement before their first
  customer touchpoint. This uses a genuinely higher-quality latent data
  distribution; no outcome is hardcoded, and the same scoring/rule
  pipeline applies to all 50 concepts identically.
- **Feature weights** are a documented Weighted Sum Model judgment call
  (a standard MCDA technique), not statistically derived — appropriate
  given no historical outcome data exists yet. See code comments in
  `feature_engineering.py` and `ml_model.py` for the reasoning behind
  each weight.
- **Outcome thresholds** were calibrated through testing on this sample
  (documented in `ml_model.py`), not fixed by an external benchmark.
- A hidden "ground truth" latent quality score is generated during data
  creation (`data/_ground_truth_debug.csv`) purely to make the synthetic
  data internally realistic — it is never used by the feature engineering
  or ML pipeline.

## Future improvements

- Recalibrate feature weights and outcome thresholds via regression once
  real historical outcome data becomes available.
- Extend the engine to score brand-new, unseen concepts live (currently
  scores the existing 50-concept portfolio).
- Deploy to Streamlit Community Cloud for a persistent public demo link.

## If something errors out

Most issues are environment-related (e.g. NumPy 2.0 removed some legacy
array methods). Check the traceback's exact line and error type first —
it's usually a one-line fix.
