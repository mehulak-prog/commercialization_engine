# AI/ML Commercialization Decision Engine

Converts noisy customer demo and product-usage signals into evidence-based commercialization recommendations — MVP Build, Customer Pilot, Reusable Asset, Incubate, or Archive — for a portfolio of product concepts.

**Live demo:** https://commercializationengine-anarnl3cxg3fk4ejl49y5y.streamlit.app/

**Tech stack:** Python, Pandas, NumPy, scikit-learn, Streamlit, Plotly, Groq API (LLM)

## What it does

- Generates synthetic product-concept data (demo sessions, sandbox usage, commercial signals, text feedback) using Beta-distributed sampling across 40 concepts
- Engineers an 18-feature table per concept from the raw signal tables
- Clusters concepts by behavioral pattern using KMeans, with silhouette-score-based selection of k
- Scores each concept on **readiness** and **confidence** via a weighted-feature model
- Applies a documented 7-rule decision waterfall (first match wins) to recommend one of 5 outcomes
- Surfaces everything in an interactive dashboard with three views:
  - **Portfolio Overview** — outcome distribution, readiness-vs-confidence scatter, cluster profiles, full ranked list
  - **Concept Explorer** — search/filter individual concepts, see per-feature score contribution, generate an AI-written narrative explanation (LLM via Groq API, with an offline template fallback when no API key is provided)
  - **Explainability** — portfolio-wide feature importance, the full decision-rule logic, and a confidence-vs-readiness decision-zone plot

## Project structure

| File                     | Description                                          |
| ------------------------ | ----------------------------------------------------- |
| `src/generate_data.py`   | Generates the synthetic dataset (Beta-distributed)     |
| `src/feature_engineering.py` | Builds the 18-feature concept table               |
| `src/ml_model.py`        | Clustering, weighted scoring, and the rule engine      |
| `src/ai_insights.py`     | LLM narrative generation (Groq API) + offline fallback |
| `src/app.py`             | Streamlit dashboard                                   |
| `data/`                  | Generated CSVs (created by the pipeline, not committed source data) |
| `requirements.txt`       | Python dependencies                                   |

## Running locally

```bash
# 1. Create a virtual environment (recommended, keeps things clean)
python3 -m venv venv

# 2. Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the pipeline (in order — each step depends on the previous one's output)

```bash
# Step 1: Generate the synthetic dataset (5 CSVs -> data/ folder)
python3 src/generate_data.py

# Step 2: Engineer concept-level features from raw data
python3 src/feature_engineering.py

# Step 3: Run clustering + scoring + rule engine
python3 src/ml_model.py

# Step 4: Launch the dashboard
streamlit run src/app.py
```

### What to check after each step

**After Step 1** — you should see in the terminal:
```
Generated 40 concepts.
  product_concepts:        40 rows
  customer_demo_signals:   ~200 rows
  sandbox_usage:           ~150-160 rows
  commercial_signals:      ~170 rows
  text_feedback:           ~200 rows
```
And these files should now exist in `data/`: `product_concepts.csv`, `customer_demo_signals.csv`, `sandbox_usage.csv`, `commercial_signals.csv`, `text_feedback.csv`, `_ground_truth_debug.csv`

**After Step 2** — terminal should print:
```
Built feature table: 40 concepts x 18 columns
```
And `data/concept_features.csv` should exist with no blank/NaN cells in the numeric feature columns.

**After Step 3** — terminal should print something like:
```
Chosen k=3 (silhouette=0.XXX)
recommended_outcome
Incubate           XX
Customer Pilot      X
...
```
And `data/scored_concepts.csv` should exist with columns including `readiness_score`, `confidence_score`, `recommended_outcome`, `cluster_label`, `top_positive_factor`, `weakest_factor`.

**After Step 4** — the dashboard opens in your browser at `localhost:8501`. Enter a Groq API key in the sidebar for LLM-generated narratives, or leave it blank to use offline template narratives.

## Things worth sanity-checking yourself

1. **Outcome distribution isn't absurd** — some spread across Archive/Incubate/Pilot/MVP Build/Reusable Asset is expected; most early-stage concepts should NOT be MVP Build.
2. **Silhouette score** — should ideally be above ~0.15-0.2. If very low or negative, the clusters aren't well separated.
3. **Spot-check one concept's logic** — pick a `concept_id` from `scored_concepts.csv`, trace its scores back against the decision rules in the Explainability tab.

## If something errors out

Copy the exact error message (the last ~15-20 lines of the traceback) before trying to guess-fix ML/pandas errors — they're usually a one-line issue but easy to misdiagnose.

## Next steps

- PPT/PDF for submission
