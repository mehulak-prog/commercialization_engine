# AI/ML Commercialization Decision Engine

## Local Setup

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

## Run the pipeline (in order — each step depends on the previous one's output)

```bash
# Step 1: Generate the synthetic dataset (5 CSVs -> data/ folder)
python3 src/generate_data.py

# Step 2: Engineer concept-level features from raw data
python3 src/feature_engineering.py

# Step 3: Run clustering + scoring + rule engine
python3 src/ml_model.py
```

## What to check after each step

**After Step 1** — you should see in the terminal:
```
Generated 40 concepts.
  product_concepts:        40 rows
  customer_demo_signals:   ~200 rows
  sandbox_usage:           ~150-160 rows
  commercial_signals:      ~170 rows
  text_feedback:           ~200 rows
```
And these files should now exist in `data/`:
`product_concepts.csv`, `customer_demo_signals.csv`, `sandbox_usage.csv`, `commercial_signals.csv`, `text_feedback.csv`, `_ground_truth_debug.csv`

**After Step 2** — terminal should print:
```
Built feature table: 40 concepts x 18 columns
```
And `data/concept_features.csv` should exist. Open it (Excel, VS Code, or `pandas.read_csv`) — every row should have no blank/NaN cells in the numeric feature columns (demand_intensity, repeatability, etc.) — that confirms missing-value imputation worked.

**After Step 3** — terminal should print something like:
```
Chosen k=3 (silhouette=0.XXX)
recommended_outcome
Incubate           XX
Customer Pilot      X
...
```
followed by a table of the top 10 concepts by readiness score. And `data/scored_concepts.csv` should exist with columns including `readiness_score`, `confidence_score`, `recommended_outcome`, `cluster_label`, `top_positive_factor`, `weakest_factor`.

## Things worth sanity-checking yourself

1. **Outcome distribution isn't absurd** — e.g. not 39 out of 40 concepts landing in the same bucket. Some spread across Archive/Incubate/Pilot/MVP Build/Reusable Asset is expected and realistic (most early-stage concepts should NOT be MVP Build — that's meant to be the exception, not the norm).
2. **Silhouette score** — printed value should ideally be above ~0.15-0.2. If it's very low (near 0) or negative, the clusters aren't well separated — flag this back and we'll discuss (could mean features need reweighting, or k range needs adjusting).
3. **Spot-check one concept's logic** — pick any `concept_id` from `scored_concepts.csv`, trace its `readiness_score`, `confidence_score`, and `recommended_outcome` back against the `decide_outcome()` rules in `ml_model.py` to confirm the logic fired the way you'd expect.


## Next steps (not built yet)

- Narrative generation layer (LLM + offline fallback)
- Streamlit dashboard (`app.py`)
- GitHub + Streamlit Cloud deployment guide
- PPT/PDF for submission
