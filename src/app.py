"""
app.py
------
Streamlit dashboard for the AI/ML Commercialization Decision Engine.

Run with:  streamlit run src/app.py

Reads the already-computed data/scored_concepts.csv (output of
ml_model.py). Does NOT re-run the ML pipeline on every interaction --
that would be slow and pointless since the underlying data doesn't
change on a filter click. AI narratives are generated on-demand per
concept (button click), not for all 40 concepts automatically, to avoid
burning API quota/rate limits unnecessarily.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from ai_insights import generate_narrative
from ml_model import CLUSTER_FEATURES, READINESS_WEIGHTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(
    page_title="Commercialization Decision Engine",
    page_icon="\U0001F4CA",
    layout="wide",
)

OUTCOME_COLORS = {
    "MVP Build": "#1a7f37",
    "Customer Pilot": "#2f81f7",
    "Reusable Asset": "#8250df",
    "Incubate": "#bf8700",
    "Archive": "#cf222e",
}


# ---------------------------------------------------------------------
# Data loading (cached so filters/interactions don't re-read from disk)
# ---------------------------------------------------------------------
@st.cache_data
def load_scored_data():
    path = DATA_DIR / "scored_concepts.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_raw_tables():
    tables = {}
    for name in ["product_concepts", "customer_demo_signals", "sandbox_usage",
                 "commercial_signals", "text_feedback"]:
        p = DATA_DIR / f"{name}.csv"
        if p.exists():
            tables[name] = pd.read_csv(p)
    return tables


def missing_data_notice():
    st.error(
        "No scored data found. Run the pipeline first:\n\n"
        "```\npython src/generate_data.py\n"
        "python src/feature_engineering.py\n"
        "python src/ml_model.py\n```"
    )
    st.stop()


# ---------------------------------------------------------------------
# Sidebar: filters + AI settings
# ---------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("Filters")

    industries = st.sidebar.multiselect(
        "Industry", sorted(df["industry"].unique()), default=[]
    )
    outcomes = st.sidebar.multiselect(
        "Recommended Outcome", sorted(df["recommended_outcome"].unique()), default=[]
    )
    clusters = st.sidebar.multiselect(
        "Behavioral Pattern", sorted(df["cluster_label"].unique()), default=[]
    )
    score_range = st.sidebar.slider(
        "Readiness score range", 0, 100, (0, 100)
    )

    st.sidebar.markdown("---")
    st.sidebar.title("AI Insight Settings")
    api_key = st.sidebar.text_input(
        "Groq API key (optional)", type="password",
        help="Leave blank to use offline template-based narratives instead of the LLM."
    )
    if api_key:
        st.sidebar.success("LLM narratives enabled.")
    else:
        st.sidebar.info("Offline narrative mode (no API key entered).")

    filtered = df.copy()
    if industries:
        filtered = filtered[filtered["industry"].isin(industries)]
    if outcomes:
        filtered = filtered[filtered["recommended_outcome"].isin(outcomes)]
    if clusters:
        filtered = filtered[filtered["cluster_label"].isin(clusters)]
    filtered = filtered[
        (filtered["readiness_score"] >= score_range[0])
        & (filtered["readiness_score"] <= score_range[1])
    ]
    return filtered, api_key


# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
def render_kpis(df: pd.DataFrame):
    cols = st.columns(5)
    cols[0].metric("Total Concepts", len(df))
    cols[1].metric("Avg Readiness Score", f"{df['readiness_score'].mean():.1f}" if len(df) else "-")
    cols[2].metric("Avg Confidence", f"{df['confidence_score'].mean():.1f}%" if len(df) else "-")
    top_outcome = df["recommended_outcome"].mode().iloc[0] if len(df) else "-"
    cols[3].metric("Most Common Outcome", top_outcome)
    mvp_count = (df["recommended_outcome"] == "MVP Build").sum()
    cols[4].metric("MVP-Ready Concepts", mvp_count)


# ---------------------------------------------------------------------
# Tab 1: Portfolio Overview
# ---------------------------------------------------------------------
def render_portfolio_tab(df: pd.DataFrame):
    if df.empty:
        st.warning("No concepts match the current filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Outcome Distribution")
        counts = df["recommended_outcome"].value_counts().reset_index()
        counts.columns = ["outcome", "count"]
        fig = px.bar(
            counts, x="outcome", y="count", color="outcome",
            color_discrete_map=OUTCOME_COLORS, text="count",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Concepts")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Readiness vs Confidence")
        fig = px.scatter(
            df, x="readiness_score", y="confidence_score",
            color="cluster_label", hover_name="concept_name",
            hover_data=["recommended_outcome", "industry"],
            labels={"readiness_score": "Readiness Score", "confidence_score": "Confidence Score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Behavioral Cluster Profiles")
    cluster_means = df.groupby("cluster_label")[CLUSTER_FEATURES].mean().reset_index()
    cluster_long = cluster_means.melt(id_vars="cluster_label", var_name="feature", value_name="value")
    fig = px.bar(
        cluster_long, x="feature", y="value", color="cluster_label",
        barmode="group", labels={"value": "Average Feature Value", "feature": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Ranked List")
    display_cols = ["concept_id", "concept_name", "industry", "readiness_score",
                     "confidence_score", "recommended_outcome", "cluster_label",
                     "top_positive_factor", "weakest_factor"]
    st.dataframe(
        df[display_cols].sort_values("readiness_score", ascending=False),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------
# Tab 2: Concept Explorer (search + detail view)
# ---------------------------------------------------------------------
def render_explorer_tab(df: pd.DataFrame, raw_tables: dict, api_key: str):
    if df.empty:
        st.warning("No concepts match the current filters.")
        return

    st.subheader("Search")
    search_query = st.text_input(
        "Search by concept name, ID, industry, or problem area",
        placeholder="e.g. CompassAI, C030, Healthcare, Forecasting",
    )

    search_results = df
    if search_query:
        q = search_query.strip().lower()
        mask = (
            df["concept_id"].str.lower().str.contains(q)
            | df["concept_name"].str.lower().str.contains(q)
            | df["industry"].str.lower().str.contains(q)
            | df["problem_area"].str.lower().str.contains(q)
        )
        search_results = df[mask]
        if search_results.empty:
            st.warning(f"No concept matches '{search_query}'. Showing full filtered list instead.")
            search_results = df

    options = search_results.apply(
        lambda r: f"{r['concept_id']} — {r['concept_name']} ({r['industry']})", axis=1
    ).tolist()
    if not options:
        st.warning("No concepts available.")
        return

    selected_label = st.selectbox("Select a concept", options)
    selected_id = selected_label.split(" — ")[0]
    row = df[df["concept_id"] == selected_id].iloc[0]

    st.markdown("---")
    header_col, badge_col = st.columns([3, 1])
    with header_col:
        st.markdown(f"## {row['concept_name']}  ·  {row['industry']} — {row['problem_area']}")
    with badge_col:
        color = OUTCOME_COLORS.get(row["recommended_outcome"], "#666")
        st.markdown(
            f"<div style='background-color:{color};color:white;padding:10px;"
            f"border-radius:8px;text-align:center;font-weight:bold;'>"
            f"{row['recommended_outcome']}</div>",
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Readiness Score", f"{row['readiness_score']:.1f} / 100")
    m2.metric("Confidence Score", f"{row['confidence_score']:.1f}%")
    m3.metric("Behavioral Pattern", row["cluster_label"])
    m4.metric("Avg Objections", f"{row['avg_objections']:.1f}")

    st.markdown("### Feature Contribution Breakdown")
    contrib = {feat: row[feat] * w * 100 for feat, w in READINESS_WEIGHTS.items()}
    contrib_df = pd.DataFrame(
        {"feature": [k.replace("_", " ") for k in contrib.keys()],
         "contribution": list(contrib.values())}
    ).sort_values("contribution", ascending=True)
    fig = px.bar(
        contrib_df, x="contribution", y="feature", orientation="h",
        color="contribution", color_continuous_scale="RdYlGn",
        labels={"contribution": "Points contributed to readiness score", "feature": ""},
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Insight")
    insight_key = f"insight_{selected_id}"
    if insight_key not in st.session_state:
        st.session_state[insight_key] = None

    if st.button("Generate AI Insight", key=f"btn_{selected_id}"):
        with st.spinner("Generating narrative..."):
            text, source = generate_narrative(row, api_key if api_key else None)
            st.session_state[insight_key] = (text, source)

    if st.session_state[insight_key]:
        text, source = st.session_state[insight_key]
        badge = "🤖 LLM-generated" if source == "llm" else "📋 Offline template"
        st.caption(badge)
        st.info(text)
    else:
        st.caption("Click the button above to generate a narrative explanation for this concept.")

    with st.expander("Raw supporting evidence"):
        for table_name, label in [
            ("customer_demo_signals", "Demo Sessions"),
            ("sandbox_usage", "Sandbox Usage"),
            ("commercial_signals", "Commercial Signals"),
            ("text_feedback", "Text Feedback"),
        ]:
            if table_name in raw_tables:
                subset = raw_tables[table_name][raw_tables[table_name]["concept_id"] == selected_id]
                if not subset.empty:
                    st.markdown(f"**{label}** ({len(subset)} records)")
                    st.dataframe(subset, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Tab 3: Explainability
# ---------------------------------------------------------------------
def render_explainability_tab(df: pd.DataFrame):
    if df.empty:
        st.warning("No concepts match the current filters.")
        return

    st.subheader("Portfolio-Wide Feature Importance")
    st.caption(
        "Average weighted point-contribution of each feature to the readiness score, "
        "across all concepts currently in view."
    )
    avg_contribs = {}
    for feat, w in READINESS_WEIGHTS.items():
        avg_contribs[feat.replace("_", " ")] = (df[feat] * w * 100).mean()
    imp_df = pd.DataFrame(
        {"feature": list(avg_contribs.keys()), "avg_contribution": list(avg_contribs.values())}
    ).sort_values("avg_contribution", ascending=True)
    fig = px.bar(
        imp_df, x="avg_contribution", y="feature", orientation="h",
        labels={"avg_contribution": "Average points contributed", "feature": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decision Path Logic")
    st.markdown(
        """
        The recommendation engine applies rules in this order (first match wins):

        1. **Confidence < 40%** → `Incubate` (if score ≥ 35) or `Archive` (if score < 35)
           — *low evidence never produces a strong go/no-go call.*
        2. **Readiness < 35** → `Archive`
        3. **Repeatability ≥ 0.55 AND segment diversity ≥ 0.5 AND readiness ≥ 55** → `Reusable Asset`
        4. **Readiness ≥ 65 AND feasibility ≥ 0.5** → `MVP Build`
        5. **Readiness ≥ 55 AND demand intensity ≥ 0.5** → `Customer Pilot`
        6. **Readiness ≥ 45** → `Incubate`
        7. Otherwise → `Archive`
        """
    )

    st.subheader("Confidence vs Readiness — Decision Zones")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=100, y0=0, y1=40, fillcolor="rgba(200,200,200,0.25)", line_width=0)
    fig.add_annotation(x=50, y=20, text="Low-confidence zone → Incubate/Archive only", showarrow=False)
    for outcome in df["recommended_outcome"].unique():
        subset = df[df["recommended_outcome"] == outcome]
        fig.add_trace(go.Scatter(
            x=subset["readiness_score"], y=subset["confidence_score"],
            mode="markers", name=outcome,
            marker=dict(color=OUTCOME_COLORS.get(outcome, "#666"), size=10),
            text=subset["concept_name"],
        ))
    fig.update_layout(
        xaxis_title="Readiness Score", yaxis_title="Confidence Score (%)",
        yaxis_range=[0, 105], xaxis_range=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    st.title("\U0001F4CA AI/ML Commercialization Decision Engine")
    st.caption(
        "Converts noisy customer demo and product-usage signals into "
        "evidence-based commercialization recommendations."
    )

    df = load_scored_data()
    if df is None:
        missing_data_notice()
    raw_tables = load_raw_tables()

    filtered_df, api_key = render_sidebar(df)

    render_kpis(filtered_df)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Concept Explorer", "Explainability"])
    with tab1:
        render_portfolio_tab(filtered_df)
    with tab2:
        render_explorer_tab(filtered_df, raw_tables, api_key)
    with tab3:
        render_explainability_tab(filtered_df)


if __name__ == "__main__":
    main()
