import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
REQUEST_TIMEOUT_SECONDS = 12

# Offline templates for fallback incase user has no API key

OFFLINE_TEMPLATES = {
    "MVP Build": (
        "This concept is recommended for MVP Build because it demonstrates "
        "strong customer demand, high repeatability, and strong commercial "
        "potential. The readiness score of {readiness_score}/100 (confidence: "
        "{confidence_score}%) indicates that the concept is mature enough for "
        "development. Its strongest evidence is {top_factor}, while "
        "{weak_factor} is comparatively weaker. While feasibility could still "
        "be improved, the available evidence supports proceeding with an MVP."
    ),
    "Customer Pilot": (
        "This concept shows promising customer interest but requires "
        "additional market validation before full-scale development. With a "
        "readiness score of {readiness_score}/100 (confidence: "
        "{confidence_score}%), the strongest signal is {top_factor}, while "
        "{weak_factor} remains a gap. Running a customer pilot will help "
        "validate assumptions and reduce commercialization risk."
    ),
    "Reusable Asset": (
        "Customer engagement is consistent across multiple use cases, making "
        "this concept suitable for reuse as a common capability or platform "
        "component instead of a standalone product. Readiness score: "
        "{readiness_score}/100 (confidence: {confidence_score}%), driven "
        "primarily by {top_factor}, with {weak_factor} as the weaker area."
    ),
    "Incubate": (
        "The concept shows potential but currently lacks sufficient "
        "supporting evidence. Readiness score is {readiness_score}/100 with "
        "confidence at only {confidence_score}% — {top_factor} is the "
        "strongest available signal, but {weak_factor} needs more support. "
        "Additional customer validation, prototype improvements, or "
        "commercial analysis is recommended before making an investment "
        "decision."
    ),
    "Archive": (
        "Current customer demand, engagement, and commercial indicators are "
        "insufficient to justify further investment. Readiness score: "
        "{readiness_score}/100 (confidence: {confidence_score}%). Even the "
        "strongest available signal, {top_factor}, is not enough to "
        "outweigh weaknesses such as {weak_factor}. The concept should be "
        "archived unless new evidence becomes available."
    ),
}


def build_offline_narrative(row: pd.Series) -> str:
    template = OFFLINE_TEMPLATES.get(row["recommended_outcome"], OFFLINE_TEMPLATES["Incubate"])
    return template.format(
        readiness_score=row["readiness_score"],
        confidence_score=row["confidence_score"],
        top_factor=row["top_positive_factor"],
        weak_factor=row["weakest_factor"],
    )

#prompt for AI insight
def build_llm_prompt(row: pd.Series) -> str:
    """Instructs the LLM to write in the same register/structure as the
    offline templates, but reasoning freshly from the concept's actual data
    rather than filling a fixed template."""
    return f"""You are a business analyst writing a short, professional
commercialization recommendation for an executive audience.

Concept: {row['concept_name']} ({row['industry']}, {row['problem_area']})
Recommended outcome: {row['recommended_outcome']}
Readiness score: {row['readiness_score']}/100
Confidence score: {row['confidence_score']}%
Strongest evidence factor: {row['top_positive_factor']} (value: {row['top_positive_value']})
Weakest evidence factor: {row['weakest_factor']} (value: {row['weakest_value']})
Cluster behavioral pattern: {row['cluster_label']}

Write a 3-4 sentence narrative explaining WHY this recommendation makes
sense, referencing the readiness score, confidence score, and both the
strongest and weakest evidence factors naturally in the explanation.
Match the tone of a concise internal business analyst memo — clear,
evidence-based, no fluff, no bullet points, no headers. Do not repeat the
raw numbers mechanically; weave them into natural sentences."""


def generate_llm_narrative(row: pd.Series, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",#Says im sending the data in JSON format
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": build_llm_prompt(row)}
        ],
        "temperature": 0.4, #controls randomness
        "max_tokens": 220, #mas token to use per ai insight
    }
    response = requests.post( #uses request library to send a HTTP post req to GROQ , and if GROQ does not reply in 12 secs
        GROQ_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip() #converts JSON to actual generated text


def generate_narrative(row: pd.Series, api_key: str | None = None) -> tuple[str, str]:
    """Returns (narrative_text, source) where source is 'llm' or 'offline'.
    Always falls back to offline on any failure — never raises."""
    if api_key:
        try:
            text = generate_llm_narrative(row, api_key)
            if text:
                return text, "llm"
        except Exception:
            pass 
    return build_offline_narrative(row), "offline"

""" Pre-generates the narratives and stores them, usefl for offline review or testing"""
def generate_all_narratives(api_key: str | None = None) -> pd.DataFrame:
    scored = pd.read_csv(DATA_DIR / "scored_concepts.csv")
    narratives, sources = [], []
    for _, row in scored.iterrows():
        text, source = generate_narrative(row, api_key)
        narratives.append(text)
        sources.append(source)
    scored["ai_insight"] = narratives
    scored["insight_source"] = sources
    scored.to_csv(DATA_DIR / "scored_concepts_with_insights.csv", index=False)
    return scored


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else None
    if key:
        print(f"Running with LLM (Groq) — key provided.")
    else:
        print("No API key provided — running fully offline (template mode).")
    result = generate_all_narratives(key)
    print(f"\nGenerated narratives for {len(result)} concepts.")
    print(f"Source breakdown:\n{result['insight_source'].value_counts()}\n")
    print("--- Sample narrative (top concept by readiness score) ---")
    top = result.sort_values("readiness_score", ascending=False).iloc[0]
    print(f"{top['concept_name']} ({top['recommended_outcome']}):")
    print(top["ai_insight"])
