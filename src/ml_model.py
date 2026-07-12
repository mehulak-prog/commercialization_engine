import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42

CLUSTER_FEATURES = [
    "demand_intensity", "repeatability", "engagement_depth",
    "segment_similarity", "revenue_potential", "feasibility",
    "strategic_fit_score",
]

# Documented weights for the readiness score (must sum to 1.0)
READINESS_WEIGHTS = {
    "demand_intensity": 0.24,
    "repeatability": 0.16,
    "engagement_depth": 0.12,
    "revenue_potential": 0.18,
    "feasibility": 0.14,
    "strategic_fit_score": 0.16,
}
assert abs(sum(READINESS_WEIGHTS.values()) - 1.0) < 1e-9

OBJECTION_PENALTY_WEIGHT = 3.0   


def pick_k(X, k_range=range(2, 6)):
    best_k, best_score = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def run_model():
    features = pd.read_csv(DATA_DIR / "concept_features.csv")

    # ---- 1. Clustering ----
    X = features[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, sil_score = pick_k(X_scaled)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    features["cluster"] = kmeans.labels_

    # Label clusters by their mean demand+repeatability so they read naturally
    cluster_profile = features.groupby("cluster")[CLUSTER_FEATURES].mean()
    cluster_rank = cluster_profile.mean(axis=1).sort_values(ascending=False)
    cluster_name_map = {}
    tiers = ["Strong Behavioral Pattern", "Promising Pattern",
             "Mixed/Weak Pattern", "Low-Signal Pattern"]
    for rank_i, cluster_id in enumerate(cluster_rank.index):
        cluster_name_map[cluster_id] = tiers[min(rank_i, len(tiers) - 1)]
    features["cluster_label"] = features["cluster"].map(cluster_name_map)

    # ---- 2. Weighted readiness score (0-100) ----
    contributions = pd.DataFrame(index=features.index)
    for feat, w in READINESS_WEIGHTS.items():
        contributions[feat] = features[feat] * w * 100

    raw_score = contributions.sum(axis=1)
    objection_penalty = (features["avg_objections"] - 1).clip(lower=0) * OBJECTION_PENALTY_WEIGHT
    readiness_score = (raw_score - objection_penalty).clip(0, 100)
    features["readiness_score"] = readiness_score.round(1)


    distances = kmeans.transform(X_scaled)
    own_cluster_dist = distances[np.arange(len(features)), features["cluster"]]
    dist_norm = (own_cluster_dist - own_cluster_dist.min()) / (np.ptp(own_cluster_dist) + 1e-9)
    cohesion_adjustment = (1 - dist_norm) * 10 - 5
    features["confidence_score"] = (features["confidence_score"] + cohesion_adjustment).clip(0, 100).round(1)

    # ---- 3. Rule-based outcome mapping ----
    def decide_outcome(row):
        score = row["readiness_score"]
        conf = row["confidence_score"]
        feasible = row["feasibility"]
        repeat = row["repeatability"]
        seg_div = row["segment_diversity"]
        demand = row["demand_intensity"]

        if conf < 40:
            return "Incubate" if score >= 35 else "Archive"
        if score < 35:
            return "Archive"
        if score >= 65 and feasible >= 0.5:
            return "MVP Build"
        if repeat >= 0.55 and seg_div >= 0.5 and score >= 55:
            return "Reusable Asset"
        if score >= 55 and demand >= 0.5:
            return "Customer Pilot"
        if score >= 45:
            return "Incubate"
        return "Archive"

    features["recommended_outcome"] = features.apply(decide_outcome, axis=1)

    # ---- 4. Explainable evidence: top contributing / detracting factors ----
    def top_evidence(idx):
        row_contrib = contributions.loc[idx].sort_values(ascending=False)
        top_pos = row_contrib.index[0]
        top_neg = row_contrib.index[-1]
        return pd.Series({
            "top_positive_factor": top_pos.replace("_", " "),
            "top_positive_value": round(features.loc[idx, top_pos], 3),
            "weakest_factor": top_neg.replace("_", " "),
            "weakest_value": round(features.loc[idx, top_neg], 3),
        })

    evidence = pd.DataFrame([top_evidence(i) for i in features.index])
    features = pd.concat([features, evidence], axis=1)

    features.to_csv(DATA_DIR / "scored_concepts.csv", index=False)

    print(f"Chosen k={best_k} (silhouette={sil_score:.3f})")
    print(features["recommended_outcome"].value_counts())
    print(features[["concept_id", "concept_name", "readiness_score",
                     "confidence_score", "recommended_outcome", "cluster_label"]]
          .sort_values("readiness_score", ascending=False).head(10))
    return features


if __name__ == "__main__":
    run_model()
