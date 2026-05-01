"""Tier 1 signal: semantic entropy via N-sample fan-out + cosine clustering.

Not active in MVP (Weeks 1-2). Wired up in Weeks 3-4.
Requires the 'tier1' extras: sentence-transformers, numpy.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def compute_semantic_entropy(
    responses: list[str],
    similarity_threshold: float = 0.85,
) -> float:
    """Farquhar et al. 2024 semantic entropy, productized.

    Embeds responses, greedily clusters by cosine similarity, computes
    Shannon entropy over the cluster distribution, and returns
    ``1 - H_norm`` as a confidence contribution.

    Args:
        responses: K decoded response strings from the LLM fan-out.
        similarity_threshold: Cosine similarity threshold for cluster
            membership (0.85 is a sensible default; tune per customer).

    Returns:
        Confidence in [0, 1]: 1 = all responses in same cluster (certain),
        0 = all in distinct clusters (maximally uncertain).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers and numpy are required for Tier 1 semantic entropy. "
            "Install with: pip install 'confidence-sidecar[tier1]'"
        ) from exc

    k = len(responses)
    if k == 0:
        return 0.5
    if k == 1:
        return 1.0

    model = _get_embedding_model()
    embeddings: np.ndarray = model.encode(responses, normalize_embeddings=True)

    cluster_ids: list[int] = [0]
    centroids: list[np.ndarray] = [embeddings[0].copy()]
    centroid_counts: list[int] = [1]

    for i in range(1, k):
        sims = [float(embeddings[i] @ c) for c in centroids]
        best_idx = max(range(len(sims)), key=lambda j: sims[j])
        if sims[best_idx] >= similarity_threshold:
            cluster_ids.append(best_idx)
            n = centroid_counts[best_idx]
            centroids[best_idx] = (centroids[best_idx] * n + embeddings[i]) / (n + 1)
            norm = float(np.linalg.norm(centroids[best_idx]))
            if norm > 0:
                centroids[best_idx] /= norm
            centroid_counts[best_idx] += 1
        else:
            cluster_ids.append(len(centroids))
            centroids.append(embeddings[i].copy())
            centroid_counts.append(1)

    counts = Counter(cluster_ids)
    probs = [counts[c] / k for c in range(len(centroids))]
    h = -sum(p * math.log(p) for p in probs if p > 1e-12)
    h_max = math.log(k)
    return 1.0 - (h / h_max)


_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from sidecar.config import settings
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model
