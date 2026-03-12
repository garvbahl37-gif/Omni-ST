"""
Omni-ST: Evaluation Metrics
==============================
Comprehensive benchmarking metrics for all supported ST tasks.

Metrics:
  Gene expression prediction:  MSE, R², Pearson r, SSIM
  Cell type classification:    Accuracy, F1, AUC
  Spatial domain:              ARI, NMI, Silhouette
  Retrieval:                   Recall@K, MRR, mAP
  Embedding alignment:         Cosine similarity, FOSCTTM
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Gene Expression Prediction Metrics
# ---------------------------------------------------------------------------

def mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2, axis=0)
    ss_tot = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - np.where(ss_tot > 0, ss_res / ss_tot, 0)
    return float(np.mean(r2))


def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-gene Pearson r across samples."""
    G = pred.shape[1]
    rs = []
    for g in range(G):
        if target[:, g].std() > 0 and pred[:, g].std() > 0:
            r, _ = pearsonr(pred[:, g], target[:, g])
            rs.append(r)
    return float(np.mean(rs)) if rs else 0.0


def cosine_similarity_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_t = torch.tensor(pred, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.float32)
    sim = F.cosine_similarity(pred_t, target_t, dim=-1)
    return float(sim.mean().item())


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    return float(np.mean(pred_labels == true_labels))


def f1_score_macro(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Clustering / Spatial Domain Metrics
# ---------------------------------------------------------------------------

def adjusted_rand_index(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(true_labels, pred_labels))


def normalized_mutual_info(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score as sk_sil
    try:
        return float(sk_sil(embeddings, labels))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int = 5,
) -> float:
    """R@K: fraction of queries where correct item is in top-K retrieved."""
    sims = query_emb @ gallery_emb.T  # [Q, G]
    topk_idx = np.argsort(-sims, axis=1)[:, :k]  # [Q, k]
    hits = 0
    for q, row in enumerate(topk_idx):
        if query_labels[q] in gallery_labels[row]:
            hits += 1
    return hits / len(query_labels)


# ---------------------------------------------------------------------------
# Alignment — FOSCTTM
# ---------------------------------------------------------------------------

def foscttm(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Fraction of Samples Closer Than the True Match (FOSCTTM).
    Lower is better (0 = perfect alignment).
    """
    dists = np.sum((emb_a[:, None] - emb_b[None]) ** 2, axis=-1)  # [N, N]
    n = len(emb_a)
    fracs = []
    for i in range(n):
        d_true = dists[i, i]
        fracs.append(np.mean(dists[i] < d_true))
    return float(np.mean(fracs))


# ---------------------------------------------------------------------------
# Unified Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """
    Unified evaluation suite that computes all relevant metrics for a task.

    Parameters
    ----------
    task : str  task identifier
    """

    TASK_METRICS = {
        "image_to_gene": ["mse", "r2", "pearson", "cosine"],
        "gene_to_celltype": ["accuracy", "f1_macro"],
        "graph_to_domain": ["ari", "nmi"],
        "text_to_spatial": ["recall_at_1", "recall_at_5"],
    }

    def __init__(self, task: str = "image_to_gene") -> None:
        self.task = task

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        results = {}
        task = self.task

        if task == "image_to_gene":
            results["mse"] = mean_squared_error(predictions, targets)
            results["r2"] = r_squared(predictions, targets)
            results["pearson_r"] = pearson_correlation(predictions, targets)
            results["cosine_sim"] = cosine_similarity_score(predictions, targets)

        elif task == "gene_to_celltype":
            results["accuracy"] = accuracy(predictions, targets)
            results["f1_macro"] = f1_score_macro(predictions, targets)

        elif task == "graph_to_domain":
            results["ari"] = adjusted_rand_index(predictions, targets)
            results["nmi"] = normalized_mutual_info(predictions, targets)
            if embeddings is not None:
                results["silhouette"] = silhouette_score(embeddings, targets)

        return results

    def print_report(self, results: Dict[str, float]) -> None:
        print(f"\n{'='*50}")
        print(f"  Omni-ST Benchmark: {self.task}")
        print(f"{'='*50}")
        for metric, value in results.items():
            print(f"  {metric:25s}: {value:.4f}")
        print(f"{'='*50}\n")
