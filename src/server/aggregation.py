"""
Robust Aggregation Module for Federated Learning
Implements TrimmedMean and AWTM (Adaptive Weighted Trimmed Mean).
See aggregation_notes.txt for full math, examples, and design rationale.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ClientUpdate:
    """Single client's model update."""
    client_id: str
    update: Dict[str, np.ndarray]
    num_samples: int
    version: int
    timestamp: float
    loss: Optional[float] = None
    reputation: float = 1.0


@dataclass
class AggregationResult:
    """Aggregation output with metadata."""
    aggregated_update: Dict[str, np.ndarray]
    clients_used: List[str]
    clients_filtered: List[str]
    method_used: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Base Aggregator
# ---------------------------------------------------------------------------

class BaseAggregator:
    """Base class with shared utilities for flatten/unflatten and distances."""

    def __init__(self, name: str = "BaseAggregator"):
        self.name = name

    @staticmethod
    def flatten_update(update: Dict[str, np.ndarray]) -> np.ndarray:
        """Dict of layer arrays -> single flat vector (sorted keys)."""
        return np.concatenate([update[k].flatten() for k in sorted(update.keys())])

    @staticmethod
    def unflatten_update(flat_update: np.ndarray,
                         reference_update: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Flat vector -> dict matching reference shapes."""
        result = {}
        idx = 0
        for key in sorted(reference_update.keys()):
            shape = reference_update[key].shape
            size = np.prod(shape)
            result[key] = flat_update[idx:idx + size].reshape(shape)
            idx += size
        return result

    @staticmethod
    def compute_pairwise_distances(updates: List[np.ndarray]) -> np.ndarray:
        """Pairwise Euclidean distance matrix."""
        n = len(updates)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(updates[i] - updates[j])
                distances[i][j] = dist
                distances[j][i] = dist
        return distances

    def aggregate(self, updates: List[ClientUpdate],
                  current_version: int = 0) -> AggregationResult:
        raise NotImplementedError("Subclasses must implement aggregate()")


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

class TrimmedMeanAggregator(BaseAggregator):
    """
    Coordinate-wise trimmed mean.
    Sorts each parameter across clients, removes top/bottom trim_ratio
    fraction, averages the rest. Breakdown point = trim_ratio.
    """

    def __init__(self, trim_ratio: float = 0.2):
        super().__init__(name="TrimmedMean")
        self.trim_ratio = trim_ratio

    def aggregate(self, updates: List[ClientUpdate],
                  current_version: int = 0) -> AggregationResult:
        if len(updates) < 3:
            # Fallback: simple weighted average
            return self._simple_avg(updates)

        n_clients = len(updates)
        n_trim = int(n_clients * self.trim_ratio)

        # Keep at least 1 client after trimming
        if 2 * n_trim >= n_clients:
            n_trim = (n_clients - 1) // 2

        layer_names = list(updates[0].update.keys())
        aggregated = {}

        for layer_name in layer_names:
            layer_updates = np.stack([u.update[layer_name] for u in updates], axis=0)
            sorted_updates = np.sort(layer_updates, axis=0)
            # Trim extremes from both ends, average the middle
            trimmed = sorted_updates[n_trim:n_clients - n_trim]
            aggregated[layer_name] = np.mean(trimmed, axis=0)

        filtered_count = 2 * n_trim
        logger.info(
            f"TrimmedMean: Trimmed {filtered_count} (ratio={self.trim_ratio}), "
            f"keeping {n_clients - filtered_count}"
        )

        return AggregationResult(
            aggregated_update=aggregated,
            clients_used=[u.client_id for u in updates],
            clients_filtered=[f"unknown_{i}" for i in range(filtered_count)],
            method_used=self.name,
            confidence=1.0 - (2 * n_trim / n_clients),
        )

    def _simple_avg(self, updates: List[ClientUpdate]) -> AggregationResult:
        """Weighted average fallback when too few clients."""
        total = sum(u.num_samples for u in updates)
        weights = [u.num_samples / total for u in updates]
        aggregated = {}
        for layer in updates[0].update:
            aggregated[layer] = sum(w * u.update[layer] for w, u in zip(weights, updates))
        return AggregationResult(
            aggregated_update=aggregated,
            clients_used=[u.client_id for u in updates],
            clients_filtered=[],
            method_used="FedAvg",
            confidence=1.0,
        )


# ---------------------------------------------------------------------------
# AWTM (Adaptive Weighted Trimmed Mean) - Our Innovation
# ---------------------------------------------------------------------------

class AWTMAggregator(BaseAggregator):
    """
    Adaptive Weighted Trimmed Mean.
    Combines: adaptive trim via DBSCAN clustering, reputation weighting,
    and staleness decay. Automatically adjusts trim ratio per round.
    """

    def __init__(self,
                 max_trim_ratio: float = 0.4,
                 staleness_coefficient: float = 0.5,
                 dbscan_eps: float = None,
                 dbscan_min_samples: int = 2,
                 safety_margin: float = 0.1):
        super().__init__(name="AWTM")
        self.max_trim = max_trim_ratio
        self.staleness_coef = staleness_coefficient
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.safety_margin = safety_margin

    # -- Step 1: Estimate malicious ratio via DBSCAN --

    def _estimate_malicious_ratio(self, flat_updates: List[np.ndarray]) -> Tuple[float, List[int]]:
        """Use DBSCAN clustering to find outlier updates."""
        n = len(flat_updates)
        if n < 3:
            return 0.0, []

        distances = self.compute_pairwise_distances(flat_updates)

        # Auto-compute eps: median nearest-neighbor distance * 1.5
        if self.dbscan_eps is None:
            nn_dists = [sorted(distances[i])[1] for i in range(n)]
            eps = np.median(nn_dists) * 1.5
        else:
            eps = self.dbscan_eps

        labels = DBSCAN(
            eps=eps,
            min_samples=self.dbscan_min_samples,
            metric='precomputed',
        ).fit_predict(distances)

        # Find largest cluster (honest majority)
        # IMPORTANT: exclude noise label (-1)
        cluster_sizes = defaultdict(int)
        for label in labels:
            cluster_sizes[label] += 1

        valid_clusters = {k: v for k, v in cluster_sizes.items() if k != -1}
        if valid_clusters:
            largest_cluster = max(valid_clusters, key=valid_clusters.get)
            largest_size = valid_clusters[largest_cluster]
        else:
            # All noise -> treat all as honest
            largest_cluster = -1
            largest_size = n

        # Everything outside the largest cluster = potentially malicious
        outlier_indices = [i for i, lbl in enumerate(labels) if lbl != largest_cluster]
        ratio = len(outlier_indices) / n

        logger.info(
            f"AWTM: DBSCAN clusters={len(cluster_sizes)}, "
            f"largest={largest_size}/{n}, outliers={len(outlier_indices)}, "
            f"estimated_malicious={ratio:.2%}"
        )
        return ratio, outlier_indices

    # -- Step 2: Compute reputation + staleness weights --

    def _compute_client_weights(self, updates: List[ClientUpdate],
                                current_version: int) -> np.ndarray:
        """weight_i = reputation_i / (1 + alpha * staleness_i)"""
        weights = []
        for u in updates:
            staleness = max(0, current_version - u.version)
            weights.append(u.reputation / (1 + self.staleness_coef * staleness))
        weights = np.array(weights)
        return weights / (weights.sum() + 1e-8)

    # -- Step 3: Main aggregation --

    def aggregate(self, updates: List[ClientUpdate],
                  current_version: int = 0) -> AggregationResult:
        if len(updates) < 3:
            return TrimmedMeanAggregator()._simple_avg(updates)

        n_clients = len(updates)

        # Flatten for clustering analysis
        flat_updates = [self.flatten_update(u.update) for u in updates]

        # Adaptive trim ratio from DBSCAN
        estimated_ratio, outlier_indices = self._estimate_malicious_ratio(flat_updates)
        adaptive_trim = min(estimated_ratio + self.safety_margin, self.max_trim)
        n_trim = int(n_clients * adaptive_trim)
        if 2 * n_trim >= n_clients:
            n_trim = max(0, (n_clients - 1) // 2)

        # Reputation + staleness weights
        client_weights = self._compute_client_weights(updates, current_version)

        # Weighted trimmed mean per layer
        layer_names = list(updates[0].update.keys())
        aggregated = {}

        for layer_name in layer_names:
            layer_updates = np.stack([u.update[layer_name] for u in updates], axis=0)

            # Use median as reference, trim farthest updates
            layer_median = np.median(layer_updates, axis=0)
            dists = np.array([np.linalg.norm(layer_updates[i] - layer_median)
                              for i in range(n_clients)])
            sorted_indices = np.argsort(dists)

            # Keep closest to median (trim only farthest outliers)
            valid_indices = sorted_indices[:n_clients - n_trim]

            valid_weights = client_weights[valid_indices]
            valid_weights = valid_weights / (valid_weights.sum() + 1e-8)

            aggregated[layer_name] = np.zeros_like(layer_updates[0])
            for idx, w in zip(valid_indices, valid_weights):
                aggregated[layer_name] += w * layer_updates[idx]

        filtered_clients = [updates[i].client_id for i in outlier_indices]
        used_clients = [u.client_id for u in updates if u.client_id not in filtered_clients]
        confidence = 1.0 - (len(outlier_indices) / n_clients) if outlier_indices else 1.0

        logger.info(
            f"AWTM: trim={adaptive_trim:.2%}, filtered={len(filtered_clients)}, "
            f"confidence={confidence:.2f}"
        )

        return AggregationResult(
            aggregated_update=aggregated,
            clients_used=used_clients,
            clients_filtered=filtered_clients,
            method_used=self.name,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Aggregation: TrimmedMean + AWTM")
    print("=" * 50)
    np.random.seed(42)

    # Synthetic data: 10 honest + 2 malicious
    updates = []
    for i in range(10):
        updates.append(ClientUpdate(
            client_id=f"honest_{i}",
            update={
                'layer1.weight': np.random.randn(10, 10) * 0.1 + 1.0,
                'layer1.bias': np.random.randn(10) * 0.1 + 0.1,
            },
            num_samples=100, version=0, timestamp=0.0, reputation=1.0,
        ))
    for i in range(2):
        updates.append(ClientUpdate(
            client_id=f"malicious_{i}",
            update={
                'layer1.weight': np.random.randn(10, 10) * 10.0,
                'layer1.bias': np.random.randn(10) * 10.0,
            },
            num_samples=100, version=0, timestamp=0.0, reputation=1.0,
        ))

    print(f"\n12 updates (10 honest, 2 malicious)")
    print("-" * 50)

    for name, agg in [("TRIMMED_MEAN", TrimmedMeanAggregator()),
                      ("AWTM", AWTMAggregator())]:
        result = agg.aggregate(updates)
        norm = np.linalg.norm(BaseAggregator.flatten_update(result.aggregated_update))
        print(f"\n{name}:")
        print(f"  Used: {len(result.clients_used)}, Filtered: {len(result.clients_filtered)}")
        print(f"  Confidence: {result.confidence:.2f}, Norm: {norm:.4f}")
