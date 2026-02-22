"""
Anomaly Detection Module for Federated Learning
Implements GradientNormDetector + ReputationSystem.
See anomaly_detection_notes.txt for full math, all methods, and design rationale.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of anomaly detection for a single update."""
    is_malicious: bool
    anomaly_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ClientProfile:
    """Historical profile for tracking a client's behavior."""
    client_id: str
    reputation: float = 1.0
    total_updates: int = 0
    flagged_updates: int = 0
    avg_gradient_norm: float = 0.0
    norm_history: List[float] = field(default_factory=list)
    last_update_time: float = 0.0
    consecutive_flags: int = 0


# ---------------------------------------------------------------------------
# Gradient Norm Detector
# ---------------------------------------------------------------------------

class GradientNormDetector:
    """
    Catches scaling and noise attacks by checking if a client's update
    norm deviates from the group. Uses z-score + IQR.
    """

    def __init__(self,
                 z_score_threshold: float = 2.5,
                 use_iqr: bool = True,
                 iqr_multiplier: float = 1.5):
        self.z_threshold = z_score_threshold
        self.use_iqr = use_iqr
        self.iqr_mult = iqr_multiplier

    @staticmethod
    def compute_gradient_norm(update: Dict[str, np.ndarray]) -> float:
        """L2 norm of the flattened update."""
        return np.sqrt(sum(np.sum(v ** 2) for v in update.values()))

    def detect(self,
               update: Dict[str, np.ndarray],
               reference_norms: List[float]) -> Tuple[bool, float, str]:
        """
        Check if this update's norm is anomalous vs the group.
        Returns (is_anomaly, score_0_to_1, reason_string).
        """
        norm = self.compute_gradient_norm(update)

        if len(reference_norms) < 3:
            # Too few peers ΓÇö use a conservative hard threshold
            if norm > 10.0:
                return True, 1.0, f"Norm {norm:.2f} is suspiciously large"
            return False, 0.0, ""

        norms = np.array(reference_norms)

        # Z-score method
        mean_n, std_n = np.mean(norms), np.std(norms) + 1e-8
        z = abs(norm - mean_n) / std_n
        z_flag = z > self.z_threshold

        # IQR method
        iqr_flag = False
        if self.use_iqr:
            q1, q3 = np.percentile(norms, 25), np.percentile(norms, 75)
            iqr = q3 - q1
            iqr_flag = norm < (q1 - self.iqr_mult * iqr) or norm > (q3 + self.iqr_mult * iqr)

        is_anomaly = z_flag or iqr_flag
        score = min(1.0, z / (2 * self.z_threshold))

        reason = ""
        if is_anomaly:
            reason = f"Norm {norm:.2f} deviates from mean {mean_n:.2f} (z={z:.2f})"

        return is_anomaly, score, reason


# ---------------------------------------------------------------------------
# Reputation System
# ---------------------------------------------------------------------------

class ReputationSystem:
    """
    Tracks client trust over time. Penalizes flagged clients, rewards
    good behavior, identifies persistent offenders.
    Formula: reputation_new = reputation_old * decay + reward
    """

    def __init__(self,
                 reputation_decay: float = 0.9,
                 min_reputation: float = 0.3,
                 flag_penalty: float = 0.2,
                 good_reward: float = 0.05,
                 consecutive_threshold: int = 3):
        self.decay = reputation_decay
        self.min_rep = min_reputation
        self.penalty = flag_penalty
        self.reward = good_reward
        self.consec_threshold = consecutive_threshold
        self.profiles: Dict[str, ClientProfile] = {}

    def get_or_create_profile(self, client_id: str) -> ClientProfile:
        if client_id not in self.profiles:
            self.profiles[client_id] = ClientProfile(client_id=client_id)
        return self.profiles[client_id]

    def update_reputation(self, client_id: str, was_flagged: bool,
                          gradient_norm: float = None) -> float:
        """Update reputation based on current round behavior."""
        profile = self.get_or_create_profile(client_id)

        profile.total_updates += 1
        if gradient_norm is not None:
            profile.norm_history.append(gradient_norm)
            if len(profile.norm_history) > 100:
                profile.norm_history = profile.norm_history[-100:]
            profile.avg_gradient_norm = np.mean(profile.norm_history)

        if was_flagged:
            profile.flagged_updates += 1
            profile.consecutive_flags += 1
            r = -self.penalty
        else:
            profile.consecutive_flags = 0
            r = self.reward

        profile.reputation = max(0.0, min(1.0, profile.reputation * self.decay + r))
        return profile.reputation

    def get_reputation(self, client_id: str) -> float:
        return self.get_or_create_profile(client_id).reputation

    def is_low_reputation(self, client_id: str) -> bool:
        return self.get_reputation(client_id) < self.min_rep

    def is_persistent_offender(self, client_id: str) -> bool:
        return self.get_or_create_profile(client_id).consecutive_flags >= self.consec_threshold

    def get_all_low_reputation_clients(self) -> List[str]:
        return [cid for cid, p in self.profiles.items() if p.reputation < self.min_rep]


# ---------------------------------------------------------------------------
# Anomaly Detector (Orchestrator)
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Combines GradientNormDetector + ReputationSystem.
    Pipeline: compute norms -> norm check -> reputation check -> flag decision.
    """

    def __init__(self,
                 norm_weight: float = 0.6,
                 reputation_weight: float = 0.4,
                 flag_threshold: float = 0.5,
                 **kwargs):
        # Accept but ignore extra kwargs for backward compatibility with server.py
        self.norm_detector = GradientNormDetector()
        self.reputation_system = ReputationSystem()
        self.weights = {'norm': norm_weight, 'reputation': reputation_weight}
        self.threshold = flag_threshold

    def detect_batch(self, updates, current_global_update=None) -> List[DetectionResult]:
        """
        Run detection on a batch of client updates.
        Returns a DetectionResult for each update.
        """
        if not updates:
            return []

        results = []

        # Compute all gradient norms
        norms = [self.norm_detector.compute_gradient_norm(u.update) for u in updates]

        # Evaluate each update
        for i, client_update in enumerate(updates):
            # Reference norms: all except current client
            ref_norms = norms[:i] + norms[i+1:]

            # Norm detection
            norm_flag, norm_score, norm_reason = self.norm_detector.detect(
                client_update.update, ref_norms
            )

            # Reputation check
            reputation = self.reputation_system.get_reputation(client_update.client_id)
            rep_flag = self.reputation_system.is_low_reputation(client_update.client_id)
            rep_score = 1.0 - reputation  # Lower rep = higher anomaly score

            # Weighted combination
            combined = (
                self.weights['norm'] * norm_score +
                self.weights['reputation'] * rep_score
            )

            # Collect reasons
            reasons = []
            if norm_reason:
                reasons.append(f"Norm: {norm_reason}")
            if rep_flag:
                reasons.append(f"Low reputation: {reputation:.2f}")

            # Final flag decision
            is_malicious = combined > self.threshold or norm_flag

            # Update reputation for next round
            self.reputation_system.update_reputation(
                client_update.client_id, is_malicious, norms[i]
            )

            results.append(DetectionResult(
                is_malicious=is_malicious,
                anomaly_scores={
                    'norm': norm_score,
                    'reputation': rep_score,
                    'combined': combined,
                },
                reasons=reasons,
                confidence=min(1.0, combined),
            ))

            if is_malicious:
                logger.warning(
                    f"Flagged {client_update.client_id}: "
                    f"score={combined:.2f}, reasons={reasons}"
                )

        n_flagged = sum(1 for r in results if r.is_malicious)
        logger.info(f"Anomaly Detection: {n_flagged}/{len(updates)} flagged")

        return results

    def get_reputation_scores(self) -> Dict[str, float]:
        """All client reputation scores."""
        return {cid: p.reputation for cid, p in self.reputation_system.profiles.items()}

    def get_malicious_clients(self) -> List[str]:
        """Clients with reputation below threshold."""
        return self.reputation_system.get_all_low_reputation_clients()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Anomaly Detection: GradientNorm + Reputation")
    print("=" * 55)

    from src.server.aggregation import ClientUpdate
    np.random.seed(42)

    updates = []

    # 10 honest clients
    for i in range(10):
        updates.append(ClientUpdate(
            client_id=f"honest_{i}",
            update={
                'layer1.weight': np.random.randn(10, 10) * 0.1 + 1.0,
                'layer1.bias': np.random.randn(10) * 0.1 + 0.1,
            },
            num_samples=100, version=0, timestamp=0.0, reputation=1.0,
        ))

    # 2 malicious clients (scaling attack)
    for i in range(2):
        updates.append(ClientUpdate(
            client_id=f"malicious_{i}",
            update={
                'layer1.weight': np.random.randn(10, 10) * 10.0,
                'layer1.bias': np.random.randn(10) * 10.0,
            },
            num_samples=100, version=0, timestamp=0.0, reputation=1.0,
        ))

    detector = AnomalyDetector()
    results = detector.detect_batch(updates)

    print("\nResults:")
    print("-" * 55)
    for u, r in zip(updates, results):
        status = "MALICIOUS" if r.is_malicious else "HONEST"
        print(f"  {u.client_id:15s}  {status:9s}  score={r.anomaly_scores['combined']:.3f}")
        if r.reasons:
            for reason in r.reasons:
                print(f"    -> {reason}")

    print("\nReputations:")
    print("-" * 55)
    for cid, rep in detector.get_reputation_scores().items():
        print(f"  {cid}: {rep:.2f}")
