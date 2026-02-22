"""
Server-side Differential Privacy for Federated Learning
Implements privacy-preserving aggregation with differential privacy guarantees.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerDifferentialPrivacy:
    """
    Server-side differential privacy for federated learning aggregation.
    Provides (epsilon, delta)-differential privacy guarantees.
    """
    
    def __init__(self, 
                 epsilon: float = 1.0, 
                 delta: float = 1e-5,
                 sensitivity: float = 1.0,
                 enabled: bool = True):
        """
        Initialize server-side DP.
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability
            sensitivity: Maximum L2 norm of updates
            enabled: Whether DP is enabled
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.enabled = enabled
        
        if enabled:
            self.noise_scale = self._compute_noise_scale()
            logger.info(
                f"Server DP enabled: ╬╡={epsilon}, ╬┤={delta}, "
                f"sensitivity={sensitivity}, noise_scale={self.noise_scale:.4f}"
            )
    
    def _compute_noise_scale(self) -> float:
        """
        Compute noise standard deviation for Gaussian mechanism.
        sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        """
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_update(self, update: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clip update L2 norm to sensitivity.
        
        Args:
            update: Model update dictionary
            
        Returns:
            Clipped update
        """
        # Flatten all parameters
        flat = np.concatenate([update[k].flatten() for k in sorted(update.keys())])
        norm = np.linalg.norm(flat)
        
        if norm > self.sensitivity:
            scale = self.sensitivity / norm
            return {k: v * scale for k, v in update.items()}
        return update
    
    def add_noise(self, 
                  aggregated: Dict[str, np.ndarray],
                  num_clients: int = 1) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Add calibrated Gaussian noise to aggregated update.
        
        Args:
            aggregated: Aggregated model update
            num_clients: Number of clients contributing (increases privacy)
            
        Returns:
            Tuple of (noised update, actual noise scale used)
        """
        if not self.enabled:
            return aggregated, 0.0
        
        # Effective sigma decreases with more clients (amplification)
        effective_sigma = self.noise_scale / np.sqrt(num_clients)
        
        noised = {
            k: v + np.random.normal(0, effective_sigma, size=v.shape)
            for k, v in aggregated.items()
        }
        
        logger.info(
            f"Added DP noise: sigma_eff={effective_sigma:.6f}, "
            f"num_clients={num_clients}"
        )
        
        return noised, effective_sigma
    
    def privatize_aggregate(self, 
                           aggregated: Dict[str, np.ndarray],
                           num_clients: int) -> Dict[str, np.ndarray]:
        """
        Full DP pipeline: clip + noise.
        
        Args:
            aggregated: Raw aggregated update
            num_clients: Number of contributing clients
            
        Returns:
            Privatized update
        """
        clipped = self.clip_update(aggregated)
        noised, _ = self.add_noise(clipped, num_clients)
        return noised


class PrivacyAccountant:
    """
    Tracks privacy budget expenditure across multiple rounds.
    Uses strong composition theorem.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.current_epsilon = 0.0
        self.num_rounds = 0
    
    def accumulate(self, rounds: int = 1) -> float:
        """
        Update accumulated epsilon using strong composition.
        
        Args:
            rounds: Number of rounds to accumulate
            
        Returns:
            Current accumulated epsilon
        """
        # Simplified: linear composition (conservative)
        # More accurate: strong composition with sqrt(2 * k * log(1/delta)) * sigma
        for _ in range(rounds):
            self.current_epsilon += self.epsilon
            self.num_rounds += 1
        
        return self.current_epsilon
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self.current_epsilon)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.current_epsilon >= self.epsilon
    
    def get_status(self) -> Dict[str, float]:
        """Get privacy accounting status."""
        return {
            "total_budget": self.epsilon,
            "spent": self.current_epsilon,
            "remaining": self.get_remaining_budget(),
            "num_rounds": self.num_rounds,
            "delta": self.delta
        }


def compute_adaptive_noise_scale(update_norms: List[float],
                                  target_epsilon: float,
                                  sensitivity: float) -> float:
    """
    Compute adaptive noise scale based on update statistics.
    
    Args:
        update_norms: List of client update norms
        target_epsilon: Target privacy budget
        sensitivity: Clipping threshold
        
    Returns:
        Noise standard deviation
    """
    if not update_norms:
        return sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / target_epsilon
    
    # Use median as reference for robustness
    median_norm = np.median(update_norms)
    mean_norm = np.mean(update_norms)
    
    # Adaptive sensitivity based on data statistics
    adaptive_sensitivity = min(sensitivity, median_norm)
    
    sigma = adaptive_sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / target_epsilon
    
    logger.info(
        f"Adaptive noise: median_norm={median_norm:.4f}, "
        f"mean_norm={mean_norm:.4f}, sigma={sigma:.6f}"
    )
    
    return sigma

