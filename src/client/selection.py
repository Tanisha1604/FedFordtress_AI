"""
Client Selection Strategies for Federated Learning
Implements various client selection algorithms for improved efficiency and privacy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientSelector:
    """
    Base class for client selection strategies.
    """
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
    
    def select(self, available_clients: List[int]) -> List[int]:
        """
        Select clients from available pool.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            List of selected client IDs
        """
        raise NotImplementedError("Subclasses must implement select()")


class RandomSelector(ClientSelector):
    """
    Random client selection - baseline strategy.
    """
    
    def __init__(self, num_clients: int, selection_ratio: float = 1.0):
        super().__init__(num_clients)
        self.selection_ratio = selection_ratio
    
    def select(self, available_clients: List[int]) -> List[int]:
        """
        Randomly select clients.
        
        Args:
            available_clients: Available client IDs
            
        Returns:
            Randomly selected clients
        """
        num_to_select = max(1, int(len(available_clients) * self.selection_ratio))
        
        if num_to_select >= len(available_clients):
            return available_clients
        
        selected = np.random.choice(available_clients, size=num_to_select, replace=False)
        return selected.tolist()


class BanditSelector(ClientSelector):
    """
    Multi-armed bandit based client selection.
    Balances exploration (random) and exploitation (best performing).
    """
    
    def __init__(self, num_clients: int, epsilon: float = 0.1):
        super().__init__(num_clients)
        self.epsilon = epsilon  # Exploration rate
        self.scores = {i: 0.0 for i in range(num_clients)}
        self.selection_counts = {i: 0 for i in range(num_clients)}
    
    def update_score(self, client_id: int, reward: float):
        """
        Update the score for a client based on observed reward.
        
        Args:
            client_id: Client ID
            reward: Observed reward (e.g., accuracy improvement)
        """
        self.scores[client_id] = reward
        self.selection_counts[client_id] += 1
    
    def select(self, available_clients: List[int]) -> List[int]:
        """
        Select clients using epsilon-greedy strategy.
        
        Args:
            available_clients: Available client IDs
            
        Returns:
            Selected clients
        """
        if np.random.random() < self.epsilon:
            # Exploration: random selection
            return np.random.choice(available_clients, 
                                   size=min(3, len(available_clients)),
                                   replace=False).tolist()
        else:
            # Exploitation: select best performing clients
            sorted_clients = sorted(available_clients, 
                                   key=lambda x: self.scores.get(x, 0),
                                   reverse=True)
            return sorted_clients[:min(3, len(sorted_clients))]


class ImportanceSelector(ClientSelector):
    """
    Importance-based client selection using data quality metrics.
    """
    
    def __init__(self, num_clients: int):
        super().__init__(num_clients)
        self.importance_scores = {i: 1.0 for i in range(num_clients)}
    
    def update_importance(self, client_id: int, importance: float):
        """
        Update importance score for a client.
        
        Args:
            client_id: Client ID
            importance: Importance metric (e.g., data size, data quality)
        """
        self.importance_scores[client_id] = importance
    
    def select(self, available_clients: List[int], 
               num_to_select: Optional[int] = None) -> List[int]:
        """
        Select clients based on importance scores.
        
        Args:
            available_clients: Available client IDs
            num_to_select: Number of clients to select (default: all)
            
        Returns:
            Selected clients
        """
        if num_to_select is None:
            num_to_select = len(available_clients)
        
        # Sort by importance score
        sorted_clients = sorted(
            available_clients,
            key=lambda x: self.importance_scores.get(x, 1.0),
            reverse=True
        )
        
        return sorted_clients[:num_to_select]


class FederatedClientSelector(ClientSelector):
    """
    Advanced client selector with reputation and staleness awareness.
    """
    
    def __init__(self, num_clients: int):
        super().__init__(num_clients)
        self.reputation = {i: 1.0 for i in range(num_clients)}
        self.staleness = {i: 0 for i in range(num_clients)}
    
    def update_reputation(self, client_id: int, reputation: float):
        """Update client reputation score."""
        self.reputation[client_id] = max(0.1, min(1.0, reputation))
    
    def update_staleness(self, client_id: int, version: int, current_version: int):
        """Update client staleness."""
        self.staleness[client_id] = max(0, current_version - version)
    
    def select(self, available_clients: List[int],
               num_to_select: int = 3) -> List[int]:
        """
        Select clients based on reputation and staleness.
        
        Args:
            available_clients: Available client IDs
            num_to_select: Number of clients to select
            
        Returns:
            Selected clients
        """
        scores = {}
        
        for client_id in available_clients:
            # Score = reputation / (1 + staleness)
            rep = self.reputation.get(client_id, 1.0)
            stal = self.staleness.get(client_id, 0)
            scores[client_id] = rep / (1 + 0.5 * stal)
        
        # Select top clients by score
        sorted_clients = sorted(available_clients, 
                                key=lambda x: scores.get(x, 0),
                                reverse=True)
        
        return sorted_clients[:num_to_select]


def select_clients_by_resource_availability(
    client_resources: Dict[int, Dict[str, float]],
    min_bandwidth: float = 1.0,
    min_compute: float = 1.0
) -> List[int]:
    """
    Select clients based on available computational resources.
    
    Args:
        client_resources: Dict mapping client_id to resource metrics
        min_bandwidth: Minimum bandwidth requirement (Mbps)
        min_compute: Minimum compute capability score
        
    Returns:
        List of eligible client IDs
    """
    eligible = []
    
    for client_id, resources in client_resources.items():
        bandwidth = resources.get('bandwidth', float('inf'))
        compute = resources.get('compute', float('inf'))
        
        if bandwidth >= min_bandwidth and compute >= min_compute:
            eligible.append(client_id)
    
    return eligible


def select_clients_by_data_quality(
    client_data_stats: Dict[int, Dict[str, float]],
    quality_threshold: float = 0.5
) -> List[int]:
    """
    Select clients based on data quality metrics.
    
    Args:
        client_data_stats: Dict mapping client_id to data quality metrics
        quality_threshold: Minimum quality score threshold
        
    Returns:
        List of eligible client IDs
    """
    eligible = []
    
    for client_id, stats in client_data_stats.items():
        quality = stats.get('quality', 1.0)
        
        if quality >= quality_threshold:
            eligible.append(client_id)
    
    return eligible

