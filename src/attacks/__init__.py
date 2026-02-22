"""
FedFortress Attacks Package
Contains implementations of various malicious attacks for federated learning.
"""

from src.attacks.malicious import (
    noise_injection,
    weight_scaling,
    random_weights,
    label_flipping
)

__all__ = [
    'noise_injection',
    'weight_scaling',
    'random_weights',
    'label_flipping'
]

