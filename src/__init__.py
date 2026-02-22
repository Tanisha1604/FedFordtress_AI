"""
FedFortress - Federated Learning Defense Platform
A secure distributed learning environment with attack detection and privacy preservation.
"""

__version__ = "1.0.0"

# Main components
from src.models import SimpleCNN, SimpleMLP
from src.client import Client
from src.data import load_cifar10, iid_split, create_dataloaders

__all__ = [
    'SimpleCNN',
    'SimpleMLP', 
    'Client',
    'load_cifar10',
    'iid_split',
    'create_dataloaders'
]

