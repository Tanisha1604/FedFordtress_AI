"""
Data Partitioning Utilities for Federated Learning
Provides IID and Non-IID data splitting strategies.
"""

import numpy as np
from typing import List, Tuple
from torch.utils.data import Subset, DataLoader
import torch


def iid_split(dataset, num_clients: int) -> List[Subset]:
    """
    Perform IID (Independent and Identically Distributed) data split.
    Each client receives an equal portion of randomly sampled data.
    
    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of clients for federated learning
        
    Returns:
        List of Subset objects, one per client
    """
    num_items = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    
    client_subsets = []
    
    for i in range(num_clients):
        start = i * num_items
        end = start + num_items
        # Handle remainder items
        if i == num_clients - 1:
            end = len(dataset)
        subset = Subset(dataset, indices[start:end])
        client_subsets.append(subset)
    
    return client_subsets


def non_iid_split(dataset, num_clients: int, classes_per_client: int = 2,
                  shards_per_client: int = 2) -> List[Subset]:
    """
    Perform Non-IID data split using shard-based distribution.
    Each client receives data from a limited number of classes.
    
    Args:
        dataset: PyTorch dataset with 'targets' attribute
        num_clients: Number of clients
        classes_per_client: Number of distinct classes per client
        shards_per_client: Number of shards per client
        
    Returns:
        List of Subset objects, one per client
    """
    # Get targets from dataset
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'data'):
        # For CIFAR-10, targets might be in different format
        if hasattr(dataset, 'train_labels'):
            targets = np.array(dataset.train_labels)
        else:
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(targets))
    num_shards = num_clients * shards_per_client
    shard_size = len(dataset) // num_shards
    
    # Sort indices by class
    sorted_indices = np.argsort(targets)
    
    # Divide into shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else len(dataset)
        shards.append(sorted_indices[start:end])
    
    # Assign shards to clients
    client_subsets = []
    np.random.shuffle(shards)
    
    for i in range(num_clients):
        client_indices = []
        for j in range(shards_per_client):
            shard_idx = (i + j * num_clients) % len(shards)
            client_indices.extend(shards[shard_idx])
        
        client_subsets.append(Subset(dataset, client_indices))
    
    return client_subsets


def pathological_non_iid_split(dataset, num_clients: int, 
                                classes_per_client: int = 2) -> List[Subset]:
    """
    Perform pathological Non-IID split.
    Each client only has data from a specific subset of classes.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        classes_per_client: Number of classes per client (typically 1 or 2)
        
    Returns:
        List of Subset objects
    """
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(targets))
    
    # Create class indices mapping
    class_indices = {}
    for cls in range(num_classes):
        class_indices[cls] = np.where(targets == cls)[0]
    
    # Assign classes to clients
    client_subsets = []
    classes_per_shard = num_classes // num_clients
    
    for i in range(num_clients):
        client_indices = []
        
        # Assign specific classes to this client
        start_cls = (i * classes_per_shard) % num_classes
        for j in range(classes_per_client):
            cls = (start_cls + j) % num_classes
            cls_idx = class_indices[cls]
            # Sample a portion of this class
            selected = np.random.choice(cls_idx, 
                                        size=len(cls_idx) // num_clients,
                                        replace=False)
            client_indices.extend(selected)
        
        client_subsets.append(Subset(dataset, client_indices))
    
    return client_subsets


def create_dataloaders(subsets, batch_size: int = 32, 
                       shuffle: bool = True) -> List[DataLoader]:
    """
    Create DataLoaders from dataset subsets.
    
    Args:
        subsets: List of dataset subsets
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        
    Returns:
        List of DataLoader objects
    """
    loaders = []
    
    for subset in subsets:
        loader = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        loaders.append(loader)
    
    return loaders


def get_data_distribution(subsets) -> dict:
    """
    Get the class distribution for each client subset.
    
    Args:
        subsets: List of dataset subsets
        
    Returns:
        Dictionary mapping client_id to class distribution
    """
    distribution = {}
    
    for i, subset in enumerate(subsets):
        if hasattr(subset.dataset, 'targets'):
            labels = [subset.dataset.targets[idx] for idx in subset.indices]
        else:
            labels = [subset.dataset[idx][1] for idx in subset.indices]
        
        unique, counts = np.unique(labels, return_counts=True)
        distribution[f"client_{i}"] = dict(zip(unique, counts.tolist()))
    
    return distribution


def print_data_distribution(subsets):
    """
    Print the class distribution for each client subset.
    
    Args:
        subsets: List of dataset subsets
    """
    for i, subset in enumerate(subsets):
        if hasattr(subset.dataset, 'targets'):
            labels = [subset.dataset.targets[idx] for idx in subset.indices]
        else:
            labels = [subset.dataset[idx][1] for idx in subset.indices]
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {i} distribution:")
        print(dict(zip(unique, counts)))
        print("-" * 40)

