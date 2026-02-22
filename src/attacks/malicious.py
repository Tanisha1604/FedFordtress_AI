import torch
import random


def noise_injection(state_dict, noise_std=0.1):
    """
    Add Gaussian noise to model weights
    """
    attacked = {}

    for key in state_dict:
        noise = torch.randn_like(state_dict[key]) * noise_std
        attacked[key] = state_dict[key] + noise

    return attacked


def weight_scaling(state_dict, scale_factor=5.0):
    """
    Multiply weights by large constant
    """
    attacked = {}

    for key in state_dict:
        attacked[key] = state_dict[key] * scale_factor

    return attacked


def random_weights(state_dict):
    """
    Replace weights with completely random values
    """
    attacked = {}

    for key in state_dict:
        attacked[key] = torch.randn_like(state_dict[key])

    return attacked


def label_flipping(targets, num_classes=10):
    """
    Flip labels randomly (0ΓåÆ9, 1ΓåÆ8 etc.)
    """
    flipped = (num_classes - 1) - targets
    return flipped
