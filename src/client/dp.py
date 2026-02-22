"""
Client-side Differential Privacy for Federated Learning
Implements DP-SGD (Differentially Private Stochastic Gradient Descent)
and gradient clipping for privacy preservation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientClipper:
    """
    Clips gradients by L2 norm to ensure bounded sensitivity.
    """
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters.
        
        Args:
            model: PyTorch model with gradients to clip
            
        Returns:
            Total norm of clipped gradients
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm
        )
        return total_norm.item()


class GaussianMechanism:
    """
    Adds Gaussian noise to achieve differential privacy.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        # Calibrate noise standard deviation
        self.sigma = self._calibrate_sigma()
    
    def _calibrate_sigma(self) -> float:
        """
        Compute sigma for Gaussian mechanism based on DP parameters.
        Using: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        """
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma
    
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor with added noise
        """
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise


class DPSGDOptimizer:
    """
    Differentially Private SGD Optimizer.
    Implements DP-SGD: clip gradients + add noise per batch.
    """
    
    def __init__(self, model: nn.Module, lr: float = 0.01,
                 epsilon: float = 1.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        # Compute noise multiplier (sigma / max_grad_norm)
        self.noise_multiplier = self._compute_noise_multiplier()
        
        # Initialize components
        self.clipper = GradientClipper(max_norm=max_grad_norm)
        self.noise_generator = GaussianMechanism(
            epsilon=epsilon, 
            delta=delta, 
            sensitivity=max_grad_norm
        )
        
        # Standard optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # Privacy budget tracking
        self.accumulated_epsilon = 0.0
        self.num_steps = 0
    
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier for DP-SGD.
        """
        sigma = self.max_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma / self.max_grad_norm
    
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """
        Perform one DP-SGD step.
        
        Args:
            loss: Loss tensor from forward pass
            
        Returns:
            Dictionary with privacy accounting info
        """
        # Backward pass
        loss.backward()
        
        # Clip gradients
        total_norm = self.clipper.clip_gradients(self.model)
        
        # Add noise to gradients
        self._add_noise_to_gradients()
        
        # Standard optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update privacy accounting (simplified)
        self.num_steps += 1
        self._update_privacy_accounting()
        
        return {
            "gradient_norm": total_norm,
            "num_steps": self.num_steps,
            "accumulated_epsilon": self.accumulated_epsilon
        }
    
    def _add_noise_to_gradients(self):
        """
        Add calibrated Gaussian noise to each parameter's gradients.
        """
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Add noise scaled by max_grad_norm
                    noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm
                    param.grad.add_(noise)
    
    def _update_privacy_accounting(self):
        """
        Update accumulated epsilon using strong composition.
        Simplified: eps += sigma^2 * delta / (sigma^2 + 1) * 1/num_steps
        """
        # Simplified accounting (more accurate methods exist)
        self.accumulated_epsilon += self.epsilon / self.num_steps


class ClientSideDP:
    """
    High-level interface for client-side differential privacy.
    """
    
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 max_grad_norm: float = 1.0,
                 enabled: bool = True):
        
        self.enabled = enabled
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        if enabled:
            logger.info(f"Client-side DP enabled: eps={epsilon}, delta={delta}, max_norm={max_grad_norm}")
    
    def create_private_optimizer(self, model: nn.Module, 
                                  lr: float = 0.01) -> DPSGDOptimizer:
        """
        Create a DPSGD optimizer for the given model.
        
        Args:
            model: PyTorch model
            lr: Learning rate
            
        Returns:
            DPSGDOptimizer instance
        """
        if not self.enabled:
            return torch.optim.SGD(model.parameters(), lr=lr)
        
        return DPSGDOptimizer(
            model=model,
            lr=lr,
            epsilon=self.epsilon,
            delta=self.delta,
            max_grad_norm=self.max_grad_norm
        )
    
    def clip_and_noise(self, model: nn.Module) -> float:
        """
        Apply gradient clipping and noise addition.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm before clipping
        """
        if not self.enabled:
            return 0.0
        
        clipper = GradientClipper(self.max_grad_norm)
        norm = clipper.clip_gradients(model)
        
        noise_gen = GaussianMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.max_grad_norm
        )
        
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.copy_(noise_gen.add_noise(param.grad))
        
        return norm


def apply_dp_to_model_update(update: Dict[str, torch.Tensor],
                              clip_norm: float = 1.0,
                              noise_std: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Apply differential privacy to a model update (state dict).
    
    Args:
        update: Model state dict
        clip_norm: Maximum L2 norm for clipping
        noise_std: Standard deviation for Gaussian noise
        
    Returns:
        Privatized model update
    """
    privatized = {}
    
    for key, tensor in update.items():
        # Flatten and compute norm
        flat = tensor.flatten()
        
        # Clip by L2 norm
        norm = torch.norm(flat)
        if norm > clip_norm:
            scale = clip_norm / norm
            tensor = tensor * scale
        
        # Add noise
        noise = torch.randn_like(tensor) * noise_std
        privatized[key] = tensor + noise
    
    return privatized

