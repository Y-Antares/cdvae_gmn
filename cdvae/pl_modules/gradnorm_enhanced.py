"""
Enhanced GradNorm implementation for dynamic task weighting.
This module separates GradNorm from multi-objective optimization methods.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

class GradNorm(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    
    This implementation allows GradNorm to be used independently from other
    multi-objective optimization techniques like weighted sums, Tchebycheff method,
    or boundary crossing.
    
    Reference: https://proceedings.mlr.press/v80/chen18a.html
    """
    
    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        initial_task_weights: Optional[Union[List[float], torch.Tensor]] = None,
        enable: bool = True,
        lr: float = 0.025,
    ):
        """
        Initialize GradNorm module.
        
        Args:
            num_tasks: Number of tasks/losses to balance
            alpha: GradNorm asymmetry parameter (higher values → more aggressive balancing)
            initial_task_weights: Initial weights for each task (default: all 1.0)
            enable: Whether to use GradNorm (if False, weights remain constant)
            lr: Learning rate for task weights optimizer
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.enable = enable
        self.lr = lr
        
        # Initialize task weights
        if initial_task_weights is None:
            initial_task_weights = torch.ones(num_tasks)
        elif isinstance(initial_task_weights, list):
            initial_task_weights = torch.tensor(initial_task_weights)
            
        self.task_weights = nn.Parameter(initial_task_weights.float())
        
        # Initialize tracking variables
        self.initial_losses = {}
        self.last_step_losses = {}
        self.train_step = 0
        self.optimizer = None
    
    def setup_optimizer(self):
        """Create optimizer for task weights if not already created."""
        if self.optimizer is None and self.enable:
            self.optimizer = torch.optim.Adam([self.task_weights], lr=self.lr)
    
    def normalize_weights(self):
        """Normalize weights to sum to number of tasks."""
        if self.enable:
            with torch.no_grad():
                normalized_weights = self.task_weights * self.num_tasks / self.task_weights.sum()
                self.task_weights.copy_(normalized_weights)
    
    def compute_weighted_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply task weights to input losses.
        
        Args:
            losses: Dictionary mapping task names to loss tensors
            
        Returns:
            Tuple of (weighted_sum_loss, weighted_losses_dict)
        """
        weighted_losses = {}
        weighted_sum = 0.0
        
        # Get task names in order
        task_names = list(losses.keys())
        
        # Ensure we have the right number of tasks
        assert len(task_names) == self.num_tasks, f"Expected {self.num_tasks} tasks, got {len(task_names)}"
        
        # Apply weights to each loss
        for i, task_name in enumerate(task_names):
            weighted_loss = self.task_weights[i] * losses[task_name]
            weighted_losses[task_name] = weighted_loss
            weighted_sum += weighted_loss
            
        return weighted_sum, weighted_losses
    
    def compute_gradnorm_loss(
        self, 
        losses: Dict[str, torch.Tensor],
        shared_parameters: nn.Parameter
    ) -> torch.Tensor:
        """
        Compute GradNorm loss for updating task weights.
        
        Args:
            losses: Dictionary of task losses
            shared_parameters: Parameters of the shared representation layer
            
        Returns:
            GradNorm loss tensor
        """
        if not self.enable:
            return torch.tensor(0.0, device=self.task_weights.device)
        
        task_names = list(losses.keys())
        
        # Get weighted losses for each task
        _, weighted_losses = self.compute_weighted_losses(losses)
        
        # Compute gradient norms for each task
        grad_norms = []
        for task_name in task_names:
            weighted_loss = weighted_losses[task_name]
            grad = torch.autograd.grad(
                weighted_loss, 
                shared_parameters, 
                retain_graph=True, 
                create_graph=True
            )[0]
            grad_norm = torch.norm(grad)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        mean_norm = torch.mean(grad_norms)
        
        # Calculate loss ratios for inverse training rates
        loss_ratios = []
        for i, task_name in enumerate(task_names):
            current_loss = losses[task_name].detach()
            
            # Store initial loss values on first step
            if self.train_step == 0:
                self.initial_losses[task_name] = current_loss.item()
                loss_ratio = 1.0
            else:
                initial_loss = self.initial_losses.get(task_name, current_loss.item())
                loss_ratio = current_loss.item() / (initial_loss + 1e-8)
            
            loss_ratios.append(loss_ratio)
        
        # Calculate relative inverse training rates
        mean_loss_ratio = np.mean(loss_ratios)
        relative_inverse_rates = [(ratio / mean_loss_ratio) ** self.alpha 
                                  for ratio in loss_ratios]
        relative_inverse_rates = torch.tensor(
            relative_inverse_rates, 
            device=grad_norms.device
        )
        
        # Calculate GradNorm loss
        target_grad_norms = mean_norm * relative_inverse_rates
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))
        
        # Update tracking variables
        self.train_step += 1
        for task_name, loss in losses.items():
            self.last_step_losses[task_name] = loss.detach().item()
        
        return gradnorm_loss
    
    def update_weights(self, gradnorm_loss: torch.Tensor):
        """
        Update the task weights using the GradNorm loss.
        
        Args:
            gradnorm_loss: GradNorm loss tensor
        """
        if not self.enable or gradnorm_loss.item() == 0:
            return
            
        # Create optimizer if needed
        self.setup_optimizer()
        
        # Update weights
        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()
        
        # Normalize weights after update
        self.normalize_weights()

    def get_weights_dict(self, task_names: List[str]) -> Dict[str, float]:
        """
        Get task weights as a dictionary.
        
        Args:
            task_names: List of task names corresponding to weight indices
            
        Returns:
            Dictionary mapping task names to weight values
        """
        weights_dict = {}
        for i, task_name in enumerate(task_names):
            weights_dict[task_name] = self.task_weights[i].item()
        return weights_dict