"""Base interface for recurrent encoders with state caching support."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseRecurrentEncoder(nn.Module, ABC):
    """
    Abstract base class for recurrent encoders with state caching capability.
    
    This interface supports various types of recurrent encoders (minGRU, xLSTM, etc.)
    and provides a unified way to:
    - Forward pass through the model
    - Get initial states for batch processing
    - Store and cache recurrent states for inference
    """
    
    @abstractmethod
    def get_initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get the initial recurrent state for the given batch size.
        
        Args:
            batch_size: The batch size
            device: The device to create the tensor on
            dtype: The dtype for the tensor
            
        Returns:
            Initial state tensor of shape (batch_size, ...) depending on the encoder type
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        gradient_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the recurrent encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            initial_state: Optional initial state tensor. If None, will use get_initial_state()
            gradient_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, feature_dim)
            final_state: Final recurrent state tensor for caching
        """
        pass
    
    def cache_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Prepare the recurrent state for caching. Can be overridden by subclasses
        if special processing is needed.
        
        Args:
            state: The recurrent state tensor
            
        Returns:
            State tensor ready for caching/storage
        """
        return state.detach().clone()
    
    def restore_state(self, cached_state: torch.Tensor) -> torch.Tensor:
        """
        Restore a cached recurrent state for use in the next forward pass.
        Can be overridden by subclasses if special processing is needed.
        
        Args:
            cached_state: The cached state tensor
            
        Returns:
            State tensor ready for the next forward pass
        """
        return cached_state
