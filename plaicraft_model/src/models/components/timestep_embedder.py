"""
Timestep embedding module for diffusion models.
"""

import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    """
    Embeds continuous timesteps into vector representations.
    
    Converts scalar timestep values (typically in [0, 1] for diffusion models)
    into high-dimensional embeddings using sinusoidal encoding followed by
    a 2-layer MLP.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Initialize timestep embedder.
        
        Args:
            hidden_size: Output embedding dimension
            frequency_embedding_size: Dimension of sinusoidal frequency embeddings
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0, max_timesteps: int = 1000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: 1-D Tensor of N timestep values in [0, 1]
            dim: Dimension of the output embeddings
            max_period: Controls the minimum frequency of the embeddings
            max_timesteps: Maximum timestep value to scale to (default: 1000)
                         Timesteps in [0, 1] are scaled to [0, max_timesteps] internally
        
        Returns:
            Tensor of shape [N, dim] containing positional embeddings
        """
        # Scale continuous timesteps [0, 1] to discrete range [0, max_timesteps]
        t_scaled = t * max_timesteps
        
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t_scaled[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps into vector representations.
        
        Args:
            t: Timestep tensor of shape [B] with float values (typically in [0, 1])
        
        Returns:
            Timestep embeddings of shape [B, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        first_linear = self.mlp[0]
        if t_freq.dtype != first_linear.weight.dtype or t_freq.device != first_linear.weight.device:
            t_freq = t_freq.to(device=first_linear.weight.device, dtype=first_linear.weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
