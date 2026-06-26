import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models.components.recurrent_encoders.base_recurrent_encoder import BaseRecurrentEncoder

_LOG_EXP_MIN = -60.0
_LOG_EXP_MAX =  20.0


def parallel_scan_log_fp32(log_coeffs, log_values):
    lc32 = log_coeffs.float()
    lv32 = log_values.float()
    a_star = F.pad(torch.cumsum(lc32, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(lv32 - a_star, dim=1)
    s = a_star[:, 1:] + log_h0_plus_b_star[:, 1:]
    return torch.exp(s.clamp(_LOG_EXP_MIN, _LOG_EXP_MAX))

def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    h = torch.exp(a_star[:, 1:] + log_h0_plus_b_star[:, 1:])
    return h

def g(x):
    return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

def inv_g(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    origd = y.dtype
    y32 = y.float()
    y_ge = (y32 >= 0.5)
    y_clamped = y32.clamp(min=eps, max=1.0 - eps)
    a_lin   = y32 - 0.5
    a_logit = torch.log(y_clamped) - torch.log1p(-y_clamped)
    out32 = torch.where(y_ge, a_lin, a_logit)
    return out32.to(origd)

def log_g(x: torch.Tensor) -> torch.Tensor:
    origd = x.dtype
    x32 = x.float()
    out32 = torch.where(x32 >= 0,
                        (F.relu(x32) + 0.5).log(),
                        -F.softplus(-x32))
    return out32.to(origd)

class MinGRUCell(nn.Module):
    def __init__(self, units, input_shape):
        super(MinGRUCell, self).__init__()
        self.units = units
        self.input_shape = input_shape

        self.linear_z = nn.Linear(self.input_shape, self.units)
        self.linear_h = nn.Linear(self.input_shape, self.units)

    def forward(self, x, h_0):
        """
        x: (B, L, input_size)
        h_0: (B, 1, hidden_size)
        """
        # keep the outer world in bf16, but do the fragile math in fp32
        origd = x.dtype      # typically torch.bfloat16 under DS bf16

        # upcast inputs to fp32
        x32  = x.float()
        h032 = h_0.float()

        # force both linears to *compute* in fp32 (even if params are bf16)
        k32 = F.linear(x32, self.linear_z.weight.float(),
                                   self.linear_z.bias.float() if self.linear_z.bias is not None else None)
        tilde_in32 = F.linear(x32, self.linear_h.weight.float(),
                                   self.linear_h.bias.float() if self.linear_h.bias is not None else None)

        # stable logs in fp32
        log_z = -F.softplus(-k32, threshold=20)    # log(sigmoid(k))
        log_coeffs = -F.softplus( k32, threshold=20)    # log(1 - sigmoid(k))
        log_h0 = log_g(h032)
        log_tilde_h = log_g(tilde_in32)

        # scan in fp32, clamp exponent to avoid INF
        h32 = parallel_scan_log_fp32(
            log_coeffs,
            torch.cat([log_h0, log_z + log_tilde_h], dim=1),
        )

        # hand back bf16 so the rest of the network stays bf16
        return h32.to(origd)

    def sequential(self, x, h_0):
        """
        x: (batch_size, seq_len, input_size)
        h_0: (batch_size, 1, hidden_size)
        """
        h = []
        h_prev = g(h_0)
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = g(self.linear_h(x))
        for i in range(0, h_tilde.shape[1]):
            h_prev = (1 - z[:, i:i+1]) * h_prev + z[:, i:i+1] * h_tilde[:, i:i+1]
            h.append(h_prev)
        h = torch.cat(h, dim=1)
        return h


class TransformerLikeGRUBlock(BaseRecurrentEncoder):
    def __init__(self, feature_dim, num_heads=4, mlp_hidden_dim_multiplier=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = self.feature_dim * mlp_hidden_dim_multiplier
        self.head_dim = self.feature_dim // self.num_heads

        self.proj_in = nn.Linear(self.feature_dim, self.feature_dim)
        self.rnn = MinGRUCell(self.head_dim, self.head_dim)
        self.proj_out = nn.Linear(self.feature_dim, self.feature_dim)

        self.ln1 = nn.LayerNorm(self.feature_dim)
        self.ln2 = nn.LayerNorm(self.feature_dim)

        self.post_rnn_MLP = nn.Sequential(
            nn.Linear(self.feature_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.feature_dim),
        )
    
    def get_initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get the initial recurrent state (h_0) for minGRU.
        Shape: (batch_size, 1, feature_dim)
        """
        return torch.zeros(batch_size, 1, self.feature_dim, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        gradient_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the recurrent block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            initial_state: Optional initial state h_0. If None, uses zeros.
            gradient_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, feature_dim)
            final_state: Final recurrent state of shape (batch_size, 1, feature_dim)
        """
        N, L, D = x.shape
        
        if initial_state is None:
            initial_state = self.get_initial_state(N, x.device, x.dtype)
        
        h_0 = initial_state

        # Empty sequence can occur when upstream context slicing/downsampling
        # removes all LTM timesteps. Keep state unchanged in that case.
        if L == 0:
            return x, h_0

        def _block(x_in, h0_in):
            rnn_input = self.ln1(x_in)
            proj_in = self.proj_in(rnn_input)                                 # [N,L,D]
            proj_in = proj_in.reshape(N, L, self.num_heads, self.head_dim)    \
                             .permute(0, 2, 1, 3).reshape(-1, L, self.head_dim)
            h0 = h0_in.reshape(N, 1, self.num_heads, self.head_dim)           \
                     .permute(0, 2, 1, 3).reshape(-1, 1, self.head_dim)

            rnn_states = self.rnn(proj_in, h0)                                 # [-1, L, head_dim]
            rnn_states = rnn_states.reshape(N, self.num_heads, L, self.head_dim) \
                                     .permute(0, 2, 1, 3).reshape(N, L, D)

            y = self.proj_out(rnn_states)
            x_skip = x_in + y
            mlp_in = self.ln2(x_skip)
            out = self.post_rnn_MLP(mlp_in)
            out = x_skip + out
            
            # Return the last block output state as final state so output-path
            # parameters (proj_out/ln2/post_rnn_MLP) remain on the training graph.
            final_state = out[:, -1:, :]  # shape: (batch_size, 1, feature_dim)
            return out, final_state

        if gradient_checkpoint:
            out, final_state = torch.utils.checkpoint.checkpoint(_block, x, h_0, use_reentrant=False)
        else:
            out, final_state = _block(x, h_0)

        return out, final_state


class MinGRUEncoder(BaseRecurrentEncoder):
    """
    Multi-layer minGRU encoder with internal layer management.
    
    This encoder manages multiple TransformerLikeGRUBlock layers internally,
    following the same pattern as xLSTMEncoder.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_multiplier: int = 4,
    ):
        """
        Initialize MinGRU encoder with multiple layers.
        
        Args:
            embedding_dim: Dimension of the embeddings (feature_dim)
            num_layers: Number of minGRU blocks/layers
            num_heads: Number of attention heads
            mlp_multiplier: Multiplier for MLP hidden dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Create multiple minGRU blocks
        self.layers = nn.ModuleList([
            TransformerLikeGRUBlock(
                feature_dim=embedding_dim,
                num_heads=num_heads,
                mlp_hidden_dim_multiplier=mlp_multiplier,
            )
            for _ in range(num_layers)
        ])
    
    def get_initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Get the initial recurrent states for all layers.
        
        Args:
            batch_size: The batch size
            device: The device to create the tensor on
            dtype: The dtype for the tensor
            
        Returns:
            Stacked initial states tensor of shape (num_layers, batch_size, 1, embedding_dim)
        """
        states = []
        for layer in self.layers:
            state = layer.get_initial_state(batch_size, device, dtype)
            states.append(state)
        return torch.stack(states, dim=0)
    
    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        gradient_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all minGRU layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            initial_state: Optional initial states from all layers
                Shape: (num_layers, batch_size, 1, embedding_dim)
            gradient_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, embedding_dim)
            final_states: Final states from all layers for caching
                Shape: (num_layers, batch_size, 1, embedding_dim)
        """
        batch_size = x.shape[0]
        
        if initial_state is None:
            initial_state = self.get_initial_state(batch_size, x.device, x.dtype)
        
        final_states = []
        output = x
        
        for i, layer in enumerate(self.layers):
            layer_initial_state = initial_state[i] if initial_state is not None else None
            output, state = layer(output, initial_state=layer_initial_state, gradient_checkpoint=gradient_checkpoint)
            final_states.append(state)
        
        # Stack all final states
        final_states = torch.stack(final_states, dim=0)
        
        return output, final_states