"""SDPA and FlexAttention backend dispatch tests for training-like inputs."""
import sys
import warnings
from pathlib import Path

# Dynamically find the 'src' directory relative to this test file and add it to the path
src_dir = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(src_dir))

from itertools import product
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange

from models.components.attention import Attention
from models.components.attention_masks import (
    create_token_level_mask,
    create_dataframe_level_mask,
)

MODALITY_LAYOUT: List[Tuple[str, int]] = [
    ("video", 2),
    ("audio_hear", 3),
    ("key_press", 1),
]

def _build_training_like_target_tokens(
    batch_size: int, num_timesteps: int, embed_dim: int,
    modality_layout: List[Tuple[str, int]], device: torch.device, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[int, int]], List[str]]:
    by_modality_tokens, by_modality_times = {}, {}
    modality_shapes, active_modality_names = {}, []

    for modality_idx, (name, tokens_per_step) in enumerate(modality_layout):
        tokens = torch.randn(batch_size, num_timesteps, tokens_per_step, embed_dim, device=device, dtype=dtype)
        t = torch.arange(num_timesteps, device=device, dtype=torch.float32)
        t = t.view(1, num_timesteps, 1).expand(batch_size, -1, tokens_per_step)
        intra = torch.arange(tokens_per_step, device=device, dtype=torch.float32)
        intra = intra.view(1, 1, tokens_per_step).expand(batch_size, num_timesteps, -1)
        times = t + (0.01 * modality_idx) + (intra / max(tokens_per_step, 1)) * 1e-3

        by_modality_tokens[name] = tokens
        by_modality_times[name] = times
        modality_shapes[name] = (num_timesteps, tokens_per_step)
        active_modality_names.append(name)

    flat_tokens, flat_times = [], []
    for t in range(num_timesteps):
        for name in active_modality_names:
            flat_tokens.append(by_modality_tokens[name][:, t, :, :])
            flat_times.append(by_modality_times[name][:, t, :])

    return torch.cat(flat_tokens, dim=1), torch.cat(flat_times, dim=1), modality_shapes, active_modality_names

def _build_target_key_padding_mask(
    frame_valid: torch.Tensor, active_modality_names: List[str], modality_shapes: Dict[str, Tuple[int, int]],
) -> torch.Tensor:
    chunks = []
    num_timesteps = int(modality_shapes[active_modality_names[0]][0])
    for t in range(num_timesteps):
        valid_t = frame_valid[:, t:t + 1]
        for name in active_modality_names:
            chunks.append(torch.repeat_interleave(valid_t, repeats=modality_shapes[name][1], dim=1))
    return torch.cat(chunks, dim=1)

def _build_history_like_context(
    batch_size: int, seq_len_k: int, embed_dim: int, device: torch.device, dtype: torch.dtype, frame_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    history_tokens = torch.randn(batch_size, seq_len_k, embed_dim, device=device, dtype=dtype)
    ltm_k = min(4, max(1, seq_len_k // 2))
    stm_len = max(0, seq_len_k - ltm_k)
    ltm_valid = torch.ones(batch_size, ltm_k, dtype=torch.bool, device=device)

    if stm_len == 0: return history_tokens, ltm_valid

    stm_context_len = min(3, frame_valid.shape[1])
    stm_frame_valid = frame_valid[:, -stm_context_len:]
    repeats = max(1, (stm_len + stm_context_len - 1) // stm_context_len)
    stm_valid_flat = torch.repeat_interleave(stm_frame_valid, repeats=repeats, dim=1)[:, :stm_len]
    return history_tokens, torch.cat([ltm_valid, stm_valid_flat], dim=1)

def _build_attn_mask(
    mask_type: str, target_time: torch.Tensor, active_modality_names: List[str],
    modality_shapes: Dict[str, Tuple[int, int]], device: torch.device,
) -> Optional[torch.Tensor]:
    if mask_type == "token_level": return create_token_level_mask(timestamps=target_time[0], device=device)
    if mask_type == "dataframe_level":
        layout = [(name, int(modality_shapes[name][1])) for name in active_modality_names]
        return create_dataframe_level_mask(
            modality_layout=layout,
            num_timesteps=int(modality_shapes[active_modality_names[0]][0]),
            device=device,
        )
    if mask_type == "no_mask": return None
    raise ValueError(f"Unknown mask type: {mask_type}")

def _run_attention_with_flash_only(attn, q_input, k_input, v_input, attn_mask, key_padding_mask) -> str:
    try:
        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False, enable_cudnn=False):
                _ = attn(q_input, k_input, v_input, attn_mask, key_padding_mask)
        torch.cuda.synchronize()
        return "YES"
    except RuntimeError as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ["no available kernel", "no viable backend", "no suitable kernel", "not supported"]):
            return "NO"
        return f"ERR: {type(exc).__name__}"

def _run_flex_attention(attn, q_input, k_input, v_input, q_time, kv_time, mask_type, key_padding_mask) -> str:
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except ImportError:
        return "N/A (PyTorch < 2.3)"

    try:
        with torch.no_grad():
            q = rearrange(attn.q_proj(q_input), 'b n (h d) -> b h n d', h=attn.num_heads)
            k = rearrange(attn.k_proj(k_input), 'b n (h d) -> b h n d', h=attn.num_heads)
            v = rearrange(attn.v_proj(v_input), 'b n (h d) -> b h n d', h=attn.num_heads)

            q_len = q.shape[2]
            kv_len = k.shape[2]

            # The Pure Math Score Mod
            def score_mod(b, h, q_idx, kv_idx):
                # Clamp indices to prevent CUDA out-of-bounds asserts from the 128-block padded grid
                safe_q = torch.clamp(q_idx, max=q_len - 1)
                safe_kv = torch.clamp(kv_idx, max=kv_len - 1)
                
                # Default valid: True, but mask out the virtual padded blocks immediately
                valid = (q_idx < q_len) & (kv_idx < kv_len)
                
                if key_padding_mask is not None:
                    valid = valid & key_padding_mask[b, safe_kv]
                    
                if mask_type in ["token_level", "dataframe_level"]:
                    valid = valid & (q_time[b, safe_q] >= kv_time[b, safe_kv])
                    
                return valid

            if mask_type == "no_mask" and key_padding_mask is None:
                block_mask = None
            else:
                block_mask = create_block_mask(score_mod, B=q.shape[0], H=1, Q_LEN=q_len, KV_LEN=kv_len, device=q.device)

            out = flex_attention(q, k, v, block_mask=block_mask)
            torch.cuda.synchronize()
            return "YES"
    except Exception as e:
        err_msg = str(e).split('\n')[0][:45]
        return f"ERR: {err_msg}..."

CASE_MATRIX = list(
    product(
        ["self_attn", "cross_attn"],
        ["all_valid_data", "with_padding"],
        ["token_level", "dataframe_level", "no_mask"],
    )
)

def check_scenario(attention_kind: str, data_condition: str, mask_type: str) -> Tuple[str, str]:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.float16

    batch_size = 2
    num_timesteps = 6
    embed_dim = 128
    num_heads = 8

    x_flat, target_time, modality_shapes, active_modality_names = _build_training_like_target_tokens(
        batch_size, num_timesteps, embed_dim, MODALITY_LAYOUT, device, dtype
    )

    attn_mask = _build_attn_mask(mask_type, target_time, active_modality_names, modality_shapes, device)

    frame_valid = torch.ones(batch_size, num_timesteps, dtype=torch.bool, device=device)
    if data_condition == "with_padding":
        frame_valid[1, -2:] = False

    target_key_padding_mask = _build_target_key_padding_mask(frame_valid, active_modality_names, modality_shapes)

    if attention_kind == "self_attn":
        q_input = k_input = v_input = x_flat
        q_time = kv_time = target_time
        key_padding_mask = target_key_padding_mask if data_condition == "with_padding" else None
    else:
        history, history_key_padding_mask = _build_history_like_context(
            batch_size, x_flat.shape[1], embed_dim, device, dtype, frame_valid
        )
        q_input = x_flat
        k_input = v_input = history
        q_time = target_time
        kv_time = torch.zeros(batch_size, history.shape[1], device=device, dtype=torch.float32)
        key_padding_mask = history_key_padding_mask

    if key_padding_mask is not None and key_padding_mask.all():
        key_padding_mask = None

    attn = Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, use_rope=False).to(device=device, dtype=dtype).eval()

    sdpa_str = _run_attention_with_flash_only(attn, q_input, k_input, v_input, attn_mask, key_padding_mask)
    flex_str = _run_flex_attention(attn, q_input, k_input, v_input, q_time, kv_time, mask_type, key_padding_mask)

    return sdpa_str, flex_str

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. FlashAttention requires a GPU.")
        sys.exit(1)

    warnings.filterwarnings("ignore")

    print("\n" + "="*115)
    print(f"{'Attention Type':<15} | {'Data Condition':<18} | {'Mask Type':<18} | {'SDPA Flash?':<12} | {'FlexAttention?'}")
    print("="*115)

    for attn_kind, data_cond, mask_t in CASE_MATRIX:
        sdpa_str, flex_str = check_scenario(attn_kind, data_cond, mask_t)
        
        sdpa_fmt = "\033[92mYES\033[0m" if sdpa_str == "YES" else "\033[91mNO\033[0m"
        
        if flex_str == "YES":
            flex_fmt = "\033[92mYES\033[0m"
        elif flex_str == "NO":
            flex_fmt = "\033[91mNO\033[0m"
        else:
            flex_fmt = f"\033[93m{flex_str}\033[0m"

        print(f"{attn_kind:<15} | {data_cond:<18} | {mask_t:<18} | {sdpa_fmt:<21} | {flex_fmt}")

    print("="*115 + "\n")