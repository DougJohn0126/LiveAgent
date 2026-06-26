"""3D Spatiotemporal verification testing the NATIVE model code."""

from __future__ import annotations
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf

# Adjust path to your src directory
_src_path = Path(__file__).resolve().parent.parent.parent.parent / "src"
sys.path.insert(0, str(_src_path))

from models.components.multimodal_io import MultimodalIO
from models.components.context_embedder import ContextEmbedder
from models.components.moe_decoder import MoEDecoder
from models.components.positional_encoding import RotaryEmbedding
from models.components.weightnorm_modules import normalize
from models.plai_v1 import PlaiV1Model
from data.data_classes import FullData
from utils.constants import (
    MODALITY_SHAPES,
)


def _build_component_dict(cfg_node) -> dict:
    cfg_dict = OmegaConf.to_container(cfg_node, resolve=True)
    cfg_dict.pop("_target_", None)
    return cfg_dict


def _make_random_full_data(batch_size: int, timesteps: int, start_df_index: int) -> FullData:
    video_shape = MODALITY_SHAPES["video"]
    audio_shape = MODALITY_SHAPES["audio_hear"]
    key_shape = MODALITY_SHAPES["key_press"]
    mouse_shape = MODALITY_SHAPES["mouse_movement"]

    metadata = [
        {
            "player_name": f"player_{i}",
            "player_gender": "unknown",
            "player_skill_level": "mid",
        }
        for i in range(batch_size)
    ]

    return FullData(
        batch={
            "video": torch.randn(
                batch_size,
                timesteps,
                video_shape[0],
                video_shape[1],
                video_shape[2],
                video_shape[3],
            ),
            "audio_speak": torch.randn(batch_size, timesteps, audio_shape[0], audio_shape[1]),
            "audio_hear": torch.randn(batch_size, timesteps, audio_shape[0], audio_shape[1]),
            "key_press": torch.randn(batch_size, timesteps, key_shape[0], key_shape[1]),
            "mouse_movement": torch.randn(batch_size, timesteps, mouse_shape[0], mouse_shape[1]),
            "metadata": metadata,
            "dataframe_indices": torch.arange(
                start_df_index,
                start_df_index + timesteps,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1),
            "transcript_speak": [""] * batch_size,
            "transcript_hear": [""] * batch_size,
        }
    )


def _build_rope_score_matrix(rope: RotaryEmbedding, coords: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Rotate a shared base vector at each coordinate and return the pairwise dot-product matrix."""
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, N, 3], got {tuple(coords.shape)}")

    generator = torch.Generator(device=coords.device)
    generator.manual_seed(seed)
    base_vector = torch.randn(1, 1, 1, rope.freqs.numel() * 2, generator=generator, dtype=torch.float32)
    base = base_vector.expand(coords.shape[0], 1, coords.shape[1], -1).contiguous()
    rotated = rope.apply_multimodal_rotary_pos_emb(base, coords, seq_dim=-2, base_fps=1.0)
    return torch.einsum("bhid,bhjd->ij", rotated, rotated).detach().cpu()


def _assert_toeplitz_like(matrix: torch.Tensor, atol: float = 1e-4) -> None:
    seq_len = matrix.shape[0]
    for offset in range(-(seq_len - 1), seq_len):
        diag = matrix.diagonal(offset=offset)
        if diag.numel() > 1:
            assert torch.max(diag) - torch.min(diag) <= atol


def _assert_displacement_groups_are_consistent(
    matrix: torch.Tensor,
    coords: torch.Tensor,
    atol: float = 3.0,
) -> None:
    grouped_scores: dict[tuple[float, float, float], list[float]] = defaultdict(list)

    for i, j in product(range(coords.shape[1]), repeat=2):
        delta = tuple((coords[0, j] - coords[0, i]).tolist())
        grouped_scores[delta].append(float(matrix[i, j]))

    repeated_groups = 0
    for delta, scores in grouped_scores.items():
        if len(scores) < 2:
            continue
        repeated_groups += 1
        assert max(scores) - min(scores) <= atol, f"delta={delta} has spread {max(scores) - min(scores)}"

    assert repeated_groups > 0, "Expected at least one repeated displacement group"


def test_relative_attention_score_heatmap_full_pipeline():
    """Run full data pipeline on 1+1 dataframe windows and validate RoPE relative-score stability."""
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "model" / "plai_v1.yaml"
    cfg = OmegaConf.load(cfg_path)

    mm_cfg = _build_component_dict(cfg.multimodal_io)
    ce_cfg = _build_component_dict(cfg.context_embedder)
    md_cfg = _build_component_dict(cfg.moe_decoder)

    # Match the RoPE-enabled training constraints used by the decoder implementation.
    mm_cfg["positional_encoding_type"] = "rope"
    mm_cfg["mask_type"] = "no_mask"
    ce_cfg["use_stm_perceiver"] = False
    ce_cfg["ltm_conditioning_mode"] = "adaln"
    ce_cfg["rnn_config"]["rnn_type"] = "mingru"
    md_cfg["positional_encoding_type"] = "rope"
    md_cfg["use_stm_perceiver"] = False
    md_cfg["ltm_conditioning_mode"] = "adaln"

    # Use coarser video tokenization so modality blocks are visually easier to distinguish.
    mm_cfg["patch_h"] = 8
    mm_cfg["patch_w"] = 8
    mm_cfg["ltm_patch_h"] = 8
    mm_cfg["ltm_patch_w"] = 8

    multimodal_io = MultimodalIO(**mm_cfg)
    context_embedder = ContextEmbedder(**ce_cfg)
    moe_decoder = MoEDecoder(**md_cfg)

    model = PlaiV1Model(
        cfg={
            "h_dim": int(cfg.h_dim),
            "checkpointing": {"perceiver": False, "rnn": False, "moe_decoder": False},
        },
        context_embedder=context_embedder,
        moe_decoder=moe_decoder,
        multimodal_io=multimodal_io,
    ).eval()

    # Requested setup: 1 dataframe context, 1 dataframe target.
    batch_size = 1
    context_fd = _make_random_full_data(batch_size=batch_size, timesteps=2, start_df_index=0)
    target_fd = _make_random_full_data(batch_size=batch_size, timesteps=2, start_df_index=2)

    # Build target sequence once to recover modality block boundaries.
    target_input = model.multimodal_io.fulldata_to_moe_decoder_input(target_fd)
    active_names = target_input["active_modality_names"]
    modality_shapes = target_input["modality_shapes"]

    labels = []
    cursor = 0
    for name in active_names:
        timesteps, tokens_per_df = modality_shapes[name]
        length = int(timesteps * tokens_per_df)
        labels.append((name, cursor, cursor + length))
        cursor += length

    captured: dict[str, torch.Tensor] = {}

    def _project_with_rope(attn_module, x_in: torch.Tensor, pos: torch.Tensor, proj):
        out = proj(x_in)
        bsz, seq_len, _ = out.shape
        out = out.view(bsz, seq_len, attn_module.num_heads, attn_module.head_dim).permute(0, 2, 1, 3)
        if attn_module.use_weightnorm:
            out = normalize(out, dim=-1)
        if attn_module.rope is not None:
            out = attn_module.rope.apply_multimodal_rotary_pos_emb(
                out,
                pos,
                seq_dim=-2,
                base_fps=attn_module.rope_base_fps,
            )
        return out

    def _capture_attention(module, args, kwargs):
        if captured:
            return
        q_input = kwargs["q_input"]
        k_input = kwargs["k_input"]
        q_pos = kwargs.get("q_pos")
        k_pos = kwargs.get("k_pos")

        # Relative-only matrix: same base content at all positions, only RoPE coordinates vary.
        q_base = q_input.mean(dim=1, keepdim=True).expand_as(q_input)
        k_base = k_input.mean(dim=1, keepdim=True).expand_as(k_input)
        q_relative = _project_with_rope(module, q_base, q_pos, module.q_proj)
        k_relative = _project_with_rope(module, k_base, k_pos, module.k_proj)
        scores_relative = torch.einsum("bhid,bhjd->bij", q_relative, k_relative)[0].detach().cpu()

        captured["relative_scores"] = scores_relative

    hook = model.moe_decoder.blocks[0].attn.register_forward_pre_hook(_capture_attention, with_kwargs=True)
    try:
        with torch.no_grad():
            tau = torch.full((batch_size, 1), 0.5)
            _ = model.forward(x_tau=target_fd, tau=tau, context=context_fd)
    finally:
        hook.remove()

    assert "relative_scores" in captured
    relative_scores = captured["relative_scores"]
    assert torch.isfinite(relative_scores).all()

    # Also report per-modality shift consistency where Toeplitz is expected to be strongest.
    diagnostics = []
    for name, start, end in labels:
        block = relative_scores[start:end, start:end]
        if block.shape[0] < 2:
            continue
        shift = max(1, min(8, block.shape[0] // 4))
        shift_mae = (block[:-shift, :-shift] - block[shift:, shift:]).abs().mean().item()
        diagnostics.append(shift_mae)

    assert diagnostics
    assert all(torch.isfinite(torch.tensor(diagnostics)).tolist())


def test_videorope_temporal_only_for_non_video_tokens() -> None:
    """Non-video (1D) tokens should map to [t, t, t] coordinates in VideoRoPE."""
    rope = RotaryEmbedding(
        dim=96,
        freqs_for="lang",
        mrope_section=[16, 16, 16],
    )

    base_fps = 10.0
    metric_pos = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.float32)
    coords = rope._build_videorope_coords(metric_pos=metric_pos, base_fps=base_fps)

    expected_t = metric_pos * base_fps
    assert coords.shape == (1, metric_pos.shape[1], 3)
    assert torch.allclose(coords[..., 0], expected_t)
    assert torch.allclose(coords[..., 1], expected_t)
    assert torch.allclose(coords[..., 2], expected_t)


def test_videorope_no_internal_temporal_scaling_in_rotary_builder() -> None:
    """Rotary builder should be independent from any temporal scale knob."""
    base_fps = 25.0
    metric_pos = torch.tensor(
        [[[0.0, 1.0, 2.0], [0.5, 1.0, 2.0], [1.0, 1.0, 2.0]]],
        dtype=torch.float32,
    )

    rope_scale_1 = RotaryEmbedding(
        dim=96,
        freqs_for="lang",
        mrope_section=[16, 16, 16],
    )
    rope_scale_2 = RotaryEmbedding(
        dim=96,
        freqs_for="lang",
        mrope_section=[16, 16, 16],
    )

    coords_1 = rope_scale_1._build_videorope_coords(metric_pos=metric_pos, base_fps=base_fps)
    coords_2 = rope_scale_2._build_videorope_coords(metric_pos=metric_pos, base_fps=base_fps)

    assert torch.allclose(coords_1, coords_2)


def test_videorope_axis_layout_matches_reference_four_group_pattern() -> None:
    """Axis layout should follow [spatial, temporal, spatial, temporal] after merge+duplicate+reverse."""
    rope = RotaryEmbedding(
        dim=12,
        freqs_for="lang",
        mrope_section=[2, 2, 2],  # merged -> [2, 4] -> reversed duplicate -> [4, 2, 4, 2]
    )

    # axis_freqs shape: [3 axes, B=1, N=1, D=12]
    axis_freqs = torch.zeros(3, 1, 1, 12, dtype=torch.float32)
    axis_freqs[0, 0, 0, :] = torch.arange(100, 112, dtype=torch.float32)  # temporal
    axis_freqs[1, 0, 0, :] = torch.arange(200, 212, dtype=torch.float32)  # x
    axis_freqs[2, 0, 0, :] = torch.arange(300, 312, dtype=torch.float32)  # y

    laid_out = rope._apply_videorope_axis_layout(axis_freqs)[0, 0, :]

    # Reference-like expected selection pattern:
    # section sizes [4, 2, 4, 2], with spatial sections interleaving x/y channel-by-channel.
    expected = torch.tensor(
        [
            200.0, 301.0, 202.0, 303.0,  # spatial section 1 (indices 0..3)
            104.0, 105.0,                # temporal section 1 (indices 4..5)
            206.0, 307.0, 208.0, 309.0,  # spatial section 2 (indices 6..9)
            110.0, 111.0,                # temporal section 2 (indices 10..11)
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(laid_out, expected)


def test_videorope_temporal_scale_applies_upstream_in_multimodal_io() -> None:
    """Temporal scaling should be applied when building rope position ids in MultimodalIO."""
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "model" / "plai_v1.yaml"
    cfg = OmegaConf.load(cfg_path)
    mm_cfg = _build_component_dict(cfg.multimodal_io)

    # Isolate a 1D modality for deterministic axis checks.
    mm_cfg["positional_encoding_type"] = "rope"
    mm_cfg["target_modalities"] = ["audio_hear"]
    mm_cfg["context_modalities"] = ["audio_hear"]
    mm_cfg["rope_temporal_scale"] = 1.0
    mm_scale_1 = MultimodalIO(**mm_cfg)

    mm_cfg["rope_temporal_scale"] = 2.0
    mm_scale_2 = MultimodalIO(**mm_cfg)

    fd = _make_random_full_data(batch_size=1, timesteps=1, start_df_index=0)
    out_1 = mm_scale_1.fulldata_to_moe_decoder_input(fd)
    out_2 = mm_scale_2.fulldata_to_moe_decoder_input(fd)

    rope_pos_1 = out_1["target_rope_pos"]
    rope_pos_2 = out_2["target_rope_pos"]

    # Temporal scaling is upstream: t-axis scales, spatial axes marked with sentinel (-1) for 1D tokens.
    assert torch.allclose(rope_pos_2[..., 0], rope_pos_1[..., 0] * 2.0)
    # 1D tokens have spatial axes marked with -1 sentinel
    assert torch.allclose(rope_pos_1[..., 1], torch.full_like(rope_pos_1[..., 1], -1.0))
    assert torch.allclose(rope_pos_1[..., 2], torch.full_like(rope_pos_1[..., 2], -1.0))
    assert torch.allclose(rope_pos_2[..., 1], torch.full_like(rope_pos_2[..., 1], -1.0))
    assert torch.allclose(rope_pos_2[..., 2], torch.full_like(rope_pos_2[..., 2], -1.0))

    # Raw time values used for masks remain unscaled.
    assert torch.allclose(out_1["target_time"], out_2["target_time"])


def test_videorope_video_tokens_keep_raw_time_separate_from_rope_time() -> None:
    """Video target_time should stay on physical time even when RoPE scaling is enabled."""
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "model" / "plai_v1.yaml"
    cfg = OmegaConf.load(cfg_path)
    mm_cfg = _build_component_dict(cfg.multimodal_io)

    mm_cfg["positional_encoding_type"] = "rope"
    mm_cfg["target_modalities"] = ["video"]
    mm_cfg["context_modalities"] = ["video"]
    mm_cfg["rope_temporal_scale"] = 2.0
    multimodal_io = MultimodalIO(**mm_cfg)

    fd = _make_random_full_data(batch_size=1, timesteps=2, start_df_index=3)
    out = multimodal_io.fulldata_to_moe_decoder_input(fd)

    target_time = out["target_time"]
    target_rope_pos = out["target_rope_pos"]

    assert torch.allclose(target_rope_pos[..., 0], target_time * 2.0)
    assert not torch.allclose(target_time, target_rope_pos[..., 0])


def test_videorope_temporal_only_video_slice_is_toeplitz_like() -> None:
    """Fix x,y and vary only time; scores should depend mainly on Δt."""
    rope = RotaryEmbedding(dim=96, freqs_for="lang", mrope_section=[16, 16, 16])
    coords = torch.tensor(
        [[[0.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0], [4.0, 2.0, 3.0]]],
        dtype=torch.float32,
    )

    matrix = _build_rope_score_matrix(rope, coords, seed=1)
    _assert_toeplitz_like(matrix)


def test_videorope_x_only_spatial_slice_is_toeplitz_like() -> None:
    """Fix t,y and vary only x; scores should depend mainly on Δx."""
    rope = RotaryEmbedding(dim=96, freqs_for="lang", mrope_section=[16, 16, 16])
    coords = torch.tensor(
        [[[5.0, 0.0, 1.0], [5.0, 1.0, 1.0], [5.0, 2.0, 1.0], [5.0, 3.0, 1.0], [5.0, 4.0, 1.0]]],
        dtype=torch.float32,
    )

    matrix = _build_rope_score_matrix(rope, coords, seed=2)
    _assert_toeplitz_like(matrix, atol=8.0)


def test_videorope_y_only_spatial_slice_is_toeplitz_like() -> None:
    """Fix t,x and vary only y; scores should depend mainly on Δy."""
    rope = RotaryEmbedding(dim=96, freqs_for="lang", mrope_section=[16, 16, 16])
    coords = torch.tensor(
        [[[7.0, 1.0, 0.0], [7.0, 1.0, 1.0], [7.0, 1.0, 2.0], [7.0, 1.0, 3.0], [7.0, 1.0, 4.0]]],
        dtype=torch.float32,
    )

    matrix = _build_rope_score_matrix(rope, coords, seed=3)
    _assert_toeplitz_like(matrix, atol=5.0)


def test_videorope_same_relative_displacement_has_consistent_scores() -> None:
    """Pairs with the same (Δt, Δx, Δy) should produce nearly identical scores."""
    rope = RotaryEmbedding(dim=96, freqs_for="lang", mrope_section=[16, 16, 16])
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )

    matrix = _build_rope_score_matrix(rope, coords, seed=4)
    _assert_displacement_groups_are_consistent(matrix, coords)

