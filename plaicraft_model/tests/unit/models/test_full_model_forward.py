"""Full plai_v1 forward interface tests (kept skipped: requires full component wiring)."""

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from data.data_classes import FullData
from utils.constants import MODALITY_SHAPES


@pytest.fixture
def model_config():
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="train.yaml", overrides=["model=plai_v1"])
        return cfg.model


@pytest.fixture
def sample_target_context():
    bsz = 2
    t_ctx = 8
    t_tgt = 4
    frames = 3

    video_shape = MODALITY_SHAPES["video"]
    channels, height, width = video_shape[1], video_shape[2], video_shape[3]

    audio_dim = MODALITY_SHAPES["audio_hear"][1]
    audio_tokens = 10
    key_dim = MODALITY_SHAPES["key_press"][1]
    key_tokens = 5
    mouse_dim = MODALITY_SHAPES["mouse_movement"][1]
    mouse_tokens = 5

    context = FullData(batch={
        "video": torch.randn(bsz, t_ctx, frames, channels, height, width),
        "audio_speak": torch.randn(bsz, t_ctx, audio_tokens, audio_dim),
        "audio_hear": torch.randn(bsz, t_ctx, audio_tokens, audio_dim),
        "key_press": torch.randn(bsz, t_ctx, key_tokens, key_dim),
        "mouse_movement": torch.randn(bsz, t_ctx, mouse_tokens, mouse_dim),
        "metadata": [
            {"name": "player1", "gender": "male", "skill_level": "intermediate", "age": 25},
            {"name": "player2", "gender": "female", "skill_level": "advanced", "age": 30},
        ],
        "padding_mask": torch.ones(bsz, t_ctx, dtype=torch.bool),
        "dataframe_indices": torch.arange(0, t_ctx, dtype=torch.long).unsqueeze(0).expand(bsz, -1),
    })

    target = FullData(batch={
        "video": torch.randn(bsz, t_tgt, frames, channels, height, width),
        "audio_speak": torch.randn(bsz, t_tgt, audio_tokens, audio_dim),
        "audio_hear": torch.randn(bsz, t_tgt, audio_tokens, audio_dim),
        "key_press": torch.randn(bsz, t_tgt, key_tokens, key_dim),
        "mouse_movement": torch.randn(bsz, t_tgt, mouse_tokens, mouse_dim),
        "metadata": [
            {"name": "player1", "gender": "male", "skill_level": "intermediate", "age": 25},
            {"name": "player2", "gender": "female", "skill_level": "advanced", "age": 30},
        ],
        "padding_mask": torch.ones(bsz, t_tgt, dtype=torch.bool),
        "dataframe_indices": torch.arange(t_ctx, t_ctx + t_tgt, dtype=torch.long).unsqueeze(0).expand(bsz, -1),
    })

    return target, context


class TestFullModelForward:
    @pytest.mark.skip(reason="Full model tests require complete component setup")
    def test_forward_pass_interface(self, model_config, sample_target_context):
        model = instantiate(model_config)
        model.eval()

        target, context = sample_target_context
        target_sequence = model.multimodal_io.fulldata_to_moe_decoder_input(target)
        x_tau, tau, _ = model.scheduler(target_sequence)

        with torch.no_grad():
            output = model.forward(x_tau=x_tau, tau=tau, context=context)

        assert isinstance(output, dict)
        assert "predictions" in output
        assert isinstance(output["predictions"], dict)

    @pytest.mark.skip(reason="Full model tests require complete component setup")
    def test_multimodal_io_uses_embedded_relative_indices(self, model_config, sample_target_context):
        model = instantiate(model_config)
        target, _ = sample_target_context

        tokenized = model.multimodal_io.fulldata_to_moe_decoder_input(target)

        assert "x_flat" in tokenized
        assert "player_emb" in tokenized


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
