"""Integration tests for stateless denoising/generation helpers."""

import torch
from unittest.mock import Mock

from data.data_classes import FullData
from inference.denoising import generate_chunk, init_noise_target


class _IdentityVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_tau, tau, memory):
        return x_tau


class TestPlaiV1ModelPipeline:

    def test_init_noise_target_shapes(self):
        batch_size = 3
        target_pred_len = 2
        target_modalities = ["video", "audio_speak", "key_press", "mouse_movement"]
        metadata = [{"sample_id": i} for i in range(batch_size)]

        x_tau = init_noise_target(
            batch_size=batch_size,
            target_pred_len=target_pred_len,
            target_modalities=target_modalities,
            metadata=metadata,
            device=torch.device("cpu"),
        )

        assert isinstance(x_tau, FullData)
        assert x_tau.video.shape == (batch_size, 2, 2, 4, 96, 160)
        assert x_tau.audio_speak.shape == (batch_size, 2, 15, 512)
        assert x_tau.key_press.shape == (batch_size, 2, 10, 512)
        assert x_tau.mouse_movement.shape == (batch_size, 2, 20, 512)
        assert x_tau.dataframe_indices.shape == (batch_size, target_pred_len)
        assert x_tau.metadata == metadata

    def test_generate_chunk_interface(self):
        batch_size = 2
        target_pred_len = 2
        target_modalities = ["video", "audio_speak", "key_press", "mouse_movement"]
        model = _IdentityVelocityModel()
        memory = {"stm": torch.zeros(batch_size, 1, 8)}

        config = Mock()
        config.denoising_type = "flow"
        config.num_denoising_steps = 2

        x_denoised = generate_chunk(
            model=model,
            config=config,
            memory=memory,
            batch_size=batch_size,
            target_pred_len=target_pred_len,
            target_modalities=target_modalities,
            metadata=None,
        )

        assert isinstance(x_denoised, FullData)
        assert x_denoised.video.shape == (batch_size, 2, 2, 4, 96, 160)
        assert x_denoised.audio_speak.shape == (batch_size, 2, 15, 512)
        assert x_denoised.key_press.shape == (batch_size, 2, 10, 512)
        assert x_denoised.mouse_movement.shape == (batch_size, 2, 20, 512)
