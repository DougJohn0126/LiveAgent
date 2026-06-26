"""Unit tests for PlaiV1Model (orchestration of backbone components)."""

import pytest
import torch
from unittest.mock import Mock

from models.plai_v1 import PlaiV1Model
from data.data_classes import FullData


class TestPlaiV1ModelInitialization:
    """Test PlaiV1Model initialization."""

    def test_init_with_defaults(self):
        model = PlaiV1Model(multimodal_io=Mock())
        assert hasattr(model, "cfg")
        assert model.h_dim == 512

    def test_init_with_config(self):
        config = {
            "h_dim": 256,
        }
        model = PlaiV1Model(cfg=config, multimodal_io=Mock())
        # Check that it extracted h_dim correctly from cfg
        assert model.h_dim == 256

    def test_init_with_components(self):
        context_embedder = Mock()
        moe_decoder = Mock()

        model = PlaiV1Model(
            context_embedder=context_embedder,
            moe_decoder=moe_decoder,
            multimodal_io=Mock(),
            h_dim=512,
        )

        assert isinstance(model, PlaiV1Model)
        assert model.context_embedder is context_embedder
        assert model.moe_decoder is moe_decoder
        assert model.h_dim == 512

    def test_cfg_stored_on_model(self):
        cfg = {"h_dim": 512}
        model = PlaiV1Model(cfg=cfg, multimodal_io=Mock())
        assert model.cfg == cfg


class TestPlaiV1ModelForward:
    """Test PlaiV1Model forward pass."""

    @pytest.fixture
    def model_with_mocks(self):
        h_dim = 256
        context_embedder = Mock(return_value=torch.randn(2, 16, h_dim))
        moe_decoder = Mock(return_value={"video": torch.randn(2, 960, h_dim)})
        multimodal_io = Mock()
        multimodal_io.prepare_transformer_inputs.side_effect = lambda full_data: {
            "modalities": {
                "video": {
                    "tokens": torch.randn(full_data.video.shape[0], full_data.video.shape[1], 960, h_dim)
                }
            },
            "player_emb": torch.randn(full_data.video.shape[0], h_dim),
        }
        multimodal_io.project_outputs.side_effect = lambda outputs: {
            "video": torch.randn(2, 2, 2, 4, 96, 160)
        }
        
        model = PlaiV1Model(
            h_dim=h_dim,
            context_embedder=context_embedder,
            moe_decoder=moe_decoder,
            multimodal_io=multimodal_io,
        )
        return model

    def test_forward_basic(self, model_with_mocks):
        batch_size = 2

        x_tau = FullData(batch={
            "video": torch.randn(batch_size, 2, 2, 4, 96, 160),
            "padding_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "dataframe_indices": torch.arange(2).unsqueeze(0).expand(batch_size, -1),
            "metadata": [
                {"player_name": "p1", "player_gender": "f", "player_skill_level": "i"},
                {"player_name": "p2", "player_gender": "m", "player_skill_level": "a"},
            ],
        })
        tau = torch.rand(batch_size, 1)

        context = FullData(batch={
            "video": torch.randn(batch_size, 8, 2, 4, 96, 160),
            "padding_mask": torch.ones(batch_size, 8, dtype=torch.bool),
            "dataframe_indices": torch.arange(8).unsqueeze(0).expand(batch_size, -1),
            "metadata": [
                {"player_name": "p1", "player_gender": "f", "player_skill_level": "i"},
                {"player_name": "p2", "player_gender": "m", "player_skill_level": "a"},
            ],
        })

        output = model_with_mocks.forward(x_tau=x_tau, tau=tau, context=context)

        assert isinstance(output, FullData)
        assert output.video.shape == (batch_size, 2, 2, 4, 96, 160)

    def test_forward_calls_components(self, model_with_mocks):
        batch_size = 2
        x_tau = FullData(batch={
            "video": torch.randn(batch_size, 2, 2, 4, 96, 160),
            "padding_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "dataframe_indices": torch.arange(2).unsqueeze(0).expand(batch_size, -1),
            "metadata": [
                {"player_name": "p1", "player_gender": "f", "player_skill_level": "i"},
                {"player_name": "p2", "player_gender": "m", "player_skill_level": "a"},
            ],
        })
        tau = torch.rand(batch_size, 1)
        context = FullData(batch={
            "video": torch.randn(batch_size, 8, 2, 4, 96, 160),
            "padding_mask": torch.ones(batch_size, 8, dtype=torch.bool),
            "dataframe_indices": torch.arange(8).unsqueeze(0).expand(batch_size, -1),
            "metadata": [
                {"player_name": "p1", "player_gender": "f", "player_skill_level": "i"},
                {"player_name": "p2", "player_gender": "m", "player_skill_level": "a"},
            ],
        })

        model_with_mocks.forward(x_tau=x_tau, tau=tau, context=context)

        model_with_mocks.context_embedder.assert_called_once()
        model_with_mocks.moe_decoder.assert_called_once()
