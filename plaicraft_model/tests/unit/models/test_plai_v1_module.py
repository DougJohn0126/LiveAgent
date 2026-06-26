"""Unit tests for PlaiV1LightningModule (Lightning orchestration)."""

import pytest
import torch
from unittest.mock import Mock
import lightning.pytorch as pl

from training.plai_v1_lightning_module import PlaiV1LightningModule
from training.noise_schedulers import FlowMatchingScheduler
from data.data_classes import FullData


class TestPlaiV1ModuleInitialization:
    """Test PlaiV1LightningModule initialization."""

    def test_init_with_scheduler(self):
        # Use a real minimal module instead of a Mock to avoid LitEma iteration errors
        net = torch.nn.Linear(10, 10)
        scheduler = FlowMatchingScheduler(target_modalities=["video"])
        
        module = PlaiV1LightningModule(
            net=net,
            noise_scheduler=scheduler,
        )
        
        assert module.net is net
        assert module.noise_scheduler is scheduler
        assert isinstance(module.noise_scheduler, FlowMatchingScheduler)

    def test_hparams_saved(self):
        net = torch.nn.Linear(10, 10)
        scheduler = FlowMatchingScheduler(target_modalities=["video"])
        
        module = PlaiV1LightningModule(
            net=net,
            noise_scheduler=scheduler,
            ema_decay=0.999
        )
        
        assert module.hparams.ema_decay == 0.999
        # Objects should be ignored in save_hyperparameters
        assert 'net' not in module.hparams
        assert 'noise_scheduler' not in module.hparams


class TestPlaiV1ModuleLossComputation:
    """Test PlaiV1LightningModule loss calculation logic."""

    def test_get_train_losses_interface(self):
        # 1. Setup mocks
        batch_size = 2
        timesteps = 4
        h_dim = 256
        
        # Net predicts velocity in flattened sequence format
        mock_preds = {
            "video": torch.randn(batch_size, timesteps * 960, h_dim),
            "audio_speak": torch.randn(batch_size, timesteps * 15, h_dim)
        }
        mock_target_velocity = {
            "video": torch.randn(batch_size, timesteps, 960, h_dim),
            "audio_speak": torch.randn(batch_size, timesteps, 15, h_dim),
        }
        mock_net = Mock(spec=torch.nn.Module)
        
        scheduler = FlowMatchingScheduler(target_modalities=["video", "audio_speak"])
        
        mock_multimodal_io = Mock()
        def mock_tokenize(full_data):
            res = {"modalities": {}}
            for mod in ["video", "audio_speak"]:
                tokens = torch.randn(batch_size, timesteps, 960 if mod == "video" else 15, h_dim)
                res["modalities"][mod] = {
                    "tokens": tokens,
                    "time": torch.zeros(batch_size, timesteps, tokens.shape[2])
                }
            return res
        mock_multimodal_io.prepare_transformer_inputs.side_effect = mock_tokenize

        mock_net.multimodal_io = mock_multimodal_io
        mock_net.forward_training.return_value = (mock_preds, mock_target_velocity)

        module = PlaiV1LightningModule(net=mock_net, noise_scheduler=scheduler)

        # 2. Create dummy batch
        target = FullData(batch={
            "video": torch.randn(batch_size, timesteps, 2, 4, 96, 160),
            "padding_mask": torch.ones(batch_size, timesteps, dtype=torch.bool)
        })
        context = FullData(batch={
            "video": torch.randn(batch_size, timesteps, 2, 4, 96, 160),
            "padding_mask": torch.ones(batch_size, timesteps, dtype=torch.bool)
        })
        batch = (target, context)

        # 3. Compute losses
        losses = module.get_train_losses(batch)

        # 4. Assertions
        assert "total_loss" in losses
        assert "video" in losses
        assert "audio_speak" in losses
        assert losses["total_loss"] > 0
        
        # Verify component calls
        mock_net.forward_training.assert_called_once()
