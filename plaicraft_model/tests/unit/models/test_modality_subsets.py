import torch
import pytest
from types import SimpleNamespace

from data.data_classes import FullData
from models.components.multimodal_io import MultimodalIO
from training.noise_schedulers import FlowMatchingScheduler
from training.plai_v1_lightning_module import PlaiV1LightningModule


def _make_full_data(batch_size: int = 1, timesteps: int = 2) -> FullData:
    return FullData(
        batch={
            "video": torch.randn(batch_size, timesteps, 2, 4, 96, 160),
            "audio_hear": torch.randn(batch_size, timesteps, 15, 128),
            "audio_speak": torch.randn(batch_size, timesteps, 15, 128),
            "key_press": torch.randn(batch_size, timesteps, 10, 16),
            "mouse_movement": torch.randn(batch_size, timesteps, 20, 2),
            "metadata": [{"player": "test"} for _ in range(batch_size)],
            "dataframe_indices": (
                torch.arange(0, timesteps, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            ),
            "transcript_speak": None,
            "transcript_hear": None,
        }
    )


def test_multimodal_io_applies_target_and_context_subsets():
    io = MultimodalIO(
        patch_h=8,
        patch_w=8,
        model_dim=64,
        player_embed_dim=64,
        stm_context_length=2,
        ltm_downsample_chunk_length=2,
        ltm_drop_modalities=[],
        context_modalities=["video", "audio_hear", "key_press"],
        target_modalities=["video", "audio_hear"],
    )
    full_data = _make_full_data()

    decoder_inputs = io.fulldata_to_moe_decoder_input(full_data)
    assert decoder_inputs["active_modality_names"] == ["audio_hear", "video"]

    context_inputs = io.fulldata_to_context_embedder_input(full_data)
    assert context_inputs["ltm_tokens"].shape[2] > 0
    assert context_inputs["stm_tokens"].shape[2] > 0



def test_flow_matching_scheduler_outputs_only_target_modalities():
    scheduler = FlowMatchingScheduler(target_modalities=["video", "key_press"])
    x_0 = _make_full_data()

    x_tau, _, eps = scheduler(x_0)
    velocity = scheduler.get_loss_target(eps=eps, x_0=x_0)

    assert x_tau.video is not None
    assert x_tau.key_press is not None
    assert x_tau.audio_hear is None
    assert x_tau.audio_speak is None
    assert x_tau.mouse_movement is None

    assert velocity.video is not None
    assert velocity.key_press is not None
    assert velocity.audio_hear is None
    assert velocity.audio_speak is None
    assert velocity.mouse_movement is None


class _DummyModel(torch.nn.Module):
    def __init__(self, decoder_modalities, target_modalities, context_modalities):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.moe_decoder = SimpleNamespace(modality_names=decoder_modalities)
        self.multimodal_io = SimpleNamespace(
            target_modalities=set(target_modalities),
            context_modalities=set(context_modalities),
        )

    def forward(self, x_tau, tau, context=None, memory=None):
        return x_tau



def test_lightning_module_validates_generated_modalities_against_experts():
    scheduler = FlowMatchingScheduler(target_modalities=["video", "audio_hear"])
    model = _DummyModel(
        decoder_modalities=["video"],
        target_modalities=["video", "audio_hear"],
        context_modalities=["video"],
    )
    datamodule = SimpleNamespace(modalities=["video", "audio_hear"])

    with pytest.raises(ValueError, match="Missing experts"):
        PlaiV1LightningModule(model=model, noise_scheduler=scheduler, datamodule=datamodule)
