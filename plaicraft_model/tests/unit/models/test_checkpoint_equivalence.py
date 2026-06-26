"""Gradient equivalence tests for checkpointed model components."""

import copy

import pytest
import torch

from models.components.context_embedder import ContextEmbedder
from models.components.moe_decoder import MoEDecoder
from models.components.perceiver import PerceiverIO
from models.components.recurrent_encoders.min_gru import MinGRUEncoder

try:
    from models.components.recurrent_encoders.xlstm import xLSTMEncoder
except ImportError:
    xLSTMEncoder = None


def _run_and_collect_grads(model: torch.nn.Module, loss_fn):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model)
    loss.backward()

    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    return loss.detach(), grads


def _assert_grad_dicts_close(grads_a, grads_b, atol=1e-6, rtol=1e-5):
    assert set(grads_a.keys()) == set(grads_b.keys())
    for name in grads_a:
        a = grads_a[name]
        b = grads_b[name]
        assert torch.allclose(a, b, atol=atol, rtol=rtol), (
            f"Gradient mismatch for {name}: max_abs_diff={(a - b).abs().max().item():.3e}"
        )


def test_perceiver_checkpoint_gradient_equivalence():
    torch.manual_seed(7)

    base = PerceiverIO(
        depth=2,
        dim=16,
        queries_dim=16,
        num_latents=8,
        latent_dim=16,
        cross_heads=2,
        latent_heads=2,
        cross_dim_head=8,
        latent_dim_head=8,
        decoder_ff=True,
        use_decoder=True,
        seq_dropout_prob=0.0,
    ).train()

    model_no_ckpt = copy.deepcopy(base)
    model_ckpt = copy.deepcopy(base)

    data = torch.randn(2, 6, 16, requires_grad=True)
    queries = torch.randn(2, 4, 16, requires_grad=True)

    def loss_no_ckpt(model):
        out = model(data=data, queries=queries, gradient_checkpoint=False)
        return out.pow(2).mean()

    def loss_ckpt(model):
        out = model(data=data, queries=queries, gradient_checkpoint=True)
        return out.pow(2).mean()

    loss_a, grads_a = _run_and_collect_grads(model_no_ckpt, loss_no_ckpt)
    loss_b, grads_b = _run_and_collect_grads(model_ckpt, loss_ckpt)

    assert torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6)
    _assert_grad_dicts_close(grads_a, grads_b)


def test_moe_decoder_checkpoint_gradient_equivalence():
    torch.manual_seed(11)

    base = MoEDecoder(
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        modality_configs={
            "video": {"hidden_dim": 64},
            "audio_speak": {"hidden_dim": 64},
        },
        dropout=0.0,
        use_cross_attention=True,
        positional_encoding_type="fourier",
        mask_type="none",
    ).train()

    model_no_ckpt = copy.deepcopy(base)
    model_ckpt = copy.deepcopy(base)

    bsz, timesteps, dim = 2, 3, 32
    x_tau = {
        "video": {
            "tokens": torch.randn(bsz, timesteps, 2, dim, requires_grad=True),
            "time": torch.arange(timesteps).view(1, timesteps, 1).expand(bsz, timesteps, 2).float(),
        },
        "audio_speak": {
            "tokens": torch.randn(bsz, timesteps, 3, dim, requires_grad=True),
            "time": torch.arange(timesteps).view(1, timesteps, 1).expand(bsz, timesteps, 3).float(),
        },
    }
    cond_emb = torch.randn(bsz, 2, dim, requires_grad=True)
    history = torch.randn(bsz, 5, dim, requires_grad=True)

    def loss_no_ckpt(model):
        out = model(
            x_tau=x_tau,
            cond_emb=cond_emb,
            history=history,
            gradient_checkpoint=False,
        )
        return sum(v.pow(2).mean() for v in out.values())

    def loss_ckpt(model):
        out = model(
            x_tau=x_tau,
            cond_emb=cond_emb,
            history=history,
            gradient_checkpoint=True,
        )
        return sum(v.pow(2).mean() for v in out.values())

    loss_a, grads_a = _run_and_collect_grads(model_no_ckpt, loss_no_ckpt)
    loss_b, grads_b = _run_and_collect_grads(model_ckpt, loss_ckpt)

    assert torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6)
    _assert_grad_dicts_close(grads_a, grads_b)


def test_mingru_encoder_checkpoint_gradient_equivalence():
    torch.manual_seed(13)

    base = MinGRUEncoder(
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_multiplier=2,
    ).train()

    model_no_ckpt = copy.deepcopy(base)
    model_ckpt = copy.deepcopy(base)

    x = torch.randn(2, 7, 32, requires_grad=True)

    def loss_no_ckpt(model):
        init_state = model.get_initial_state(x.shape[0], x.device, x.dtype)
        out, _ = model(x, initial_state=init_state, gradient_checkpoint=False)
        return out.pow(2).mean()

    def loss_ckpt(model):
        init_state = model.get_initial_state(x.shape[0], x.device, x.dtype)
        out, _ = model(x, initial_state=init_state, gradient_checkpoint=True)
        return out.pow(2).mean()

    loss_a, grads_a = _run_and_collect_grads(model_no_ckpt, loss_no_ckpt)
    loss_b, grads_b = _run_and_collect_grads(model_ckpt, loss_ckpt)

    assert torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6)
    _assert_grad_dicts_close(grads_a, grads_b)


def test_context_embedder_checkpoint_gradient_equivalence_mingru_path():
    torch.manual_seed(17)

    base = ContextEmbedder(
        model_dim=16,
        num_latents=4,
        perceiver_depth=1,
        perceiver_cross_heads=1,
        perceiver_latent_heads=1,
        perceiver_cross_dim_head=16,
        perceiver_latent_dim_head=16,
        perceiver_seq_dropout=0.0,
        stm_context_length=3,
        chunk_len=0,
        k_ltm=2,
        rnn_config={
            "rnn_type": "mingru",
            "embedding_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "mlp_multiplier": 2,
        },
    ).train()

    model_no_ckpt = copy.deepcopy(base)
    model_ckpt = copy.deepcopy(base)

    modalities = {
        "video": {"tokens": torch.randn(2, 4, 2, 16, requires_grad=True)},
        "audio_speak": {"tokens": torch.randn(2, 4, 3, 16, requires_grad=True)},
    }

    def loss_no_ckpt(model):
        out = model(modalities, gradient_checkpoint=False)
        return out.pow(2).mean()

    def loss_ckpt(model):
        out = model(modalities, gradient_checkpoint=True)
        return out.pow(2).mean()

    loss_a, grads_a = _run_and_collect_grads(model_no_ckpt, loss_no_ckpt)
    loss_b, grads_b = _run_and_collect_grads(model_ckpt, loss_ckpt)

    assert torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6)
    _assert_grad_dicts_close(grads_a, grads_b)


@pytest.mark.skipif(xLSTMEncoder is None, reason="xLSTM dependencies are not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="xLSTM requires CUDA GPU")
def test_xlstm_checkpoint_gradient_equivalence():
    torch.manual_seed(19)

    try:
        base = xLSTMEncoder(
            embedding_dim=64,
            num_heads=4,
            num_blocks=2,
            dropout=0.0,
            context_length=128,
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
        ).train().cuda()
    except RuntimeError as exc:
        pytest.skip(f"xLSTM backend unavailable: {exc}")

    model_no_ckpt = copy.deepcopy(base)
    model_ckpt = copy.deepcopy(base)

    x = torch.randn(2, 6, 64, device="cuda", requires_grad=True)

    def loss_no_ckpt(model):
        out, _ = model(x, initial_state=None, gradient_checkpoint=False)
        return out.pow(2).mean()

    def loss_ckpt(model):
        out, _ = model(x, initial_state=None, gradient_checkpoint=True)
        return out.pow(2).mean()

    loss_a, grads_a = _run_and_collect_grads(model_no_ckpt, loss_no_ckpt)
    loss_b, grads_b = _run_and_collect_grads(model_ckpt, loss_ckpt)

    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=1e-5)
    _assert_grad_dicts_close(grads_a, grads_b, atol=1e-5, rtol=1e-4)
