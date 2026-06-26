import torch
import lightning.pytorch as pl
from torch import nn
import base64

def _encode_param_name(name: str) -> str:
    """Encode arbitrary parameter names into safe buffer names."""
    encoded = base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii").rstrip("=")
    return f"ema_{encoded}"

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        
        self.m_name2s_name = dict()
        self.s_name2m_name = dict()
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int)
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = _encode_param_name(name)
                self.m_name2s_name[name] = s_name
                self.s_name2m_name[s_name] = name
                # Rigorous fix: Use detach().clone() without .data
                self.register_buffer(s_name, p.detach().clone())

        self.collected_params = list()

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_((1.0 - decay) * (shadow_params[sname] - m_param[key]))
                else:
                    assert key not in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        with torch.no_grad():
            for key in m_param:
                if m_param[key].requires_grad:
                    # Copy EMA values without tracking autograd on leaf parameters.
                    m_param[key].copy_(shadow_params[self.m_name2s_name[key]])
                else:
                    assert key not in self.m_name2s_name

    def store(self, parameters):
        # Rigorous fix: Ensure cloning does not drag gradients without .data
        self.collected_params = [param.detach().clone() for param in parameters]

    def restore(self, parameters):
        with torch.no_grad():
            for c_param, param in zip(self.collected_params, parameters):
                # Restore original train weights without recording gradients.
                param.copy_(c_param)

    def _legacy_shadow_name(self, model_param_name: str) -> str:
        """Legacy mapping used by older checkpoints (dot stripping)."""
        return model_param_name.replace(".", "")

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load EMA state with backward compatibility for legacy shadow-buffer names."""
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        adapted = dict(state_dict)
        for m_name, s_name in self.m_name2s_name.items():
            if s_name in adapted:
                continue
            legacy_name = self._legacy_shadow_name(m_name)
            if legacy_name in adapted:
                adapted[s_name] = adapted[legacy_name]

        return super().load_state_dict(adapted, strict=strict)


class EMACallback(pl.Callback):
    """EMA callback backed by LitEma with checkpoint-safe state persistence."""

    def __init__(
        self,
        decay: float = 0.9999,
        evaluate_ema_weights_instead: bool = True,
        use_num_updates: bool = True,
    ):
        super().__init__()
        self.decay = float(decay)
        self.evaluate_ema_weights_instead = bool(evaluate_ema_weights_instead)
        self.use_num_updates = bool(use_num_updates)
        self._ema: LitEma | None = None
        self._loaded_state: dict | None = None
        self._ema_applied_for_eval = False

    def _model_for_ema(self, pl_module):
        return getattr(pl_module, "model", pl_module)

    def _trainable_parameters(self, pl_module):
        model = self._model_for_ema(pl_module)
        return [p for p in model.parameters() if p.requires_grad]

    def _init_ema_if_needed(self, pl_module) -> None:
        if self._ema is not None:
            return

        model = self._model_for_ema(pl_module)
        self._ema = LitEma(model=model, decay=self.decay, use_num_upates=self.use_num_updates)

        if self._loaded_state:
            self._ema.load_state_dict(self._loaded_state, strict=False)
            self._loaded_state = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._init_ema_if_needed(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._ema is None:
            self._init_ema_if_needed(pl_module)
        if self._ema is None:
            return
        self._ema(self._model_for_ema(pl_module))

    def _swap_in_ema_weights(self, pl_module) -> None:
        if not self.evaluate_ema_weights_instead or self._ema is None or self._ema_applied_for_eval:
            return
        self._ema.store(self._trainable_parameters(pl_module))
        self._ema.copy_to(self._model_for_ema(pl_module))
        self._ema_applied_for_eval = True

    def _restore_train_weights(self, pl_module) -> None:
        if not self._ema_applied_for_eval or self._ema is None:
            return
        self._ema.restore(self._trainable_parameters(pl_module))
        self._ema_applied_for_eval = False

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._swap_in_ema_weights(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._restore_train_weights(pl_module)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._swap_in_ema_weights(pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._restore_train_weights(pl_module)

    def on_predict_start(self, trainer, pl_module) -> None:
        self._swap_in_ema_weights(pl_module)

    def on_predict_end(self, trainer, pl_module) -> None:
        self._restore_train_weights(pl_module)

    def state_dict(self) -> dict:
        if self._ema is None:
            return {}
        return {"ema_state": self._ema.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._loaded_state = state_dict.get("ema_state")
