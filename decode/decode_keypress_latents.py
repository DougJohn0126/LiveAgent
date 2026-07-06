from __future__ import annotations
 
import numpy as np
import torch
 
# --------------------------------------------------------------------------- #
#  Layout constants (utils/insights-gg/constants.py + AE construction)
# --------------------------------------------------------------------------- #
BIN_MS = 10                      # ms per time bin (keys and mouse alike)
WINDOW_MS = 100                  # one AE window
VIDEO_FRAMES_PER_UNIT = 2        # 1 unit = 200 ms = 2 windows
LATENT_DIM = 16                  # KeyPressAutoencoder latent_dim
LATENT_SEQ_LEN = 5               # KeyPressAutoencoder latent_seq_len
INPUT_DIM = 79                   # channels: 75 keys + 4 mouse actions
BINS_PER_WINDOW = 10             # original_seq_len: 10 x 10 ms = 100 ms
KEY_ON_THRESH = 0.5              # activation -> pressed
CHECKPOINT_DIR = "../plaimodel/encode_key_press/checkpoints/keyencoder_16_5_best_checkpoint.pt"
 
# --------------------------------------------------------------------------- #
#  Channel map — VERBATIM from encode_key_press/scripts/constants.py.
#  Order matters: it fixes which of the 79 decoder rows is which key.
# --------------------------------------------------------------------------- #
_FIXED_KEYS = {
    "32": "space", "65": "a", "66": "b", "67": "c", "68": "d", "69": "e", "70": "f",
    "71": "g", "72": "h", "73": "i", "74": "j", "75": "k", "76": "l", "77": "m",
    "78": "n", "79": "o", "80": "p", "81": "q", "82": "r", "83": "s", "84": "t",
    "85": "u", "86": "v", "87": "w", "88": "x", "89": "y", "90": "z", "48": "0",
    "49": "1", "50": "2", "51": "3", "52": "4", "53": "5", "54": "6", "55": "7",
    "56": "8", "57": "9", "256": "Escape", "257": "Return", "258": "Tab", "259": "BackSpace",
    "260": "Insert", "261": "Delete", "262": "Right", "263": "Left", "264": "Down",
    "265": "Up", "266": "Page_Up", "267": "Page_Down", "268": "Home", "269": "End",
    "340": "Shift_L", "341": "Control_L", "342": "Alt_L", "343": "Super_L", "344": "Shift_R",
    "345": "Control_R", "346": "Alt_R", "347": "Super_R", "348": "Menu", "91": "bracketleft",
    "93": "bracketright", "92": "backslash", "59": "semicolon", "39": "apostrophe",
    "44": "comma", "46": "period", "47": "slash", "45": "minus", "61": "equal", "96": "grave",
    "290": "F1", "292": "F3", "294": "F5", "280": "Caps_Lock",
}
_FIXED_MOUSE_BUTTONS = {
    "left": "mouse_left",
    "right": "mouse_right",
    "scroll_up": "scroll_up",
    "scroll_down": "scroll_down",
}
 
INDEX_TO_NAME: dict[int, str] = {}
MOUSE_BUTTON_INDICES: set[int] = set()
for _kid, _name in _FIXED_KEYS.items():
    INDEX_TO_NAME[len(INDEX_TO_NAME)] = _name
for _bid, _name in _FIXED_MOUSE_BUTTONS.items():
    MOUSE_BUTTON_INDICES.add(len(INDEX_TO_NAME))
    INDEX_TO_NAME[len(INDEX_TO_NAME)] = _name
assert len(INDEX_TO_NAME) == INPUT_DIM, "channel map drifted from INPUT_DIM"
 
 
def _build_autoencoder(ckpt_path: str, device: str = "cpu"):
    """Construct the KeyPressAutoencoder exactly as encode_key_press/main.py
    does and load weights. The class is imported from the preprocessing repo;
    make sure it is on sys.path (repo root)."""
    try:
        from encode_key_press.scripts.key_press_encoder import KeyPressAutoencoder
    except ImportError:
        from plaicraft_model.encode_key_press.scripts.key_press_encoder import KeyPressAutoencoder  # alt layout
    ae = KeyPressAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        latent_seq_len=LATENT_SEQ_LEN,
        original_seq_len=BINS_PER_WINDOW,
        num_gru_layers=2,
        conv_dropout=0.1,
        gru_dropout=0.1,
    ).to(device)
    ae.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    ae.eval()
    return ae
 
 
class FullDataDecoder:
    def __init__(self, device: str = "cpu",
                 key_on_thresh: float = KEY_ON_THRESH, mm_stats: dict | None = None,
                 ae=None):
        """
        ae        : optional preloaded KeyPressAutoencoder (shared with the
                    encode path). If None, loads CHECKPOINT_DIR from disk.
        mm_stats  : None if the model consumes RAW summed pixel deltas (what
                    preprocessing produces). Pass {"mode":"zscore","mean":..,
                    "std":..} or {"mode":"minmax","min":..,"max":..} ONLY if
                    the model repo's datamodule normalizes mouse_movement.
        """
        self.device = device
        self.ae = ae if ae is not None else _build_autoencoder(CHECKPOINT_DIR, device)
        self.thresh = float(key_on_thresh)
        self.mm_stats = mm_stats
 
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode(self, fd=None, *, key_press: torch.Tensor | None = None,
               mouse_movement: torch.Tensor | None = None):
        """fd: FullData from inference (batch size 1), or pass tensors directly.
        Returns (key_events, mouse_rows) — see module docstring."""
        if fd is not None:
            key_press = getattr(fd, "key_press", None) if key_press is None else key_press
            mouse_movement = (getattr(fd, "mouse_movement", None)
                              if mouse_movement is None else mouse_movement)
        key_events = self._decode_keys(key_press) if key_press is not None else []
        mouse_rows = self._decode_mouse(mouse_movement) if mouse_movement is not None else []
        return key_events, mouse_rows
 
    __call__ = decode
 
    # ------------------------------------------------------------------ #
    def _decode_keys(self, kp: torch.Tensor) -> list[dict]:
        """kp: (1, U, 10, 16) -> merged, time-sorted press events."""
        if kp.dim() != 4 or kp.shape[0] != 1:
            raise ValueError(f"expected key_press (1, U, 10, 16), got {tuple(kp.shape)}")
        U = kp.shape[1]
        if (kp.shape[2] != VIDEO_FRAMES_PER_UNIT * LATENT_SEQ_LEN
                or kp.shape[3] != LATENT_DIM):
            raise ValueError(
                f"key_press unit shape {tuple(kp.shape[2:])} != "
                f"({VIDEO_FRAMES_PER_UNIT * LATENT_SEQ_LEN}, {LATENT_DIM})"
            )
        # (1,U,10,16) -> (Nwin, 5, 16): each 200 ms unit holds its two 100 ms
        # windows stacked along the 10-axis in time order [w0; w1].
        win = kp[0].reshape(U * VIDEO_FRAMES_PER_UNIT, LATENT_SEQ_LEN, LATENT_DIM)
        # Model-native (5,16) -> the decoder's documented (B, 16, 5).
        z = win.transpose(-1, -2).float().contiguous().to(self.device)
        acts = self.ae.decoder(z)                       # (Nwin, 79, 10), one batched call
        pressed = (acts >= self.thresh).cpu().numpy()   # bool (Nwin, 79, 10)
 
        # (Nwin,79,10) -> (79, Nwin*10): windows are consecutive in time.
        flat = np.transpose(pressed, (1, 0, 2)).reshape(INPUT_DIM, -1)
        total_bins = flat.shape[1]
 
        events: list[dict] = []
        for idx in range(INPUT_DIM):
            row = flat[idx]
            if not row.any():
                continue
            name = INDEX_TO_NAME[idx]
            kind = "mouse_button" if idx in MOUSE_BUTTON_INDICES else "keyboard"
            # run-length encode press intervals
            padded = np.concatenate(([False], row, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            for start, end in zip(edges[::2], edges[1::2]):
                events.append({
                    "key_name": name,
                    "kind": kind,
                    "start_ms": int(start * BIN_MS),
                    "end_ms": int(end * BIN_MS),
                })
        events.sort(key=lambda e: (e["start_ms"], e["key_name"]))
        # sanity: nothing beyond the rollout span
        assert all(e["end_ms"] <= total_bins * BIN_MS for e in events)
        return events
 
    # ------------------------------------------------------------------ #
    def _decode_mouse(self, mm: torch.Tensor) -> list[tuple[int, int, int]]:
        """mm: (1, U, 20, 2) -> [(time_ms, dx, dy)] per 10 ms bin."""
        if mm.dim() != 4 or mm.shape[0] != 1:
            raise ValueError(f"expected mouse_movement (1, U, 20, 2), got {tuple(mm.shape)}")
        if (mm.shape[2] != VIDEO_FRAMES_PER_UNIT * BINS_PER_WINDOW
                or mm.shape[3] != 2):
            raise ValueError(
                f"mouse_movement unit shape {tuple(mm.shape[2:])} != "
                f"({VIDEO_FRAMES_PER_UNIT * BINS_PER_WINDOW}, 2)"
            )
        # (1,U,20,2) -> (total_bins, 2), time-major; then channel-major (2, T).
        arr = mm[0].reshape(-1, 2).float().cpu().numpy().T        # (2, U*20)
        if self.mm_stats is not None:
            mode = str(self.mm_stats.get("mode", "")).lower()
            if mode == "zscore":
                mean = np.asarray(self.mm_stats["mean"], np.float32).reshape(-1, 1)
                std = np.asarray(self.mm_stats["std"], np.float32).reshape(-1, 1)
                arr = arr * std + mean
            elif mode == "minmax":
                vmin = np.asarray(self.mm_stats["min"], np.float32).reshape(-1, 1)
                vmax = np.asarray(self.mm_stats["max"], np.float32).reshape(-1, 1)
                arr = arr * (vmax - vmin) + vmin
            else:
                raise ValueError(f"unknown mm_stats mode: {mode!r}")
        arr = np.rint(arr).astype(int)
        dx, dy = arr[0], arr[1]
        return [(t * BIN_MS, int(dx[t]), int(dy[t])) for t in range(arr.shape[1])]