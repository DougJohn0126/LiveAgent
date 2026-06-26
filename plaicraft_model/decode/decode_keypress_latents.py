#!/usr/bin/env python3
"""
decode_keypress_latents.py
==========================

Decode stored *keypress latents* -> human-readable key / mouse-button events (JSON).

WHAT THIS DECODES
-----------------
Your rows look like:

    ("session0123", start_ms, end_ms, <pickled np.ndarray (16, 5) float32>)

Each (16, 5) array is a KeyPress autoencoder *latent* (latent_dim=16, latent_seq_len=5),
i.e. the compressed form of a (79, 10) "key/mouse-button activation" block:

    decoder : (16, 5)  ->  (79, 10)        # 79 channels x 10 time-bins
              79 channels = keyboard keys + mouse-button/scroll actions
              10 bins     = 10 x BIN_MS (=10ms) = the 100ms window

This mirrors inference/decode.py exactly:
    k_latent.transpose(0,1).unsqueeze(0) -> decoder -> (79, 10) -> threshold -> 0/1 -> events
The only difference: your blobs are already stored in (16, 5) orientation, so no transpose.

WHAT THIS DOES NOT DECODE
-------------------------
Mouse *movement* (dx/dy pointer trajectory) is a DIFFERENT modality in your pipeline
("mouse_movement", shape (2, 10), decoded directly from m_pred/m_gt -- no autoencoder).
It is NOT contained in these (16, 5) latents. If you have those blobs, decode them with
`decode_mouse_movement()` at the bottom (denorm + round, no AE needed).

WHAT YOU MUST PROVIDE (repo-specific / learned)
-----------------------------------------------
1. The trained autoencoder + weights:
       from keypress_autoencoder.model import KeyPressAutoencoder
       checkpoint: keyencoder_16_5_best_checkpoint.pt
2. The channel<->name mapping:
       from keypress_autoencoder.constants import id_to_index, id_to_name
   (this fixes which of the 79 rows is which key -- it must match training)

Run from your repo root (so the imports resolve), e.g.:

    python decode_keypress_latents.py \
        --rows rows.pkl \
        --ckpt keypress_autoencoder/checkpoints/keyencoder_16_5_best_checkpoint.pt \
        --out-dir decoded_out

`--rows` may be a pickle of the list, OR a .py file holding the literal list (the exact
thing you pasted), OR omitted with --dry-run to just inspect structure.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = "plaicraft_model/encode_key_press/checkpoints/keyencoder_16_5_best_checkpoint.pt"

# --------------------------------------------------------------------------- #
#  CONFIG  (defaults mirror decode.py / the AE construction; override as needed)
# --------------------------------------------------------------------------- #
LATENT_DIM = 16          # KeyPressAutoencoder latent_dim
LATENT_SEQ_LEN = 5       # KeyPressAutoencoder latent_seq_len
INPUT_DIM = 79           # KeyPressAutoencoder input_dim (channels: keys + mouse buttons)
ORIGINAL_SEQ_LEN = 10    # KeyPressAutoencoder original_seq_len (bins per window)
BIN_MS = 10              # ms per bin  (10 bins * 10ms = 100ms window)
KEY_ON_THRESH = 0.5      # DECODE_KEY_ON_THRESH: activation -> pressed

# DECODE_MOUSE_ACTION_NAMES from decode.py's draw_mouse_clicks usage.
MOUSE_ACTION_NAMES = {"mouse_left", "mouse_right", "scroll_up", "scroll_down"}


# --------------------------------------------------------------------------- #
#  LOADING ROWS
# --------------------------------------------------------------------------- #
def load_rows(path: str | Path) -> list[tuple]:
    """
    Return a list of (session_id, start_ms, end_ms, blob_bytes).

    Accepts:
      * a pickle file containing the list, or
      * a .py/.txt file containing the Python literal you pasted.
    """
    path = Path(path)
    data = path.read_bytes()
    # Try pickle first.
    try:
        rows = pickle.loads(data)
    except Exception:
        # Fall back to evaluating a Python literal (handles b'...' byte literals).
        import ast
        rows = ast.literal_eval(data.decode("utf-8", errors="strict"))
    if not isinstance(rows, (list, tuple)):
        raise ValueError(f"Expected a list of rows, got {type(rows)}")
    return list(rows)


def unpickle_latent(blob: bytes) -> np.ndarray:
    """Unpickle one blob and validate it is a (LATENT_DIM, LATENT_SEQ_LEN) latent."""
    arr = pickle.loads(blob)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (LATENT_DIM, LATENT_SEQ_LEN):
        raise ValueError(
            f"Expected latent shape {(LATENT_DIM, LATENT_SEQ_LEN)}, got {arr.shape}. "
            "If your blobs are transposed (5,16), set transpose=True in group_by_session()."
        )
    return arr


def group_by_session(rows: list[tuple], transpose: bool = False) -> dict[str, list[dict]]:
    """
    Group rows by session_id and sort each session by start_ms.
    Returns: { session_id: [ {start_ms, end_ms, latent (16,5)}, ... time-sorted ] }
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if len(row) != 4:
            raise ValueError(f"Expected 4-tuple (session, start, end, blob), got len {len(row)}")
        session_id, start_ms, end_ms, blob = row
        latent = unpickle_latent(blob)
        if transpose:
            latent = latent.T
        grouped[session_id].append(
            {"start_ms": int(start_ms), "end_ms": int(end_ms), "latent": latent}
        )
    for sid in grouped:
        grouped[sid].sort(key=lambda r: r["start_ms"])
    return dict(grouped)


# --------------------------------------------------------------------------- #
#  AUTOENCODER DECODE  (latent (16,5) -> activations (79,10))
# --------------------------------------------------------------------------- #
def build_autoencoder(ckpt_path: str | Path, device: str = "cpu"):
    """
    Construct the KeyPressAutoencoder exactly as decode.py does and load weights.
    Imported lazily so the loader / parser are usable without torch installed.
    """
    import torch
    from encode_key_press.scripts.key_press_encoder import KeyPressAutoencoder

    ae = KeyPressAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        latent_seq_len=LATENT_SEQ_LEN,
        original_seq_len=ORIGINAL_SEQ_LEN,
        num_gru_layers=2,
        conv_dropout=0.1,
        gru_dropout=0.1,
    ).to(device)
    ae.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    ae.eval()
    return ae


def decode_latents_to_activations(ae, latents: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    latents: (N, 16, 5) -> activations (N, 79, 10) continuous, via ae.decoder.
    Matches decode.py: latent.unsqueeze(0) -> decoder -> (1,79,10).
    """
    import torch

    out = []
    with torch.no_grad():
        for lat in latents:
            t = torch.from_numpy(np.asarray(lat, dtype=np.float32).T).unsqueeze(0).to(device)
            dec = ae.decoder(t)                       # (1, 79, 10)
            out.append(dec.squeeze(0).cpu().numpy())  # (79, 10)
    return np.stack(out, axis=0)                      # (N, 79, 10)


def binarize(activations: np.ndarray, thresh: float = KEY_ON_THRESH) -> np.ndarray:
    """(N,79,10) continuous -> (N,79,10) int 0/1."""
    return (activations >= thresh).astype(np.int32)


# --------------------------------------------------------------------------- #
#  NAME MAPPING  (requires your repo constants; falls back with a warning)
# --------------------------------------------------------------------------- #
def load_name_maps():
    """
    Return (index_to_name: dict[int,str], mouse_indices: set[int]).
    Prefers keypress_autoencoder.constants (authoritative row ordering).
    """
    try:
        from encode_key_press.scripts.constants import id_to_index, id_to_name
        index_to_id = {idx: kid for kid, idx in id_to_index.items()}
        index_to_name = {
            idx: id_to_name.get(kid, f"Key_{idx}") for idx, kid in index_to_id.items()
        }
        mouse_indices = {
            idx for idx, name in index_to_name.items() if name in MOUSE_ACTION_NAMES
        }
        return index_to_name, mouse_indices
    except Exception as e:  # pragma: no cover
        print(
            f"[WARN] Could not import keypress_autoencoder.constants ({e}).\n"
            "       Channel->key mapping will be generic ('Key_<row>'). Run from your\n"
            "       repo root so the real id_to_index/id_to_name are used."
        )
        index_to_name = {i: f"Key_{i}" for i in range(INPUT_DIM)}
        return index_to_name, set()


# --------------------------------------------------------------------------- #
#  EVENT PARSING  (ported from decode.py:_parse_keypress_events, with timestamps)
# --------------------------------------------------------------------------- #
def parse_events(
    bin_frames: np.ndarray,
    index_to_name: dict[int, str],
    mouse_indices: set[int],
    base_start_ms: int,
):
    """
    bin_frames: (N, 79, 10) int 0/1, time-ordered.
    Flatten to (79, N*10) and emit press intervals. Each event carries:
        * t_ms      : ms from the start of THIS session's first window (base_start_ms)
        * abs_ms    : absolute timestamp = base_start_ms + t_ms
    Splits into keyboard vs mouse-button events.
    """
    if bin_frames.size == 0:
        return {"keyboard_events": [], "mouse_button_events": [],
                "bin_ms": BIN_MS, "total_ms": 0}

    n, ch, b = bin_frames.shape
    # (N,79,10) -> (79, N*10)
    flat = np.transpose(bin_frames, (1, 0, 2)).reshape(ch, n * b)
    total_bins = flat.shape[1]
    pressed = flat >= 1

    keyboard_events, mouse_events = [], []
    for idx in range(ch):
        name = index_to_name.get(idx, f"Key_{idx}")
        row = pressed[idx]
        on, start = False, 0
        for t in range(total_bins):
            if row[t] and not on:
                on, start = True, t
            elif not row[t] and on:
                on = False
                _emit(keyboard_events, mouse_events, idx, mouse_indices,
                      name, start, t, base_start_ms)
        if on:
            _emit(keyboard_events, mouse_events, idx, mouse_indices,
                  name, start, total_bins, base_start_ms)

    keyboard_events.sort(key=lambda e: (e["start_ms"], e["key_name"]))
    mouse_events.sort(key=lambda e: (e["start_ms"], e["key_name"]))
    return {
        "keyboard_events": keyboard_events,
        "mouse_button_events": mouse_events,
        "bin_ms": BIN_MS,
        "total_ms": int(total_bins * BIN_MS),
    }


def _emit(kb_list, ms_list, idx, mouse_indices, name, start_bin, end_bin, base_start_ms):
    start_ms = int(start_bin * BIN_MS)
    end_ms = int(end_bin * BIN_MS)
    ev = {
        "key_name": name,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "abs_start_ms": base_start_ms + start_ms,
        "abs_end_ms": base_start_ms + end_ms,
    }
    (ms_list if idx in mouse_indices else kb_list).append(ev)


# --------------------------------------------------------------------------- #
#  NATIVE decode.py-COMPATIBLE key_press.json (per-frame 79x10 0/1 + events)
# --------------------------------------------------------------------------- #
def write_native_keypress_json(path: Path, bin_frames: np.ndarray, index_to_name: dict[int, str]):
    """Reproduces decode.py's key_press.json schema (raw_decoded + parsed_decoded)."""
    n = bin_frames.shape[0]
    # inline frames block: one key per line
    frame_blocks = ["["]
    for i in range(n):
        fr = bin_frames[i]
        frame_blocks.append("  {")
        for idx in range(fr.shape[0]):
            name = index_to_name.get(idx, f"Key_{idx}")
            arr_text = ",".join(str(int(x)) for x in fr[idx].tolist())
            comma = "," if idx < fr.shape[0] - 1 else ""
            frame_blocks.append(f'    "{name}": [{arr_text}]{comma}')
        frame_blocks.append("  }" + ("," if i < n - 1 else ""))
    frame_blocks.append("]")
    frames_inline = "\n".join(frame_blocks)

    # parsed events in decode.py's flat {"key_name","start_ms","end_ms"} form
    flat = np.transpose(bin_frames, (1, 0, 2)).reshape(bin_frames.shape[1], -1)
    total_bins = flat.shape[1]
    events = []
    for idx in range(flat.shape[0]):
        name = index_to_name.get(idx, f"Key_{idx}")
        row = flat[idx] >= 1
        on, start = False, 0
        for t in range(total_bins):
            if row[t] and not on:
                on, start = True, t
            elif not row[t] and on:
                on = False
                print(f"Emitting event: {name} [{start*BIN_MS}ms .. {t*BIN_MS}ms]")
                events.append({"key_name": name, "start_ms": int(start * BIN_MS), "end_ms": int(t * BIN_MS)})
        if on:
            events.append({"key_name": name, "start_ms": int(start * BIN_MS), "end_ms": int(total_bins * BIN_MS)})

    obj = {
        "schema_version": 1,
        "bin_ms": int(BIN_MS),
        "raw_decoded": {"shape_per_frame": [INPUT_DIM, ORIGINAL_SEQ_LEN], "frames": "__INLINE__"},
        "parsed_decoded": {"events": events, "bin_ms": int(BIN_MS), "total_ms": int(total_bins * BIN_MS)},
    }
    text = json.dumps(obj, indent=2, ensure_ascii=False).replace('"__INLINE__"', frames_inline)
    path.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
#  OPTIONAL: mouse MOVEMENT decode (separate modality; only if you have it)
# --------------------------------------------------------------------------- #
def decode_mouse_movement(frames_2x10, mm_stats=None):
    """
    frames_2x10: list of (2,10) arrays (the 'mouse_movement' modality, model space).
    Returns decode.py-style {"series":[{"time_ms","dx","dy"}...]}.
    NOTE: this modality is NOT in the (16,5) keypress latents.
    """
    if not frames_2x10:
        return {"series": [], "bin_ms": int(BIN_MS), "total_ms": 0}
    arr = np.array(frames_2x10, dtype=np.float32)             # (S,2,10)
    arr = np.transpose(arr, (1, 0, 2)).reshape(2, -1)         # (2, S*10)
    if mm_stats is not None:
        mode = str(mm_stats.get("mode", "")).lower()
        if mode == "zscore":
            mean = np.asarray(mm_stats["mean"], np.float32).reshape(-1, 1)
            std = np.asarray(mm_stats["std"], np.float32).reshape(-1, 1)
            arr = arr * std + mean
        elif mode == "minmax":
            vmin = np.asarray(mm_stats["min"], np.float32).reshape(-1, 1)
            vmax = np.asarray(mm_stats["max"], np.float32).reshape(-1, 1)
            arr = arr * (vmax - vmin) + vmin
    arr = np.rint(arr).astype(int)
    dx, dy = arr[0], arr[1]
    series = [{"time_ms": int(t * BIN_MS), "dx": int(dx[t]), "dy": int(dy[t])} for t in range(arr.shape[1])]
    return {"series": series, "bin_ms": int(BIN_MS), "total_ms": int(arr.shape[1] * BIN_MS)}


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def main(rows):
    '''
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", help="pickle or .py-literal file of (session,start,end,blob) rows")
    ap.add_argument("--ckpt", help="path to keyencoder_16_5_best_checkpoint.pt")
    ap.add_argument("--out-dir", default=r"encode_key_press\checkpoints\keyencoder_16_5_best_checkpoint.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresh", type=float, default=KEY_ON_THRESH)
    ap.add_argument("--transpose", action="store_true", help="set if blobs are stored (5,16)")
    ap.add_argument("--dry-run", action="store_true", help="inspect structure only; no AE/torch")
    '''
    sessions = group_by_session(rows, transpose="cpu")
    print(f"Loaded {len(rows)} rows across {len(sessions)} session(s).")
    for sid, frames in sessions.items():
        span = frames[-1]["end_ms"] - frames[0]["start_ms"]
        print(f"  {sid}: {len(frames)} windows, span {span} ms "
              f"[{frames[0]['start_ms']} .. {frames[-1]['end_ms']}]")


    out_dir = Path("decoded_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    index_to_name, mouse_indices = load_name_maps()
    ae = build_autoencoder(CHECKPOINT_DIR, device="cpu")
    keys = None
    combined = {}
    for sid, frames in sessions.items():
        latents = np.stack([f["latent"] for f in frames], axis=0)              # (N,16,5)
        acts = decode_latents_to_activations(ae, latents, device="cpu")  # (N,79,10)
        bin_frames = binarize(acts, thresh= KEY_ON_THRESH)                         # (N,79,10)
        base = frames[0]["start_ms"]

        parsed = parse_events(bin_frames, index_to_name, mouse_indices, base)
        parsed["session_id"] = sid
        parsed["base_start_ms"] = base
        combined[sid] = parsed

        sdir = out_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        keys = write_native_keypress_json(sdir / "key_press.json", bin_frames, index_to_name)
        (sdir / "events.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {sdir/'key_press.json'} and {sdir/'events.json'} "
              f"({len(parsed['keyboard_events'])} key + {len(parsed['mouse_button_events'])} mouse-button events)")

    (out_dir / "all_sessions.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote combined summary -> {out_dir/'all_sessions.json'}")
    if (keys == None):
        print("no keys decoded")
        return ""
    return keys


if __name__ == "__main__":
    main()
