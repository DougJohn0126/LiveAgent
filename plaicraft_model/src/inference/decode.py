# samplers/decode.py
import sys
import os
import subprocess
import tempfile
from pathlib import Path
import json
from dotenv import load_dotenv

import cv2
import ffmpeg
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL
from scipy.io import wavfile
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  project-local imports
# ------------------------------------------------------------------ #
from data.data_classes import FullData
from data.datamodule import DataModule
from utils.constants import (
    BIN_MS,
    ENCODEC_SAMPLE_RATE,
    DECODE_BINS_PER_SUBFRAME,
    DECODE_FINAL_FRAME_SIZE,
    DECODE_KEY_BOX_HEIGHT,
    DECODE_KEY_BOX_PADDING_X,
    DECODE_KEY_BOX_PADDING_Y,
    DECODE_KEY_FONT_SCALE,
    DECODE_KEY_FONT_THICKNESS,
    DECODE_KEY_ON_THRESH,
    DECODE_KEY_ROW_GAP,
    DECODE_KEY_TOKENS_PER_SUBFRAME,
    DECODE_LEFT_SECTION_MAX_W,
    DECODE_MOUSE_ACTION_NAMES,
    DECODE_MOUSE_ARROW_TIP_LEN,
    DECODE_MOUSE_LINE_COLOR,
    DECODE_MOUSE_LINE_THICKNESS,
    DECODE_SUBFRAMES_PER_FRAME,
    DECODE_TOP_BAR_HEIGHT,
    DECODE_USE_FP16,
    DECODE_VIDEO_FPS,
)
from keypress_autoencoder.model import KeyPressAutoencoder
from keypress_autoencoder.constants import id_to_index, id_to_name

device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video latents decode/display rate (content only)
FRAME_DURATION_MS   = 1000.0 / DECODE_VIDEO_FPS           # 100 ms per frame (video latent)

# ------------------------------------------------------------------ #
#  KEY / MOUSE MAPPINGS
# ------------------------------------------------------------------ #
FIXED_KEYS = {
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
    "290": "F1", "292": "F3", "294": "F5", "280": "Caps_Lock"
}

index_to_id        = {idx: key_id for key_id, idx in id_to_index.items()}
keyboard_indices, mouse_indices = [], []
for idx, key_id in index_to_id.items():
    name = id_to_name.get(key_id, "")
    (mouse_indices if name in DECODE_MOUSE_ACTION_NAMES else keyboard_indices).append(idx)

# ------------------------------------------------------------------ #
#  MODEL INITIALISATION (lazy, module-scoped singletons)
# ------------------------------------------------------------------ #
_vae = None
_encodec_model = None
_keypress_ae = None


def _init_models():
    global _vae, _encodec_model, _keypress_ae
    if _vae is None:
        print("Loading VAE …")
        _vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16 if DECODE_USE_FP16 else torch.float32
        ).to(device)
        if DECODE_USE_FP16:
            _vae.half()

    if _encodec_model is None:
        print("Loading Encodec …")
        from transformers import EncodecModel
        _encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        _encodec_model = _encodec_model.to(device).eval()

    if _keypress_ae is None:
        print("Loading KeyPressAutoencoder …")
        REPO_ROOT = Path(__file__).resolve().parents[1]
        
        # Use environment variable if set, otherwise use relative path
        keypress_ckpt_env = os.getenv('KEYPRESS_CHECKPOINT_PATH')
        if keypress_ckpt_env:
            keypress_ckpt = Path(keypress_ckpt_env)
        else:
            keypress_ckpt = REPO_ROOT / "keypress_autoencoder" / "checkpoints" / "keyencoder_16_5_best_checkpoint.pt"
        
        if not keypress_ckpt.exists():
            raise FileNotFoundError(
                f"KeyPressAutoencoder checkpoint not found at {keypress_ckpt}. "
                "Expected path is relative to this file: ../keypress_autoencoder/checkpoints/…\n"
                "You can set KEYPRESS_CHECKPOINT_PATH env var to override the default location."
            )
        _keypress_ae = KeyPressAutoencoder(
            input_dim=79,
            latent_dim=16,
            latent_seq_len=5,
            original_seq_len=10,
            num_gru_layers=2,
            conv_dropout=0.1,
            gru_dropout=0.1
        ).to(device)
        _keypress_ae.load_state_dict(torch.load(keypress_ckpt, map_location=device))
        _keypress_ae.eval()


# ------------------------------------------------------------------ #
#  HELPERS
# ------------------------------------------------------------------ #
def decode_latents(vae_model, latents):
    latents = latents.half() if DECODE_USE_FP16 else latents
    latents = latents / 0.13025
    with torch.no_grad():
        imgs = vae_model.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


def _round_mouse(arr_dn: np.ndarray) -> np.ndarray:
    """
    Round denormalized mouse deltas to the nearest integer (min unit = 1),
    so tiny values like 0.03 → 0. Used for both saved JSON and overlay.
    """
    return np.rint(arr_dn).astype(np.int32)


def draw_keyboard_keys(bar, pressed_indices, frame_w):
    if not pressed_indices:
        return
    key_x, key_y = 10, 10
    max_w = int(frame_w * DECODE_LEFT_SECTION_MAX_W)
    for idx in pressed_indices:
        key_id   = index_to_id.get(idx)
        key_name = id_to_name.get(key_id, f"Key_{idx}")
        (text_w, text_h), _ = cv2.getTextSize(
            key_name, cv2.FONT_HERSHEY_SIMPLEX,
            DECODE_KEY_FONT_SCALE, DECODE_KEY_FONT_THICKNESS
        )
        box_w = text_w + 2 * DECODE_KEY_BOX_PADDING_X
        if key_x + box_w > max_w:
            key_x = 10
            key_y += DECODE_KEY_BOX_HEIGHT + DECODE_KEY_ROW_GAP
            if key_y + DECODE_KEY_BOX_HEIGHT > DECODE_TOP_BAR_HEIGHT:
                break
        cv2.rectangle(bar,
                      (key_x, key_y),
                      (key_x + box_w, key_y + DECODE_KEY_BOX_HEIGHT),
                      (0, 0, 0), 2)
        text_x = key_x + (box_w - text_w)//2
        text_y = key_y + (DECODE_KEY_BOX_HEIGHT + text_h)//2
        cv2.putText(bar, key_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, DECODE_KEY_FONT_SCALE,
                    (0, 0, 0), DECODE_KEY_FONT_THICKNESS, cv2.LINE_AA)
        key_x += box_w + DECODE_KEY_ROW_GAP


def draw_mouse_clicks(bar, pressed_mouse_names, frame_w):
    mouse_x0 = frame_w - 120
    mouse_y0 = 10
    mouse_w, mouse_h = 100, 60

    # outline + center split
    cv2.rectangle(bar, (mouse_x0, mouse_y0), (mouse_x0 + mouse_w, mouse_y0 + mouse_h), (0, 0, 0), 2)
    cv2.line(bar, (mouse_x0 + mouse_w // 2, mouse_y0), (mouse_x0 + mouse_w // 2, mouse_y0 + mouse_h // 2), (0, 0, 0), 2)

    # left button fill
    if "mouse_left" in pressed_mouse_names:
        cv2.rectangle(
            bar,
            (mouse_x0 + 2, mouse_y0 + 2),
            (mouse_x0 + mouse_w // 2 - 2, mouse_y0 + mouse_h // 2 - 2),
            (0, 255, 0),
            -1,
        )

    # right button fill (fixed: pt1/pt2 are tuples)
    if "mouse_right" in pressed_mouse_names:
        cv2.rectangle(
            bar,
            (mouse_x0 + mouse_w // 2 + 2, mouse_y0 + 2),
            (mouse_x0 + mouse_w - 2,      mouse_y0 + mouse_h // 2 - 2),
            (0, 255, 0),
            -1,
        )

    if "scroll_up" in pressed_mouse_names:
        cv2.arrowedLine(
            bar,
            (mouse_x0 + mouse_w // 2, mouse_y0 + mouse_h // 2 + 10),
            (mouse_x0 + mouse_w // 2, mouse_y0 + 5),
            (255, 0, 0),
            2,
            tipLength=0.4,
        )
    if "scroll_down" in pressed_mouse_names:
        cv2.arrowedLine(
            bar,
            (mouse_x0 + mouse_w // 2, mouse_y0 + mouse_h // 2 - 10),
            (mouse_x0 + mouse_w // 2, mouse_y0 + mouse_h - 5),
            (255, 0, 0),
            2,
            tipLength=0.4,
        )


def overlay_data_on_frame(frame, mouse_data_rounded, key_data, _, __, ___, keypress_threshold=0.5):
    content = frame.copy()
    c_h, c_w = content.shape[:2]

    # mouse movement arrows on content (uses already-rounded model-space units)
    if mouse_data_rounded is not None and np.any(mouse_data_rounded != 0):
        cx, cy = c_w // 2, c_h // 2
        prev = (cx, cy)
        for i in range(mouse_data_rounded.shape[1]):
            dx, dy = int(mouse_data_rounded[0, i]), int(mouse_data_rounded[1, i])
            dx_scaled = int(dx * (c_w / 1920) * 2)
            dy_scaled = int(dy * (c_h / 1080) * 2)
            nx = np.clip(prev[0] + dx_scaled, 0, c_w - 1)
            ny = np.clip(prev[1] + dy_scaled, 0, c_h - 1)
            cv2.arrowedLine(content, prev, (int(nx), int(ny)),
                            DECODE_MOUSE_LINE_COLOR, DECODE_MOUSE_LINE_THICKNESS,
                            cv2.LINE_AA, tipLength=DECODE_MOUSE_ARROW_TIP_LEN)
            prev = (int(nx), int(ny))

    # top bar
    top_bar = np.full((DECODE_TOP_BAR_HEIGHT, c_w, 3), 255, dtype=np.uint8)
    pressed_keyboard_indices, pressed_mouse_names = [], []
    if key_data is not None:
        union_pressed = np.any(key_data >= keypress_threshold, axis=1)
        pressed_indices = np.where(union_pressed)[0]
        pressed_keyboard_indices = [p for p in pressed_indices if p in keyboard_indices]
        pressed_mouse_names = [id_to_name.get(index_to_id.get(p), "") for p in pressed_indices if p in mouse_indices]
    draw_keyboard_keys(top_bar, pressed_keyboard_indices, c_w)
    draw_mouse_clicks(top_bar, pressed_mouse_names, c_w)

    return np.vstack((top_bar, content))


# ------------------------------------------------------------------ #
#  VIDEO WRITER
# ------------------------------------------------------------------ #
def save_video(frames, output_path, fps, audio_speak=None, audio_hear=None):
    h, w, _ = frames[0].shape
    
    # 1. Force dimensions to be even numbers (strictly required for yuv420p)
    even_h = h - (h % 2)
    even_w = w - (w % 2)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        prefix=f"{output_path.stem}_temp_",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp_file:
        temp_path = Path(tmp_file.name)
    proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{even_w}x{even_h}", framerate=fps)
        .output(str(temp_path), pix_fmt="yuv420p", vcodec="libx264", preset="fast", r=fps)
        .overwrite_output()
        # 2. Capture stderr to surface actual FFmpeg errors if they happen
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )
    
    try:
        for fr in frames:
            if fr.shape[2] == 4:
                fr = cv2.cvtColor(fr, cv2.COLOR_RGBA2RGB)
            
            # Resize slightly if we cropped dimensions to make them even
            if fr.shape[0] != even_h or fr.shape[1] != even_w:
                fr = cv2.resize(fr, (even_w, even_h))
                
            proc.stdin.write(fr.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        # 3. Read the actual error from FFmpeg instead of failing silently
        stderr_output = proc.stderr.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"FFmpeg crashed immediately. FFmpeg error log:\n{stderr_output}")

    if audio_speak is None and audio_hear is None:
        os.replace(temp_path, output_path)
        print(f"Saved video to {output_path}")
        return

    if audio_speak is None and audio_hear is not None:
        audio_speak = np.zeros_like(audio_hear)
    if audio_hear is None and audio_speak is not None:
        audio_hear = np.zeros_like(audio_speak)

    min_len = min(len(audio_speak), len(audio_hear))
    audio_speak, audio_hear = audio_speak[:min_len], audio_hear[:min_len]
    mixed = audio_speak + audio_hear
    m = np.max(np.abs(mixed)) if mixed.size else 1.0
    if m > 1.0:
        mixed /= m
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix=f"{output_path.stem}_audio_",
        dir=str(output_path.parent),
        delete=False,
    ) as wav_tmp_file:
        wav_path = Path(wav_tmp_file.name)
    wavfile.write(str(wav_path), ENCODEC_SAMPLE_RATE, np.int16(mixed * 32767))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(temp_path),
            "-i", str(wav_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", str(ENCODEC_SAMPLE_RATE),
            "-shortest",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if temp_path.exists():
        temp_path.unlink()
    if wav_path.exists():
        wav_path.unlink()
    print(f"Saved video to {output_path}")

# ------------------------------------------------------------------ #
#  OPTIONAL AUDIO PLOTS
# ------------------------------------------------------------------ #
def plot_wave_compare(gt_audio, gen_audio, sample_rate, save_path,
                      title_gt="Ground-truth", title_gen="Generated",
                      duration_s=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    gt_audio = np.asarray(gt_audio, dtype=np.float32).squeeze()
    gen_audio = np.asarray(gen_audio, dtype=np.float32).squeeze()

    if duration_s is None or duration_s <= 0:
        duration_s = max(len(gt_audio), len(gen_audio)) / float(sample_rate)

    t_gt = np.linspace(0.0, duration_s, len(gt_audio), endpoint=False)
    t_gen = np.linspace(0.0, duration_s, len(gen_audio), endpoint=False)

    y_abs_max = float(max(np.max(np.abs(gt_audio)), np.max(np.abs(gen_audio)), 1e-6))

    axes[0].plot(t_gt, gt_audio)
    axes[0].set_title(title_gt)
    axes[0].set_ylabel("Amp")
    axes[0].set_xlim(0.0, duration_s)
    axes[0].set_ylim(-y_abs_max, y_abs_max)

    axes[1].plot(t_gen, gen_audio)
    axes[1].set_title(title_gen)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amp")
    axes[1].set_xlim(0.0, duration_s)
    axes[1].set_ylim(-y_abs_max, y_abs_max)

    fig.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def plot_mel_spectrogram(audio, sample_rate, save_path, title="Mel Spectrogram"):
    audio = np.squeeze(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_dB, sr=sample_rate, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB"); plt.title(title); plt.tight_layout()
    plt.savefig(str(save_path)); plt.close()
    print(f"Saved mel spectrogram plot to {save_path}")


def plot_mouse_compare(gt_frames_list, gen_frames_list, mm_stats, save_path, duration_s=None):
    """Plot mouse dx/dy against time for GT and generated sequences."""
    def _frames_to_dxdy(frames):
        if not frames:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        arrs = [np.array(fr, dtype=np.float32) for fr in frames]
        arr = np.concatenate(arrs, axis=1)  # [2, T]
        arr_dn = _denorm_array(arr, mm_stats)
        arr_r = _round_mouse(arr_dn).astype(np.int32)
        return arr_r[0], arr_r[1]

    dx_gt, dy_gt = _frames_to_dxdy(gt_frames_list)
    dx_gen, dy_gen = _frames_to_dxdy(gen_frames_list)

    max_bins = max(len(dx_gt), len(dx_gen), len(dy_gt), len(dy_gen))
    if duration_s is None or duration_s <= 0:
        duration_s = (max_bins * BIN_MS) / 1000.0 if max_bins > 0 else 1.0

    t_gt = np.linspace(0.0, duration_s, len(dx_gt), endpoint=False) if len(dx_gt) else np.array([])
    t_gen = np.linspace(0.0, duration_s, len(dx_gen), endpoint=False) if len(dx_gen) else np.array([])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    if len(dx_gt):
        axes[0].plot(t_gt, dx_gt, label="GT dx", color="tab:blue")
    if len(dx_gen):
        axes[0].plot(t_gen, dx_gen, label="Gen dx", color="tab:orange", alpha=0.9)
    axes[0].set_ylabel("dx")
    axes[0].set_title("Mouse dx over time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    if len(dy_gt):
        axes[1].plot(t_gt, dy_gt, label="GT dy", color="tab:blue")
    if len(dy_gen):
        axes[1].plot(t_gen, dy_gen, label="Gen dy", color="tab:orange", alpha=0.9)
    axes[1].set_ylabel("dy")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Mouse dy over time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[1].set_xlim(0.0, duration_s)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    print(f"Saved mouse comparison plot to {save_path}")


def plot_keypress_compare(gt_frames_list, gen_frames_list, save_path, threshold=DECODE_KEY_ON_THRESH, duration_s=None):
    """Create GT/Gen keypress raster plots with active key-name annotations only."""
    num_keys = len(index_to_id)

    def _frames_to_matrix(frames):
        if not frames:
            return np.zeros((num_keys, 0), dtype=np.int8)
        arrs = [np.array(fr, dtype=np.float32) for fr in frames]
        arr = np.concatenate(arrs, axis=1)  # [K, T]
        mat = (arr >= threshold).astype(np.int8)
        return mat

    gt_mat = _frames_to_matrix(gt_frames_list)
    gen_mat = _frames_to_matrix(gen_frames_list)
    time_bins = max(gt_mat.shape[1], gen_mat.shape[1])

    if duration_s is None or duration_s <= 0:
        duration_s = (time_bins * BIN_MS) / 1000.0 if time_bins > 0 else 1.0

    gt_active = np.where(gt_mat.sum(axis=1) > 0)[0] if gt_mat.shape[1] > 0 else np.array([], dtype=np.int64)
    gen_active = np.where(gen_mat.sum(axis=1) > 0)[0] if gen_mat.shape[1] > 0 else np.array([], dtype=np.int64)
    active_idx = np.unique(np.concatenate([gt_active, gen_active])) if (gt_active.size or gen_active.size) else np.array([], dtype=np.int64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    def _render(ax, mat, title):
        if mat.shape[1] == 0 or active_idx.size == 0:
            ax.text(0.5, 0.5, "No keypress data", ha="center", va="center")
            ax.set_ylabel("Keys")
            ax.set_xlim(0.0, duration_s)
            return

        mat_active = mat[active_idx]
        ax.imshow(
            mat_active,
            aspect="auto",
            interpolation="nearest",
            cmap="gray_r",
            origin="lower",
            extent=(0.0, duration_s, -0.5, mat_active.shape[0] - 0.5),
        )
        ax.set_ylabel("Keys")
        ax.set_title(title)

        key_labels = []
        for i in active_idx:
            key_id = index_to_id.get(int(i))
            key_name = id_to_name.get(key_id, f"Key_{int(i)}")
            key_labels.append(key_name)
        ax.set_yticks(np.arange(len(active_idx)))
        ax.set_yticklabels(key_labels, fontsize=8)

    _render(axes[0], gt_mat, "GT keypresses")
    _render(axes[1], gen_mat, "Gen keypresses")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0.0, duration_s)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
    print(f"Saved keypress comparison plot to {save_path}")


def _decode_audio_tokens(batch_src):
    """batch_src: [B, L, 15, 128] or None -> list[np.ndarray|None] length B"""
    if batch_src is None:
        return [None]
    _init_models()
    B_, Lc, T_, C_ = batch_src.shape
    codes = batch_src.permute(0, 3, 1, 2).reshape(B_, 128, Lc * T_)

    model_dtype = next(_encodec_model.parameters()).dtype

    outs = []
    with torch.no_grad():
        for i in range(B_):
            inp = codes[i].unsqueeze(0).to(device=device, dtype=model_dtype)
            rec = _encodec_model.decoder(inp)
            rec = rec.clamp(-0.99, 0.99).squeeze(0).T.float().cpu().numpy()  # [T, C]
            outs.append(rec)
    return outs


def _unpack_fd(fd: FullData):
    vid = getattr(fd, "video", None)
    ai  = getattr(fd, "audio_speak", None)
    ao  = getattr(fd, "audio_hear", None)
    mouse = getattr(fd, "mouse_movement", None)
    keys = getattr(fd, "key_press", None)
    meta = getattr(fd, "metadata", None)
    return vid, ai, ao, mouse, keys, meta


def _frames_len_from(*xs):
    for x in xs:
        if isinstance(x, torch.Tensor) and x is not None:
            return x.size(1)
    raise ValueError("Could not infer sequence length L from any modality.")


def _save_wav(path: Path, x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(str(path), ENCODEC_SAMPLE_RATE, (x * 32767).astype(np.int16))


def _save_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _save_json_pretty(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------------------------- denorm helpers ---------------------------- #
def _extract_stats_from_metadata(metadata, modality_key):
    """
    Best-effort: try to find stats in metadata for the given modality.
    Expected shapes:
      - zscore: {"mode":"zscore","mean":[...], "std":[...]}
      - minmax: {"mode":"minmax","min":[...], "max":[...]}
    Returns dict or None.
    """
    metadata = FullData._unwrap_non_tensor(metadata)
    if metadata is None:
        return None
    # metadata may be dict, list, nested; walk it
    def _walk(obj):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _walk(v)
    for node in _walk(metadata):
        if modality_key in node and isinstance(node[modality_key], dict):
            mk = node[modality_key]
            if "mode" in mk and (("mean" in mk and "std" in mk) or ("min" in mk and "max" in mk) or mk["mode"] == "none"):
                return mk
    return None


def _denorm_array(arr, stats):
    """
    arr: np.ndarray [D, T]
    stats: {"mode": "...", arrays per-D}
    """
    if stats is None:
        return arr
    mode = stats.get("mode", "").lower()
    if mode == "zscore":
        mean = np.asarray(stats.get("mean"), dtype=np.float32).reshape(-1, 1)
        std  = np.asarray(stats.get("std"),  dtype=np.float32).reshape(-1, 1)
        return arr * std + mean
    if mode == "minmax":
        vmin = np.asarray(stats.get("min"), dtype=np.float32).reshape(-1, 1)
        vmax = np.asarray(stats.get("max"), dtype=np.float32).reshape(-1, 1)
        return arr * (vmax - vmin) + vmin
    # "none" or unknown
    return arr


# ---------------------------- parsing helpers ---------------------------- #
def _flatten_bins(frames_list, expect_shape):
    """
    frames_list: list of per-subframe arrays (lists) shaped expect_shape
    Returns (arr[D, T], total_bins=int).
    """
    if not frames_list:
        return None, 0
    arr = np.array(frames_list, dtype=np.float32)  # [S, D, 10]
    assert arr.shape[1:] == tuple(expect_shape), f"Expected per-frame shape {expect_shape}, got {arr.shape[1:]}"
    arr = np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1)  # [D, S*10]
    total_bins = arr.shape[1]
    return arr, total_bins


def _parse_keypress_events(keys_frames_list, _stats_unused=None):
    """
    keys_frames_list: list of [79,10] arrays (lists), expected already 0/1.
    Returns: {"events":[{"key_name","start_ms","end_ms"}...], "bin_ms":10, "total_ms":...}
    """
    arr, total_bins = _flatten_bins(keys_frames_list, expect_shape=(79, 10))
    if arr is None:
        return {"events": [], "bin_ms": int(BIN_MS), "total_ms": 0}

    pressed = arr >= 0.5  # binary already; keep threshold to be robust
    events = []
    for idx in range(pressed.shape[0]):
        key_id = index_to_id.get(idx)
        key_name = id_to_name.get(key_id, f"Key_{idx}")
        row = pressed[idx]
        on = False
        start = 0
        for t in range(total_bins):
            if row[t] and not on:
                on = True
                start = t
            elif not row[t] and on:
                on = False
                events.append({
                    "key_name": key_name,
                    "start_ms": int(start * BIN_MS),
                    "end_ms":   int(t * BIN_MS)
                })
        if on:
            events.append({
                "key_name": key_name,
                "start_ms": int(start * BIN_MS),
                "end_ms":   int(total_bins * BIN_MS)
            })
    return {"events": events, "bin_ms": int(BIN_MS), "total_ms": int(total_bins * BIN_MS)}


def _parse_mouse_bins(mouse_frames_list, stats):
    """
    mouse_frames_list: list of [2,10] arrays (lists).
    Returns: {"series":[{"time_ms","dx","dy"}...], "bin_ms":10, "total_ms":...}
    """
    arr, total_bins = _flatten_bins(mouse_frames_list, expect_shape=(2, 10))
    if arr is None:
        return {"series": [], "bin_ms": int(BIN_MS), "total_ms": 0}

    # denorm then round to integer minimal unit = 1
    arr = _denorm_array(arr, stats)  # [2, T]
    arr = _round_mouse(arr).astype(int)

    dx = arr[0]; dy = arr[1]
    series = [{"time_ms": int(t * BIN_MS), "dx": int(dx[t]), "dy": int(dy[t])}
              for t in range(total_bins)]
    return {"series": series, "bin_ms": int(BIN_MS), "total_ms": int(total_bins * BIN_MS)}


def _name_key_rows_inline(fr_np_int):
    """
    fr_np_int: np.ndarray (79,10) with ints (0/1).
    Return a list of lines for pretty JSON with arrays on a single line.
    """
    lines = []
    n_keys = fr_np_int.shape[0]
    for idx in range(n_keys):
        key_id = index_to_id.get(idx)
        key_name = id_to_name.get(key_id, f"Key_{idx}")
        arr = fr_np_int[idx].astype(int).tolist()
        arr_text = ",".join(str(int(x)) for x in arr)
        comma = "," if idx < n_keys - 1 else ""
        lines.append(f'    "{key_name}": [{arr_text}]{comma}')
    return lines


def _format_keypress_frames_inline(frames_bin_int):
    """
    frames_bin_int: list of np.ndarray (79,10) with ints 0/1
    Returns a JSON list string with each key on its own line and arrays inline.
    """
    out_lines = ["["]
    for i, fr in enumerate(frames_bin_int):
        out_lines.append("  {")
        out_lines.extend(_name_key_rows_inline(fr))
        out_lines.append("  }" + ("," if i < len(frames_bin_int) - 1 else ""))
    out_lines.append("]")
    return "\n".join(out_lines)


# ------------------------------------------------------------------ #
#  PUBLIC: decode FullData → overlay video, plus optional per-modality dumps
# ------------------------------------------------------------------ #
def decode_and_save(pred_fd: FullData,
                    gt_fd: FullData,
                    out_dir: Path,
                    *,
                    window_length: int,
                    video_filename: str = "full_modality_overlay.mp4",
                    make_audio_plots: bool = True,
                    store_decoded_generated: bool = False,
                    store_decoded_gt: bool = False,
                    metadata=None) -> None:
    """
    1) Render a side-by-side GT(top) vs Pred(bottom) overlay video at out_dir/video_filename.
    2) Optionally (flags):
         out_dir/generated/{video.mp4,audio_speak.wav,audio_hear.wav,key_press.json,mouse_movement.json}
         out_dir/gt/{video.mp4,audio_speak.wav,audio_hear.wav,key_press.json,mouse_movement.json}

       JSON is structured as **original binned, pre-normalization** format:
       - key_press: per-frame 79×10 strictly 0/1 (arrays inline per key), BIN_MS=10
       - mouse_movement: per-frame 2×10 denormalized dx/dy **rounded to int**, BIN_MS=10
       - Both subframes are preserved (so each latent frame contributes 2×(79×10 / 2×10) bins).
    """
    _init_models()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_mp4 = out_dir / video_filename

    # stats for denorm (provided by caller via metadata; no on-the-fly estimation)
    kp_stats = _extract_stats_from_metadata(metadata, "key_press")          # usually "none"
    mm_stats = _extract_stats_from_metadata(metadata, "mouse_movement")     # zscore/minmax with params

    # Inverse normalization before decoding back to human-interpretable outputs.
    # This is centralized in DataModule to mirror normalize_full_data.
    pred_fd = DataModule.denormalize_full_data(pred_fd)
    gt_fd = DataModule.denormalize_full_data(gt_fd)
    # Mouse denorm is already applied via denormalize_full_data in this case.
    mm_stats = None

    # ---- unpack tensors ----
    v_pred, ai_pred, ao_pred, m_pred, k_pred, _ = _unpack_fd(pred_fd)
    v_gt,   ai_gt,   ao_gt,   m_gt,   k_gt,   _ = _unpack_fd(gt_fd)

    L = _frames_len_from(v_pred, ai_pred, ao_pred, m_pred, k_pred, v_gt, ai_gt, ao_gt, m_gt, k_gt)

    # ---- decode audio for per-modality dumps + overlay mux ----
    audio_speak_gen  = _decode_audio_tokens(ai_pred)[0] if ai_pred is not None else None
    audio_hear_gen = _decode_audio_tokens(ao_pred)[0] if ao_pred is not None else None
    audio_speak_gt   = _decode_audio_tokens(ai_gt)[0]   if ai_gt   is not None else None
    audio_hear_gt  = _decode_audio_tokens(ao_gt)[0]   if ao_gt   is not None else None

    # ---- build overlay frames and simultaneously collect content frames
    overlay_frames = []
    gen_frames_content = []
    gt_frames_content  = []

    # decoded per-subframe arrays (to be flattened & saved)
    gen_keys_frames = []
    gen_mouse_frames = []
    gt_keys_frames = []
    gt_mouse_frames = []

    # ---- decode frames (video latents, keypresses, mouse) with progress bar ----
    print(f"Decoding {L} frames of multimodal data...")
    for t in tqdm(range(L), desc="Decoding Clips", unit="frame"):
        for f in range(DECODE_SUBFRAMES_PER_FRAME):
            idx = t * DECODE_SUBFRAMES_PER_FRAME + f
            mouse_start = f * DECODE_BINS_PER_SUBFRAME
            mouse_end = mouse_start + DECODE_BINS_PER_SUBFRAME
            key_start = f * DECODE_KEY_TOKENS_PER_SUBFRAME
            key_end = key_start + DECODE_KEY_TOKENS_PER_SUBFRAME

            # -------- GT --------
            if v_gt is not None:
                latent = v_gt[0, t, f].unsqueeze(0).to(device)
                img = decode_latents(_vae, latent)
                fr_gt_content = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                fr_gt_content = cv2.resize(fr_gt_content, DECODE_FINAL_FRAME_SIZE)
            else:
                fr_gt_content = np.full((DECODE_FINAL_FRAME_SIZE[1], DECODE_FINAL_FRAME_SIZE[0], 3), 255, np.uint8)

            m_gt_chunk = m_gt[0, t, mouse_start:mouse_end].transpose(0, 1).detach().cpu().numpy() if m_gt is not None else np.zeros((2, DECODE_BINS_PER_SUBFRAME))
            m_gt_chunk_draw = _round_mouse(_denorm_array(m_gt_chunk, mm_stats))  # denorm + round for overlay
            if k_gt is not None:
                with torch.no_grad():
                    k_gt_latent = k_gt[0, t, key_start:key_end].transpose(0, 1).unsqueeze(0).to(device)
                    dec = _keypress_ae.decoder(k_gt_latent)
                k_gt_chunk = dec.squeeze(0).detach().cpu().numpy()  # (79, 10) continuous @100Hz
            else:
                k_gt_chunk = np.zeros((len(keyboard_indices) + len(mouse_indices), DECODE_BINS_PER_SUBFRAME))  # 79,10

            fr_gt_overlay = overlay_data_on_frame(fr_gt_content, m_gt_chunk_draw, k_gt_chunk, None, None, idx)
            h, w, _ = fr_gt_overlay.shape
            cv2.rectangle(fr_gt_overlay, (0, 0), (w - 1, h - 1), (255, 0, 0), 4)

            # -------- Pred --------
            if v_pred is not None:
                latent = v_pred[0, t, f].unsqueeze(0).to(device)
                img = decode_latents(_vae, latent)
                fr_pr_content = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                fr_pr_content = cv2.resize(fr_pr_content, DECODE_FINAL_FRAME_SIZE)
            else:
                fr_pr_content = np.full((DECODE_FINAL_FRAME_SIZE[1], DECODE_FINAL_FRAME_SIZE[0], 3), 255, np.uint8)

            m_pr_chunk = m_pred[0, t, mouse_start:mouse_end].transpose(0, 1).detach().cpu().numpy() if m_pred is not None else np.zeros((2, DECODE_BINS_PER_SUBFRAME))
            m_pr_chunk_draw = _round_mouse(_denorm_array(m_pr_chunk, mm_stats))  # denorm + round for overlay
            if k_pred is not None:
                with torch.no_grad():
                    k_pr_latent = k_pred[0, t, key_start:key_end].transpose(0, 1).unsqueeze(0).to(device)
                    dec = _keypress_ae.decoder(k_pr_latent)
                k_pr_chunk = dec.squeeze(0).detach().cpu().numpy()  # (79, 10)
            else:
                k_pr_chunk = np.zeros((len(keyboard_indices) + len(mouse_indices), DECODE_BINS_PER_SUBFRAME))

            fr_pr_overlay = overlay_data_on_frame(fr_pr_content, m_pr_chunk_draw, k_pr_chunk, None, None, idx)
            if idx < window_length * DECODE_SUBFRAMES_PER_FRAME:
                cv2.rectangle(fr_pr_overlay, (0, 0), (w - 1, h - 1), (255, 0, 0), 4)

            # collect overlay/content
            overlay_frames.append(cv2.vconcat([fr_gt_overlay, fr_pr_overlay]))
            gt_frames_content.append(fr_gt_content)
            gen_frames_content.append(fr_pr_content)

            # collect arrays for JSON (per-subframe) — model-space arrays; writers will denorm/round
            gt_keys_frames.append(k_gt_chunk.tolist())
            gt_mouse_frames.append(m_gt_chunk.tolist())
            gen_keys_frames.append(k_pr_chunk.tolist())
            gen_mouse_frames.append(m_pr_chunk.tolist())

    # ---- render the overlay video (always when --decode) ----
    dur = len(overlay_frames) / DECODE_VIDEO_FPS
    ns  = int(dur * ENCODEC_SAMPLE_RATE)
    a_in  = audio_speak_gen  if audio_speak_gen  is not None else np.zeros(ns, dtype=np.float32)
    a_out = audio_hear_gen if audio_hear_gen is not None else np.zeros(ns, dtype=np.float32)

    print(f"Rendering ⇢  {overlay_mp4}")
    save_video(overlay_frames, overlay_mp4, DECODE_VIDEO_FPS, a_in, a_out)

    # ---- writers that emit **original binned / denormed** formats ----
    def _write_kp_json(path, frames_list):
        """
        frames_list: list of (79,10) float arrays from AE decoder (continuous).
        Save strictly 0/1 per-bin (original binned format), plus parsed events.
        Keys are one per line, arrays kept on a single line per your preference.
        """
        # binarize frames → ints
        bin_frames_int = []
        for fr in frames_list:
            fr_np = np.array(fr, dtype=np.float32)                  # (79,10)
            fr_bin = (fr_np >= DECODE_KEY_ON_THRESH).astype(np.int32)      # 0/1 ints
            bin_frames_int.append(fr_bin)

        # pretty inline "frames" block
        frames_inline = _format_keypress_frames_inline(bin_frames_int)

        # parsed events use the same binarized frames (cast back to float for parser expectations)
        parsed = _parse_keypress_events([fr.astype(np.float32).tolist() for fr in bin_frames_int], _stats_unused=None)

        # build JSON text with placeholder replacement so only arrays are inline
        obj = {
            "schema_version": 1,
            "bin_ms": int(BIN_MS),
            "raw_decoded": {
                "shape_per_frame": [79, 10],
                "frames": "__INLINE_FRAMES__"
            },
            "parsed_decoded": parsed
        }
        text = json.dumps(obj, indent=2, ensure_ascii=False)
        text = text.replace('"__INLINE_FRAMES__"', frames_inline)
        _save_text(path, text)

    def _write_mm_json(path, frames_list):
        """
        frames_list: list of (2,10) arrays in model space.
        Save denormalized **rounded** dx/dy (int) per frame, and a flattened time-series.
        """
        # raw: named rows, denormalized + rounded values
        raw_frames_named = []
        for fr in frames_list:
            fr_np = np.array(fr, dtype=np.float32)     # (2,10) normalized
            fr_dn = _denorm_array(fr_np, mm_stats)     # denorm
            fr_dn_r = _round_mouse(fr_dn)              # round to int
            raw_frames_named.append({"dx": fr_dn_r[0].astype(int).tolist(),
                                     "dy": fr_dn_r[1].astype(int).tolist()})

        obj = {
            "schema_version": 1,
            "bin_ms": int(BIN_MS),
            "raw_decoded": {
                "shape_per_frame": [2, 10],
                "frames": raw_frames_named
            },
            # Flattened series, denormalized & rounded inside parser
            "parsed_decoded": _parse_mouse_bins(frames_list, mm_stats)
        }
        _save_json_pretty(path, obj)

    # ---- optionally save per-modality content for generated ----
    gen_dir = out_dir / "generated" if store_decoded_generated else None
    if gen_dir:
        gen_dir.mkdir(exist_ok=True, parents=True)
        if gen_frames_content:
            save_video(gen_frames_content, gen_dir / "video.mp4", DECODE_VIDEO_FPS)
        if audio_speak_gen is not None:  _save_wav(gen_dir / "audio_speak.wav",  audio_speak_gen)
        if audio_hear_gen is not None: _save_wav(gen_dir / "audio_hear.wav", audio_hear_gen)
        _write_kp_json(gen_dir / "key_press.json", gen_keys_frames)
        _write_mm_json(gen_dir / "mouse_movement.json", gen_mouse_frames)

    # ---- optionally save per-modality content for GT ----
    gt_dir = out_dir / "gt" if store_decoded_gt else None
    if gt_dir:
        gt_dir.mkdir(exist_ok=True, parents=True)
        if gt_frames_content:
            save_video(gt_frames_content, gt_dir / "video.mp4", DECODE_VIDEO_FPS)
        if audio_speak_gt is not None:   _save_wav(gt_dir  / "audio_speak.wav",  audio_speak_gt)
        if audio_hear_gt is not None:  _save_wav(gt_dir  / "audio_hear.wav", audio_hear_gt)
        _write_kp_json(gt_dir / "key_press.json", gt_keys_frames)
        _write_mm_json(gt_dir / "mouse_movement.json", gt_mouse_frames)

    # ---- save metadata JSON for evaluation metrics lookup ----
    if metadata is not None:
        metadata_dict = FullData._unwrap_non_tensor(metadata)
        if metadata_dict and isinstance(metadata_dict, list) and len(metadata_dict) > 0:
            # Extract the first metadata dict from the list
            meta_entry = metadata_dict[0] if isinstance(metadata_dict, list) else metadata_dict
            metadata_path = out_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(meta_entry, f, indent=2, ensure_ascii=False, default=str)

    # optional waveform plots (independent of store flags)
    if make_audio_plots and audio_speak_gt is not None and audio_speak_gen is not None:
        plot_wave_compare(audio_speak_gt, audio_speak_gen,
                          ENCODEC_SAMPLE_RATE,
                          out_dir / "audio_speak_wave_cmp.png",
                          title_gt="GT audio_speak", title_gen="Gen audio_speak")
    if make_audio_plots and audio_hear_gt is not None and audio_hear_gen is not None:
        plot_wave_compare(audio_hear_gt, audio_hear_gen,
                          ENCODEC_SAMPLE_RATE,
                          out_dir / "audio_hear_wave_cmp.png",
                          title_gt="GT audio_hear", title_gen="Gen audio_hear")

    # ---- additional visual comparisons ----
    try:
        mouse_cmp_path = out_dir / "mouse_path_cmp.png"
        keypress_cmp_path = out_dir / "key_press_raster_cmp.png"
        plot_mouse_compare(gt_mouse_frames, gen_mouse_frames, mm_stats, mouse_cmp_path)
        plot_keypress_compare(gt_keys_frames, gen_keys_frames, keypress_cmp_path)
    except Exception as e:
        print(f"Warning: Failed to save additional visual comparisons: {e}")
