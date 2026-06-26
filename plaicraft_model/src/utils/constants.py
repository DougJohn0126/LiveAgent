"""Canonical Plaicraft data constants.

This file is the source of truth for all modality frame rates and tensor shapes.

Smallest aligned training unit:
  - duration: 200 ms
  - video: 2 frames at 10 Hz
  - audio: 15 tokens at 75 Hz
  - keyboard: 10 tokens at 50 Hz, represented as [10, 16]
  - mouse: 20 tokens at 100 Hz, represented as [20, 2]

After dataloader collation, modalities are shaped as:
  - video: [B, T, 2, 4, 96, 160]
  - audio_{speak,hear}: [B, T, 15, 128]
  - key_press: [B, T, 10, 16]
  - mouse_movement: [B, T, 20, 2]
"""

from math import prod


UNIT_DURATION_SECONDS = 0.2
UNIT_DURATION_MS = int(UNIT_DURATION_SECONDS * 1000)

VIDEO_FPS = 10
AUDIO_TOKEN_FPS = 75
KEYBOARD_FPS = 50
MOUSE_FPS = 100

VIDEO_FRAMES_PER_UNIT = int(VIDEO_FPS * UNIT_DURATION_SECONDS)  # 2
AUDIO_TOKENS_PER_UNIT = int(AUDIO_TOKEN_FPS * UNIT_DURATION_SECONDS)  # 15
KEYBOARD_TOKENS_PER_UNIT = int(KEYBOARD_FPS * UNIT_DURATION_SECONDS)  # 10
MOUSE_TOKENS_PER_UNIT = int(MOUSE_FPS * UNIT_DURATION_SECONDS)  # 20

VIDEO_LATENT_CHANNELS = 4
VIDEO_LATENT_HEIGHT = 96
VIDEO_LATENT_WIDTH = 160
VIDEO_LATENT_SHAPE = (VIDEO_LATENT_CHANNELS, VIDEO_LATENT_HEIGHT, VIDEO_LATENT_WIDTH)

AUDIO_FEATURE_DIM = 128
KEYBOARD_FEATURE_DIM = 16
MOUSE_FEATURE_DIM = 2

KEYBOARD_TOKENS_PER_VIDEO_FRAME = KEYBOARD_TOKENS_PER_UNIT // VIDEO_FRAMES_PER_UNIT  # 5
MOUSE_TOKENS_PER_VIDEO_FRAME = MOUSE_TOKENS_PER_UNIT // VIDEO_FRAMES_PER_UNIT  # 10

# Sampling / decode constants
ENCODEC_SAMPLE_RATE = 24000
BIN_MS = UNIT_DURATION_MS // MOUSE_TOKENS_PER_UNIT  # 10 ms

# Decode pipeline runtime/layout constants
DECODE_USE_FP16 = True
DECODE_VIDEO_FPS = float(VIDEO_FPS)
DECODE_FRAME_DURATION_MS = 1000.0 / DECODE_VIDEO_FPS
DECODE_SUBFRAMES_PER_FRAME = VIDEO_FRAMES_PER_UNIT
DECODE_BINS_PER_SUBFRAME = MOUSE_TOKENS_PER_VIDEO_FRAME
DECODE_KEY_TOKENS_PER_SUBFRAME = KEYBOARD_TOKENS_PER_VIDEO_FRAME

# Decode rendering constants
DECODE_FINAL_FRAME_SIZE = (1280, 768)  # (W, H)
DECODE_TOP_BAR_HEIGHT = 100
DECODE_KEY_BOX_HEIGHT = 90
DECODE_KEY_BOX_PADDING_X = 10
DECODE_KEY_BOX_PADDING_Y = 10
DECODE_KEY_ROW_GAP = 5
DECODE_LEFT_SECTION_MAX_W = 0.45
DECODE_KEY_FONT_SCALE = 1.5
DECODE_KEY_FONT_THICKNESS = 2
DECODE_MOUSE_LINE_COLOR = (0, 255, 255)
DECODE_MOUSE_LINE_THICKNESS = 3
DECODE_MOUSE_ARROW_TIP_LEN = 0.2
DECODE_KEY_ON_THRESH = 0.5
DECODE_MOUSE_ACTION_NAMES = ("mouse_left", "mouse_right", "scroll_up", "scroll_down")

MODALITY_SHAPES = {
    "video": (VIDEO_FRAMES_PER_UNIT, *VIDEO_LATENT_SHAPE),
    "audio_speak": (AUDIO_TOKENS_PER_UNIT, AUDIO_FEATURE_DIM),
    "audio_hear": (AUDIO_TOKENS_PER_UNIT, AUDIO_FEATURE_DIM),
    "key_press": (KEYBOARD_TOKENS_PER_UNIT, KEYBOARD_FEATURE_DIM),
    "mouse_movement": (MOUSE_TOKENS_PER_UNIT, MOUSE_FEATURE_DIM),
}

VALID_MODALITIES = set(MODALITY_SHAPES.keys())

MODALITY_FLAT_DIMS = {
    name: prod(shape) for name, shape in MODALITY_SHAPES.items()
}

MODALITY_TO_LATENT = {
    "video": "frame_latent",
    "audio_speak": "audio_speak_latent",
    "audio_hear": "audio_hear_latent",
    "key_press": "keyboard_latent",
    "mouse_movement": "mouse_latent",
}
