#!/usr/bin/env python3
"""
evdev_keys.py — Wayland-proof keyboard + mouse-button capture.

The recorder's on_press/on_release/on_click use pynput, which is X11-based and
goes SILENT under Wayland — so key_press/click context never reaches the model
even though evdev mouse MOTION works. This reads EV_KEY straight from the kernel
input layer (same source as the motion thread) and appends the EXACT same buffer
entries the encoders expect:

    keyboard: {"timestamp", "key": <GLFW code>, "action": "PRESS"|"RELEASE"}
    click:    {"timestamp", "action": "LEFT_PRESS" | "LEFT_RELEASE" | ...}

Start it from run_live.py INSTEAD OF the pynput listeners:

    import evdev_keys
    evdev_keys.start()           # call after rec._recording.set()

Needs read access to /dev/input/event*  (be in the 'input' group, or run as root).
"""
import threading
import selectors

import minecraft_input_recorder as recorder

try:
    import evdev
    from evdev import ecodes
except Exception:
    evdev = None


# --- Linux evdev keycode -> the GLFW code rec._key_code() emits -------------- #
# Letters/digits map by character; specials match _key_code's _SPECIAL table.
_GLFW_SPECIAL = {
    ecodes.KEY_SPACE: 32, ecodes.KEY_ENTER: 257, ecodes.KEY_TAB: 258,
    ecodes.KEY_BACKSPACE: 259, ecodes.KEY_INSERT: 260, ecodes.KEY_DELETE: 261,
    ecodes.KEY_RIGHT: 262, ecodes.KEY_LEFT: 263, ecodes.KEY_DOWN: 264,
    ecodes.KEY_UP: 265, ecodes.KEY_PAGEUP: 266, ecodes.KEY_PAGEDOWN: 267,
    ecodes.KEY_HOME: 268, ecodes.KEY_END: 269, ecodes.KEY_CAPSLOCK: 280,
    ecodes.KEY_SCROLLLOCK: 281, ecodes.KEY_NUMLOCK: 282, ecodes.KEY_SYSRQ: 283,
    ecodes.KEY_PAUSE: 284, ecodes.KEY_F1: 290, ecodes.KEY_F2: 291,
    ecodes.KEY_F3: 292, ecodes.KEY_F4: 293, ecodes.KEY_F5: 294, ecodes.KEY_F6: 295,
    ecodes.KEY_F7: 296, ecodes.KEY_F8: 297, ecodes.KEY_F9: 298, ecodes.KEY_F10: 299,
    ecodes.KEY_F11: 300, ecodes.KEY_F12: 301, ecodes.KEY_LEFTSHIFT: 340,
    ecodes.KEY_RIGHTSHIFT: 344, ecodes.KEY_LEFTCTRL: 341, ecodes.KEY_RIGHTCTRL: 345,
    ecodes.KEY_LEFTALT: 342, ecodes.KEY_RIGHTALT: 346, ecodes.KEY_ESC: 256,
} if evdev is not None else {}

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
_CHAR = {}
if evdev is not None:
    for _c in _LETTERS:
        _CHAR[getattr(ecodes, f"KEY_{_c}")] = ord(_c)            # rec uses upper()
    for _d in _DIGITS:
        _CHAR[getattr(ecodes, f"KEY_{_d}")] = ord(_d)

_BTN = {
    ecodes.BTN_LEFT:   "LEFT",
    ecodes.BTN_RIGHT:  "RIGHT",
    ecodes.BTN_MIDDLE: "MIDDLE",
} if evdev is not None else {}


def _glfw_code(keycode):
    if keycode in _GLFW_SPECIAL:
        return _GLFW_SPECIAL[keycode]
    if keycode in _CHAR:
        return _CHAR[keycode]
    return -1   # unmapped; matches _key_code's fallback






def start():
    """Start the evdev key/click reader in a daemon thread. Returns True if it
    will actually capture (evdev present), False otherwise."""
    if evdev is None:
        print("[evdev_keys] python-evdev not installed — keys/clicks NOT captured "
              "on Wayland. Install:  pip install evdev")
        return False
    threading.Thread(target=_loop, daemon=True).start()
    return True


if __name__ == "__main__":
    # Quick standalone check: prints captured key/click events for a few seconds.
    recorder._recording.set()
    start()
    import time
    print("Press keys / click for 8s…")
    t0 = time.time()
    while time.time() - t0 < 8:
        time.sleep(1)
        with recorder._lock:
            print(f"  keys={len(recorder._keyboard_buf)}  clicks={len(recorder._click_buf)}")
