"""
inputInjector.py
----------------
Replays key press and mouse movement JSON files into a game window.
  - Mouse frames are single (dx, dy) values per bin, not lists — fixed zip
  - Frame timing uses perf_counter + absolute target times, not relative sleeps
  - 1ms sleep resolution requested on Windows (no-op / not needed on Linux)
  - Cross-platform input backend:
      * Windows: pydirectinput (DirectInput) — bypasses the Windows message
        queue and works with games that ignore SendInput.
      * Linux:   evdev uinput — creates a kernel-level virtual input device,
        the closest analog to DirectInput. Works with games that read raw
        input / ignore X11 (XTEST) synthetic events.
 
Linux requirements:
    pip install evdev
    Access to /dev/uinput. Either run as root, or grant your user access
    with a udev rule, e.g. create /etc/udev/rules.d/99-uinput.rules with:
        KERNEL=="uinput", GROUP="input", MODE="0660"
    add yourself to the `input` group, then `sudo modprobe uinput` and re-login.
 
Usage:
    python inputInjector.py --keys key_press.json --mouse mouse_movement.json
"""
 
import json
import time
import threading
import argparse
import sys
import sqlite3
import scripts_replay.convert_mouse_movement 
import scripts_replay.convert_keyboard_movement
import scripts_replay.data_preprocessor as data_preprocessor
from pathlib import Path
 
#from encode_key_press import main as encode_key_press_main
#from encode_video import main as encode_video_main
#from decode import decode_keypress_latents
 
OUTPUT_DIR = "processed_data"
 
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
 
# ── High-resolution timer shim ────────────────────────────────────────────────
# On Windows the default timer granularity is ~15.6ms, so we request 1ms.
# On Linux nanosleep already provides sub-millisecond resolution, so these are
# no-ops.
if IS_WINDOWS:
    import ctypes
    _winmm = ctypes.windll.winmm
 
    def _timer_begin():
        _winmm.timeBeginPeriod(1)
 
    def _timer_end():
        _winmm.timeEndPeriod(1)
else:
    def _timer_begin():
        pass
 
    def _timer_end():
        pass
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Input backends
# ──────────────────────────────────────────────────────────────────────────────
#
# Each backend exposes the same small interface so the replay logic below stays
# platform-agnostic:
#     key_down(key_name) / key_up(key_name)
#     mouse_down(button)  / mouse_up(button)   button is "left" or "right"
#     move_rel(dx, dy)
#     close()
#
# key_name uses the PLAICraft schema (e.g. "space", "Shift_L", "w", "1").
 
class WindowsBackend:
    # PLAICraft schema → pydirectinput key names
    KEY_MAP = {
        "space":      "space",
        "Return":     "enter",
        "Escape":     "esc",
        "Tab":        "tab",
        "BackSpace":  "backspace",
        "Delete":     "delete",
        "Shift_L":    "shiftleft",
        "Shift_R":    "shiftright",
        "Control_L":  "ctrlleft",
        "Control_R":  "ctrlright",
        "Alt_L":      "altleft",
        "Alt_R":      "altright",
        "Super_L":    "winleft",
        "Caps_Lock":  "capslock",
        "Up":         "up",
        "Down":       "down",
        "Left":       "left",
        "Right":      "right",
        "F1":         "f1",
        "F3":         "f3",
        "F5":         "f5",
        # letters and digits map 1:1 (a→a, w→w, 1→1 etc.)
    }
 
    def __init__(self):
        import pydirectinput as pdi
        pdi.PAUSE = 0  # disable pydirectinput's built-in inter-call delay
        self._pdi = pdi
 
    def _map(self, key_name):
        return self.KEY_MAP.get(key_name, key_name)
 
    def key_down(self, key_name):
        mapped = self._map(key_name)
        if mapped is None:
            return
        self._pdi.keyDown(mapped)
 
    def key_up(self, key_name):
        mapped = self._map(key_name)
        if mapped is None:
            return
        self._pdi.keyUp(mapped)
 
    def mouse_down(self, button):
        self._pdi.mouseDown(button=button)
 
    def mouse_up(self, button):
        self._pdi.mouseUp(button=button)
 
    def move_rel(self, dx, dy):
        self._pdi.moveRel(dx, dy, relative=True)
 
    def close(self):
        pass
 
 
class LinuxBackend:
    """Kernel-level virtual input device via evdev/uinput."""
 
    def __init__(self):
        try:
            from evdev import UInput, ecodes as e
        except ImportError as exc:
            raise RuntimeError(
                "The 'evdev' package is required on Linux. Install it with: "
                "pip install evdev"
            ) from exc
 
        self._e = e
        self.KEY_MAP = self._build_keymap(e)
 
        # uinput writes from the key thread and the mouse thread can interleave,
        # so serialize all writes.
        self._lock = threading.Lock()
 
        # Declare every capability up front; uinput devices are fixed at
        # creation time.
        key_codes = sorted({code for code in self.KEY_MAP.values()
                             if code is not None})
        key_codes += [e.BTN_LEFT, e.BTN_RIGHT]
        capabilities = {
            e.EV_KEY: key_codes,
            e.EV_REL: [e.REL_X, e.REL_Y],
        }
 
        try:
            self._ui = UInput(capabilities, name="inputInjector-virtual")
        except PermissionError as exc:
            raise RuntimeError(
                "Permission denied opening /dev/uinput. Run as root, or add a "
                "udev rule granting access and add yourself to the 'input' "
                "group (see the module docstring)."
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "/dev/uinput not found. Load the kernel module with: "
                "sudo modprobe uinput"
            ) from exc
 
        # Give the desktop/input stack a moment to register the new device
        # before we start sending events to it.
        time.sleep(0.5)
 
    @staticmethod
    def _build_keymap(e):
        m = {
            "space":      e.KEY_SPACE,
            "Return":     e.KEY_ENTER,
            "Escape":     e.KEY_ESC,
            "Tab":        e.KEY_TAB,
            "BackSpace":  e.KEY_BACKSPACE,
            "Delete":     e.KEY_DELETE,
            "Shift_L":    e.KEY_LEFTSHIFT,
            "Shift_R":    e.KEY_RIGHTSHIFT,
            "Control_L":  e.KEY_LEFTCTRL,
            "Control_R":  e.KEY_RIGHTCTRL,
            "Alt_L":      e.KEY_LEFTALT,
            "Alt_R":      e.KEY_RIGHTALT,
            "Super_L":    e.KEY_LEFTMETA,
            "Caps_Lock":  e.KEY_CAPSLOCK,
            "Up":         e.KEY_UP,
            "Down":       e.KEY_DOWN,
            "Left":       e.KEY_LEFT,
            "Right":      e.KEY_RIGHT,
            "F1":         e.KEY_F1,
            "F3":         e.KEY_F3,
            "F5":         e.KEY_F5,
        }
        for c in "abcdefghijklmnopqrstuvwxyz":
            m[c] = getattr(e, f"KEY_{c.upper()}")
        for d in "0123456789":
            m[d] = getattr(e, f"KEY_{d}")
        return m
 
    def _emit_key(self, code, value):
        e = self._e
        with self._lock:
            self._ui.write(e.EV_KEY, code, value)
            self._ui.syn()
 
    def key_down(self, key_name):
        code = self.KEY_MAP.get(key_name)
        if code is None:
            print(f"[warn] no Linux mapping for key {key_name!r}, skipping")
            return
        self._emit_key(code, 1)
 
    def key_up(self, key_name):
        code = self.KEY_MAP.get(key_name)
        if code is None:
            return
        self._emit_key(code, 0)
 
    def mouse_down(self, button):
        code = self._e.BTN_LEFT if button == "left" else self._e.BTN_RIGHT
        self._emit_key(code, 1)
 
    def mouse_up(self, button):
        code = self._e.BTN_LEFT if button == "left" else self._e.BTN_RIGHT
        self._emit_key(code, 0)
 
    def move_rel(self, dx, dy):
        e = self._e
        with self._lock:
            if dx:
                self._ui.write(e.EV_REL, e.REL_X, dx)
            if dy:
                self._ui.write(e.EV_REL, e.REL_Y, dy)
            self._ui.syn()
 
    def close(self):
        try:
            self._ui.close()
        except Exception:
            pass
 
 
def make_backend():
    if IS_WINDOWS:
        return WindowsBackend()
    if IS_LINUX:
        return LinuxBackend()
    raise RuntimeError(f"Unsupported platform: {sys.platform}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Key replay
# ──────────────────────────────────────────────────────────────────────────────
 
def replay_keys(key_events, start_perf, backend):
    # 1. Explode each event into two scheduled actions.
    actions = []
    for evt in key_events:
        actions.append((evt["start_ms"], "down", evt["key_name"]))
        actions.append((evt["end_ms"],   "up",   evt["key_name"]))
 
    # 2. Sort by time. On a tie, do releases before presses so a key that
    #    ends exactly as another begins doesn't stomp on it.
    actions.sort(key=lambda a: (a[0], 0 if a[1] == "up" else 1))
 
    # 3. Fire each action at its absolute time on the shared clock.
    for t_ms, action, key in actions:
        wait = (start_perf + t_ms / 1000.0) - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
 
        print(key)
        if key in ("mouse_left", "mouse_right"):
            button = "left" if key == "mouse_left" else "right"
            (backend.mouse_down if action == "down" else backend.mouse_up)(button)
        else:
            (backend.key_down if action == "down" else backend.key_up)(key)
 
# ──────────────────────────────────────────────────────────────────────────────
# Mouse movement replay
# ──────────────────────────────────────────────────────────────────────────────
 
def replay_mouse(mouse_frames: list, bin_ms: float, start_perf: float, backend):
    print("Replay starting")
    """
    mouse_frames: list of frames, each frame is one bin of bin_ms milliseconds.
 
    Each frame can be:
      - {"dx": float, "dy": float}          ← single value per bin (most common)
      - {"dx": [f,f,...], "dy": [f,f,...]}  ← list of sub-samples per bin
 
    Movement within a bin is spread evenly across its duration.
    """
    for frame_idx, frame in enumerate(mouse_frames):
 
        dx_raw = frame[1]
        dy_raw = frame[2]
 
        # Normalise to lists
        if isinstance(dx_raw, (int, float)):
            dx_list = [float(dx_raw)]
            dy_list = [float(dy_raw)]
        else:
            dx_list = [float(v) for v in dx_raw]
            dy_list = [float(v) for v in dy_raw]
 
        n_subs = len(dx_list)
        if n_subs == 0:
            continue
 
        sub_interval_s = bin_ms / 1000.0  # → 0.010s (10ms) ✓
        bin_start_s    = frame_idx * n_subs * sub_interval_s
 
        for sub_idx, (dx, dy) in enumerate(zip(dx_list, dy_list)):
            # Absolute target time for this sub-sample
            target = start_perf + bin_start_s + sub_idx * sub_interval_s
 
            wait = target - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
 
            dx_i = int(round(dx))
            dy_i = int(round(dy))
            if dx_i != 0 or dy_i != 0:
                backend.move_rel(dx_i, dy_i)
    print("Replay complete")
 
  
# ──────────────────────────────────────────────────────────────────────────────
# Replay(deprecated)
# ──────────────────────────────────────────────────────────────────────────────
 
def replay(mousedata: str, clickdata: str, keydata: str, videopath: Path):
 
    db, cur, new_video_path = data_preprocessor.preprocess_data(mousedata, clickdata, keydata, videopath, OUTPUT_DIR)
 
    cur.execute(""" SELECT timestamp, mouseDX, mouseDY FROM mouse_movement """)
 
    mouse_data = cur.fetchall()
    encoded_data = encode_key_press_main.main(db, 32, "cpu", False)
    encoded_video = encode_video_main.main(new_video_path, 32,  False);
    
 
    keys_data = decode_keypress_latents.main(encoded_data)
    if (keys_data == ""):
        key_events  = []
    else:
        keys_data = json.loads(keys_data)
        key_events  = keys_data["parsed_decoded"]["events"]
    mouse_events = mouse_data
    bin_ms       = 10
 
 
    backend = make_backend()
    _timer_begin()
 
    start_perf = time.perf_counter()
    t_keys  = threading.Thread(target=replay_keys,
                                args=(key_events, start_perf, backend), daemon=True)
    t_mouse = threading.Thread(target=replay_mouse,
                                args=(mouse_events, bin_ms, start_perf, backend), daemon=True)
    print("Replay starting.")
    t_keys.start()
    t_mouse.start()
    #t_keys.join()
    #t_mouse.join()
 
    _timer_end()
 
 
 
'''
def replay(mousedata: str, clickdata: str, keydata: str  ):
 
    mouse_data = json.loads(convert_mouse_movement.convertmousemovement(mousedata))
    mouse_events = mouse_data["raw_decoded"]["frames"]
    bin_ms       = mouse_data["bin_ms"]
    keys_data = json.loads(convert_keyboard_movement.convertkeyboardmovement(keydata, clickdata))
    key_events  = keys_data["parsed_decoded"]["events"]
    print(key_events)
 
    start_perf = time.perf_counter()
    t_keys  = threading.Thread(target=replay_keys,
                                args=(key_events, start_perf), daemon=True)
    t_mouse = threading.Thread(target=replay_mouse,
                                args=(mouse_events, bin_ms, start_perf), daemon=True)
    t_keys.start()
    t_mouse.start()
    t_keys.join()
    t_mouse.join()
 
    _winmm.timeEndPeriod(1)
    print("Replay complete.")
 
 
'''
 
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main (deprecated)
# ──────────────────────────────────────────────────────────────────────────────
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys",  default="key_press.json",       help="Key press JSON file")
    ap.add_argument("--mouse", default="mouse_movement.json",  help="Mouse movement JSON file")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="Seconds to wait before starting (switch to game window)")
    args = ap.parse_args()
 
    with open(args.keys) as f:
        keys_data = json.load(f)
    with open(args.mouse) as f:
        mouse_data = json.load(f)
 
    key_events  = keys_data["parsed_decoded"]["events"]
    mouse_frames = mouse_data["raw_decoded"]["frames"]
    bin_ms       = mouse_data["bin_ms"]
 
    print(f"Loaded {len(key_events)} key events, "
          f"{len(mouse_frames)} mouse frames ({bin_ms}ms/frame)")
 
    # Build the input backend before the countdown so any setup/permission
    # errors surface immediately.
    backend = make_backend()
 
    print(f"Starting in {args.delay:.0f}s — switch to your game now...")
    time.sleep(args.delay)
 
    # Request 1ms timer resolution for accurate sleeps (no-op on Linux)
    _timer_begin()
 
    start_perf = time.perf_counter()
 
    t_keys  = threading.Thread(target=replay_keys,
                                args=(key_events, start_perf, backend), daemon=True)
    t_mouse = threading.Thread(target=replay_mouse,
                                args=(mouse_frames, bin_ms, start_perf, backend), daemon=True)
 
    t_keys.start()
    t_mouse.start()
    t_keys.join()
    t_mouse.join()
 
    _timer_end()
    backend.close()
    print("Replay complete.")
 
 
if __name__ == "__main__":
    main()
 