"""
minecraft_input_recorder.py
----------------
This version captures mouse MOVEMENT from the Windows Raw Input API (WM_INPUT):
device-level relative deltas, independent of cursor lock/clamping. This is the
SAME signal Minecraft reads, and the same kind your injector emits with
pydirectinput.moveRel(relative=True). So record and replay are now symmetric.
A real half-spin should now show ~1200 px net horizontal travel at 100% sens
(the recorder prints net dx/dy every flush so you can verify immediately).
 
Mouse CLICKS and the KEYBOARD are still captured via pynput — those are not
affected by cursor grab.
 
Output format is (so convert_mouse_movement / inputInjector work as-is):
  mouse_movement_data.json  [{"timestamp","mouseX","mouseY","mouseDX","mouseDY"}]
  mouse_click_data.json     [{"timestamp","action"}]
  keyboard_data.json        [{"timestamp","key","action"}]
 
NOTES
-----
* Keep Minecraft's Raw Input setting ON — this recorder matches it.
* Also turn OFF Windows "Enhance pointer precision" so deltas stay linear.
* Record and replay on the SAME machine/session with the SAME Minecraft
  sensitivity, or the pixel->degree mapping won't match.
 
Requirements:
  pip install pynput pywin32
Usage:
  python minecraft_input_recorder.py
  Press Ctrl+C to stop.
"""
 
import ctypes
import ctypes.wintypes as wt   # importable on Linux too; only the WinDLL/WINFUNCTYPE calls are Windows-only
import json
import os
import sys
import threading
import time
import traceback
import subprocess
import signal
import imageio_ffmpeg
from pathlib import Path
from pynput import mouse, keyboard
 
# --------------------------------------------------------------------------- #
# Platform detection — this recorder now runs on Windows AND Linux (X11).
# --------------------------------------------------------------------------- #
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX   = sys.platform.startswith("linux")
 
if IS_WINDOWS:
    import win32gui
elif IS_LINUX:
    import selectors
    try:
        import evdev
        from evdev import ecodes
    except Exception:                # python-evdev not installed yet
        evdev = None
 
# The replay/injection side (inputInjector) is a separate, currently
# Windows-oriented module. Import it lazily so the recorder still runs on
# Linux even when the injector can't be imported — replay is just skipped.
try:
    import scripts_replay.inputInjector as inputInjector
except Exception as _inj_err:
    inputInjector = None
    print(f"[warn] inputInjector unavailable ({_inj_err!r}); replay disabled.")
 
 
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
WINDOW_TITLE    = os.environ.get("PLAICRAFT_WINDOW_TITLE", "x11grab").lower()
OUTPUT_DIR      = Path("recordings")
FLUSH_INTERVAL  = 5      # seconds
REPLAY_ON_FLUSH = True   # set False to record only (no immediate replay loop)
WRITE_ON_FLUSH = True
VIDEO_FPS = 10
FFMPEG_BIN = "ffmpeg"
 
# --------------------------------------------------------------------------- #
# Shared buffers
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_movement_buf: list[dict] = []
_click_buf: list[dict] = []
_keyboard_buf: list[dict] = []
 
# Gates all capture. Cleared during replay so we don't (a) re-record the
# injected motion (Raw Input sees synthetic moves too) or (b) let the blocking
# replay inflate the next window past FLUSH_INTERVAL.
_recording = threading.Event()
 
# Running virtual position. The OS cursor is frozen under grab, so we integrate
# raw deltas to keep mouseX/mouseY meaningful and continuous.
_vx = 0.0
_vy = 0.0
 
_flush_count= 0
_session_filepath: Path | None = None
 
 
def _now_ms() -> int:
    """Current time in milliseconds (matches the sample data format)."""
    return int(time.time() * 1000)
 
# --------------------------------------------------------------------------- #
# Raw Input (WM_INPUT) mouse-movement capture
# --------------------------------------------------------------------------- #
if IS_WINDOWS:
    user32   = ctypes.WinDLL("user32",   use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
 
LRESULT = wt.LPARAM  # pointer-sized signed
 
WM_INPUT            = 0x00FF
RID_INPUT           = 0x10000003
RIDEV_INPUTSINK     = 0x00000100   # receive raw input even when NOT foreground
RIM_TYPEMOUSE       = 0
MOUSE_MOVE_ABSOLUTE = 0x01
 
HID_USAGE_PAGE_GENERIC  = 0x01
HID_USAGE_GENERIC_MOUSE = 0x02
 
 
class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wt.USHORT),
                ("usUsage",     wt.USHORT),
                ("dwFlags",     wt.DWORD),
                ("hwndTarget",  wt.HWND)]
 
 
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType",  wt.DWORD),
                ("dwSize",  wt.DWORD),
                ("hDevice", wt.HANDLE),
                ("wParam",  wt.WPARAM)]
 
 
class _BUTTONS(ctypes.Structure):
    _fields_ = [("usButtonFlags", wt.USHORT),
                ("usButtonData",  wt.USHORT)]
 
 
class _MOUSE_U(ctypes.Union):
    _fields_ = [("ulButtons", wt.ULONG),
                ("buttons",   _BUTTONS)]
 
 
class RAWMOUSE(ctypes.Structure):
    _fields_ = [("usFlags",            wt.USHORT),
                ("u",                  _MOUSE_U),
                ("ulRawButtons",       wt.ULONG),
                ("lLastX",             wt.LONG),
                ("lLastY",             wt.LONG),
                ("ulExtraInformation", wt.ULONG)]
 
 
class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER),
                ("mouse",  RAWMOUSE)]
 
 
# WINFUNCTYPE only exists on Windows; CFUNCTYPE is an import-safe stand-in on
# Linux (this callback type is only ever actually used on Windows).
_FUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
WNDPROC = _FUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
 
 
class WNDCLASS(ctypes.Structure):
    _fields_ = [("style",        wt.UINT),
                ("lpfnWndProc",   WNDPROC),
                ("cbClsExtra",    ctypes.c_int),
                ("cbWndExtra",    ctypes.c_int),
                ("hInstance",     wt.HINSTANCE),
                ("hIcon",         wt.HANDLE),
                ("hCursor",       wt.HANDLE),
                ("hbrBackground", wt.HANDLE),
                ("lpszMenuName",  wt.LPCWSTR),
                ("lpszClassName", wt.LPCWSTR)]
 
 
class MSG(ctypes.Structure):
    _fields_ = [("hwnd",    wt.HWND),
                ("message", wt.UINT),
                ("wParam",  wt.WPARAM),
                ("lParam",  wt.LPARAM),
                ("time",    wt.DWORD),
                ("pt_x",    wt.LONG),
                ("pt_y",    wt.LONG)]
 
 
# --- prototypes (REQUIRED on 64-bit so handles/pointers aren't truncated) --- #
if IS_WINDOWS:
    user32.DefWindowProcW.restype  = LRESULT
    user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
 
    user32.GetRawInputData.restype  = wt.UINT
    user32.GetRawInputData.argtypes = [wt.HANDLE, wt.UINT, wt.LPVOID,
                                       ctypes.POINTER(wt.UINT), wt.UINT]
 
    user32.RegisterRawInputDevices.restype  = wt.BOOL
    user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE),
                                               wt.UINT, wt.UINT]
 
    user32.RegisterClassW.restype  = wt.ATOM
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
 
    user32.CreateWindowExW.restype  = wt.HWND
    user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, wt.HWND, wt.HMENU,
                                       wt.HINSTANCE, wt.LPVOID]
 
    user32.GetMessageW.restype  = ctypes.c_int
    user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), wt.HWND, wt.UINT, wt.UINT]
 
    user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.restype  = LRESULT
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
 
    kernel32.GetModuleHandleW.restype  = wt.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
 
 
def _record_move(dx: int, dy: int, timestamp_ms: int | None = None):
    if not _recording.is_set():
        return
    global _vx, _vy
    with _lock:
        _vx += dx
        _vy += dy
        _movement_buf.append({
            "timestamp": timestamp_ms if timestamp_ms is not None else _now_ms(),
            "mouseX": float(_vx),
            "mouseY": float(_vy),
            "mouseDX": float(dx),
            "mouseDY": float(dy),
        })
        print(f"time {timestamp_ms} dx={dx} dy={dy}")
 
 
def _handle_raw_input(hrawinput):
    size = wt.UINT(0)
    # 1) query required buffer size (returns 0 on success when pData is NULL)
    if user32.GetRawInputData(hrawinput, RID_INPUT, None, ctypes.byref(size),
                              ctypes.sizeof(RAWINPUTHEADER)) != 0:
        return
    if size.value == 0:
        return
    # 2) read the actual data
    buf = (ctypes.c_byte * size.value)()
    got = user32.GetRawInputData(hrawinput, RID_INPUT, buf, ctypes.byref(size),
                                 ctypes.sizeof(RAWINPUTHEADER))
    if got != size.value:
        return
    raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
    if raw.header.dwType != RIM_TYPEMOUSE:
        return
    # Ignore absolute-mode devices (touchpads / RDP) — we want relative deltas.
    if raw.mouse.usFlags & MOUSE_MOVE_ABSOLUTE:
        return
    dx, dy = raw.mouse.lLastX, raw.mouse.lLastY
    if dx or dy:
        _record_move(dx, dy)
 
 
def _wndproc(hwnd, msg, wparam, lparam):
    if msg == WM_INPUT:
        _handle_raw_input(lparam)
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
 
 
# Keep a module-level reference so the callback isn't garbage-collected.
_WNDPROC_PTR = WNDPROC(_wndproc)
_CLASS_NAME = "MCRawInputRecorder"
 
 
def _raw_input_thread():
    """Create a hidden window, register for raw mouse input, pump messages."""

 
 
# --------------------------------------------------------------------------- #
# Linux relative-motion capture 
#
# Reads EV_REL/REL_X/REL_Y straight from the kernel input layer, i.e.
# device-level relative deltas independent of pointer acceleration or the
# cursor grab/clamp — the same kind of signal Raw Input gives on Windows.
# Requires read access to /dev/input/event* (be in the 'input' group, or run
# with sudo). Deltas are accumulated per SYN_REPORT to match one hardware
# report == one _record_move() call.
# --------------------------------------------------------------------------- #
def _find_relative_mice():
    """Return evdev devices that emit relative X/Y motion and no keyboard events (real mice)."""
    mice = []
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        rel = device.capabilities().get(ecodes.EV_REL, [])
        if ecodes.REL_X in rel and ecodes.REL_Y in rel and ecodes.KEY_A not in rel:
            mice.append(device)
    return mice

def _find_keyclick_devices():
    """Devices that emit EV_KEY (keyboards + mice with buttons)."""
    devs = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        if ecodes.EV_KEY in dev.capabilities():
            devs.append(dev)
    return devs
 
 
def _linux_raw_mouse_input_thread():
    """Linux counterpart of _raw_input_thread(): pump evdev relative motion."""
    if evdev is None:
        print("[linux] python-evdev not installed — mouse MOVEMENT will not be "
              "captured. Install it with:  pip install evdev")
        return
 
    try:
        mice = _find_relative_mice()
    except PermissionError:
        mice = []
 
    if not mice:
        print("[linux] No readable relative-motion mouse found in /dev/input.\n"
              "        Add yourself to the 'input' group and re-login:\n"
              "            sudo usermod -aG input $USER\n"
              "        (or run this recorder with sudo). Clicks/keys still work.")
        return
 
    print(f"[linux] Capturing raw motion from: "
          f"{', '.join(d.name for d in mice)}")
 
    #initializes the event monitoring system (selector engine 'epoll')
    sel = selectors.DefaultSelector()
    for mouse in mice:
        try:
            #signs up the specific mouse to be watched by the selector.
            sel.register(mouse, selectors.EVENT_READ)
        except Exception:
            pass
    
    REPORT_INTERVAL_MS = 10
    REPORT_INTERVAL = REPORT_INTERVAL_MS / 1000.0

    dx = dy = 0
    tick = 0  # bin index — bin 0 = t=0, bin 1 = t=10, bin 2 = t=20, ...
    start_mono = time.monotonic()
    next_emit_mono = start_mono + REPORT_INTERVAL

    while True:
        timeout = next_emit_mono - time.monotonic()
        if timeout < 0:
            timeout = 0
        # calling from the selector engine allows the process to sleep during moments when it is
        events_ready = sel.select(timeout=timeout)

        for key, _mask in events_ready:
            dev = key.fileobj
            try:
                for event in dev.read():
                    if event.type == ecodes.EV_REL:
                        if event.code == ecodes.REL_X:
                            dx += event.value
                        elif event.code == ecodes.REL_Y:
                            dy += event.value
            except BlockingIOError:
                pass
            except OSError:
                try:
                    sel.unregister(dev)
                except Exception:
                    pass

        now = time.monotonic()
        if now >= next_emit_mono:
            tick += 1
            _record_move(dx, dy, timestamp_ms=tick * REPORT_INTERVAL_MS)
            dx = dy = 0

            next_emit_mono += REPORT_INTERVAL
            if next_emit_mono <= now:
                # we fell behind — jump tick/next_emit forward together
                # instead of letting it re-derive from "now" (which drifts)
                missed = int((now - next_emit_mono) / REPORT_INTERVAL) + 1
                tick += missed
                next_emit_mono += missed * REPORT_INTERVAL

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




def _linux_raw_key_click_input_thread():
    devices = _find_keyclick_devices()
    if not devices:
        print("[evdev_keys] No readable EV_KEY devices in /dev/input — keys/clicks "
              "will be EMPTY (model gets no key context). Add yourself to 'input':\n"
              "    sudo usermod -aG input $USER   # then re-login")
        return
    print(f"[evdev_keys] capturing keys/clicks from: {', '.join(d.name for d in devs)}")

    sel = selectors.DefaultSelector()
    for dev in devices:
        try:
            sel.register(dev, selectors.EVENT_READ)
        except Exception:
            pass

    while True:
        for key, _mask in sel.select():
            dev = key.fileobj
            try:
                for ev in dev.read():
                    if ev.type != ecodes.EV_KEY:
                        continue
                    if not rec._recording.is_set():
                        continue
                    # value: 1=down, 0=up, 2=autorepeat (ignore repeats)
                    if ev.value == 2:
                        continue
                    ts = rec._now_ms()
                    if ev.code in _BTN:                      # mouse button -> click_buf
                        lbl = _BTN[ev.code]
                        act = f"{lbl}_PRESS" if ev.value == 1 else f"{lbl}_RELEASE"
                        with rec._lock:
                            _click_buf.append({"timestamp": ts, "action": act})
                    else:                                    # keyboard -> keyboard_buf
                        act = "PRESS" if ev.value == 1 else "RELEASE"
                        with rec._lock:
                            rec._keyboard_buf.append({"timestamp": ts,
                                                      "key": _glfw_code(ev.code),
                                                      "action": act})
            except BlockingIOError:
                pass
            except OSError:
                try:
                    sel.unregister(dev)
                except Exception:
                    pass
 
 
# --------------------------------------------------------------------------- #
# Mouse click listener (pynput — unaffected by cursor grab)
# --------------------------------------------------------------------------- #
_BUTTON_MAP = {
    mouse.Button.left:   "LEFT",
    mouse.Button.right:  "RIGHT",
    mouse.Button.middle: "MIDDLE",
}
 
 
def on_click(x: float, y: float, button: mouse.Button, pressed: bool):
    if not _recording.is_set():
        return
    label  = _BUTTON_MAP.get(button, button.name.upper())
    action = f"{label}_PRESS" if pressed else f"{label}_RELEASE"
    with _lock:
        _click_buf.append({"timestamp": _now_ms(), "action": action})
 
 
# --------------------------------------------------------------------------- #
# Keyboard listener (pynput)
# --------------------------------------------------------------------------- #
def _key_code(key) -> int:
    """Numeric key code matching GLFW / Minecraft values."""
    _SPECIAL = {
        keyboard.Key.space: 32, keyboard.Key.enter: 257, keyboard.Key.tab: 258,
        keyboard.Key.backspace: 259, keyboard.Key.insert: 260,
        keyboard.Key.delete: 261, keyboard.Key.right: 262, keyboard.Key.left: 263,
        keyboard.Key.down: 264, keyboard.Key.up: 265, keyboard.Key.page_up: 266,
        keyboard.Key.page_down: 267, keyboard.Key.home: 268, keyboard.Key.end: 269,
        keyboard.Key.caps_lock: 280, keyboard.Key.scroll_lock: 281,
        keyboard.Key.num_lock: 282, keyboard.Key.print_screen: 283,
        keyboard.Key.pause: 284, keyboard.Key.f1: 290, keyboard.Key.f2: 291,
        keyboard.Key.f3: 292, keyboard.Key.f4: 293, keyboard.Key.f5: 294,
        keyboard.Key.f6: 295, keyboard.Key.f7: 296, keyboard.Key.f8: 297,
        keyboard.Key.f9: 298, keyboard.Key.f10: 299, keyboard.Key.f11: 300,
        keyboard.Key.f12: 301, keyboard.Key.shift: 340, keyboard.Key.shift_r: 344,
        keyboard.Key.ctrl: 341, keyboard.Key.ctrl_r: 345, keyboard.Key.alt: 342,
        keyboard.Key.alt_r: 346, keyboard.Key.esc: 256,
    }
    if isinstance(key, keyboard.Key):
        return _SPECIAL.get(key, -1)
    if getattr(key, "char", None) is not None:
        return ord(key.char.upper())
    if getattr(key, "vk", None) is not None:
        return key.vk
    return -1
 
 
def on_press(key):
    if not _recording.is_set():
        return
    with _lock:
        _keyboard_buf.append({"timestamp": _now_ms(),
                              "key": _key_code(key), "action": "PRESS"})
 
 
def on_release(key):
    if not _recording.is_set():
        return
    with _lock:
        _keyboard_buf.append({"timestamp": _now_ms(),
                              "key": _key_code(key), "action": "RELEASE"})
        
# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
class ScreenRecorder:
    """Records a fixed screen region to a single .mkv via ffmpeg (Windows gdigrab).
 
    start(path) spawns ffmpeg; stop() finalizes that clip and returns its path.
    Designed to be driven once per FLUSH_INTERVAL window so each clip lines up
    1:1 with the JSON for that window.
    """
 
    def __init__(self, region: tuple[int, int, int, int], fps: int, ffmpeg_bin: str = "ffmpeg"):
        left, top, width, height = region
        # libx264 + yuv420p needs even dimensions.
        self.left = int(left)
        self.top = int(top)
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.fps = fps
        self.ffmpeg = ffmpeg_bin   # resolved absolute path (or "ffmpeg" if on PATH)
        self._proc: subprocess.Popen | None = None
        self._path: Path | None = None
 
    def _cmd(self, out_path):
        # Prefer an explicit binary (FFMPEG_BINARY env) so users can point at a
        # system ffmpeg if the bundled one lacks the needed grabber.
        ffmpeg = os.environ.get("FFMPEG_BINARY") or imageio_ffmpeg.get_ffmpeg_exe()
 
        out_opts = [
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", str(self.fps),
            str(out_path),
        ]
 
        if IS_WINDOWS:
            return [
                ffmpeg,
                "-hide_banner", "-loglevel", "warning", "-y",
                "-f", "gdigrab",
                "-framerate", str(self.fps),
                "-offset_x", str(self.left),
                "-offset_y", str(self.top),
                "-video_size", f"{self.width}x{self.height}",
                "-i", "desktop",
                *out_opts,
            ]
 
        # Linux / X11: grab a region of the X display starting at (left, top).
        # NOTE: requires an X11 (or XWayland) session; pure Wayland needs
        # kmsgrab/pipewire instead. If you hit "Unknown input format 'x11grab'",
        # install a full ffmpeg (sudo apt install ffmpeg) and set
        # FFMPEG_BINARY=/usr/bin/ffmpeg.
        display = os.environ.get("DISPLAY", ":0.0")
        return [
            ffmpeg,
            "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "x11grab",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", f"{display}+{self.left},{self.top}",
            *out_opts,
        ]
 
    def start(self, out_path):
        self._path = out_path  
        cmd = self._cmd(out_path)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
 
    def stop(self) -> Path | None:
        """Finalize the current clip cleanly and return its path."""
        proc, path = self._proc, self._path
        self._proc, self._path = None, None
        if proc is None:
            return None
        try:
            if proc.stdin:
                proc.stdin.write(b"q")   # graceful flush -> valid moov/index
                proc.stdin.flush()
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return path
 
 
# --------------------------------------------------------------------------- #
# Periodic flush to disk (+ optional immediate replay)
# --------------------------------------------------------------------------- #
def _write_json(path: Path, data: list[dict]):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=None, separators=(",\n", ":"))
    tmp.replace(path)  # atomic rename
 
 
def _send_to_injector(mousedata: str, clickdata: str, keydata: str, videopath: Path):
    if inputInjector is None:
        return
    try:
        inputInjector.replay(mousedata, clickdata, keydata, videopath)
    except Exception as e:
        print(traceback.print_exc() )
 
 
def _flush():
    global _flush_count
 
    with _lock: 
        movement_snapshot = _movement_buf.copy()
        click_snapshot    = _click_buf.copy()
        keyboard_snapshot = _keyboard_buf.copy()
        _movement_buf.clear()
        _click_buf.clear()
        _keyboard_buf.clear()
 
    if (WRITE_ON_FLUSH):
        _write_json(_session_filepath / f"mouse_movement_data_{_flush_count}.json", movement_snapshot)
        _write_json(_session_filepath / f"mouse_click_data_{_flush_count}.json",    click_snapshot)
        _write_json(_session_filepath / f"keyboard_data_{_flush_count}.json",       keyboard_snapshot)
 
    """
    # Sanity readout: a real 180° spin should be ~1200 px net dx at 100% sens.
    net_dx = sum(e["mouseDX"] for e in movement_snapshot)
    net_dy = sum(e["mouseDY"] for e in movement_snapshot)
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] Flushed — {len(movement_snapshot)} moves "
          f"(net dx={net_dx:.0f}px dy={net_dy:.0f}px), "
          f"{len(click_snapshot)} clicks, {len(keyboard_snapshot)} keys")
    """
    _flush_count += 1
 
    return movement_snapshot, click_snapshot, keyboard_snapshot
 
 
def _record_loop(screen: "ScreenRecorder | None"):
    """Record for exactly FLUSH_INTERVAL, then flush, then optionally replay.
 
    Recording is PAUSED around the flush + replay, so:
      * the recorder never captures the injected replay motion (no feedback), and
      * the blocking replay (inputInjector joins its thread) can't push events
        into the next window — every window is bounded to FLUSH_INTERVAL.
    """
    global _vx, _vy
    global _flush_count
    global _session_filepath
    while True:
        # ---- RECORD phase: exactly FLUSH_INTERVAL of capture ----
        with _lock:
            _movement_buf.clear()
            _click_buf.clear()
            _keyboard_buf.clear()
        
        if screen is not None and _session_filepath is not None:
            screen.start(_session_filepath / f"clip_{_flush_count}.mkv")
        
        _recording.set()
        _record_move(0,0)              # flush starting 0,0
        time.sleep(FLUSH_INTERVAL)
        _record_move(0,0)              # flush ending 0,0
        _recording.clear()            # pause before snapshot + replay
 
        # ---- stop video for this window (clip now aligns with the JSON) ----
        clip_path = screen.stop() if screen is not None else None
        print(screen)
 
        print(clip_path)
 
        # ---- FLUSH ----
        mv, ck, kb = _flush()
 
        # ---- REPLAY phase (recording paused -> no feedback, no inflation) ----
        if REPLAY_ON_FLUSH:
            _send_to_injector(json.dumps(mv), json.dumps(ck), json.dumps(kb), clip_path)
 
 
# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #
def _win_find_window(sub):
    found = {}
 
    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if sub.lower() in t.lower():
                r = win32gui.GetWindowRect(hwnd)
                found.update(hwnd=hwnd, title=t, left=r[0], top=r[1],
                             width=r[2] - r[0], height=r[3] - r[1])
 
    win32gui.EnumWindows(cb, None)
    return found or None
 
 
def _win_focus_window(hwnd):
    try:
        if win32gui.GetForegroundWindow() != hwnd:
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
 
 
# --- Linux (X11) window helpers, via wmctrl + xwininfo ---------------------- #
def _xwininfo_geometry(wid):
    """Absolute on-screen geometry for an X11 window id, via xwininfo."""
    try:
        out = subprocess.check_output(["xwininfo", "-id", str(wid)], text=True)
    except FileNotFoundError:
        print("[linux] 'xwininfo' not found — install with: sudo apt install x11-utils")
        return None
    except subprocess.CalledProcessError:
        return None
    info = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Absolute upper-left X:"):
            info["left"] = int(line.split(":")[1])
        elif line.startswith("Absolute upper-left Y:"):
            info["top"] = int(line.split(":")[1])
        elif line.startswith("Width:"):
            info["width"] = int(line.split(":")[1])
        elif line.startswith("Height:"):
            info["height"] = int(line.split(":")[1])
    if {"left", "top", "width", "height"} <= info.keys():
        return info
    return None
 
 
def _linux_find_window(sub):
    try:
        out = subprocess.check_output(["wmctrl", "-l"], text=True)
    except FileNotFoundError:
        print("[linux] 'wmctrl' not found — install with: sudo apt install wmctrl")
        return None
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        # columns: <id> <desktop> <host> <title...>
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        wid, _desktop, _host, title = parts
        if sub.lower() in title.lower():
            geo = _xwininfo_geometry(wid)
            if geo is None:
                continue
            return {"hwnd": wid, "title": title, **geo}
    return None
 
 
def _linux_focus_window(wid, x=100, y=100, width=1280, height=720):
    for cmd in (["wmctrl", "-i", "-a", str(wid)],
                ["xdotool", "windowactivate", str(wid)]):
        try:
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    # 1. Force-remove Fullscreen and Maximized states via wmctrl
    # Many window managers block resizing if these properties are active
    subprocess.run([
        "wmctrl", "-i", "-r", str(wid), "-b", 
        "remove,maximized_vert,maximized_horz,fullscreen"
    ], check=False)
    time.sleep(0.1)
    print ("success?")
    """Resizes and moves a window using xdotool."""
    try:
        # Move the window
        subprocess.check_call(["xdotool", "windowmove", wid, str(x), str(y)])
        # Resize the window
        subprocess.check_call(["xdotool", "windowsize", wid, str(width), str(height)])
        print(f"Successfully positioned window {wid} to {width}x{height} at ({x}, {y})")
        time.sleep(0.1)
        return (x, y, width, height)
    except subprocess.CalledProcessError:
        print(f"xdotool failed to manipulate window {wid}")
        return False
    except FileNotFoundError:
        print("xdotool is not installed. Run: sudo apt install xdotool")
        return False
 
 
# --- Platform dispatchers --------------------------------------------------- #
def find_window(sub):
    if IS_WINDOWS:
        return _win_find_window(sub)
    if IS_LINUX:
        return _linux_find_window(sub)
    return None
 
 
def focus_window(hwnd):
    if IS_WINDOWS:
        return _win_focus_window(hwnd)
    if IS_LINUX:
        return _linux_focus_window(hwnd)
 
 
# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Minecraft Input Repeater (Raw Input)")
    print(f"  Flush every : {FLUSH_INTERVAL}s")
    print("  Keep Minecraft 'Raw Input' = ON. Press Ctrl+C to stop.\n")
 
    print(f"Looking for '{WINDOW_TITLE}'…")
    win = None
    while win is None:
        win = find_window(WINDOW_TITLE)
        if win is None:
            time.sleep(1)
    print(f"Found: '{win['title']}'  {win['width']}x{win['height']}")
    focus_window(win["hwnd"])
 
    if (WRITE_ON_FLUSH):
        global _session_filepath
        _session_filepath = OUTPUT_DIR / str(_now_ms())
        _session_filepath.mkdir(parents=True, exist_ok=True)
        print(f"  Output dir  : {OUTPUT_DIR.resolve()}")
 
    # Build the screen recorder from the window's rect (captured once).
    screen = None
    screen = ScreenRecorder(
        region=(win["left"], win["top"], win["width"], win["height"]),
        fps=VIDEO_FPS,
    )
    
    # Clicks + keyboard via pynput.
    click_listener    = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    click_listener.start()
    keyboard_listener.start()
 
    # Mouse-movement capture thread (device-level relative deltas).
    if IS_WINDOWS:
        # Raw Input: creates its own hidden window + message loop.
        print ("windows not supported anymore")
    elif IS_LINUX:
        # evdev: reads relative motion straight from /dev/input.
        threading.Thread(target=_linux_raw_input_thread, daemon=True).start()
    # Record / flush / replay cycle thread.
    threading.Thread(target=_record_loop, args=(screen,), daemon=True).start()
 
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping — flushing final events …")
        if screen is not None:
            screen.stop()          # finalize any in-progress clip
        _flush()
        print("Done.")
 
 
if __name__ == "__main__":
    main()
 