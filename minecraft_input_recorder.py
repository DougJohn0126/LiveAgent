import os
import sys
import threading
import subprocess
import time
import queue
import imageio_ffmpeg
import shutil
import json
from pathlib import Path

import numpy as np
import evdev
from evdev import ecodes
import selectors

import keycode_mappings
import live_agent 
 
# --------------------------------------------------------------------------- #
# Platform detection — this recorder now runs on Windows AND Linux (X11).
# --------------------------------------------------------------------------- #
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX   = sys.platform.startswith("linux")
 
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
WINDOW_TITLE    = os.environ.get("PLAICRAFT_WINDOW_TITLE", "Minecraft 1.21.4").lower()
OUTPUT_DIR      = Path("recordings")
FRAME_QUEUE_MAX = 100

def _ffmpeg_bin():
    if os.environ.get("FFMPEG_BINARY"):
        return os.environ["FFMPEG_BINARY"]
    sys_ff = shutil.which("ffmpeg")     # prefer the system build (has pulse + x11grab)
    if sys_ff:
        return sys_ff
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

FFMPEG_PATH = _ffmpeg_bin()

# --------------------------------------------------------------------------- #
# Shared buffers
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_movement_buf: list[dict] = []
_click_buf: list[dict] = []
_keyboard_buf: list[dict] = []
_hear_buf, _hear_lock = [], threading.Lock()
_speak_buf, _speak_lock = [], threading.Lock()
_frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
 
# Gates all capture. Cleared during replay so we don't (a) re-record the
# injected motion (Raw Input sees synthetic moves too) or (b) let the blocking
# replay inflate the next window past FLUSH_INTERVAL.
_recording = threading.Event()

_stop_video = threading.Event()
 
# Running virtual mouse position. The OS cursor is frozen under grab, so we integrate
# raw deltas to keep mouseX/mouseY meaningful and continuous.
_vx = 0.0
_vy = 0.0
 
# --------------------------------------------------------------------------- #
# Deprecated
# --------------------------------------------------------------------------- #
_flush_count= 0
_session_filepath: Path | None = None
 
def _now_ms() -> int:
    """Current time in milliseconds (matches the sample data format)."""
    return int(time.time() * 1000)

def _record_move(dx: int, dy: int, timestamp_ms: int | None = None):
    """Records mousemovement to the movement buffer"""
    if not _recording.is_set():
        return
    global _vx, _vy
    with _lock:
        _vx += dx
        _vy += dy
        _movement_buf.append({
            "timestamp": timestamp_ms,
            "mouseX": float(_vx),
            "mouseY": float(_vy),
            "mouseDX": float(dx),
            "mouseDY": float(dy),
        })


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

# --------------------------------------------------------------------------- #
# Linux relative-motion capture 
#
# Reads EV_REL/REL_X/REL_Y straight from the kernel input layer, i.e.
# device-level relative deltas independent of pointer acceleration or the
# cursor grab/clamp.
# Requires read access to /dev/input/event* (be in the 'input' group, or run
# with sudo). Deltas are accumulated per SYN_REPORT to match one hardware
# report == one _record_move() call.
# --------------------------------------------------------------------------- #
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
 
    # initializes the event monitoring system (selector engine 'epoll')
    sel = selectors.DefaultSelector()
    for mouse in mice:
        try:
            # signs up the specific mouse to be watched by the selector.
            sel.register(mouse, selectors.EVENT_READ)
        except Exception:
            pass
    
    REPORT_INTERVAL_MS = 10
    REPORT_INTERVAL = REPORT_INTERVAL_MS / 1000.0

    dx = dy = 0
    tick = 0  # bin index — bin 0 = t=0, bin 1 = t=10, bin 2 = t=20, ...
    start_ms = _now_ms()
    start_mono = time.monotonic()
    next_emit_mono = start_mono + REPORT_INTERVAL

    while True:
        timeout = next_emit_mono - time.monotonic()
        if timeout < 0:
            timeout = 0
        # calling from the selector engine allows the process to sleep during moments when there is no mouse movement
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
            _record_move(dx, dy, timestamp_ms= int (start_ms + tick * REPORT_INTERVAL_MS))
            dx = dy = 0

            next_emit_mono += REPORT_INTERVAL
            if next_emit_mono <= now:
                # we fell behind — jump tick/next_emit forward together
                # instead of letting it re-derive from "now" (which drifts)
                missed = int((now - next_emit_mono) / REPORT_INTERVAL) + 1
                tick += missed
                next_emit_mono += missed * REPORT_INTERVAL


def _find_keyclick_devices():
    """Return devices that emit EV_KEY (keyboards + mice with buttons)."""
    devs = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        if ecodes.EV_KEY in dev.capabilities():
            devs.append(dev)
    return devs

# --------------------------------------------------------------------------- #
# Linux key/click capture 
#
# Reads key/click inputs straight from the kernel input layer.
# Requires read access to /dev/input/event* (be in the 'input' group, or run
# with sudo). Deltas are accumulated per SYN_REPORT to match one hardware
# report == one _record_move() call.
# --------------------------------------------------------------------------- #
def _linux_raw_key_click_input_thread():
    """Linux counterpart of _raw_click_thread(): pump evdev key presses/releases."""
    if evdev is None:
        print("[linux] python-evdev not installed — mouse MOVEMENT will not be "
              "captured. Install it with:  pip install evdev")
        return
    
    try:
        devices = _find_keyclick_devices()
    except PermissionError:
        devices = []

    if not devices:
        print("[evdev_keys] No readable EV_KEY devices in /dev/input — keys/clicks "
              "will be EMPTY (model gets no key context). Add yourself to 'input':\n"
              "    sudo usermod -aG input $USER   # then re-login")
        return

    print(f"[evdev_keys] capturing keys/clicks from: {', '.join(d.name for d in devices)}")

    sel = selectors.DefaultSelector()
    for device in devices:
        try:
            sel.register(device, selectors.EVENT_READ)
        except Exception:
            pass

    while True:
        # calling from the selector engine allows the process to sleep during moments when keys are not pressed
        for key, _mask in sel.select():
            dev = key.fileobj
            try:
                for ev in dev.read():
                    # ev.type must equal EV_KEY
                    if ev.type != ecodes.EV_KEY:
                        continue
                    # value: 1=down, 0=up, 2=autorepeat (ignore repeats)
                    if ev.value == 2:
                        continue
                    ts = _now_ms()
                    if ev.code in keycode_mappings.CLICK_BTN:   # mouse button -> click_buf
                        lbl = keycode_mappings.CLICK_BTN[ev.code]
                        act = f"{lbl}_PRESS" if ev.value == 1 else f"{lbl}_RELEASE"
                        with _lock:
                            _click_buf.append({"timestamp": ts, "action": act})
                    else:                       # keyboard -> keyboard_buf
                        act = "PRESS" if ev.value == 1 else "RELEASE"
                        with _lock:
                            _keyboard_buf.append({"timestamp": ts, "key": keycode_mappings.glfw_code(ev.code), "action": act})
            except BlockingIOError:
                pass
            except OSError:
                try:
                    sel.unregister(dev)
                except Exception:
                    pass
 

# ----------------------------------------------------------------------------- #
# Cideo config (capture-side only; model/sampler config comes from live_agent's env)
# ----------------------------------------------------------------------------- #
LATENT_FPS       = int(os.environ.get("PLAICRAFT_FPS", "10"))
FRAMES_PER_UNIT  = 2
GRAB_WH          = (1280, 720)
# every frame is rescaled to 720p, BGR, 3 bytes/pixel
FRAME_BYTES      = GRAB_WH[0] * GRAB_WH[1] * 3
QUEUE_MAX        = 100
UNIT_MS          = FRAMES_PER_UNIT * (1000 // LATENT_FPS)                 # 200

# Video and audio come off separate ffmpeg pipes with different buffering lag, so
# they need *independent* time-offset corrections (the old single FRAME_LATENCY_MS
# was wrong for audio). Tune each by watching an obvious action vs its frame/sound.
FRAME_LATENCY_MS = int(os.environ.get("PLAICRAFT_FRAME_LATENCY_MS", "100"))
AUDIO_LATENCY_MS = int(os.environ.get("PLAICRAFT_AUDIO_LATENCY_MS", "100"))

HEAR_DEVICE   = os.environ.get("PLAICRAFT_AUDIO_HEAR_DEVICE", "")   # sink .monitor
SPEAK_DEVICE  = os.environ.get("PLAICRAFT_AUDIO_SPEAK_DEVICE", "")  # mic source

# --------------------------------------------------------------------------- #
# Linux video capture 
#
# Reads pixel coordinates given a dimension and coordinates (region) using ffmpeg 
# and pumps the raw uncompressed BGR frames/pixel data to std.out which later gets 
# transformed to frames.
# --------------------------------------------------------------------------- #
def _linux_raw_video_input_thread():
    """Pumps raw pixel data from ffmpeg to std.out and reads one frame's worth of data
      at a time to form a frame which gets pushed onto a thread safe queue."""
    region = _resolve_region();
    proc = subprocess.Popen(_video_cmd(region), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=FRAME_BYTES)
    print("[capture] video grabber started (x11grab / XWayland).", flush=True)
    try:
        while not _stop_video.is_set():
            raw = _read_exact(proc.stdout, FRAME_BYTES)
            if raw is None:
                print("[capture] video pipe ended — wrong DISPLAY/region, or the "
                      "window is native-Wayland (try PLAICRAFT_VIDEO_BACKEND=portal).",
                      flush=True)
                break
            t = _now_ms()
            frame = np.frombuffer(raw, np.uint8).reshape(GRAB_WH[1], GRAB_WH[0], 3)
            try:
                _frame_q.put_nowait((t, frame))
            except queue.Full:                 # stay current: drop oldest
                try:
                    _frame_q.get_nowait()
                    _frame_q.put_nowait((t, frame))
                except queue.Empty:
                    pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass

# test function for measuring fps capable by ffmpeg
def _measure(seconds=10):
    region = _resolve_region()
    proc = subprocess.Popen(_video_cmd(region), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=FRAME_BYTES * 8)
    n, t0, last = 0, time.time(), time.time()
    while time.time() - t0 < seconds:
        raw = _read_exact(proc.stdout, FRAME_BYTES)
        if raw is None:
            break
        n += 1
        now = time.time()
        if now - last >= 1.0:                 # report once per second, not per frame
            print(f"{n} frames, {n / (now - t0):.1f} fps avg")
            last = now
    proc.kill()
    print(f"final: {n / (time.time() - t0):.1f} fps over {seconds}s")

def _read_exact(stream, n):
    """ Read the from the steam of bytes exactly one frame's (n) worth of bytes"""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def _resolve_region():
    """Region resolution: env override -> recorder's window finder -> full-ish default"""
    try:
        win = find_window(WINDOW_TITLE)
        print(f"[capture] found window '{win['title']}")
        if win:   
            try:
                moved = focus_window(win["hwnd"])
                if moved:
                    win['left']=moved[0]
                    win['top']=moved[1]
                    win['width']=moved[2]
                    win['height']=moved[3]
                
            except Exception:
                pass
            return (win["left"], win["top"], win["width"], win["height"])
    except Exception as e:
        print(f"[capture] find_window failed ({e}); falling back to default region")
    print("[capture] using default region 0,0 1280x720 — set "
          "PLAICRAFT_CAPTURE_REGION=x,y,w,h to capture the Minecraft window exactly")
    return (0, 0, 1280, 720)


def _video_cmd(region):
    """Video capture -> raw BGR frames on a queue (no JPEG). \n
    x11grab works under XWayland; portal stub for native Wayland."""
    left, top, w, h = region
    w -= w % 2
    h -= h % 2
    out = ["-r", str(LATENT_FPS),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    display = os.environ.get("DISPLAY", ":0")
    return [FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-framerate", str(LATENT_FPS),
            "-video_size", f"{w}x{h}", "-i", f"{display}+{left},{top}", *out]

# test function for video capture
def _video_cmd_test(region):
    left, top, w, h = region
    w -= w % 2
    h -= h % 2
    out = ["-r", str(LATENT_FPS),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", "output.mp4"]
    display = os.environ.get("DISPLAY", ":0")
    print (display)
    return [FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-framerate", str(LATENT_FPS),
            "-video_size", f"{w}x{h}", "-i", f"{display}+{left},{top}", *out]

# --- Platform video dispatchers --------------------------------------------------- #
def find_window(sub):
    if IS_LINUX:
        return _linux_find_window(sub)
    return None
 
def focus_window(hwnd):
    if IS_LINUX:
        return _linux_focus_window(hwnd)
 
# Linux (X11) window helpers, via wmctrl + xwininfo ---------------------- #
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
        print (title)
        if sub.lower() in title.lower():
            geo = _xwininfo_geometry(wid)
            if geo is None:
                continue
            print (parts)
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


AUDIO_SR               = int(getattr(live_agent, "AUDIO_SR", 24000))
AUDIO_SAMPLES_PER_UNIT = int(getattr(live_agent, "AUDIO_SAMPLES_PER_UNIT", AUDIO_SR // 5))

def _pactl(*args):
    """Run `pactl <args>` and return stripped stdout, or '' on any failure. \n
        pactl is a command-line utility used to control the PulseAudio sound server. \n
        It provides a scriptable interface for managing audio devices (sinks/sources), \n
        volume levels, active streams, and loaded modules.
    """
    if not shutil.which("pactl"):
        return ""
    try:
        return subprocess.run(["pactl", *args],
                              capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""

def _default_source():
    """Returns he default capture source — the mic."""
    src = _pactl("get-default-source")
    if src:
        return src
    info = _pactl("info")
    for line in info.splitlines():
        if line.lower().startswith("default source:"):
            return line.split(":", 1)[1].strip()
    print("[capture] PLAICRAFT_AUDIO_SPEAK_DEVICE unset — silence for audio_speak")
    return ""

def _default_monitor():
    """Returns the .monitor source for the default sink — the Minecraft window"""
    sink = _pactl("get-default-sink")
    if sink:
        return f"{sink}.monitor"
    # older pactl without get-default-sink: parse `info`
    info = _pactl("info")
    for line in info.splitlines():
        if line.lower().startswith("default sink:"):
            return line.split(":", 1)[1].strip() + ".monitor"
    print("[capture] PLAICRAFT_AUDIO_HEAR_DEVICE unset — silence for audio_hear")
    return ""

# ----------------------------------------------------------------------------- #
# Linux audio capture (PipeWire/Pulse) -> rolling timestamped buffers
#    -f pulse -i source — input is the pulse audio device named source (a sink .monitor for "hear," or a mic source for "speak" — the HEAR_DEVICE/SPEAK_DEVICE from config).
#    -ac 1 — downmix to 1 channel (mono).
#    -ar str(AUDIO_SR) — resample to 24 kHz, matching what the model expects.
#    -f f32le — output format 32-bit little-endian float PCM (raw samples in [-1, 1]).
# ----------------------------------------------------------------------------- #
def _linux_raw_audio_input_thread(label):
    if (label == "speak"):
        buf = _hear_buf
        lock = _hear_lock
        src = _default_source()
    else:
        buf = _speak_buf
        lock = _speak_lock
        src = _default_monitor()

    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
           "-f", "pulse", "-i", src,
           "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
    print(f"[capture] audio '{label}' started from pulse source: {src}", flush=True)
    CHUNK = AUDIO_SR // 50  # 20 ms of audio per read
    try:
        while not _stop_video.is_set():
            raw = _read_exact(proc.stdout, CHUNK * 4)   # float32 = 4 bytes
            if raw is None:
                print(f"[capture] audio '{label}' pipe ended — bad pulse source name? "
                      f"check `pactl list sources short`", flush=True)
                break
            t = _now_ms()
            c = np.frombuffer(raw, np.float32).copy()
            with lock:
                buf.append((t, c))
                cutoff = t - 2000
                while buf and buf[0][0] < cutoff:
                    buf.pop(0)
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def start_recording():
    """
     Start input capture (continuous). On Wayland ALL of it must come from evdev:
       - mouse MOTION  -> rec._linux_raw_input_thread (REL_X/REL_Y)
       - keys + clicks -> evdev_keys (EV_KEY) 
       - pynput is X11-only and goes silent on Wayland, which would feed the model zero key context (degenerate).
    """
    threading.Thread(target=_linux_raw_key_click_input_thread, daemon=True).start()
    threading.Thread(target=_linux_raw_mouse_input_thread, daemon=True).start()
    threading.Thread(target=_linux_raw_video_input_thread, daemon=True).start()
    threading.Thread(target=_linux_raw_audio_input_thread, args=("speak",),daemon=True).start()
    threading.Thread(target=_linux_raw_audio_input_thread, args=("hear",),daemon=True).start()

  
# --------------------------------------------------------------------------- #
# Periodic flush to disk + optional immediate replay (not used for current pipeline)
# --------------------------------------------------------------------------- #
def _flush(is_write_on_flush: bool):
    global _flush_count
 
    with _lock: 
        movement_snapshot = _movement_buf.copy()
        click_snapshot    = _click_buf.copy()
        keyboard_snapshot = _keyboard_buf.copy()
        _movement_buf.clear()
        _click_buf.clear()
        _keyboard_buf.clear()
 
    if (is_write_on_flush):
        _write_json(_session_filepath / f"mouse_movement_data_{_flush_count}.json", movement_snapshot)
        _write_json(_session_filepath / f"mouse_click_data_{_flush_count}.json",    click_snapshot)
        _write_json(_session_filepath / f"keyboard_data_{_flush_count}.json",       keyboard_snapshot)
 
    _flush_count += 1
 
    return movement_snapshot, click_snapshot, keyboard_snapshot

def _write_json(path: Path, data: list[dict]):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=None, separators=(",\n", ":"))
    tmp.replace(path)  # atomic rename
# --------------------------------------------------------------------------- #
# Main (not used for current pipeline)
# --------------------------------------------------------------------------- #
def main():
    FLUSH_INTERVAL  = 5      # seconds
    WRITE_ON_FLUSH = True
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
 
    _recording.set()
    start_recording()
    
    # Record / flush / replay cycle thread.
    #threading.Thread(target=_record_loop, args=(screen,), daemon=True).start()
 
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping — flushing final events …")
        _flush()
        print("Done.")
 
 
if __name__ == "__main__":
    main()