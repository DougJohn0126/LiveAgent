#!/usr/bin/env python3
"""
run_live.py — single-process Linux live agent (Wayland-aware).

Collapses the old win_client + wsl_server + ipc.py socket design into ONE process.
Everything runs in one address space against the one GPU:

    capture (video+audio+input) -> encode -> push -> generate -> inject

The model, the sampler loop, and action injection are all owned by
live_agent.init(): it loads the model/VAE/Encodec, starts the sampler thread, and
(with make_backend=True + the default on_output) injects predictions locally via
uinput. This file's ONLY job is Linux capture, feeding one 200 ms dataframe per
tick into the same encode->assemble->push path the server used to run — minus the
JPEG/socket round-trip, which is gone now that it's all in-process.

Run:
    export PLAICRAFT_MODEL_REPO=/path/to/plaicraft-model-pi0
    export PLAICRAFT_CKPT="$PLAICRAFT_MODEL_REPO/last_fp32/pytorch_model.bin"
    export PLAICRAFT_VAE=fast
    export PLAICRAFT_TARGET_MODALITIES=key_press,mouse_movement
    export PLAICRAFT_DENOISE_STEPS=8
    export PLAICRAFT_GEN_SECONDS=1.0
    export PLAICRAFT_CADENCE_SECONDS=1.5
    python run_live.py

================================  WAYLAND NOTES  ===============================
Three things differ from X11; read these before first run:

1. VIDEO CAPTURE. This uses ffmpeg x11grab, which on Wayland only sees windows
   that run under *XWayland*. Java Minecraft (LWJGL/GLFW) almost always runs under
   XWayland, so x11grab on $DISPLAY works. If your capture is black/empty, your
   Minecraft is a native-Wayland window — set PLAICRAFT_VIDEO_BACKEND=portal and
   capture via a PipeWire ScreenCast (you'll need to wire a pipewiresrc/portal
   pipeline; there's a stub + pointer in _video_grab_loop). x11grab is the default
   because it's the case that actually works for Minecraft.

2. INPUT INJECTION (predictions -> game). Handled by live_agent's uinput backend.
   uinput injects at the kernel evdev level, BELOW the compositor, so it works on
   Wayland — but needs permission on /dev/uinput:
       sudo modprobe uinput
       sudo usermod -aG input "$USER"      # then re-login
       # or a udev rule granting your user rw on /dev/uinput
   If make_backend fails, that's almost always the cause.

3. INPUT CAPTURE (your play -> context). pynput's listeners are unreliable on
   Wayland (X11-based). The reliable path is the recorder's evdev raw thread
   (rec._linux_raw_input_thread), which reads /dev/input/event* directly and works
   on Wayland. It also needs read access to /dev/input/* (the 'input' group again).

AUDIO is NOT a Wayland concern — it goes through PipeWire/PulseAudio regardless.
List sources with `pactl list sources short`; set the env vars below. Game audio
(audio_hear) is a sink *monitor* source; mic (audio_speak) is your input source.
If unset, silence is sent (fine if audio isn't in context_modalities).
==============================================================================
"""
import os
import sys
import time
import queue
import threading

import numpy as np

import live_agent
import online_sampler
import minecraft_input_recorder as rec

# ----------------------------------------------------------------------------- #
# Config (capture-side only; model/sampler config comes from live_agent's env)
# ----------------------------------------------------------------------------- #
LATENT_FPS       = int(os.environ.get("PLAICRAFT_FPS", "10"))
FRAMES_PER_UNIT  = int(getattr(live_agent, "VIDEO_FRAMES_PER_UNIT", 2))   # 2
GRAB_WH          = (1280, 720)
# 
FRAME_BYTES      = GRAB_WH[0] * GRAB_WH[1] * 3
QUEUE_MAX        = 50
UNIT_MS          = FRAMES_PER_UNIT * (1000 // LATENT_FPS)                 # 200
GENERATED_SECONDS  = int(getattr(online_sampler, "GEN_SECONDS", 2))   # 2

AUDIO_SR               = int(getattr(live_agent, "AUDIO_SR", 24000))
AUDIO_SAMPLES_PER_UNIT = int(getattr(live_agent, "AUDIO_SAMPLES_PER_UNIT", AUDIO_SR // 5))

# Video and audio come off separate ffmpeg pipes with different buffering lag, so
# they need *independent* time-offset corrections (the old single FRAME_LATENCY_MS
# was wrong for audio). Tune each by watching an obvious action vs its frame/sound.
FRAME_LATENCY_MS = int(os.environ.get("PLAICRAFT_FRAME_LATENCY_MS", "100"))
AUDIO_LATENCY_MS = int(os.environ.get("PLAICRAFT_AUDIO_LATENCY_MS", "100"))

VIDEO_BACKEND = os.environ.get("PLAICRAFT_VIDEO_BACKEND", "x11grab").lower()
HEAR_DEVICE   = os.environ.get("PLAICRAFT_AUDIO_HEAR_DEVICE", "")   # sink .monitor
SPEAK_DEVICE  = os.environ.get("PLAICRAFT_AUDIO_SPEAK_DEVICE", "")  # mic source

_stop = threading.Event()
_frame_q = queue.Queue(maxsize=QUEUE_MAX)
_hear_buf, _hear_lock = [], threading.Lock()
_speak_buf, _speak_lock = [], threading.Lock()


def _ffmpeg_bin():
    if os.environ.get("FFMPEG_BINARY"):
        return os.environ["FFMPEG_BINARY"]
    import shutil
    sys_ff = shutil.which("ffmpeg")     # prefer the system build (has pulse + x11grab)
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ----------------------------------------------------------------------------- #
# Region resolution: env override -> recorder's window finder -> full-ish default
# ----------------------------------------------------------------------------- #
def _resolve_region():
    try:
        win = rec.find_window(rec.WINDOW_TITLE)
        print(f"[capture] found window '{win['title']}")
        if win:
            
            try:
                moved = rec.focus_window(win["hwnd"])
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


# ----------------------------------------------------------------------------- #
# Video capture -> raw BGR frames on a queue (no JPEG; in-process feeds the VAE
# directly). x11grab works under XWayland; portal stub for native Wayland.
# ----------------------------------------------------------------------------- #
def _video_cmd(region):
    left, top, w, h = region
    w -= w % 2
    h -= h % 2
    out = ["-vf", f"scale={GRAB_WH[0]}:{GRAB_WH[1]}", "-r", str(LATENT_FPS),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    display = os.environ.get("DISPLAY", ":0")
    return [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-framerate", str(LATENT_FPS),
            "-video_size", f"{w}x{h}", "-i", f"{display}+{left},{top}", *out]


def _video_grab_loop(region):
    import subprocess
    if VIDEO_BACKEND == "portal":
        print("[capture] PLAICRAFT_VIDEO_BACKEND=portal: native-Wayland capture is "
              "setup-specific. Wire a PipeWire ScreenCast here (xdg-desktop-portal "
              "ScreenCast -> pipewiresrc, or `wf-recorder` on wlroots). x11grab "
              "(default) is what works for XWayland'd Minecraft.", flush=True)
        return
    proc = subprocess.Popen(_video_cmd(region), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=FRAME_BYTES)
    print("[capture] video grabber started (x11grab / XWayland).", flush=True)
    try:
        while not _stop.is_set():
            raw = _read_exact(proc.stdout, FRAME_BYTES)
            if raw is None:
                print("[capture] video pipe ended — wrong DISPLAY/region, or the "
                      "window is native-Wayland (try PLAICRAFT_VIDEO_BACKEND=portal).",
                      flush=True)
                break
            t = rec._now_ms()
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


# ----------------------------------------------------------------------------- #
# Audio capture (PipeWire/Pulse) -> rolling timestamped buffers
# ----------------------------------------------------------------------------- #
def _audio_grab_loop(source, buf, lock, label):
    import subprocess
    cmd = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
           "-f", "pulse", "-i", source,
           "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
    print(f"[capture] audio '{label}' started from pulse source: {source}", flush=True)
    CHUNK = AUDIO_SR // 50  # 20 ms
    try:
        while not _stop.is_set():
            raw = _read_exact(proc.stdout, CHUNK * 4)   # float32 = 4 bytes
            if raw is None:
                print(f"[capture] audio '{label}' pipe ended — bad pulse source name? "
                      f"check `pactl list sources short`", flush=True)
                break
            t = rec._now_ms()
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


def _slice_audio(buf, lock, t0, t_end):
    with lock:
        chunks = [c for (ts, c) in buf if t0 <= ts < t_end]
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    n = AUDIO_SAMPLES_PER_UNIT
    pcm = np.pad(pcm, (0, n - len(pcm))) if len(pcm) < n else pcm[:n]
    return pcm.astype(np.float32)


# ----------------------------------------------------------------------------- #
# Input event slicing (reads the recorder's rolling buffers, same as before)
# ----------------------------------------------------------------------------- #
def _prune(buf, t_end):
    buf[:] = [e for e in buf if int(e["timestamp"]) >= t_end]


def _slice_and_prune(t0, t_end):
    with rec._lock:
        ms = [e for e in rec._movement_buf if t0 <= int(e["timestamp"]) < t_end]
        cs = [e for e in rec._click_buf    if t0 <= int(e["timestamp"]) < t_end]
        ks = [e for e in rec._keyboard_buf if t0 <= int(e["timestamp"]) < t_end]
        _prune(rec._movement_buf, t_end)
        _prune(rec._click_buf, t_end)
        _prune(rec._keyboard_buf, t_end)
    return ms, cs, ks


# ----------------------------------------------------------------------------- #
# Ingest loop: pair frames into a 200 ms unit, encode every modality in-process,
# assemble one dataframe, push. Skips entirely while a rollout is generating
# (same single-GPU contention gate the server used).
# ----------------------------------------------------------------------------- #
def _ingest_loop():
    sampler = live_agent._S["sampler"]
    print("[ingest] started.", flush=True)
    pending = []
    have_audio = bool(HEAR_DEVICE or SPEAK_DEVICE)
    n = 0
    interval = GENERATED_SECONDS * 1000 / 200
    while not _stop.is_set():
        try:
            item = _frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        pending.append(item)
        if len(pending) < FRAMES_PER_UNIT:
            continue
        unit = pending[:FRAMES_PER_UNIT]
        pending = pending[FRAMES_PER_UNIT:]

        # Don't encode while the denoiser owns the GPU — it timeslices both and
        # stretches generation. Drop this unit; capture stays current regardless.
        #if sampler.busy.is_set():
            #continue

        t0      = unit[0][0]
        ev_t0   = t0 - FRAME_LATENCY_MS
        frames  = [f for (_t, f) in unit]

        if have_audio:
            a0 = t0 - AUDIO_LATENCY_MS
            pcm_hear  = _slice_audio(_hear_buf,  _hear_lock,  a0, a0 + UNIT_MS)
            pcm_speak = _slice_audio(_speak_buf, _speak_lock, a0, a0 + UNIT_MS)
            audio_h, audio_s = live_agent._encode_audio_inmemory(pcm_hear, pcm_speak)
        else:
            audio_h = audio_s = None    # -> _assemble_dataframe fills zeros

        ms, cs, ks = _slice_and_prune(ev_t0, ev_t0 + UNIT_MS)

        video  = live_agent._encode_frames(frames)
        key_by = live_agent._encode_keys_inmemory(ks, cs, ev_t0, FRAMES_PER_UNIT)
        mouse_by = live_agent._mouse_bins_inmemory(ms, ev_t0, FRAMES_PER_UNIT)
        fd = live_agent._assemble_dataframe(0, video, key_by, mouse_by, audio_h, audio_s)
        sampler.push_dataframe(fd)

        n += 1
        #if n % interval == 0:
            #print(f"[ingest] pushed {n} dataframes ({n * UNIT_MS / 1000:.1f}s) for {GENERATED_SECONDS} seconds", flush=True)


# ----------------------------------------------------------------------------- #
def main():
    print("[run_live] loading world model (this can take a minute)…", flush=True)
    # init() loads model + VAE + Encodec + keypress AE, builds the uinput backend,
    # and starts the sampler thread whose default on_output injects predictions
    # locally. We pass on_output=None so that default local-injection path is used.
    #live_agent.init(make_backend=True)

    # Start input capture (continuous). On Wayland ALL of it must come from evdev:
    #   - mouse MOTION  -> rec._linux_raw_input_thread (REL_X/REL_Y)
    #   - keys + clicks -> evdev_keys (EV_KEY) — pynput is X11-only and goes silent
    #     on Wayland, which would feed the model zero key context (degenerate).
    rec._recording.set()

    if getattr(rec, "IS_LINUX", True):
        threading.Thread(target=rec._linux_raw_mouse_input_thread, daemon=True).start()
        threading.Thread(target=rec._linux_raw_key_click_input_thread, daemon=True).start()


    

    # Capture threads.

    region = _resolve_region()
    threading.Thread(target=_video_grab_loop, args=(region,), daemon=True).start()
    """
    if HEAR_DEVICE:
        threading.Thread(target=_audio_grab_loop,
                         args=(HEAR_DEVICE, _hear_buf, _hear_lock, "hear"),
                         daemon=True).start()
    else:
        print("[capture] PLAICRAFT_AUDIO_HEAR_DEVICE unset — silence for audio_hear")
    if SPEAK_DEVICE:
        threading.Thread(target=_audio_grab_loop,
                         args=(SPEAK_DEVICE, _speak_buf, _speak_lock, "speak"),
                         daemon=True).start()
    else:
        print("[capture] PLAICRAFT_AUDIO_SPEAK_DEVICE unset — silence for audio_speak")

    # Ingest runs on the main thread; the sampler + injection run in their own
    # threads inside live_agent. Ctrl-C stops everything.
    print("[run_live] live. Move to the Minecraft window. Ctrl-C to stop.", flush=True)
    try:
        _ingest_loop()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        try:
            live_agent._S["sampler"].stop()
        except Exception:
            pass
        print("\n[run_live] stopped.")
    """
    while True:
        pass
    


if __name__ == "__main__":
    main()
