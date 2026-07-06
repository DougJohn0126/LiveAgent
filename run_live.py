import os
import queue
import threading

import numpy as np

import minecraft_input_recorder as recorder
import live_agent
import online_sampler

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

HEAR_DEVICE   = os.environ.get("PLAICRAFT_AUDIO_HEAR_DEVICE", "")   # sink .monitor
SPEAK_DEVICE  = os.environ.get("PLAICRAFT_AUDIO_SPEAK_DEVICE", "")  # mic source

_stop = threading.Event()

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
    with recorder._lock:
        ms = [e for e in recorder._movement_buf if t0 <= int(e["timestamp"]) < t_end]
        cs = [e for e in recorder._click_buf    if t0 <= int(e["timestamp"]) < t_end]
        ks = [e for e in recorder._keyboard_buf if t0 <= int(e["timestamp"]) < t_end]
        _prune(recorder._movement_buf, t_end)
        _prune(recorder._click_buf, t_end)
        _prune(recorder._keyboard_buf, t_end)
    return ms, cs, ks


# ----------------------------------------------------------------------------- #
# Ingest loop: pair frames into a 200 ms unit, encode every modality in-process,
# assemble one dataframe, push. Skips entirely while a rollout is generating
# (same single-GPU contention gate the server used).
# ----------------------------------------------------------------------------- #

# Raw units held (not dropped!) while the denoiser owns the GPU, encoded as soon
# as it's free. Dropping them compressed the context timeline: dataframe_indices
# are contiguous downstream, so every dropped 200 ms unit was a hidden time-jump
# the model never saw in training — and at CADENCE~0.4s with ~0.5s generations
# that was a large fraction of ALL units. RAM cost: ~5.5 MB per raw unit.
BACKLOG_MAX_UNITS = int(os.environ.get("PLAICRAFT_BACKLOG_UNITS", "15"))


def _encode_and_push_unit(sampler, unit, have_audio):
    """Encode ONE 200 ms unit (video + keys + mouse + audio) and push it.
    Event slicing happens at capture-timestamp time, so encoding late (after a
    backlog wait) still bins events onto the correct frames."""
    t0      = unit[0][0]
    ev_t0   = t0 - FRAME_LATENCY_MS
    frames  = [f for (_t, f) in unit]

    ms, cs, ks = _slice_and_prune(ev_t0, ev_t0 + UNIT_MS)

    key_by   = live_agent.encode_keys_inmemory(ks, cs, ev_t0, FRAMES_PER_UNIT)
    mouse_by = live_agent.mouse_bins_inmemory(ms, ev_t0, FRAMES_PER_UNIT)
    video    = live_agent.encode_frames(frames)

    if have_audio:
        a0 = t0 - AUDIO_LATENCY_MS
        pcm_hear  = _slice_audio(recorder._hear_buf,  recorder._hear_lock,  a0, a0 + UNIT_MS)
        pcm_speak = _slice_audio(recorder._speak_buf, recorder._speak_lock, a0, a0 + UNIT_MS)
        audio_h, audio_s = live_agent.encode_audio_inmemory(pcm_hear, pcm_speak)
    else:
        audio_h = audio_s = None    # -> _assemble_dataframe fills zeros

    fd = live_agent._assemble_dataframe(0, video, key_by, mouse_by, audio_h, audio_s)
    sampler.push_dataframe(fd)


def _ingest_loop():
    sampler = live_agent._S["sampler"]
    print("[ingest] started.", flush=True)
    pending = []
    backlog = []
    have_audio = bool(HEAR_DEVICE or SPEAK_DEVICE)
    while not _stop.is_set():
        try:
            item = recorder._frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        # Add the frame; once we have 2, slice off the first 2 as a 200 ms unit
        # and keep any remainder (normally empty) in pending.
        pending.append(item)
        if len(pending) < FRAMES_PER_UNIT:
            continue
        unit = pending[:FRAMES_PER_UNIT]
        pending = pending[FRAMES_PER_UNIT:]

        # Don't encode while the denoiser owns the GPU (it timeslices both and
        # stretches generation) — but do NOT drop the unit: queue it and encode
        # the moment the GPU is free, so the real timeline stays gap-free.
        if sampler.busy.is_set():
            backlog.append(unit)
            if len(backlog) > BACKLOG_MAX_UNITS:
                backlog.pop(0)
                print("[ingest][warn] backlog overflow — dropped oldest unit "
                      "(timeline gap). Generation is outrunning ingest; "
                      "raise PLAICRAFT_CADENCE_SECONDS or lower denoise steps.",
                      flush=True)
            continue

        for u in backlog:
            _encode_and_push_unit(sampler, u, have_audio)
        backlog.clear()
        _encode_and_push_unit(sampler, unit, have_audio)


# ----------------------------------------------------------------------------- #
def main():
    print("[run_live] loading world model (this can take a minute)…", flush=True)
    # init() loads the model + VAE + Encodec + keypress AE, builds the uinput backend,
    # and starts the sampler thread whose default on_output injects predictions
    # lback to the game.
    live_agent.init(make_backend=True)

    recorder._recording.set()
    recorder.start_recording()
    
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

if __name__ == "__main__":
    main()