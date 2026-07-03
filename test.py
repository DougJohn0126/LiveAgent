import threading
import minecraft_input_recorder as rec
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

    #if getattr(rec, "IS_LINUX", True):
        #threading.Thread(target=rec._linux_raw_mouse_input_thread, daemon=True).start()
        #threading.Thread(target=rec._linux_raw_key_click_input_thread, daemon=True).start()
        #threading.Thread(target=rec._linux_raw_video_input_thread, daemon=True).start()
    tthread = threading.Thread(target=rec._linux_raw_video_input_thread, daemon=True)
    tthread.start()
    tthread.join()

if __name__ == "__main__":
    main()
    