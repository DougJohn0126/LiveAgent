# --------------------------------------------------------------------------- #
# INJECT HALF — sampler output -> decode -> drive the game
# --------------------------------------------------------------------------- #
# Expected in _S (set up once at startup):
#   _S["dkl"]           : the decode_keypress_latents module
#   _S["ae"]            : built KeyPressAutoencoder (cpu, eval)
#   _S["index_to_name"] : dict[int,str] from dkl.load_name_maps()
#   _S["mouse_indices"] : set[int]      from dkl.load_name_maps()
#   _S["mm_stats"]      : the TRAINING-TIME mouse normalization stats, e.g.
#                         {"mode": "zscore", "mean": [...], "std": [...]} or
#                         {"mode": "minmax", "min": [...], "max": [...]}
#   _S["latents_native"]: optional bool override; auto-probed if absent.


def _probe_latent_orientation(ae, dkl) -> bool:
    """Decide, once, which orientation to feed dkl.decode_latents_to_activations.

    Returns True  -> pass model-native (Nwin, 5, 16) windows unchanged.
    Returns False -> pre-transpose to (Nwin, 16, 5) first.

    Why probe instead of trusting the docs: decode_latents_to_activations
    applies one .T before ae.decoder, its docstring says input is (N,16,5),
    the module header says "decoder: (16,5) -> (79,10)", and dkl.main() has a
    truthy `transpose="cpu"` bug — these cannot all be right. The decoder
    itself is unambiguous: latent_dim(16) != latent_seq_len(5), so exactly one
    input orientation runs and produces (1, 79, 10). We test both with zeros.
    """
    import torch
    p = next(ae.parameters())
    results = {}
    for key, shape in (
        ("dec_16x5", (1, dkl.LATENT_DIM, dkl.LATENT_SEQ_LEN)),   # decoder fed (16,5)
        ("dec_5x16", (1, dkl.LATENT_SEQ_LEN, dkl.LATENT_DIM)),   # decoder fed (5,16)
    ):
        try:
            with torch.no_grad():
                out = ae.decoder(torch.zeros(shape, device=p.device, dtype=p.dtype))
            results[key] = tuple(out.shape) == (1, dkl.INPUT_DIM, dkl.ORIGINAL_SEQ_LEN)
        except Exception:
            results[key] = False
    if results["dec_16x5"] == results["dec_5x16"]:
        raise RuntimeError(
            f"Cannot infer decoder latent orientation (probe results: {results}). "
            "Set _S['latents_native'] manually: True = feed (5,16) windows as-is, "
            "False = pre-transpose to (16,5)."
        )
    # decode_latents_to_activations .T's its input before the decoder, so:
    #   decoder accepts (16,5)  -> we must pass (5,16)  -> native, no transpose
    #   decoder accepts (5,16)  -> we must pass (16,5)  -> pre-transpose
    return results["dec_16x5"]


def _decode_prediction(fd, base_ms: int = 0):
    """fd (~2 s rollout) -> (key_events, mouse_rows). No game backend needed, so
    this runs on the headless WSL2 server too.

    key_events: [{key_name, start_ms, end_ms, abs_start_ms, abs_end_ms}, ...],
                merged keyboard + mouse-button, globally sorted by start_ms.
    mouse_rows: [(time_ms, dx, dy), ...] denormalized pixel deltas.

    ALL returned times are relative to the start of THIS rollout; `base_ms`
    only feeds the abs_* fields on key events. The injector must offset by the
    playhead when scheduling.
    """
    dkl = _S["dkl"]
    ae = _S["ae"]

    # ---- one-time setup checks (fail loudly, not at the game's expense) ----
    if "latents_native" not in _S:
        _S["latents_native"] = _probe_latent_orientation(ae, dkl)
        print(f"[decode] latent orientation: "
              f"{'model-native (5,16)' if _S['latents_native'] else 'pre-transpose to (16,5)'}")
    if not _S["mouse_indices"]:
        # load_name_maps() fell back to generic Key_<row> names with an empty
        # mouse set. Fine for offline JSON dumps; for injection it means wrong
        # key names and mouse buttons emitted as keyboard presses. Refuse.
        raise RuntimeError(
            "mouse_indices is empty: dkl.load_name_maps() used its fallback "
            "mapping. Run from the repo root so encode_key_press constants "
            "import; do not inject with generic channel names."
        )

    # ---- keys: [1,U,10,16] -> (Nwin,5,16) window latents -> events ----
    key_events = []
    kp = fd.key_press
    if kp is not None:
        if kp.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got key_press shape {tuple(kp.shape)}")
        U = kp.shape[1]
        wins_per_unit = kp.shape[2] // dkl.LATENT_SEQ_LEN
        if (kp.shape[2] != wins_per_unit * dkl.LATENT_SEQ_LEN
                or kp.shape[3] != dkl.LATENT_DIM
                or wins_per_unit != VIDEO_FRAMES_PER_UNIT):
            raise ValueError(
                f"key_press unit shape {tuple(kp.shape[2:])} does not match "
                f"{VIDEO_FRAMES_PER_UNIT} windows x ({dkl.LATENT_SEQ_LEN},{dkl.LATENT_DIM})"
            )
        # Assumes the 10-axis is [window0's 5 latent steps; window1's 5 steps],
        # matching the encoder-side packing. (Nwin, 5, 16), model-native.
        win = kp[0].reshape(U * VIDEO_FRAMES_PER_UNIT, dkl.LATENT_SEQ_LEN, dkl.LATENT_DIM)
        if not _S["latents_native"]:
            win = win.transpose(-1, -2)                       # -> (Nwin, 16, 5)
        latents = win.float().contiguous().cpu().numpy()
        acts = dkl.decode_latents_to_activations(ae, latents, device="cpu")   # (Nwin,79,10)
        bins = dkl.binarize(acts, thresh=dkl.KEY_ON_THRESH)                   # (Nwin,79,10)
        parsed = dkl.parse_events(bins, _S["index_to_name"], _S["mouse_indices"], base_ms)
        # Merge AND re-sort: each list is sorted internally, but the injector
        # needs one globally time-ordered stream.
        key_events = parsed["keyboard_events"] + parsed["mouse_button_events"]
        key_events.sort(key=lambda e: (e["start_ms"], e["key_name"]))

    # ---- mouse: [1,U,20,2] -> per-100ms (2,10) -> denormalized (t,dx,dy) ----
    mouse_rows = []
    mm = fd.mouse_movement
    if mm is not None:
        if _S.get("mm_stats") is None:
            # decode_mouse_movement(mm_stats=None) skips denormalization and
            # rints normalized values -> nearly all dx/dy collapse to 0. A dead
            # mouse with no error is worse than a crash; require the stats.
            raise RuntimeError(
                "mm_stats missing: mouse_movement is stored normalized and "
                "must be denormalized with the training-time stats before "
                "rounding. Set _S['mm_stats']."
            )
        if mm.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got mouse_movement shape {tuple(mm.shape)}")
        U = mm.shape[1]
        if mm.shape[2] != VIDEO_FRAMES_PER_UNIT * BINS_PER_FRAME or mm.shape[3] != 2:
            raise ValueError(
                f"mouse_movement unit shape {tuple(mm.shape[2:])} does not match "
                f"{VIDEO_FRAMES_PER_UNIT} windows x ({BINS_PER_FRAME},2)"
            )
        frames = mm[0].reshape(U * VIDEO_FRAMES_PER_UNIT, BINS_PER_FRAME, 2)  # (Nwin,10,2)
        per_window = list(frames.float().permute(0, 2, 1).cpu().numpy())     # Nwin x (2,10)
        series = dkl.decode_mouse_movement(per_window, mm_stats=_S["mm_stats"])["series"]
        mouse_rows = [(s["time_ms"], s["dx"], s["dy"]) for s in series]

    return key_events, mouse_rows
