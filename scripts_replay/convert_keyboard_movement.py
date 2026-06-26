"""
convert_keyboard_movement.py
=============================
Combines the recorder's keyboard_data.json + mouse_click_data.json into the
key_press.json format used by inputInjector.

Input formats (from minecraft_input_recorder*.py)
-------------------------------------------------
keyboard_data.json : [{"timestamp", "key" (numeric GLFW/ASCII code), "action": PRESS|RELEASE}]
mouse_click_data.json : [{"timestamp", "action": LEFT_PRESS|LEFT_RELEASE|RIGHT_PRESS|RIGHT_RELEASE|...}]

Output format (matches key_press.json)
--------------------------------------
{
  "schema_version": 1,
  "bin_ms": 10,
  "raw_decoded": {
      "shape_per_frame": [79, 10],
      "frames": [ {key_name: [0/1 x10], ...79 keys...}, ... ]   # 1 = held in that 10ms sample
  },
  "parsed_decoded": {
      "events": [ {"key_name", "start_ms", "end_ms"}, ... ],
      "bin_ms": 10,
      "total_ms": <int>
  }
}

Programmatic use (mirrors convert_mouse_movement.convertmousemovement):
    import convert_keyboard_movement as ckm
    key_press_json = ckm.convertkeyboardmovement(keyboard_json_str, click_json_str)

CLI:
    python convert_keyboard_movement.py --keyboard keyboard_data.json \
        --clicks mouse_click_data.json --out key_press.json

TIME-SYNC NOTE
--------------
Timestamps are wall-clock ms, so they are normalised to a t0. By default t0 is
the earliest timestamp across BOTH input files. To keep the keyboard track in
sync with the mouse track, pass the SAME t0_ms / total_ms that the mouse
converter uses (e.g. the recording window's start timestamp and duration).
"""

import argparse
import json
import math

# --------------------------------------------------------------------------- #
# Fixed 79-key schema — order MUST match key_press.json exactly.
# --------------------------------------------------------------------------- #
KEY_ORDER = [
    "space",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "Escape", "Return", "Tab", "BackSpace", "Insert", "Delete",
    "Right", "Left", "Down", "Up",
    "Page_Up", "Page_Down", "Home", "End",
    "Shift_L", "Control_L", "Alt_L", "Super_L",
    "Shift_R", "Control_R", "Alt_R", "Super_R", "Menu",
    "bracketleft", "bracketright", "backslash", "semicolon", "apostrophe",
    "comma", "period", "slash", "minus", "equal", "grave",
    "F1", "F3", "F5", "Caps_Lock",
    "mouse_left", "mouse_right", "scroll_up", "scroll_down",
]
_KEY_SET = set(KEY_ORDER)

# GLFW special codes (as emitted by the recorder's _key_code) -> schema name.
_GLFW_SPECIAL = {
    32: "space", 256: "Escape", 257: "Return", 258: "Tab", 259: "BackSpace",
    260: "Insert", 261: "Delete", 262: "Right", 263: "Left", 264: "Down",
    265: "Up", 266: "Page_Up", 267: "Page_Down", 268: "Home", 269: "End",
    280: "Caps_Lock", 290: "F1", 292: "F3", 294: "F5",
    340: "Shift_L", 344: "Shift_R", 341: "Control_L", 345: "Control_R",
    342: "Alt_L", 346: "Alt_R", 343: "Super_L", 347: "Super_R", 348: "Menu",
}

# Printable-ASCII punctuation (recorder emits ord(char.upper())) -> schema name.
_PUNCT = {
    91: "bracketleft", 93: "bracketright", 92: "backslash", 59: "semicolon",
    39: "apostrophe", 44: "comma", 46: "period", 47: "slash", 45: "minus",
    61: "equal", 96: "grave",
}

_CLICK_MAP = {"LEFT": "mouse_left", "RIGHT": "mouse_right"}


# --------------------------------------------------------------------------- #
# Mapping helpers
# --------------------------------------------------------------------------- #
def keycode_to_name(code):
    """Map a recorder key code to a schema key_name, or None if unsupported."""
    if code is None or code < 0:
        return None
    if 48 <= code <= 57:          # '0'..'9'
        return chr(code)
    if 65 <= code <= 90:          # 'A'..'Z' -> lowercase schema name
        return chr(code).lower()
    if code in _GLFW_SPECIAL:
        return _GLFW_SPECIAL[code]
    if code in _PUNCT:
        return _PUNCT[code]
    return None                   # unknown / unmapped (e.g. -1, ctrl-combos)


def click_action_to_name(action):
    """LEFT_PRESS/RIGHT_RELEASE/... -> mouse_left/mouse_right (None otherwise)."""
    if not action:
        return None
    base = action.rsplit("_", 1)[0]   # "LEFT_PRESS" -> "LEFT"
    return _CLICK_MAP.get(base)


def _load(data):
    """Accept a parsed list, a JSON string, or a file path."""
    if isinstance(data, list):
        return data
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        s = data.strip()
        if s[:1] in ("[", "{"):
            return json.loads(s)
        with open(data, encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"Unsupported input type: {type(data).__name__}")


# --------------------------------------------------------------------------- #
# Press/release pairing
# --------------------------------------------------------------------------- #
def _pair_events(raw, t0, total_ms, bin_ms, lone_release_from_start=True):
    """
    raw: list of (timestamp, key_name, is_press).
    Pairs each down-transition with its release. OS key-repeat (repeated PRESS
    while held) is ignored. Unpaired presses are held to total_ms; a release
    with no matching press (key held across a previous flush window) is treated
    as held from 0.
    """
    # press before release when timestamps tie (handles instant taps)
    raw.sort(key=lambda r: (r[0], 0 if r[2] else 1))

    open_press = {}            # key_name -> start_ms (relative)
    events = []
    for ts, name, is_press in raw:
        rel = ts - t0
        if is_press:
            if name not in open_press:        # down transition only
                open_press[name] = rel
        else:
            if name in open_press:
                events.append([name, open_press.pop(name), rel])
            elif lone_release_from_start:
                events.append([name, 0, rel])

    # keys still held when the window ended
    for name, start in open_press.items():
        events.append([name, start, total_ms])

    # clamp, ensure positive duration, emit dicts
    out = []
    for name, s, e in events:
        s = max(0, min(int(s), total_ms))
        e = max(0, min(int(e), total_ms))
        if e <= s:
            e = min(s + bin_ms, total_ms)     # minimum 1-bin tap
        out.append({"key_name": name, "start_ms": s, "end_ms": e})
    out.sort(key=lambda ev: (ev["start_ms"], ev["key_name"]))
    return out


# --------------------------------------------------------------------------- #
# Raw binary grid
# --------------------------------------------------------------------------- #
def _build_frames(events, total_ms, bin_ms, subsamples):
    n_samples = total_ms // bin_ms
    n_frames = max(1, math.ceil(n_samples / subsamples))
    n_padded = n_frames * subsamples

    timelines = {k: [0] * n_padded for k in KEY_ORDER}
    for ev in events:
        tl = timelines.get(ev["key_name"])
        if tl is None:
            continue
        s = ev["start_ms"] // bin_ms
        e = math.ceil(ev["end_ms"] / bin_ms)
        for i in range(max(0, s), min(e, n_padded)):
            tl[i] = 1

    frames = []
    for f in range(n_frames):
        a = f * subsamples
        frames.append({k: timelines[k][a:a + subsamples] for k in KEY_ORDER})
    return frames


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def convertkeyboardmovement(keyboarddata, clickdata,
                            t0_ms=None, total_ms=None,
                            bin_ms=10, subsamples=10):
    """Convert recorder keyboard + click data into key_press.json format.

    Returns a JSON string (matching convert_mouse_movement's style).
    """
    kb = _load(keyboarddata)
    ck = _load(clickdata)

    all_ts = [e["timestamp"] for e in kb if "timestamp" in e]
    all_ts += [e["timestamp"] for e in ck if "timestamp" in e]

    t0 = (min(all_ts) if all_ts else 0) if t0_ms is None else t0_ms

    frame_ms = bin_ms * subsamples
    if total_ms is None:
        span = (max(all_ts) - t0) if all_ts else 0
        total = int(math.ceil((span + 1) / frame_ms) * frame_ms) if span > 0 else frame_ms
    else:
        total = int(math.ceil(total_ms / frame_ms) * frame_ms)   # whole frames

    raw = []
    skipped = []
    for e in kb:
        name = keycode_to_name(e.get("key"))
        if name is None or name not in _KEY_SET:
            skipped.append(("key", e.get("key")))
            continue
        raw.append((e["timestamp"], name, e.get("action") == "PRESS"))
    for e in ck:
        name = click_action_to_name(e.get("action", ""))
        if name is None or name not in _KEY_SET:
            skipped.append(("click", e.get("action")))
            continue
        raw.append((e["timestamp"], name, e.get("action", "").endswith("PRESS")))

    events = _pair_events(raw, t0, total, bin_ms)
    frames = _build_frames(events, total, bin_ms, subsamples)

    result = {
        "schema_version": 1,
        "bin_ms": bin_ms,
        "raw_decoded": {
            "shape_per_frame": [len(KEY_ORDER), subsamples],
            "frames": frames,
        },
        "parsed_decoded": {
            "events": events,
            "bin_ms": bin_ms,
            "total_ms": total,
        },
    }
    # stash skips on the function for optional debugging (not serialized)
    convertkeyboardmovement.last_skipped = skipped
    return json.dumps(result)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Convert keyboard + click data to key_press.json format")
    ap.add_argument("--keyboard", default="keyboard_data.json")
    ap.add_argument("--clicks",   default="mouse_click_data.json")
    ap.add_argument("--out",      default="key_press.json")
    ap.add_argument("--t0-ms",    type=int, default=None,
                    help="Time origin (ms). Pass the same value as the mouse converter to stay in sync.")
    ap.add_argument("--total-ms", type=int, default=None,
                    help="Recording duration (ms). Defaults to the data span.")
    args = ap.parse_args()

    js = convertkeyboardmovement(args.keyboard, args.clicks,
                                 t0_ms=args.t0_ms, total_ms=args.total_ms)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(js)

    data = json.loads(js)
    ev = data["parsed_decoded"]["events"]
    print(f"Wrote {args.out}")
    print(f"  events : {len(ev)}")
    print(f"  frames : {len(data['raw_decoded']['frames'])}")
    print(f"  total  : {data['parsed_decoded']['total_ms']} ms")
    if convertkeyboardmovement.last_skipped:
        print(f"  skipped {len(convertkeyboardmovement.last_skipped)} unmapped event(s): "
              f"{convertkeyboardmovement.last_skipped[:10]}")
    for e in ev[:10]:
        print(f"    {e['key_name']:>12}  {e['start_ms']:>7} -> {e['end_ms']}")


if __name__ == "__main__":
    main()
