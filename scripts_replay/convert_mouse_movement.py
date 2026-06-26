"""
convert_mouse_movement.py

Converts mouse_movement_data.json (raw timestamped events) into the
binned frame format of mouse_movement.json.

Output structure:
  schema_version : 1
  bin_ms          : 10
  raw_decoded:
    shape_per_frame : [2, BINS_PER_FRAME]   (2 = dx+dy channels)
    frames          : list of {dx: [...], dy: [...]}  (one per frame)
  parsed_decoded:
    series          : [{time_ms, dx, dy}, ...]   (one entry per 10ms bin)
    bin_ms          : 10
    total_ms        : total duration

Rules:
  - Each bin is exactly 10 ms wide.
  - All dx/dy values that fall inside a bin are SUMMED into that bin.
  - A "frame" groups BINS_PER_FRAME consecutive bins (default 10 → 100 ms/frame).
  - The first event's timestamp is t=0.
  - The input file may be truncated (missing closing `]`); the script repairs it.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

# ── configuration ────────────────────────────────────────────────────────────
INPUT_FILE   = Path("mouse_movement_data.json")
OUTPUT_FILE  = Path("mouse_movement_converted.json")
BIN_MS       = 10          # milliseconds per bin
BINS_PER_FRAME = 10        # bins grouped into one frame
SCHEMA_VERSION = 1
# ─────────────────────────────────────────────────────────────────────────────


def load_events(path: Path) -> list[dict]:
    """Load events, repairing a truncated JSON array if needed."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw.endswith("]"):
        last_brace = raw.rfind("}")
        if last_brace == -1:
            raise ValueError("Cannot find any complete JSON object in input.")
        raw = raw[: last_brace + 1] + "]"
    return json.loads(raw)


def bin_events(events: list[dict], bin_ms: int) -> dict[int, tuple[float, float]]:
    """
    Assign each event to a bin index and accumulate dx/dy.

    Returns a dict  { bin_index: (total_dx, total_dy) }
    The first event establishes t=0; its own dx/dy IS included.
    """
    if not events:
        return {}

    t0 = events[0]["timestamp"]
    accumulator: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])

    for ev in events:
        rel_ms  = ev["timestamp"] - t0
        bin_idx = int(rel_ms // bin_ms)
        accumulator[bin_idx][0] += ev.get("mouseDX", 0.0)
        accumulator[bin_idx][1] += ev.get("mouseDY", 0.0)

    return {k: (v[0], v[1]) for k, v in accumulator.items()}


def build_output(
    binned: dict[int, tuple[float, float]],
    bin_ms: int,
    bins_per_frame: int,
) -> dict:
    """Assemble the full output structure."""
    if not binned:
        total_bins = 0
    else:
        total_bins = max(binned.keys()) + 1

    # ── raw_decoded ──────────────────────────────────────────────────────────
    num_frames = math.ceil(total_bins / bins_per_frame)
    frames = []
    for f in range(num_frames):
        dx_list, dy_list = [], []
        for b in range(bins_per_frame):
            bin_idx = f * bins_per_frame + b
            dx, dy  = binned.get(bin_idx, (0.0, 0.0))
            dx_list.append(int(round(dx)))
            dy_list.append(int(round(dy)))
        frames.append({"dx": dx_list, "dy": dy_list})

    raw_decoded = {
        "shape_per_frame": [2, bins_per_frame],
        "frames": frames,
    }

    # ── parsed_decoded ───────────────────────────────────────────────────────
    series = []
    for bin_idx in range(total_bins):
        dx, dy = binned.get(bin_idx, (0.0, 0.0))
        series.append({
            "time_ms": bin_idx * bin_ms,
            "dx":      int(round(dx)),
            "dy":      int(round(dy)),
        })

    total_ms = total_bins * bin_ms
    parsed_decoded = {
        "series":   series,
        "bin_ms":   bin_ms,
        "total_ms": total_ms,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "bin_ms":         bin_ms,
        "raw_decoded":    raw_decoded,
        "parsed_decoded": parsed_decoded,
    }

def convertmousemovement(data: str): 
    events = json.loads(data)
    print(f"  Events loaded : {len(events)}")

    binned = bin_events(events, BIN_MS)
    print(f"  Bins occupied : {len(binned)}  (bin_ms={BIN_MS})")

    output = build_output(binned, BIN_MS, BINS_PER_FRAME)
    return json.dumps(output, indent=2)





def main() -> None:
    print(f"Reading  : {INPUT_FILE}")
    events = load_events(INPUT_FILE)
    print(f"  Events loaded : {len(events)}")

    binned = bin_events(events, BIN_MS)
    print(f"  Bins occupied : {len(binned)}  (bin_ms={BIN_MS})")

    output = build_output(binned, BIN_MS, BINS_PER_FRAME)

    frames      = output["raw_decoded"]["frames"]
    total_bins  = len(output["parsed_decoded"]["series"])
    total_ms    = output["parsed_decoded"]["total_ms"]
    print(f"  Total bins    : {total_bins}  ({total_ms} ms)")
    print(f"  Frames        : {len(frames)}  ({BINS_PER_FRAME} bins/frame)")

    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2)
    )
    print(f"Written  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
