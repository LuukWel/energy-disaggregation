"""Finding the real switching step near each tag and cutting out its transient."""

import numpy as np
import pandas as pd
from scipy.signal import medfilt

from nilm.config import (
    LEVEL_CHANGE_TOLERANCE_WATTS,
    LEVEL_WINDOW_SECONDS,
    MAX_TRANSIENT_SECONDS,
    MIN_STEP_WATTS,
    MIN_TRANSIENT_SAMPLES,
    NOISY_BASELINE_WATTS,
    SMOOTH_WINDOW,
    STEP_LEVEL_SECONDS,
    STEP_SEARCH_SECONDS,
)

# (transition name, tag column to search around, expected step direction)
TRANSITIONS = (("on", "start_ts", "up"), ("off", "end_ts", "down"))


def smooth_power(power, window=SMOOTH_WINDOW):
    """Median filter. Removes single-sample spikes but keeps steps sharp."""
    return medfilt(power, kernel_size=window)


def sampling_rate(ts):
    """Average samples per second over the segment."""
    return len(ts) / (ts[-1] - ts[0])


def _level_changes(power, lo, hi, width):
    """Median of the next `width` samples minus the median of the previous `width`,
    for each boundary i -> i+1 with lo <= i < hi."""
    return np.array([
        np.median(power[i + 1:i + 1 + width]) - np.median(power[max(0, i + 1 - width):i + 1])
        for i in range(lo, hi)
    ])


def find_real_step(ts, power, tag_ts, direction, fs, method="jump",
                   search_seconds=STEP_SEARCH_SECONDS, min_step=MIN_STEP_WATTS):
    """Find the strongest transition in the expected direction near a tagged time.

    Tags are only accurate to about 30 s, so the search covers `search_seconds`
    either side of the tag. What counts as strongest depends on `method`:

    - "jump": the largest sample-to-sample jump.
    - "level": the largest difference between the median level of the 2 s
      before and the 2 s after a sample. A brief dip barely moves those medians,
      and a switch spread over several samples counts in full. The step is then
      placed at the largest single jump within that change.

    Returns a dict describing the step, or None if nothing clears the threshold.
    """
    half = int(search_seconds * fs)
    centre = np.searchsorted(ts, tag_ts)
    lo, hi = max(1, centre - half), min(len(power) - 1, centre + half)
    if hi - lo < 5:
        return None

    window = power[lo:hi + 1]
    # At least 30 W, and at least 5% of the local range so that a busy signal
    # with other loads cycling doesn't produce false steps.
    threshold = max(min_step, 0.05 * np.ptp(window))
    diff = np.diff(window)
    sign = 1 if direction == "up" else -1

    if method == "jump":
        best = int(np.argmax(sign * diff))
        size = sign * diff[best]
    elif method == "level":
        width = max(int(STEP_LEVEL_SECONDS * fs), 2)
        change = sign * _level_changes(power, lo, hi, width)
        k = int(np.argmax(change))
        size = change[k]
        near = slice(max(0, k - width), min(len(diff), k + width + 1))
        best = near.start + int(np.argmax(sign * diff[near]))
    else:
        raise ValueError(f"Unknown step method {method!r}; expected 'jump' or 'level'")
    if size <= threshold:
        return None

    # The jump happens between step_idx and step_idx + 1.
    step_idx = lo + best
    n_level = int(LEVEL_WINDOW_SECONDS * fs)
    before = power[max(0, step_idx - n_level):step_idx]
    after = power[step_idx + 1:min(len(power), step_idx + 1 + n_level)]

    baseline_std = float(np.std(before)) if len(before) > 3 else 0.0
    if baseline_std > NOISY_BASELINE_WATTS:
        # Another appliance is busy before the step. For a turn-on the lower
        # envelope is closer to the true baseline (and the upper one for a turn-off).
        pre_level = np.percentile(before, 10 if direction == "up" else 90)
    elif len(before) > 0:
        pre_level = np.median(before)
    else:
        pre_level = power[step_idx]
    post_level = np.median(after) if len(after) > 0 else power[step_idx + 1]

    return {
        "step_idx": step_idx,
        "step_ts": ts[step_idx],
        "step_watts": float(diff[best]),
        "tag_offset_sec": ts[step_idx] - tag_ts,
        "pre_level": float(pre_level),
        "post_level": float(post_level),
        "baseline_std": baseline_std,
        "baseline_stable": baseline_std <= NOISY_BASELINE_WATTS,
    }


def _is_calm(power, start, window, hold, max_std):
    """True if the rolling std stays below max_std for `hold` samples from `start`."""
    for k in range(0, hold, max(window // 2, 1)):
        chunk = power[start + k:start + k + window]
        if len(chunk) < 3:
            break
        if np.std(chunk) >= max_std:
            return False
    return True


def extract_transient(power, step, fs, max_seconds=MAX_TRANSIENT_SECONDS, pre_seconds=1.0):
    """Cut out the waveform from just before the step until the signal settles.

    The length is not fixed: a resistive load like a toaster settles almost
    immediately, while a computer keeps ramping for several seconds. The cut
    ends at the first point (at least 2 s after the step) where the 1 s rolling
    std stays below max(5 W, 2% of the step) for about a second.

    Returns (waveform relative to the pre-step level, duration in seconds), or None.
    """
    idx = step["step_idx"]
    start = max(0, idx - int(pre_seconds * fs))
    max_end = min(len(power), idx + int(max_seconds * fs))
    if max_end - start < 5:
        return None

    window = max(int(fs), 4)
    calm_std = max(5.0, 0.02 * abs(step["step_watts"]))  # big steps carry more ripple
    first_check = idx + int(2 * fs)                       # skip the jump itself
    hold = int(fs)

    end = max_end
    for j in range(first_check, max_end - hold):
        if j + window > len(power):
            break
        if _is_calm(power, j, window, hold, calm_std):
            end = j
            break

    waveform = power[start:end] - step["pre_level"]
    return waveform, (end - start) / fs


def compute_rise_time(waveform):
    """Samples between 10% and 90% of the total change (0 for an instant step)."""
    if len(waveform) < 3:
        return 0
    total = waveform[-1] - waveform[0]
    if abs(total) < 1:
        return 0

    def first_crossing(fraction, default):
        level = waveform[0] + fraction * total
        crossed = waveform >= level if total > 0 else waveform <= level
        return int(np.argmax(crossed)) if crossed.any() else default

    return max(0, first_crossing(0.9, len(waveform) - 1) - first_crossing(0.1, 0))


def implausible_level_change(step, direction):
    """True if the pre/post levels contradict the expected direction."""
    change = step["post_level"] - step["pre_level"]
    if direction == "up":
        return change < -LEVEL_CHANGE_TOLERANCE_WATTS
    return change > LEVEL_CHANGE_TOLERANCE_WATTS


def extract_event_transients(events, signals, step_method="jump"):
    """Try to extract a turn-on and a turn-off transient for every event.

    `signals` maps (house, segment) to a PowerSignal. Returns one row per tag
    time (the transients table in nilm/schema.py); `status` is "ok" or the
    reason it failed. The in-memory frame also carries the event's house,
    segment, appliance and overlap flag, and a `waveform` column with the
    variable-length waveform of the successful ones.
    """
    # Smooth each segment once; every event in it searches the same signal.
    smoothed = {
        key: {"ts": s.ts, "smooth": smooth_power(s.power), "fs": sampling_rate(s.ts)}
        for key, s in signals.items()
    }
    rows = []
    for event in events.itertuples(index=False):
        signal = smoothed.get((event.house, event.segment))
        for transition, tag_column, direction in TRANSITIONS:
            tag_ts = getattr(event, tag_column)
            row = {
                "transient_id": len(rows),
                "event_id": event.event_id,
                "house": event.house,
                "segment": event.segment,
                "appliance": event.appliance,
                "transition": transition,
                "tag_ts": tag_ts,
                "has_overlap": event.has_overlap,
                "status": "ok",
                "waveform": None,
            }
            rows.append(row)

            if signal is None:
                row["status"] = "no_signal"
                continue

            step = find_real_step(signal["ts"], signal["smooth"], tag_ts, direction, signal["fs"], step_method)
            if step is None:
                row["status"] = "no_step_found"
                continue
            row.update({k: v for k, v in step.items() if k != "step_idx"})

            cut = extract_transient(signal["smooth"], step, signal["fs"])
            if cut is None or len(cut[0]) < MIN_TRANSIENT_SAMPLES:
                row["status"] = "too_short"
                continue
            if implausible_level_change(step, direction):
                row["status"] = "implausible_power"
                continue

            waveform, duration = cut
            row.update({
                "waveform": waveform,
                "n_samples": len(waveform),
                "transient_duration": duration,
                "rise_time": compute_rise_time(waveform),
            })

    return pd.DataFrame(rows)


def waveform_table(transients):
    """The extracted waveforms in long format: one row per sample (the waveforms table)."""
    ok = transients[transients["status"] == "ok"]
    frames = [
        pd.DataFrame({"transient_id": row.transient_id, "sample": np.arange(len(row.waveform)),
                      "watts": row.waveform})
        for row in ok.itertuples()
    ]
    if not frames:
        return pd.DataFrame({"transient_id": [], "sample": [], "watts": []})
    return pd.concat(frames, ignore_index=True)
