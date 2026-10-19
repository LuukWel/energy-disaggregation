"""Reading the raw Belkin files: tag lists, timestamps and voltage/current phasors."""

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TAG_COLUMNS = ["appliance_id", "appliance", "start_ts", "end_ts"]
POWER_FILES = ("TimeTicks1.csv", "LF1V.csv", "LF1I.csv", "LF2V.csv", "LF2I.csv")
# Part of every cache fingerprint. Increase it when load_aggregate_power changes,
# so existing cache files are rebuilt.
CACHE_FORMAT = 1


@dataclass(frozen=True)
class PowerSignal:
    """Aggregate real power of one segment, plus what loading had to fix."""
    ts: np.ndarray          # Unix seconds, strictly increasing
    power: np.ndarray       # W, all available phases summed
    n_phases: int
    n_rows_trimmed: int     # rows of the longest file beyond the common length
    n_out_of_order: int     # samples dropped because the clock went backwards


def house_label(house_dir):
    """Short house name: 'H1_CSV' or 'H1_CSV(1)' -> 'H1'."""
    name = Path(house_dir).name
    match = re.match(r"H\d+", name)
    return match.group(0) if match else name


def discover_segments(data_root):
    """List the tagged training segments under the dataset root."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    rows = []
    house_dirs = {}
    for house_dir in sorted(p for p in data_root.glob("H*") if p.is_dir()):
        house = house_label(house_dir)
        if house in house_dirs:
            raise ValueError(f"Two folders for house {house}: {house_dirs[house].name} and {house_dir.name}")
        house_dirs[house] = house_dir

        # Only the Tagged_Training folders come with a TaggingInfo.csv; Testing_* is unlabelled.
        for segment_dir in sorted(house_dir.glob("CSV_OUT/Tagged_Training*")):
            if segment_dir.is_dir():
                rows.append({"house": house, "segment": segment_dir.name, "segment_dir": str(segment_dir)})

    if not rows:
        raise FileNotFoundError(f"No H*/CSV_OUT/Tagged_Training* folders under {data_root}")
    segments = pd.DataFrame(rows)
    segments.insert(0, "segment_id", np.arange(len(segments)))
    return segments


def load_tag_events(segments):
    """Read TaggingInfo.csv from every segment into one event table."""
    frames = []
    for seg in segments.itertuples(index=False):
        path = Path(seg.segment_dir) / "TaggingInfo.csv"
        if not path.exists():
            logger.warning("No TaggingInfo.csv in %s/%s, skipping segment", seg.house, seg.segment)
            continue

        tags = pd.read_csv(path, header=None, names=TAG_COLUMNS)
        tags["appliance_id"] = pd.to_numeric(tags["appliance_id"], errors="coerce")
        tags["appliance"] = tags["appliance"].astype(str).str.strip()
        tags["start_ts"] = pd.to_numeric(tags["start_ts"], errors="coerce")
        tags["end_ts"] = pd.to_numeric(tags["end_ts"], errors="coerce")

        bad = tags["start_ts"].isna() | tags["end_ts"].isna()
        if bad.any():
            logger.warning("%s/%s: skipped %d tag rows without valid timestamps", seg.house, seg.segment, bad.sum())
            tags = tags[~bad]

        tags.insert(0, "segment_id", seg.segment_id)
        tags.insert(1, "house", seg.house)
        tags.insert(2, "segment", seg.segment)
        tags["segment_dir"] = seg.segment_dir
        frames.append(tags)

    if not frames:
        raise FileNotFoundError("None of the segments has a TaggingInfo.csv")

    events = pd.concat(frames, ignore_index=True)
    events.insert(0, "event_id", np.arange(len(events)))
    return events


def load_timestamps(segment_dir, phase=1):
    """Unix timestamps (seconds) of the low-frequency samples."""
    path = Path(segment_dir) / f"TimeTicks{phase}.csv"
    ticks = pd.to_numeric(pd.read_csv(path, header=None).iloc[:, 0], errors="coerce")
    # Dropping a bad row here would shift every later sample against the LF files.
    if ticks.isna().any():
        raise ValueError(f"{path.name}: {ticks.isna().sum()} rows are not numeric")
    return ticks.to_numpy(dtype=float)


def increasing_mask(ts):
    """True for samples whose timestamp is later than every sample before it."""
    if len(ts) == 0:
        return np.zeros(0, dtype=bool)
    # Use the running maximum: after the clock jumps backwards, the next few dozen
    # samples are still earlier than samples already seen.
    previous_max = np.maximum.accumulate(ts)[:-1]
    return np.concatenate([[True], ts[1:] > previous_max])


def load_complex_csv(path):
    """Parse an LF file: one row per sample, one complex value ('97.1+74.0i') per harmonic."""
    path = Path(path)
    rows = []
    n_cols = None
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = [complex(value.replace("i", "j")) for value in line.split(",")]
            except ValueError as exc:
                raise ValueError(f"{path.name}: cannot parse line {line_no}") from exc
            if n_cols is None:
                n_cols = len(row)
            elif len(row) != n_cols:
                raise ValueError(f"{path.name}: line {line_no} has {len(row)} values, expected {n_cols}")
            rows.append(row)

    if not rows:
        raise ValueError(f"{path.name} contains no samples")
    return np.array(rows, dtype=np.complex128)


def compute_real_power(voltage, current):
    """Real power per sample: sum over harmonics of Re(V * conj(I))."""
    n_rows = min(len(voltage), len(current))
    n_cols = min(voltage.shape[1], current.shape[1])
    power = np.zeros(n_rows)
    for h in range(n_cols):
        power += np.real(voltage[:n_rows, h] * np.conj(current[:n_rows, h]))
    return power


def load_aggregate_power(segment_dir):
    """Aggregate real power (W) for one segment, as a PowerSignal.

    Phase 1 is required; phase 2 is added when its files are present and readable.
    """
    segment_dir = Path(segment_dir)
    ts = load_timestamps(segment_dir, phase=1)
    v1 = load_complex_csv(segment_dir / "LF1V.csv")
    i1 = load_complex_csv(segment_dir / "LF1I.csv")
    lengths = [len(ts), len(v1), len(i1)]
    n_phases = 1

    # The houses have a split-phase supply, phase 1 misses whatever is
    # on the other leg (and half of every 240 V appliance).
    try:
        v2 = load_complex_csv(segment_dir / "LF2V.csv")
        i2 = load_complex_csv(segment_dir / "LF2I.csv")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("%s: phase 2 not usable (%s); using phase 1 only", segment_dir.name, exc)
    else:
        lengths += [len(v2), len(i2)]
        n_phases = 2

    # The files differ in length by a few rows.
    n = min(lengths)
    power = compute_real_power(v1[:n], i1[:n])
    if n_phases == 2:
        power = power + compute_real_power(v2[:n], i2[:n])

    keep = increasing_mask(ts[:n])
    return PowerSignal(
        ts=ts[:n][keep],
        power=power[keep],
        n_phases=n_phases,
        n_rows_trimmed=max(lengths) - n,
        n_out_of_order=int((~keep).sum()),
    )


def _fingerprint(segment_dir):
    """Size and modification time of the source files. Changes whenever the data does."""
    parts = [f"format={CACHE_FORMAT}"]
    for name in POWER_FILES:
        path = segment_dir / name
        if path.exists():
            stat = path.stat()
            parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return ";".join(parts)


def load_power_cached(segment_dir, cache_dir):
    """load_aggregate_power, reusing a saved .npz copy while the source files are unchanged.

    Parsing the LF text files takes most of the pipeline's run time. Returns
    (signal, source), where source is "cache" or "parsed".
    """
    segment_dir = Path(segment_dir)
    cache_file = Path(cache_dir) / f"{house_label(segment_dir.parent.parent)}_{segment_dir.name}.npz"
    fingerprint = _fingerprint(segment_dir)

    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                if str(cached["fingerprint"]) == fingerprint:
                    return PowerSignal(
                        ts=cached["ts"],
                        power=cached["power"],
                        n_phases=int(cached["n_phases"]),
                        n_rows_trimmed=int(cached["n_rows_trimmed"]),
                        n_out_of_order=int(cached["n_out_of_order"]),
                    ), "cache"
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            logger.warning("Ignoring unreadable cache file %s (%s)", cache_file.name, exc)

    signal = load_aggregate_power(segment_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first so an interrupted run can't leave half a cache file.
    tmp_file = cache_file.with_suffix(".tmp")
    with open(tmp_file, "wb") as fh:
        np.savez(fh, fingerprint=fingerprint, ts=signal.ts, power=signal.power, n_phases=signal.n_phases,
                 n_rows_trimmed=signal.n_rows_trimmed, n_out_of_order=signal.n_out_of_order)
    tmp_file.replace(cache_file)
    return signal, "parsed"
