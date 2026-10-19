import logging
import os

import numpy as np
import pytest

from nilm.data import (
    compute_real_power,
    discover_segments,
    increasing_mask,
    load_aggregate_power,
    load_complex_csv,
    load_power_cached,
    load_tag_events,
)


def write_lf(path, values):
    """Write complex values in Belkin's 'a+bi' notation, one row per sample."""
    lines = [",".join(f"{v.real}{v.imag:+}i" for v in row) for row in np.atleast_2d(values)]
    path.write_text("\n".join(lines) + "\n")


def make_segment(segment_dir, n_ticks, v1, i1, v2=None, i2=None):
    segment_dir.mkdir(parents=True)
    (segment_dir / "TimeTicks1.csv").write_text("\n".join(str(1000 + k / 6) for k in range(n_ticks)) + "\n")
    write_lf(segment_dir / "LF1V.csv", v1)
    write_lf(segment_dir / "LF1I.csv", i1)
    if v2 is not None:
        write_lf(segment_dir / "LF2V.csv", v2)
        write_lf(segment_dir / "LF2I.csv", i2)
    return segment_dir


def test_load_complex_csv_reads_belkin_notation(tmp_path):
    path = tmp_path / "LF1V.csv"
    path.write_text("-112.25+48.5i,0.01-0.02i\n1.5+9.1541143774520606e-05i,0-1i\n\n")

    values = load_complex_csv(path)

    assert values.shape == (2, 2)
    assert values[0, 0] == complex(-112.25, 48.5)
    assert values[1, 1] == complex(0, -1)
    assert values[1, 0].imag == pytest.approx(9.1541e-05, rel=1e-4)


def test_load_complex_csv_reports_bad_line(tmp_path):
    path = tmp_path / "LF1I.csv"
    path.write_text("1+2i,3+4i\n1+2i,garbage\n")

    with pytest.raises(ValueError, match="LF1I.csv: cannot parse line 2"):
        load_complex_csv(path)


def test_real_power_uses_in_phase_part_only():
    voltage = np.array([[230 + 0j, 10 + 0j],
                        [230 + 0j, 0j]])
    current = np.array([[2 + 0j, 1 + 0j],   # resistive: 460 W + 10 W from the harmonic
                        [0 + 2j, 0j]])      # 90 degrees out of phase: no real power
    np.testing.assert_allclose(compute_real_power(voltage, current), [470.0, 0.0])


def test_increasing_mask_drops_samples_after_clock_jump():
    ts = np.array([0.0, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.0])

    keep = increasing_mask(ts)

    # 2.5 is later than the sample before it but still behind 3.0, so it has to go too.
    assert keep.tolist() == [True, True, True, True, False, False, True, True]
    assert np.all(np.diff(ts[keep]) > 0)


def test_two_phases_are_summed_and_trimmed_to_common_length(tmp_path):
    ones = np.ones((4, 1), dtype=complex)
    seg = make_segment(tmp_path / "seg", n_ticks=4, v1=ones * 100, i1=ones * 2, v2=ones[:3] * 100, i2=ones[:3] * 3)

    signal = load_aggregate_power(seg)

    assert signal.n_phases == 2
    assert len(signal.ts) == len(signal.power) == 3
    assert signal.n_rows_trimmed == 1
    np.testing.assert_allclose(signal.power, 500.0)


def test_missing_phase_two_falls_back_to_phase_one(tmp_path, caplog):
    ones = np.ones((4, 1), dtype=complex)
    seg = make_segment(tmp_path / "seg", n_ticks=4, v1=ones * 100, i1=ones * 2)

    with caplog.at_level(logging.WARNING):
        signal = load_aggregate_power(seg)

    assert signal.n_phases == 1
    np.testing.assert_allclose(signal.power, 200.0)
    assert "phase 2" in caplog.text


def test_cache_is_reused_until_a_source_file_changes(tmp_path):
    ones = np.ones((4, 1), dtype=complex)
    seg = make_segment(tmp_path / "H1_CSV" / "CSV_OUT" / "Tagged_Training_1", n_ticks=4,
                       v1=ones * 100, i1=ones * 2, v2=ones * 100, i2=ones * 3)
    cache = tmp_path / "cache"

    first, source1 = load_power_cached(seg, cache)
    second, source2 = load_power_cached(seg, cache)
    assert (source1, source2) == ("parsed", "cache")
    np.testing.assert_array_equal(first.ts, second.ts)
    np.testing.assert_array_equal(first.power, second.power)
    assert second.n_phases == 2

    # Once a source file changes, the cache must be rebuilt.
    write_lf(seg / "LF2I.csv", ones * 5)
    stat = (seg / "LF2I.csv").stat()
    os.utime(seg / "LF2I.csv", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    third, source3 = load_power_cached(seg, cache)
    assert source3 == "parsed"
    np.testing.assert_allclose(third.power, 700.0)


def test_corrupt_cache_file_is_rebuilt(tmp_path, caplog):
    ones = np.ones((4, 1), dtype=complex)
    seg = make_segment(tmp_path / "H1_CSV" / "CSV_OUT" / "Tagged_Training_1", n_ticks=4, v1=ones, i1=ones)
    cache = tmp_path / "cache"
    load_power_cached(seg, cache)
    next(cache.glob("*.npz")).write_bytes(b"not a zip file")

    with caplog.at_level(logging.WARNING):
        signal, source = load_power_cached(seg, cache)

    assert source == "parsed"
    assert "unreadable cache" in caplog.text
    np.testing.assert_allclose(signal.power, 1.0)


def test_discover_segments_and_tags(tmp_path):
    seg1 = tmp_path / "H1_CSV(1)" / "CSV_OUT" / "Tagged_Training_04_13_1334300401"
    seg2 = tmp_path / "H2_CSV" / "CSV_OUT" / "Tagged_Training_02_15_1360915201"
    (tmp_path / "H2_CSV" / "CSV_OUT" / "Testing_07_17_1342508401").mkdir(parents=True)
    seg1.mkdir(parents=True)
    seg2.mkdir(parents=True)
    (seg1 / "TaggingInfo.csv").write_text('35,"Toaster",1334356440,1334356470\n16,"Dryer",1334356020,1334356110\n')
    (seg2 / "TaggingInfo.csv").write_text('14,"Hair Dryer",1360951590,1360951620\n')

    segments = discover_segments(tmp_path)
    events = load_tag_events(segments)

    assert segments["house"].tolist() == ["H1", "H2"]
    assert segments["segment_id"].tolist() == [0, 1]
    assert events["appliance"].tolist() == ["Toaster", "Dryer", "Hair Dryer"]
    assert events["event_id"].tolist() == [0, 1, 2]
    assert events["segment_id"].tolist() == [0, 0, 1]
    assert events.loc[0, "end_ts"] - events.loc[0, "start_ts"] == 30
