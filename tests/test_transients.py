import numpy as np
import pytest

from nilm.transients import (
    compute_rise_time,
    extract_transient,
    find_real_step,
    sampling_rate,
    smooth_power,
)

FS = 6.0
STEP_AT = 900  # first sample after the jump


def synthetic_signal(baseline=200.0, step=800.0, n=1800, seed=0):
    """Five minutes at 6 Hz with a little noise and one step at sample 900 (t = 1150 s)."""
    rng = np.random.default_rng(seed)
    ts = 1000.0 + np.arange(n) / FS
    power = baseline + rng.normal(0, 1.0, n)
    power[STEP_AT:] += step
    return ts, power


def test_finds_upward_step_and_tag_offset():
    ts, power = synthetic_signal()

    step = find_real_step(ts, power, tag_ts=1140.0, direction="up", fs=sampling_rate(ts))

    assert step["step_idx"] == STEP_AT - 1
    assert step["tag_offset_sec"] == pytest.approx(10.0, abs=0.5)   # real switch-on 10 s after the tag
    assert step["step_watts"] == pytest.approx(800, abs=10)
    assert step["pre_level"] == pytest.approx(200, abs=2)
    assert step["post_level"] == pytest.approx(1000, abs=2)
    assert step["baseline_stable"]


def test_finds_downward_step_before_the_tag():
    ts, power = synthetic_signal(baseline=1000.0, step=-800.0)

    step = find_real_step(ts, power, tag_ts=1170.0, direction="down", fs=sampling_rate(ts))

    assert step["tag_offset_sec"] == pytest.approx(-20.0, abs=0.5)
    assert step["step_watts"] == pytest.approx(-800, abs=10)


def test_no_step_returns_none():
    ts, power = synthetic_signal(step=0.0)
    assert find_real_step(ts, power, 1150.0, "up", sampling_rate(ts)) is None


def test_step_in_the_wrong_direction_is_ignored():
    ts, power = synthetic_signal()
    assert find_real_step(ts, power, 1150.0, "down", sampling_rate(ts)) is None


def test_step_outside_search_window_is_not_found():
    ts, power = synthetic_signal()
    # The step is at 1150 s; a tag at 1250 s is 100 s away.
    assert find_real_step(ts, power, 1250.0, "up", sampling_rate(ts), search_seconds=45) is None


def test_smoothing_stops_a_single_spike_looking_like_a_switch():
    ts, power = synthetic_signal(step=0.0)
    power[1000] += 500

    assert find_real_step(ts, power, 1160.0, "up", sampling_rate(ts)) is not None
    assert find_real_step(ts, smooth_power(power), 1160.0, "up", sampling_rate(ts)) is None


def test_clean_step_settles_after_about_three_seconds():
    ts, power = synthetic_signal()
    fs = sampling_rate(ts)
    step = find_real_step(ts, power, 1150.0, "up", fs)

    waveform, duration = extract_transient(power, step, fs)

    # 1 s before the step plus the 2 s that are always skipped.
    assert duration == pytest.approx(3.0, abs=0.2)
    assert waveform[0] == pytest.approx(0, abs=5)       # relative to the pre-step level
    assert waveform[-1] == pytest.approx(800, abs=5)


def test_slow_ramp_gives_a_longer_transient():
    ts, power = synthetic_signal(step=400.0)
    power[STEP_AT:STEP_AT + 48] += np.linspace(0, 600, 48)    # 8 s ramp after the jump
    power[STEP_AT + 48:] += 600
    fs = sampling_rate(ts)
    step = find_real_step(ts, power, 1150.0, "up", fs)

    waveform, duration = extract_transient(power, step, fs)

    assert duration > 8.0
    assert waveform[-1] > 900   # the cut covers (nearly) the whole ramp up to +1000 W


def test_busy_baseline_is_flagged_and_uses_lower_envelope():
    ts, power = synthetic_signal()
    # Another appliance cycling +-50 W in the seconds before the switch-on.
    power[840:STEP_AT] += np.where((np.arange(60) // 3) % 2 == 0, 50.0, -50.0)

    step = find_real_step(ts, power, 1150.0, "up", sampling_rate(ts))

    assert not step["baseline_stable"]
    assert step["baseline_std"] > 30
    assert step["pre_level"] < 160    # 10th percentile, not the median (~200)


def test_level_method_prefers_a_staged_switch_on_over_a_bigger_single_jump():
    # Modelled on an H3 oven: the switch-on rises in four 1000 W stages; 10 s later
    # the heating element drops out for 4 s and comes back in one 1500 W jump.
    ts, power = synthetic_signal(step=0.0)
    for k in range(4):
        power[STEP_AT + k:] += 1000
    power[STEP_AT + 60:STEP_AT + 84] -= 1500
    fs = sampling_rate(ts)

    jump = find_real_step(ts, power, 1140.0, "up", fs, method="jump")
    level = find_real_step(ts, power, 1140.0, "up", fs, method="level")

    assert jump["tag_offset_sec"] == pytest.approx(24, abs=0.5)    # the element coming back
    assert level["tag_offset_sec"] == pytest.approx(10, abs=1.0)   # the real switch-on


def test_level_method_ignores_a_brief_dip():
    # A brief dip, as in the H4 oven turn-offs, is the biggest single drop near
    # the tag, but the level is the same before and after it.
    ts, power = synthetic_signal(baseline=2200.0, step=-2000.0)
    power[780:782] -= 2500
    fs = sampling_rate(ts)

    jump = find_real_step(ts, power, 1140.0, "down", fs, method="jump")
    level = find_real_step(ts, power, 1140.0, "down", fs, method="level")

    assert jump["tag_offset_sec"] == pytest.approx(-10, abs=0.5)
    assert level["tag_offset_sec"] == pytest.approx(10, abs=0.5)
    assert level["post_level"] == pytest.approx(200, abs=5)


def test_both_methods_agree_on_a_clean_step():
    ts, power = synthetic_signal()
    fs = sampling_rate(ts)
    jump = find_real_step(ts, power, 1140.0, "up", fs, method="jump")
    level = find_real_step(ts, power, 1140.0, "up", fs, method="level")
    assert jump["step_idx"] == level["step_idx"] == STEP_AT - 1


def test_rise_time():
    assert compute_rise_time(np.array([0, 0, 0, 800, 800, 800.0])) == 0
    assert compute_rise_time(np.linspace(0, 100, 11)) == 8
    assert compute_rise_time(np.array([5.0, 5.2, 5.1])) == 0   # no real change
