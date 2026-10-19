import numpy as np
import pandas as pd
import pytest

from nilm.features import (
    STAT_FEATURES,
    build_class_prototypes,
    build_feature_matrix,
    dtw_distance,
    resample_waveform,
    select_medoid,
    statistical_features,
)


def transient(waveform, appliance="Toaster", pre_level=200.0, step_watts=800.0):
    waveform = np.asarray(waveform, dtype=float)
    return {
        "waveform": waveform,
        "appliance": appliance,
        "pre_level": pre_level,
        "step_watts": step_watts,
        "transient_duration": len(waveform) / 6,
        "rise_time": 1,
    }


def test_dtw_identity_and_symmetry():
    a = np.array([0, 0, 800, 790, 805.0])
    b = np.array([0, 400, 800, 800.0])

    assert dtw_distance(a, a) == 0
    assert dtw_distance(a, b) == dtw_distance(b, a)
    assert dtw_distance(a, b) > 0


def test_dtw_handles_different_lengths():
    # Same shape at a different speed: DTW aligns them at no cost.
    assert dtw_distance([0, 0, 0, 5, 5], [0, 5]) == 0
    # The extra 3 at the end has to be matched against the last 2.
    assert dtw_distance([0, 1, 2], [0, 1, 2, 3]) == 1


def test_resample_keeps_endpoints():
    resampled = resample_waveform(np.array([0.0, 10.0, 20.0]), n=40)
    assert len(resampled) == 40
    assert resampled[0] == 0 and resampled[-1] == 20


def test_medoid_is_the_most_central_waveform():
    waveforms = [np.full(5, level) for level in (0.0, 5.0, 1.5, 1.0, 2.0)]
    assert select_medoid(waveforms) == 2
    assert select_medoid([np.ones(3)]) == 0


def test_prototype_is_a_real_training_example():
    rows = pd.DataFrame([
        transient([0, 800, 800]),
        transient([0, 790, 800]),
        transient([0, 300, 800]),
        transient([0, 100, 100], appliance="Computer"),
    ])

    prototypes = build_class_prototypes(rows, ["Toaster", "Computer", "Oven"])

    assert list(prototypes) == ["Toaster", "Computer"]   # no oven examples, no prototype
    assert any(prototypes["Toaster"] is w for w in rows["waveform"])


def test_statistical_features_are_named_and_finite():
    features = statistical_features(transient(np.r_[np.zeros(6), np.full(12, 800.0)]))

    assert list(features) == STAT_FEATURES
    assert all(np.isfinite(v) for v in features.values())


def test_relative_features_use_pre_step_level():
    step = np.r_[np.zeros(4), np.full(16, 400.0)]

    features = statistical_features(transient(step, pre_level=200.0, step_watts=400.0))
    assert features["step_relative"] == pytest.approx(2.0)
    assert features["peak_relative"] == pytest.approx(2.0)
    assert features["mean_relative"] == pytest.approx(features["mean"] / 200.0)

    # A baseline near 0 W is clamped to 1 W so the ratios stay finite.
    features = statistical_features(transient(step, pre_level=0.2, step_watts=400.0))
    assert features["step_relative"] == pytest.approx(400.0)


def test_feature_matrix_has_one_dtw_column_per_prototype():
    rows = pd.DataFrame([transient([0, 800, 800, 800]), transient([0, 50, 90, 100, 100], appliance="Computer")])
    prototypes = {"Toaster": rows.loc[0, "waveform"], "Computer": rows.loc[1, "waveform"]}

    matrix = build_feature_matrix(rows, prototypes)

    assert list(matrix.columns) == STAT_FEATURES + ["dtw_Toaster", "dtw_Computer"]
    assert matrix.loc[0, "dtw_Toaster"] == 0
    assert matrix.loc[0, "dtw_Computer"] > 0
