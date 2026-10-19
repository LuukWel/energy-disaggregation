"""Turning extracted transients into feature vectors."""

import numpy as np
import pandas as pd
from scipy import stats

from nilm.config import RESAMPLE_LENGTH

STAT_FEATURES = [
    "mean",
    "std",
    "skew",
    "kurtosis",
    "step_change",
    "peak",
    "transient_duration",
    "rise_time",
    "step_relative",
    "mean_relative",
    "peak_relative",
]


def dtw_distance(a, b):
    """Dynamic time warping distance with |a_i - b_j| as the local cost.

    The sequences may have different lengths, so transients are compared at the
    length they were cut, without resampling.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    inf = float("inf")
    previous = [0.0] + [inf] * len(b)
    for ai in a:
        current = [inf] * (len(b) + 1)
        for j, bj in enumerate(b, start=1):
            current[j] = abs(ai - bj) + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[-1]


def resample_waveform(waveform, n=RESAMPLE_LENGTH):
    """Linear interpolation onto n evenly spaced points."""
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(waveform)), waveform)


def statistical_features(transient):
    """Shape and level features for one transient (a row of the transient table)."""
    # Resample to a fixed length so the statistics don't depend on how long the cut
    # was. The DTW features use the waveform as it was cut.
    resampled = resample_waveform(transient["waveform"])
    # Houses have very different background loads (60 W to over 2 kW), so also
    # express the step and the waveform relative to the level before the step.
    baseline = max(abs(transient["pre_level"]), 1.0)

    mean = np.mean(resampled)
    peak = np.max(np.abs(resampled))
    return {
        "mean": mean,
        "std": np.std(resampled),
        "skew": stats.skew(resampled),
        "kurtosis": stats.kurtosis(resampled),
        "step_change": transient["step_watts"],
        "peak": peak,
        "transient_duration": transient["transient_duration"],
        "rise_time": transient["rise_time"],
        "step_relative": abs(transient["step_watts"]) / baseline,
        "mean_relative": mean / baseline,
        "peak_relative": peak / baseline,
    }


def dtw_matrix(waveforms):
    """Symmetric matrix of pairwise DTW distances."""
    n = len(waveforms)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = dtw_distance(waveforms[i], waveforms[j])
    return matrix


def select_medoid(waveforms):
    """Index of the waveform with the smallest total DTW distance to the others."""
    if len(waveforms) == 1:
        return 0
    matrix = dtw_matrix(waveforms)
    costs = [sum(row) for row in matrix]
    return int(np.argmin(costs))


def build_class_prototypes(transients, appliances):
    """One medoid waveform per appliance.

    A medoid is a real example, so its switching edge stays sharp. An average of
    waveforms with different lengths would smear that edge.
    """
    prototypes = {}
    for appliance in appliances:
        waveforms = list(transients.loc[transients["appliance"] == appliance, "waveform"])
        if waveforms:
            prototypes[appliance] = waveforms[select_medoid(waveforms)]
    return prototypes


def build_feature_matrix(transients, prototypes):
    """Statistical features plus the DTW distance to each class prototype."""
    rows = []
    for _, transient in transients.iterrows():
        features = statistical_features(transient)
        for appliance, prototype in prototypes.items():
            features[f"dtw_{appliance}"] = dtw_distance(transient["waveform"], prototype)
        rows.append(features)

    columns = STAT_FEATURES + [f"dtw_{appliance}" for appliance in prototypes]
    return pd.DataFrame(rows, index=transients.index, columns=columns)
