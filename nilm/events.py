"""Cleaning and flagging the tagged switching events.

Nothing is deleted here. Every tag stays in the table and gets a `status`:
"ok", or the name of the first check it failed. That way data/events.csv
accounts for every row of every TaggingInfo.csv. The checks run in the order
of the functions below; the meaning of each status is in nilm/schema.py.
"""

import logging

import numpy as np
import pandas as pd

from nilm.config import MIN_EVENTS_PER_APPLIANCE

logger = logging.getLogger(__name__)


def _with_status(events):
    events = events.copy()
    if "status" not in events.columns:
        events["status"] = "ok"
    return events


def _exclude(events, mask, reason):
    """Set the status of rows that are still ok and match the mask."""
    events.loc[mask & (events["status"] == "ok"), "status"] = reason


def usable(events):
    """Rows that passed every check."""
    return events[events["status"] == "ok"]


def select_appliances(events, appliances):
    """Mark tags of other appliances as not_target (names must match exactly)."""
    events = _with_status(events)
    missing = sorted(set(appliances) - set(events["appliance"]))
    if missing:
        logger.warning("No tags found for: %s", ", ".join(missing))
    _exclude(events, ~events["appliance"].isin(appliances), "not_target")
    return events


def check_durations(events):
    """Add durations and exclude tags that cannot describe an on/off cycle."""
    events = _with_status(events)
    events["duration_sec"] = events["end_ts"] - events["start_ts"]
    # Several tags have end == start. There is no off-time to search around, so they can't be used.
    _exclude(events, events["duration_sec"] == 0, "zero_duration")
    _exclude(events, events["duration_sec"] < 0, "negative_duration")
    return events


def flag_overlaps(events):
    """Flag tags whose interval overlaps another tag in the same segment.

    Every appliance with a positive duration takes part, not just the selected
    ones: a kettle running during a toaster tag still ends up in the toaster's
    transient. Overlapping events are kept, only flagged.
    """
    events = events.copy()
    events["has_overlap"] = False
    positive = events[events["end_ts"] > events["start_ts"]]

    for _, group in positive.groupby(["house", "segment"]):
        starts = group["start_ts"].to_numpy()
        ends = group["end_ts"].to_numpy()
        index = group.index.to_numpy()
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if starts[i] < ends[j] and starts[j] < ends[i]:
                    events.loc[[index[i], index[j]], "has_overlap"] = True
    return events


def exclude_rare_appliances(events, min_events=MIN_EVENTS_PER_APPLIANCE):
    """Exclude appliances with fewer than `min_events` usable tags."""
    events = _with_status(events)
    counts = usable(events)["appliance"].value_counts()
    rare = counts[counts < min_events].index
    for appliance in rare:
        logger.warning("Excluding %s: only %d usable event(s)", appliance, counts[appliance])
    _exclude(events, events["appliance"].isin(rare), "too_few_events")
    return events


def count_by_house(events):
    """Appliance x house table of event counts."""
    return pd.crosstab(events["appliance"], events["house"]).astype(np.int64)
