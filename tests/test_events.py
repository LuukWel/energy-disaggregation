import pandas as pd

from nilm.events import check_durations, exclude_rare_appliances, flag_overlaps, select_appliances, usable


def make_events(rows):
    """Event table from (house, segment, appliance, start, end) tuples."""
    events = pd.DataFrame(rows, columns=["house", "segment", "appliance", "start_ts", "end_ts"])
    events.insert(0, "event_id", range(len(events)))
    return events


def test_zero_duration_tags_are_excluded_not_deleted():
    events = check_durations(make_events([
        ("H1", "s1", "Toaster", 100, 160),
        ("H2", "s2", "Microwave", 200, 200),
        ("H2", "s2", "Microwave", 300, 290),
    ]))

    assert len(events) == 3
    assert events["status"].tolist() == ["ok", "zero_duration", "negative_duration"]
    assert usable(events)["event_id"].tolist() == [0]


def test_other_appliances_are_marked_not_target():
    events = select_appliances(make_events([
        ("H1", "s1", "Toaster", 0, 30),
        ("H1", "s1", "Toaster Oven", 100, 130),
        ("H3", "s3", "Hair Dryer", 200, 230),
        ("H4", "s4", "Hairdryer", 300, 330),
    ]), ["Toaster", "Hair Dryer"])

    assert events["status"].tolist() == ["ok", "not_target", "ok", "not_target"]


def test_status_is_the_first_check_that_failed():
    events = make_events([
        ("H1", "s1", "Kettle", 50, 50),     # not a target and zero duration: not_target wins
        ("H1", "s1", "Toaster", 70, 70),
    ])

    events = check_durations(select_appliances(events, ["Toaster"]))

    assert events["status"].tolist() == ["not_target", "zero_duration"]


def test_overlap_detection_includes_other_appliances():
    events = flag_overlaps(select_appliances(make_events([
        ("H1", "s1", "Toaster", 0, 100),
        ("H1", "s1", "Kettle", 50, 150),        # not a target, but it does overlap the toaster
        ("H1", "s1", "Lights", 150, 200),       # starts exactly when the kettle ends: no overlap
        ("H1", "s2", "Microwave", 60, 90),      # same times, different segment
        ("H1", "s1", "Microwave", 400, 400),    # zero duration, not part of the check
        ("H1", "s1", "Oven", 390, 500),
    ]), ["Toaster", "Microwave", "Oven"]))

    assert events["has_overlap"].tolist() == [True, True, False, False, False, False]


def test_appliances_with_too_few_events_are_excluded():
    events = check_durations(make_events([
        ("H1", "s1", "Toaster", 0, 30),
        ("H2", "s2", "Toaster", 0, 30),
        ("H3", "s3", "Oven", 0, 300),
        ("H3", "s3", "Oven", 500, 500),   # zero duration, so only one usable oven event
    ]))

    events = exclude_rare_appliances(events, min_events=2)

    assert events["status"].tolist() == ["ok", "ok", "too_few_events", "zero_duration"]
