from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nilm.schema import (
    EVENTS,
    SEGMENTS,
    TRANSIENT_STATUS,
    SchemaError,
    check_references,
    data_dictionary,
    er_diagram,
    quality_report,
    read_table,
    validate,
    write_table,
)

ROOT = Path(__file__).parents[1]


def segments_table():
    return pd.DataFrame({
        "segment_id": [0, 1],
        "house": ["H1", "H2"],
        "segment": ["Tagged_Training_1", "Tagged_Training_2"],
        "n_tags": [2, 1],
        "n_usable_events": [1, 1],
        "status": ["loaded", "not_needed"],
        "source": ["cache", None],
        "n_phases": [2, None],
        "n_samples": [518942, None],
        "sampling_rate_hz": [6.0, None],
        "n_rows_trimmed": [3, None],
        "n_out_of_order": [37, None],
        "start_time": pd.to_datetime([0, None], unit="s"),
        "end_time": pd.to_datetime([86400, None], unit="s"),
        "error": [None, None],
    })


def events_table():
    return pd.DataFrame({
        "event_id": [0, 1, 2],
        "segment_id": [0, 0, 1],
        "appliance_id": [35.0, np.nan, 14.0],          # nullable int, so float in memory
        "appliance": ["Toaster", "Kettle", "Hair Dryer"],
        "start_ts": [100.0, 200.0, 300.0],
        "end_ts": [160.0, 200.0, 330.0],
        "start_time": pd.to_datetime([100, 200, 300], unit="s"),
        "end_time": pd.to_datetime([160, 200, 330], unit="s"),
        "duration_sec": [60.0, 0.0, 30.0],
        "has_overlap": [False, False, False],
        "status": ["ok", "zero_duration", "ok"],
        "segment_dir": ["/somewhere/local"] * 3,      # in memory only, never written
    })


def test_valid_tables_pass():
    validate(segments_table(), SEGMENTS)
    validate(events_table(), EVENTS)
    check_references({SEGMENTS: segments_table(), EVENTS: events_table()})


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns="status"), "missing columns"),
    (lambda df: df.assign(status=["ok", "typo", "ok"]), "unexpected values"),
    (lambda df: df.assign(event_id=[0, 0, 2]), "duplicate values of the primary key"),
    (lambda df: df.assign(appliance=["Toaster", None, "Hair Dryer"]), "empty values"),
    (lambda df: df.assign(start_ts=["a", "b", "c"]), "should be float"),
    (lambda df: df.assign(event_id=[0.5, 1, 2]), "should be int"),
])
def test_violations_are_reported(change, message):
    with pytest.raises(SchemaError, match=message):
        validate(change(events_table()), EVENTS)


def test_duplicate_natural_key_is_reported():
    segments = segments_table().assign(segment=["Tagged_Training_1"] * 2, house=["H1", "H1"])
    with pytest.raises(SchemaError, match="unique key"):
        validate(segments, SEGMENTS)


def test_foreign_key_to_a_missing_row_is_reported():
    events = events_table().assign(segment_id=[0, 0, 7])
    with pytest.raises(SchemaError, match=r"events.segment_id: 1 values not found in segments.segment_id"):
        check_references({SEGMENTS: segments_table(), EVENTS: events})


def test_write_and_read_round_trip(tmp_path):
    path = write_table(events_table(), EVENTS, tmp_path)

    assert path == tmp_path / "data" / "events.csv"
    assert "segment_dir" not in path.read_text()          # only schema columns are written
    assert "35.0" not in path.read_text()                 # nullable ints stay ints
    back = read_table(tmp_path, EVENTS)
    assert list(back.columns) == EVENTS.column_names
    assert back["appliance_id"].tolist()[0] == 35 and pd.isna(back["appliance_id"][1])
    assert back["start_time"].iloc[0] == pd.Timestamp(100, unit="s")


def test_quality_report_accounts_for_every_row():
    events = events_table().assign(has_overlap=[True, False, False])
    transients = pd.DataFrame({
        "status": ["ok", "ok", "no_step_found", "ok"],
        "baseline_stable": [True, False, None, True],
    })

    report = quality_report(segments_table(), events, transients).set_index(["stage", "check"])["count"]

    assert report["events", "ok"] + report["events", "zero_duration"] == len(events)
    assert report["events", "has_overlap"] == 1
    assert report["segments", "out_of_order_timestamps"] == 37
    assert report["transients", "no_step_found"] == 1
    assert report["transients", "unstable_baseline"] == 1
    assert sum(report["transients", s] for s in TRANSIENT_STATUS) == len(transients)


def test_data_dictionary_doc_is_up_to_date():
    doc = ROOT / "docs" / "data_dictionary.md"
    assert doc.read_text(encoding="utf-8") == data_dictionary(), \
        "regenerate with: python -m nilm.schema docs/data_dictionary.md"


def test_readme_er_diagram_matches_the_schema():
    assert er_diagram() in (ROOT / "README.md").read_text(encoding="utf-8"), \
        "copy the output of nilm.schema.er_diagram() into the README"
