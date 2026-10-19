"""Definitions of the tables the pipeline writes, and the checks that enforce them.

Each table has a grain (what one row is), a primary key and typed columns, plus
allowed values, unique keys and foreign keys where they apply. `write_tables`
checks every table and the references between them before anything is written.
docs/data_dictionary.md is generated from this module:

    python -m nilm.schema docs/data_dictionary.md
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nilm.config import (
    LEVEL_CHANGE_TOLERANCE_WATTS,
    MIN_EVENTS_PER_APPLIANCE,
    MIN_TRANSIENT_SAMPLES,
    NOISY_BASELINE_WATTS,
    STEP_SEARCH_SECONDS,
)


class SchemaError(ValueError):
    """A table does not match its definition."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str              # int, float, str, bool or datetime
    description: str
    nullable: bool = False
    allowed: tuple = ()
    references: str = ""    # foreign key, as "table.column"


@dataclass(frozen=True)
class Table:
    name: str
    path: str               # relative to the output directory
    grain: str              # what one row is
    key: tuple              # primary key
    columns: tuple
    unique: tuple = ()      # natural key, when the primary key is a surrogate id

    @property
    def column_names(self):
        return [c.name for c in self.columns]


# Status values. A row that fails a check stays in its table with the check as
# its status, so every input row can be accounted for.
SEGMENT_STATUS = {
    "loaded": "power loaded, either parsed from the LF files or read from the cache",
    "not_needed": "no usable event in this segment, so its power is not loaded",
    "load_failed": "phase-1 files missing or unreadable",
}
EVENT_STATUS = {
    "ok": "used",
    "not_target": "appliance is not one of the selected appliances",
    "zero_duration": "tag end equals tag start, so there is no off-time to search around",
    "negative_duration": "tag end is before tag start",
    "too_few_events": f"fewer than {MIN_EVENTS_PER_APPLIANCE} usable events for this appliance",
}
TRANSIENT_STATUS = {
    "ok": "used",
    "no_signal": "the segment's power could not be loaded",
    "no_step_found": f"no step above the threshold within +/-{STEP_SEARCH_SECONDS} s of the tag",
    "too_short": f"fewer than {MIN_TRANSIENT_SAMPLES} samples could be cut",
    "implausible_power": f"level moved more than {LEVEL_CHANGE_TOLERANCE_WATTS} W against the expected direction",
}
MODES = ("turn_on", "turn_off", "combined")
TRANSITIONS = ("on", "off")

# Every data check, in pipeline order: (stage, check, action, unit, description).
# Actions: use = row passes, exclude = row kept in its table but not used,
# flag = used but marked, fix = corrected while loading.
QUALITY_CHECKS = [
    ("segments", "loaded", "use", "segments", SEGMENT_STATUS["loaded"]),
    ("segments", "not_needed", "exclude", "segments", SEGMENT_STATUS["not_needed"]),
    ("segments", "load_failed", "exclude", "segments", SEGMENT_STATUS["load_failed"]),
    ("segments", "phase_2_missing", "flag", "segments", "only phase 1 available, so part of the load is missing"),
    ("segments", "rows_trimmed", "fix", "samples", "rows beyond the shortest LF/TimeTicks file, cut so the files line up"),
    ("segments", "out_of_order_timestamps", "fix", "samples", "samples dropped because the clock went backwards"),
    *[("events", s, "use" if s == "ok" else "exclude", "tags", d) for s, d in EVENT_STATUS.items()],
    ("events", "has_overlap", "flag", "tags", "used tag overlaps another tag in the same segment"),
    *[("transients", s, "use" if s == "ok" else "exclude", "tag times", d) for s, d in TRANSIENT_STATUS.items()],
    ("transients", "unstable_baseline", "flag", "tag times",
     f"std of the 6 s before the step is above {NOISY_BASELINE_WATTS} W (another load is busy)"),
]

_mode = Column("mode", "str", "Evaluation mode: turn-on transients, turn-off transients, or both.", allowed=MODES)
_model = Column("model", "str", "Model name; see the README for the list.")

SEGMENTS = Table(
    name="segments", path="data/segments.csv", grain="tagged day folder",
    key=("segment_id",), unique=("house", "segment"),
    columns=(
        Column("segment_id", "int", "Segment id, numbered in the order the folders are found."),
        Column("house", "str", "House label, H1 to H4."),
        Column("segment", "str", "Name of the Tagged_Training_* folder (one day of data)."),
        Column("n_tags", "int", "Rows in this segment's TaggingInfo.csv."),
        Column("n_usable_events", "int", "Tags with status ok in this segment."),
        Column("status", "str", "Whether the power was loaded.", allowed=tuple(SEGMENT_STATUS)),
        Column("source", "str", "Where the power came from.", nullable=True, allowed=("parsed", "cache")),
        Column("n_phases", "int", "Phases summed into the aggregate power (1 or 2).", nullable=True),
        Column("n_samples", "int", "Samples after trimming and dropping out-of-order timestamps.", nullable=True),
        Column("sampling_rate_hz", "float", "Samples per second over the day.", nullable=True),
        Column("n_rows_trimmed", "int", "Rows of the longest file beyond the common length.", nullable=True),
        Column("n_out_of_order", "int", "Samples dropped because their timestamp went backwards.", nullable=True),
        Column("start_time", "datetime", "First sample (UTC).", nullable=True),
        Column("end_time", "datetime", "Last sample (UTC).", nullable=True),
        Column("error", "str", "Why loading failed.", nullable=True),
    ),
)

EVENTS = Table(
    name="events", path="data/events.csv", grain="tag (one appliance use)", key=("event_id",),
    columns=(
        Column("event_id", "int", "Event id, numbered in the order the tags are read."),
        Column("segment_id", "int", "Segment the tag belongs to.", references="segments.segment_id"),
        Column("appliance_id", "int", "Belkin's appliance id within the house.", nullable=True),
        Column("appliance", "str", "Appliance name as written in TaggingInfo.csv."),
        Column("start_ts", "float", "Tagged start, Unix seconds."),
        Column("end_ts", "float", "Tagged end, Unix seconds."),
        Column("start_time", "datetime", "Tagged start (UTC)."),
        Column("end_time", "datetime", "Tagged end (UTC)."),
        Column("duration_sec", "float", "end_ts - start_ts."),
        Column("has_overlap", "bool", "Interval overlaps another tag (any appliance) in the same segment."),
        Column("status", "str", "ok, or the first check the tag failed.", allowed=tuple(EVENT_STATUS)),
    ),
)

TRANSIENTS = Table(
    name="transients", path="data/transients.csv", grain="tag time (turn-on or turn-off of a used event)",
    key=("transient_id",), unique=("event_id", "transition"),
    columns=(
        Column("transient_id", "int", "Transient id, numbered in extraction order."),
        Column("event_id", "int", "Event the tag time belongs to.", references="events.event_id"),
        Column("transition", "str", "Which edge of the event.", allowed=TRANSITIONS),
        Column("tag_ts", "float", "Tagged time searched around (start_ts for on, end_ts for off), Unix seconds."),
        Column("status", "str", "ok, or why no transient was extracted.", allowed=tuple(TRANSIENT_STATUS)),
        Column("step_ts", "float", "Detected switching time: last sample before the jump, Unix seconds.", nullable=True),
        Column("tag_offset_sec", "float", "step_ts - tag_ts. Positive means the switch came after the tag.", nullable=True),
        Column("step_watts", "float", "Size of the jump in W (negative for turn-off).", nullable=True),
        Column("pre_level", "float", "Power level before the step (the baseline), W.", nullable=True),
        Column("post_level", "float", "Median power over the 6 s after the step, W.", nullable=True),
        Column("baseline_std", "float", "Std of the 6 s before the step, W.", nullable=True),
        Column("baseline_stable", "bool", f"baseline_std <= {NOISY_BASELINE_WATTS} W.", nullable=True),
        Column("n_samples", "int", "Length of the extracted waveform (rows in waveforms.csv).", nullable=True),
        Column("transient_duration", "float", "Length of the extracted waveform, seconds.", nullable=True),
        Column("rise_time", "int", "Samples between 10% and 90% of the change.", nullable=True),
    ),
)

WAVEFORMS = Table(
    name="waveforms", path="data/waveforms.csv", grain="sample of an extracted transient",
    key=("transient_id", "sample"),
    columns=(
        Column("transient_id", "int", "Transient the sample belongs to.", references="transients.transient_id"),
        Column("sample", "int", "Position in the waveform. The cut starts about 1 s before the step; "
                                "seconds = sample * transient_duration / n_samples."),
        Column("watts", "float", "Smoothed power minus pre_level, W."),
    ),
)

PREDICTIONS = Table(
    name="predictions", path="results/predictions.csv",
    grain="held-out prediction (one transient, one model, one seed)",
    key=("mode", "model", "seed", "transient_id"),
    columns=(
        _mode, _model,
        Column("seed", "int", "Random seed of the run (models without randomness give the same result for every seed)."),
        Column("transient_id", "int", "Transient that was classified.", references="transients.transient_id"),
        Column("house", "str", "House held out in this fold, which is the transient's own house."),
        Column("appliance_true", "str", "Tagged appliance the prediction is scored against."),
        Column("appliance_predicted", "str", "Predicted appliance."),
        Column("correct", "bool", "appliance_true == appliance_predicted."),
    ),
)

METRICS = Table(
    name="metrics", path="results/metrics.csv", grain="model in one mode, summarised over seeds",
    key=("mode", "model"),
    columns=(
        _mode, _model,
        Column("n_seeds", "int", "Seeds the model was run with."),
        Column("accuracy_mean", "float", "Leave-one-house-out accuracy, mean over seeds."),
        Column("accuracy_std", "float", "Std over seeds (empty for a single seed).", nullable=True),
        Column("accuracy_min", "float", "Lowest accuracy over seeds."),
        Column("accuracy_max", "float", "Highest accuracy over seeds."),
        Column("f1_macro_mean", "float", "Macro F1, mean over seeds."),
        Column("f1_macro_std", "float", "Std over seeds (empty for a single seed).", nullable=True),
        Column("n_test", "int", "Held-out transients per seed."),
        Column("n_folds", "int", "Houses held out."),
    ),
)

FOLD_METRICS = Table(
    name="fold_metrics", path="results/fold_metrics.csv", grain="model in one mode on one held-out house",
    key=("mode", "model", "house"),
    columns=(
        _mode, _model, Column("house", "str", "Held-out house."),
        Column("accuracy_mean", "float", "Accuracy on this house, mean over seeds."),
        Column("accuracy_std", "float", "Std over seeds (empty for a single seed).", nullable=True),
        Column("f1_macro_mean", "float", "Macro F1 on this house, mean over seeds."),
        Column("n_test", "int", "Held-out transients from this house."),
    ),
)

FEATURE_IMPORTANCE = Table(
    name="feature_importance", path="results/feature_importance.csv",
    grain="feature of the full random forest in one mode", key=("mode", "feature"),
    columns=(
        _mode,
        Column("feature", "str", "Feature name; see the README."),
        Column("importance_mean", "float", "Drop in held-out accuracy when the feature is shuffled, "
                                           "mean over folds and seeds."),
        Column("importance_std", "float", "Std over folds and seeds.", nullable=True),
        Column("n_runs", "int", "Folds x seeds averaged."),
    ),
)

QUALITY_REPORT = Table(
    name="quality_report", path="data/quality_report.csv", grain="data check", key=("stage", "check"),
    columns=(
        Column("stage", "str", "Pipeline stage the check belongs to.", allowed=("segments", "events", "transients")),
        Column("check", "str", "Check name; for exclusions this is the status value."),
        Column("action", "str", "What happens to affected rows.", allowed=("use", "exclude", "flag", "fix")),
        Column("count", "int", "Rows (or samples) affected in this run."),
        Column("unit", "str", "What count counts."),
        Column("description", "str", "What the check means."),
    ),
)

TABLES = (SEGMENTS, EVENTS, TRANSIENTS, WAVEFORMS, QUALITY_REPORT,
          PREDICTIONS, METRICS, FOLD_METRICS, FEATURE_IMPORTANCE)


def _type_ok(values, dtype):
    """Check the non-null values of a column against a schema type."""
    if values.empty:
        return True
    if dtype == "datetime":
        return pd.api.types.is_datetime64_any_dtype(values)
    if dtype == "str":
        return all(isinstance(v, str) for v in values)
    if dtype == "bool":
        return pd.api.types.is_bool_dtype(values) or all(isinstance(v, (bool, np.bool_)) for v in values)
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
    if dtype == "float":
        return numeric
    # int: a nullable int column is float in memory, so accept whole numbers.
    return numeric and bool((values.astype(float) % 1 == 0).all())


def validate(df, table):
    """Raise SchemaError if df doesn't match the table definition. Extra columns are allowed."""
    missing = [c for c in table.column_names if c not in df.columns]
    if missing:
        raise SchemaError(f"{table.name}: missing columns {missing}")

    problems = []
    for col in table.columns:
        values = df[col.name].dropna()
        n_null = len(df) - len(values)
        if n_null and not col.nullable:
            problems.append(f"{col.name} has {n_null} empty values")
        if not _type_ok(values, col.dtype):
            problems.append(f"{col.name} should be {col.dtype} (found {df[col.name].dtype})")
        if col.allowed:
            unexpected = sorted(set(values) - set(col.allowed), key=str)
            if unexpected:
                problems.append(f"{col.name} has unexpected values {unexpected}")

    for columns, kind in ((table.key, "primary key"), (table.unique, "unique key")):
        if columns:
            duplicates = int(df.duplicated(list(columns)).sum())
            if duplicates:
                problems.append(f"{duplicates} duplicate values of the {kind} {columns}")
    if problems:
        raise SchemaError(f"{table.name}: " + "; ".join(problems))


def check_references(tables):
    """Raise SchemaError if a foreign key points at a row that doesn't exist.

    `tables` maps each Table to its DataFrame and must include every referenced table.
    """
    frames = {table.name: df for table, df in tables.items()}
    problems = []
    for table, df in tables.items():
        for col in table.columns:
            if not col.references:
                continue
            parent, parent_column = col.references.split(".")
            values = df[col.name].dropna()
            orphans = int((~values.isin(set(frames[parent][parent_column]))).sum())
            if orphans:
                problems.append(f"{table.name}.{col.name}: {orphans} values not found in {col.references}")
    if problems:
        raise SchemaError("; ".join(problems))


def write_table(df, table, output_dir):
    """Validate, keep only the schema columns (in schema order) and write the CSV."""
    validate(df, table)
    out = df[table.column_names].copy()
    for col in table.columns:
        # Without this a nullable int column is written as 18.0.
        if col.dtype == "int":
            out[col.name] = out[col.name].astype("Int64")
        elif col.dtype == "bool" and col.nullable:
            out[col.name] = out[col.name].astype("boolean")
    path = Path(output_dir) / table.path
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def write_tables(tables, output_dir):
    """Validate every table and the foreign keys between them, then write them all."""
    for table, df in tables.items():
        validate(df, table)
    check_references(tables)
    return [write_table(df, table, output_dir) for table, df in tables.items()]


def read_table(output_dir, table):
    """Read a table written by write_table back with the right types."""
    pandas_types = {"int": "Int64", "float": "float64", "str": "string", "bool": "boolean"}
    df = pd.read_csv(
        Path(output_dir) / table.path,
        dtype={c.name: pandas_types[c.dtype] for c in table.columns if c.dtype != "datetime"},
        parse_dates=[c.name for c in table.columns if c.dtype == "datetime"],
    )
    validate(df, table)
    return df


def quality_report(segments, events, transients):
    """Count how many rows each data check affected in this run."""
    loaded = segments[segments["status"] == "loaded"]
    used = events[events["status"] == "ok"]
    extracted = transients[transients["status"] == "ok"]
    counts = {
        **{("segments", s): (segments["status"] == s).sum() for s in SEGMENT_STATUS},
        ("segments", "phase_2_missing"): (loaded["n_phases"] == 1).sum(),
        ("segments", "rows_trimmed"): loaded["n_rows_trimmed"].sum(),
        ("segments", "out_of_order_timestamps"): loaded["n_out_of_order"].sum(),
        **{("events", s): (events["status"] == s).sum() for s in EVENT_STATUS},
        ("events", "has_overlap"): used["has_overlap"].sum(),
        **{("transients", s): (transients["status"] == s).sum() for s in TRANSIENT_STATUS},
        ("transients", "unstable_baseline"): (~extracted["baseline_stable"].astype(bool)).sum(),
    }
    rows = [
        {"stage": stage, "check": check, "action": action, "count": int(counts[(stage, check)]),
         "unit": unit, "description": description}
        for stage, check, action, unit, description in QUALITY_CHECKS
    ]
    return pd.DataFrame(rows)


def _key_labels(table, col):
    labels = []
    if col.name in table.key:
        labels.append("PK")
    if col.references:
        labels.append("FK")
    if col.name in table.unique:
        labels.append("UK")
    return labels


def er_diagram():
    """Mermaid ER diagram of the tables linked by foreign keys, with their key columns."""
    children = [t for t in TABLES if any(c.references for c in t.columns)]
    parents = {c.references.split(".")[0] for t in children for c in t.columns if c.references}
    linked = [t for t in TABLES if t in children or t.name in parents]

    lines = ["erDiagram"]
    for table in children:
        for col in table.columns:
            if col.references:
                parent = col.references.split(".")[0]
                lines.append(f'    {parent.upper()} ||--o{{ {table.name.upper()} : "{col.name}"')
    for table in linked:
        lines.append(f"    {table.name.upper()} {{")
        for col in table.columns:
            labels = _key_labels(table, col)
            if labels:
                lines.append(f"        {col.dtype} {col.name} {', '.join(labels)}")
        lines.append("    }")
    return "\n".join(lines)


def data_dictionary():
    """Markdown description of every table and data check."""
    lines = [
        "# Data dictionary",
        "",
        "Generated from `nilm/schema.py` with `python -m nilm.schema docs/data_dictionary.md`.",
        "Paths are relative to `--output-dir`. Before anything is written, every table is checked",
        "against this definition (columns, types, empty values, allowed values, primary and unique",
        "keys) and every foreign key is checked against the table it points to.",
        "",
        "PK = primary key, FK = foreign key, UK = unique key.",
        "",
        "```mermaid",
        er_diagram(),
        "```",
        "",
    ]
    for table in TABLES:
        key_text = f"Primary key: {', '.join(f'`{k}`' for k in table.key)}."
        if table.unique:
            key_text += f" Unique: {', '.join(f'`{k}`' for k in table.unique)}."
        lines += [
            f"## {table.name}",
            "",
            f"`{table.path}`: one row per {table.grain}. {key_text}",
            "",
            "| Column | Type | Key | Description |",
            "|---|---|---|---|",
        ]
        for col in table.columns:
            dtype = col.dtype + (", nullable" if col.nullable else "")
            key = ", ".join(_key_labels(table, col))
            if col.references:
                key += f" -> `{col.references}`"
            description = col.description
            if col.allowed:
                description += " One of: " + ", ".join(f"`{v}`" for v in col.allowed) + "."
            lines.append(f"| `{col.name}` | {dtype} | {key} | {description} |")
        lines.append("")

    lines += [
        "## Data checks",
        "",
        "Counted for each run in `data/quality_report.csv`.",
        "",
        "| Stage | Check | Action | Counts | Meaning |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {stage} | `{check}` | {action} | {unit} | {description} |"
              for stage, check, action, unit, description in QUALITY_CHECKS]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = data_dictionary()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)
