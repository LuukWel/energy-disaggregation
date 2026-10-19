"""Run the whole pipeline: raw Belkin files -> transients -> leave-one-house-out results.

    python -m nilm.pipeline --data-root path/to/Root --output-dir outputs

What ends up in the output directory (every table is described in docs/data_dictionary.md):

    run_info.json   settings, library versions and row counts of this run
    cache/          parsed power per segment (.npz), reused by later runs
    data/           cleaned and validated data: segments, events, transients, waveforms, quality_report
    results/        evaluation: predictions, metrics, fold_metrics, feature_importance
    figures/        only with --plots
"""

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from nilm import config
from nilm.config import DEFAULT_SEEDS, STEP_METHODS, STEP_SEARCH_SECONDS, TARGET_APPLIANCES
from nilm.data import discover_segments, load_aggregate_power, load_power_cached, load_tag_events
from nilm.evaluation import MAIN_MODEL, run_loho_evaluation
from nilm.events import (
    check_durations,
    count_by_house,
    exclude_rare_appliances,
    flag_overlaps,
    select_appliances,
    usable,
)
from nilm.schema import (
    EVENTS,
    FEATURE_IMPORTANCE,
    FOLD_METRICS,
    METRICS,
    PREDICTIONS,
    QUALITY_REPORT,
    SEGMENTS,
    TRANSIENTS,
    WAVEFORMS,
    quality_report,
    write_tables,
)
from nilm.transients import extract_event_transients, sampling_rate, waveform_table

logger = logging.getLogger("nilm")


def run_pipeline(data_root, output_dir, appliances=TARGET_APPLIANCES, seeds=DEFAULT_SEEDS,
                 step_method="jump", cache_dir=None, make_plots=False):
    """Run every stage, write the tables to `output_dir` and return the metrics table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the segments and read every tag.
    segments = discover_segments(data_root)
    logger.info("Found %d houses / %d tagged segments", segments["house"].nunique(), len(segments))
    events = load_tag_events(segments)

    # Clean the tags. Each check sets a status instead of dropping rows.
    events = select_appliances(events, appliances)
    events = check_durations(events)
    events = flag_overlaps(events)
    events = exclude_rare_appliances(events)
    log_event_summary(events)

    # Load power only where it's needed, then cut out the transients.
    signals, segments = load_signals(segments, events, cache_dir)
    transients = extract_event_transients(usable(events), signals, step_method)
    log_extraction_summary(transients)

    # Evaluate on the extracted transients, holding out one house at a time.
    extracted = transients[transients["status"] == "ok"]
    present = [a for a in appliances if a in set(extracted["appliance"])]
    metrics, fold_metrics, predictions, importance = run_loho_evaluation(extracted, present, seeds)
    log_results(metrics, fold_metrics)

    # Tables and the foreign keys between them are validated against nilm/schema.py before writing.
    tables = {
        SEGMENTS: segments,
        EVENTS: events.assign(start_time=pd.to_datetime(events["start_ts"], unit="s"),
                              end_time=pd.to_datetime(events["end_ts"], unit="s")),
        TRANSIENTS: transients,
        WAVEFORMS: waveform_table(transients),
        QUALITY_REPORT: quality_report(segments, events, transients),
        PREDICTIONS: predictions,
        METRICS: metrics,
        FOLD_METRICS: fold_metrics,
        FEATURE_IMPORTANCE: importance,
    }
    settings = {"appliances": list(appliances), "seeds": list(seeds), "step_method": step_method}
    save_outputs(output_dir, tables, settings)

    if make_plots:
        from nilm import plots  # matplotlib is only needed here

        figures = plots.save_figures(output_dir / "figures", signals, extracted, predictions, metrics,
                                     importance, present)
        logger.info("Saved %d figures to %s", len(figures), output_dir / "figures")
    return metrics


def _describe_error(exc):
    """Short error text for the segments table, without local paths."""
    filename = getattr(exc, "filename", None)
    return f"{type(exc).__name__}: {Path(filename).name}" if filename else str(exc)


def load_signals(segments, events, cache_dir=None):
    """Load aggregate power for every segment with usable events.

    Returns the signals ((house, segment) -> PowerSignal) and the segments table
    with one row per discovered segment, loaded or not.
    """
    n_tags = events.groupby(["house", "segment"]).size()
    n_usable = usable(events).groupby(["house", "segment"]).size()
    signals, rows = {}, []

    for seg in segments.itertuples(index=False):
        key = (seg.house, seg.segment)
        row = {"segment_id": seg.segment_id, "house": seg.house, "segment": seg.segment,
               "n_tags": int(n_tags.get(key, 0)), "n_usable_events": int(n_usable.get(key, 0)),
               "status": "not_needed"}
        rows.append(row)
        if row["n_usable_events"] == 0:
            continue

        try:
            if cache_dir is None:
                signal, source = load_aggregate_power(seg.segment_dir), "parsed"
            else:
                signal, source = load_power_cached(seg.segment_dir, cache_dir)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping segment %s/%s, phase-1 data unusable: %s", seg.house, seg.segment, exc)
            row.update(status="load_failed", error=_describe_error(exc))
            continue

        signals[key] = signal
        fs = sampling_rate(signal.ts)
        row.update(
            status="loaded", source=source, n_phases=signal.n_phases, n_samples=len(signal.ts),
            sampling_rate_hz=round(fs, 4), n_rows_trimmed=signal.n_rows_trimmed,
            n_out_of_order=signal.n_out_of_order,
            start_time=pd.to_datetime(signal.ts[0], unit="s"), end_time=pd.to_datetime(signal.ts[-1], unit="s"),
        )
        logger.info("Loaded %s/%s (%s): %s samples at %.2f Hz, %d phase(s), %d out-of-order samples dropped",
                    seg.house, seg.segment, source, f"{len(signal.ts):,}", fs, signal.n_phases,
                    signal.n_out_of_order)

    segments = pd.DataFrame(rows)
    # Columns that only exist for loaded segments still have to be there for the schema.
    for column in SEGMENTS.column_names:
        if column not in segments:
            segments[column] = None
    logger.info("Loaded %d of %d segments", len(signals), len(segments))
    return signals, segments


def log_event_summary(events):
    counts = events["status"].value_counts()
    used = usable(events)
    logger.info("Read %d tags: %d used, %s", len(events), len(used),
                ", ".join(f"{n} {status}" for status, n in counts.items() if status != "ok"))
    logger.info("Used events per appliance and house:\n%s", count_by_house(used).to_string())
    logger.info("Used events overlapping another tag: %d", used["has_overlap"].sum())


def log_extraction_summary(transients):
    attempts = len(transients)
    status = transients["status"].value_counts()
    found = attempts - status.get("no_step_found", 0) - status.get("no_signal", 0)
    logger.info("Steps found for %d of %d tag times (+/-%d s search)", found, attempts, STEP_SEARCH_SECONDS)
    for row in transients[transients["status"] == "no_step_found"].itertuples():
        logger.warning("No turn-%s step found for %s event %d in %s", row.transition, row.appliance,
                       row.event_id, row.house)

    ok = transients[transients["status"] == "ok"]
    rejected = status.drop(["ok", "no_step_found", "no_signal"], errors="ignore")
    logger.info("Extracted %d transients, rejected %d (%s)", len(ok), rejected.sum(),
                ", ".join(f"{k}: {v}" for k, v in rejected.items()) or "none")
    if ok.empty:
        return
    logger.info("Unstable baseline (kept, flagged): %d; overlapping events: %d",
                (~ok["baseline_stable"].astype(bool)).sum(), ok["has_overlap"].sum())

    summary = ok.groupby(["appliance", "transition"]).agg(
        n=("event_id", "size"),
        median_tag_offset_s=("tag_offset_sec", "median"),
        median_duration_s=("transient_duration", "median"),
    ).round(1)
    logger.info("Transients per appliance:\n%s", summary.to_string())


def log_results(metrics, fold_metrics):
    for row in metrics.itertuples():
        spread = (f" +/- {100 * row.accuracy_std:4.1f} ({100 * row.accuracy_min:.0f}-{100 * row.accuracy_max:.0f})"
                  if row.n_seeds > 1 else "")
        logger.info("%-9s %-30s accuracy %5.1f%%%s  macro-F1 %.2f  (%d examples, %d folds, %d seeds)",
                    row.mode, row.model, 100 * row.accuracy_mean, spread, row.f1_macro_mean,
                    row.n_test, row.n_folds, row.n_seeds)
    main_on = fold_metrics[(fold_metrics["mode"] == "turn_on") & (fold_metrics["model"] == MAIN_MODEL)]
    logger.info("Turn-on %s per held-out house: %s", MAIN_MODEL,
                ", ".join(f"{r.house} {100 * r.accuracy_mean:.0f}% (n={r.n_test})" for r in main_on.itertuples()))


def save_outputs(output_dir, tables, settings):
    """Validate and write every table, plus run_info.json describing the run."""
    write_tables(tables, output_dir)

    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arguments": settings,
        "config": {k.lower(): v for k, v in vars(config).items() if k.isupper()},
        "versions": {"python": platform.python_version(),
                     **{lib: version(lib) for lib in ("numpy", "pandas", "scipy", "scikit-learn")}},
        "rows": {table.path: len(df) for table, df in tables.items()},
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=list), encoding="utf-8")
    logger.info("Wrote %d validated tables and run_info.json to %s", len(tables), output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transient-based appliance classification on the Belkin dataset.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="folder that contains the H1_CSV ... H4_CSV house folders")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--appliances", nargs="+", default=TARGET_APPLIANCES, metavar="NAME",
                        help="appliance names as written in TaggingInfo.csv (default: the five used in the report)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), metavar="SEED",
                        help="random seeds to average over (default: 0-9)")
    parser.add_argument("--step-method", choices=STEP_METHODS, default="jump",
                        help="how the real switching step is found (default: jump)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where parsed power is cached (default: OUTPUT_DIR/cache)")
    parser.add_argument("--no-cache", action="store_true", help="always parse the raw LF files")
    parser.add_argument("--plots", action="store_true", help="also write figures to OUTPUT_DIR/figures")
    args = parser.parse_args(argv)

    cache_dir = None if args.no_cache else (args.cache_dir or args.output_dir / "cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    run_pipeline(args.data_root, args.output_dir, args.appliances, args.seeds, args.step_method,
                 cache_dir, args.plots)


if __name__ == "__main__":
    main()
