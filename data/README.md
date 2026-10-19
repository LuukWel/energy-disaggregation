# Data

The pipeline uses the Belkin Energy Disaggregation Competition dataset
(Belkin1 et al., 2013), available from the
[competition page on Kaggle](https://www.kaggle.com/competitions/belkin-energy-disaggregation-competition).
It is not included in this repository. The full citation is in the
[main README](../README.md#citation).

Unpack it anywhere and pass the folder that holds the house folders with
`--data-root`. If you unpack it inside this `data/` folder, Git ignores it.

## Layout the code expects

```text
<data-root>/
├── H1_CSV/
│   ├── AllTaggingInfo.csv              (not used, same rows as the per-segment files)
│   └── CSV_OUT/
│       ├── Tagged_Training_04_13_1334300401/
│       │   ├── TaggingInfo.csv         appliance tags for this day
│       │   ├── TimeTicks1.csv          Unix timestamp per low-frequency sample
│       │   ├── LF1V.csv  LF1I.csv      phase 1 voltage / current phasors
│       │   ├── LF2V.csv  LF2I.csv      phase 2 voltage / current phasors
│       │   ├── TimeTicks2.csv          (not used)
│       │   └── HF.csv  TimeTicksHF.csv (high-frequency noise spectrum, not used)
│       ├── Tagged_Training_.../
│       └── Testing_.../                (no tags, not used)
├── H2_CSV/
├── H3_CSV/
└── H4_CSV/
```

House folders are found with `H*` and named by their leading `H<number>`, so
`H1_CSV` and a re-downloaded `H1_CSV(1)` both become `H1`. Only the
`CSV_OUT/Tagged_Training*` segments are read. Each one covers one day
(about 518,900 samples at roughly 6 Hz).

## File formats

- **TaggingInfo.csv** has no header: `appliance_id, "appliance name", start_unix, end_unix`.
  Times are whole seconds, and in practice they sit on a 30 s grid.
- **TimeTicks1.csv** has one Unix timestamp (float seconds) per line.
- **LF\*V.csv / LF\*I.csv** have one row per sample and six complex values per row, written
  like `-112.27+48.20i` (the fundamental plus harmonics). Real power is
  `sum over columns of Re(V * conj(I))`, summed over both phases.

## Quirks the code deals with

| What | How it shows up | What the code does |
|---|---|---|
| Phase files differ in length | LF2 files are a few rows longer or shorter than LF1 / TimeTicks1 | trims all of them to the common length, assuming they start at the same sample |
| Clock jumps | around midnight, 7 of the 15 tagged segments jump back by 0.3–6 s | drops every sample that is not later than all earlier ones (2–37 samples per segment) |
| Zero-duration tags | 3 of the 39 target-appliance tags have `end == start` | kept in `data/events.csv` with status `zero_duration`, not used |
| Inaccurate tags | the detected step is up to about 40 s from the tag (median 7.5 s for turn-on, 20 s for turn-off) | searches ±45 s around the tag for the actual step |
| Missing phase 2 | not seen in this data | warns and continues with phase 1 only |
| Bad LF row | not seen in this data | marks the segment `load_failed` with the file name and line number, instead of shifting all later rows |

Per-segment counts (rows trimmed, out-of-order samples, phases) are written to
`data/segments.csv`, and the totals to `data/quality_report.csv`. See the
[data dictionary](../docs/data_dictionary.md).
