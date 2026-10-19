"""Shared defaults. The dataset path is passed on the command line."""

TARGET_APPLIANCES = [
    "Toaster",
    "Microwave",
    "Hair Dryer",
    "Oven",
    "Computer",
]

# Every model is run once per seed and the results are averaged.
DEFAULT_SEEDS = tuple(range(10))

# An appliance needs at least this many usable tags to be classified.
MIN_EVENTS_PER_APPLIANCE = 2

# Signal processing. The low-frequency data is sampled at roughly 6 Hz.
SMOOTH_WINDOW = 7            # median filter length in samples (medfilt needs an odd size)
STEP_SEARCH_SECONDS = 45     # tags sit on a 30 s grid, so search a bit wider than that
MIN_STEP_WATTS = 30
LEVEL_WINDOW_SECONDS = 6     # seconds either side of a step used for the pre/post levels
NOISY_BASELINE_WATTS = 30    # std of the pre-step window above which the baseline counts as unstable
MAX_TRANSIENT_SECONDS = 30
MIN_TRANSIENT_SAMPLES = 5
LEVEL_CHANGE_TOLERANCE_WATTS = 20   # how far the level may move against the expected direction

# How the switching step near a tag is located:
#   "jump"  - the largest single-sample jump
#   "level" - the largest change between the median levels of the STEP_LEVEL_SECONDS
#             before and after a sample; brief dips barely count and a switch spread
#             over several samples counts in full
STEP_METHODS = ("jump", "level")
STEP_LEVEL_SECONDS = 2

# Fixed-length copy of each waveform, only used for the summary statistics.
RESAMPLE_LENGTH = 40

KNN_NEIGHBOURS = (3, 5)
RF_TREES = 100
RF_MAX_DEPTH = 10
