"""Leave-one-house-out evaluation.

Every fold holds out one house completely. Anything learned from data (the DTW
prototypes, the feature scaling and the classifier) is fitted on the other
houses only, so the score says how well the method carries over to a home it
has never seen.

The random forest depends on its seed and only sees ~25 training examples per
fold, so every model is run with several seeds and the results are averaged.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from nilm.config import DEFAULT_SEEDS, KNN_NEIGHBOURS, RF_MAX_DEPTH, RF_TREES
from nilm.features import STAT_FEATURES, build_class_prototypes, build_feature_matrix

logger = logging.getLogger(__name__)

# Evaluation mode -> which transitions it uses.
MODES = {"turn_on": ["on"], "turn_off": ["off"], "combined": ["on", "off"]}
MAIN_MODEL = "Random Forest"


class Model(NamedTuple):
    name: str
    estimator: object
    features: tuple | None = None   # feature columns the model sees; None means all of them


def make_models(seed):
    """Models compared in every fold. The first three are baselines for the main random forest."""
    forest = RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, random_state=seed)
    return [
        # Always predicts the most common appliance in the training houses.
        Model("Majority class", DummyClassifier(strategy="most_frequent")),
        # Sees only the size of the step in watts.
        Model("Random Forest, step size only", forest, ("step_change",)),
        # Sees the summary statistics but not the DTW distances.
        Model("Random Forest, no DTW", forest, tuple(STAT_FEATURES)),
        *[Model(f"k-NN (k={k})", KNeighborsClassifier(n_neighbors=k)) for k in KNN_NEIGHBOURS],
        Model(MAIN_MODEL, forest),
    ]


def select_mode(transients, mode):
    """Examples for one evaluation mode, sorted by appliance within each transition.

    The random forest's bootstrap samples depend on row order, so the order is
    part of what makes a result reproducible.
    """
    parts = [
        transients[transients["transition"] == transition].sort_values("appliance")
        for transition in MODES[mode]
    ]
    return pd.concat(parts, ignore_index=True)


def loho_splits(examples):
    """Yield (held-out house, training rows, test rows), one fold per house."""
    for house in sorted(examples["house"].unique()):
        train = examples[examples["house"] != house]
        test = examples[examples["house"] == house]
        if len(train) < 3:
            logger.warning("Fold %s skipped: only %d training examples", house, len(train))
            continue

        # An appliance that never occurs in the training houses can't be predicted.
        unseen = ~test["appliance"].isin(set(train["appliance"]))
        if unseen.any():
            logger.warning("Fold %s: %d test example(s) dropped, appliance not in training houses: %s",
                           house, unseen.sum(), ", ".join(sorted(test.loc[unseen, "appliance"].unique())))
            test = test[~unseen]
        if test.empty:
            continue
        yield house, train, test


def fold_features(train, test, appliances):
    """Scaled feature matrices for one fold, using only the training houses."""
    # Prototypes come from the training houses only. Building them on all houses
    # would let the held-out house shape its own DTW features.
    prototypes = build_class_prototypes(train, appliances)
    train_features = build_feature_matrix(train, prototypes)
    test_features = build_feature_matrix(test, prototypes)

    x_train = np.nan_to_num(train_features.to_numpy(dtype=float))
    x_test = np.nan_to_num(test_features.to_numpy(dtype=float))
    # The scaler is likewise fitted on the training houses only.
    scaler = StandardScaler().fit(x_train)
    return {
        "X_train": scaler.transform(x_train),
        "X_test": scaler.transform(x_test),
        "feature_names": list(train_features.columns),
        "prototypes": prototypes,
        "scaler": scaler,
    }


def held_out_importance(estimator, x_test, y_test, seed, n_repeats=10):
    """Permutation importance: drop in held-out accuracy when one feature is shuffled.

    Works like sklearn's permutation_importance, but all shuffled copies are
    predicted in a single call, which is much faster for test sets of a few rows.
    Returns one value per feature, averaged over `n_repeats` shuffles.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_features = x_test.shape
    copies = np.repeat(x_test[None], n_features * n_repeats, axis=0)
    for j in range(n_features):
        for r in range(n_repeats):
            copies[j * n_repeats + r, :, j] = rng.permutation(x_test[:, j])
    predicted = estimator.predict(copies.reshape(-1, n_features)).reshape(n_features, n_repeats, n_rows)
    baseline = np.mean(estimator.predict(x_test) == y_test)
    return baseline - (predicted == y_test).mean(axis=(1, 2))


def fit_and_predict(model, fold, y_train):
    """Fit a fresh copy of the model on the fold's training rows and predict its test rows."""
    columns =[fold["feature_names"].index(f) for f in model.features] if model.features else slice(None)
    estimator = clone(model.estimator)
    if isinstance(estimator, KNeighborsClassifier):
        # Small folds can have fewer training examples than k.
        estimator.set_params(n_neighbors=max(1, min(estimator.n_neighbors, len(y_train) - 1)))
    estimator.fit(fold["X_train"][:, columns], y_train)
    return estimator, estimator.predict(fold["X_test"][:, columns])


def run_loho_evaluation(transients, appliances, seeds=DEFAULT_SEEDS):
    """Leave-one-house-out evaluation for every mode, model and seed.

    Returns (metrics, fold_metrics, predictions, importance), matching the
    tables of the same name in nilm/schema.py.
    """
    predictions, importance = [], []

    for mode in MODES:
        examples = select_mode(transients, mode)
        if examples["house"].nunique() < 2:
            logger.warning("Mode %s skipped: needs examples from at least two houses", mode)
            continue

        for house, train, test in loho_splits(examples):
            # Features don't depend on the seed, so they're built once per fold.
            fold = fold_features(train, test, appliances)
            y_train = train["appliance"].to_numpy()
            y_test = test["appliance"].to_numpy()

            for seed in seeds:
                for model in make_models(seed):
                    estimator, predicted = fit_and_predict(model, fold, y_train)
                    predictions.append(pd.DataFrame({
                        "mode": mode,
                        "model": model.name,
                        "seed": seed,
                        "transient_id": test["transient_id"].to_numpy(),
                        "house": house,
                        "appliance_true": y_test,
                        "appliance_predicted": predicted,
                    }))

                    # Importance is measured on the held-out house; on the training
                    # data the forest is near-perfect and every feature looks useful.
                    if model.name == MAIN_MODEL and len(test) >= 2:
                        importance.append(pd.DataFrame({
                            "mode": mode,
                            "feature": fold["feature_names"],
                            "importance": held_out_importance(estimator, fold["X_test"], y_test, seed),
                        }))

    if not predictions:
        raise ValueError("No fold could be evaluated; need labelled transients from at least two houses")

    predictions = pd.concat(predictions, ignore_index=True)
    predictions["correct"] = predictions["appliance_true"] == predictions["appliance_predicted"]
    metrics, fold_metrics = summarise_predictions(predictions)
    return metrics, fold_metrics, predictions, summarise_importance(importance)


def _scores(predictions, keys):
    """Accuracy and macro F1 for each group of predictions."""
    rows = []
    for key, group in predictions.groupby(keys, sort=False):
        rows.append({
            **dict(zip(keys, key)),
            "accuracy": accuracy_score(group["appliance_true"], group["appliance_predicted"]),
            "f1_macro": f1_score(group["appliance_true"], group["appliance_predicted"],
                                 average="macro", zero_division=0),
            "n_test": len(group),
            "n_folds": group["house"].nunique(),
        })
    return pd.DataFrame(rows)


def summarise_predictions(predictions):
    """The metrics and fold_metrics tables: scores per seed, then mean/spread over seeds."""
    per_seed = _scores(predictions, ["mode", "model", "seed"])
    metrics = per_seed.groupby(["mode", "model"], sort=False).agg(
        n_seeds=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        n_test=("n_test", "first"),
        n_folds=("n_folds", "first"),
    ).reset_index()

    per_house = _scores(predictions, ["mode", "model", "house", "seed"])
    fold_metrics = per_house.groupby(["mode", "model", "house"], sort=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        n_test=("n_test", "first"),
    ).reset_index()
    return metrics.round(4), fold_metrics.round(4)


def summarise_importance(importance):
    """Average the permutation importances over folds and seeds."""
    columns = ["mode", "feature", "importance_mean", "importance_std", "n_runs"]
    if not importance:
        return pd.DataFrame(columns=columns)
    table = pd.concat(importance, ignore_index=True)
    summary = (table.groupby(["mode", "feature"], sort=False)["importance"]
               .agg(importance_mean="mean", importance_std="std", n_runs="count")
               .reset_index())
    summary = summary.sort_values(["mode", "importance_mean"], ascending=[True, False], ignore_index=True)
    return summary[columns].round(4)
