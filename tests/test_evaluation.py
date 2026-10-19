import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from nilm import evaluation
from nilm.evaluation import (
    MAIN_MODEL,
    fit_and_predict,
    fold_features,
    held_out_importance,
    loho_splits,
    make_models,
    run_loho_evaluation,
    select_mode,
)
from nilm.schema import FEATURE_IMPORTANCE, FOLD_METRICS, METRICS, PREDICTIONS, validate

APPLIANCES = ["Toaster", "Computer"]


def synthetic_transients(houses=("H1", "H2", "H3"), per_class=3, seed=0):
    """A sharp 800 W step (toaster) and a slow 100 W ramp (computer) in every house."""
    rng = np.random.default_rng(seed)
    rows = []
    event_id = 0
    for house in houses:
        for appliance, watts, ramp in [("Toaster", 800.0, 1), ("Computer", 100.0, 20)]:
            for _ in range(per_class):
                shape = np.r_[np.zeros(6), np.linspace(0, watts, ramp), np.full(6, watts)]
                for transition, sign in [("on", 1), ("off", -1)]:
                    rows.append({
                        "transient_id": len(rows),
                        "event_id": event_id,
                        "house": house,
                        "appliance": appliance,
                        "transition": transition,
                        "waveform": sign * shape + rng.normal(0, 2, len(shape)),
                        "pre_level": 200 + rng.normal(0, 20),
                        "step_watts": sign * watts,
                        "transient_duration": len(shape) / 6,
                        "rise_time": ramp,
                    })
                event_id += 1
    return pd.DataFrame(rows)


def test_each_house_is_held_out_exactly_once():
    examples = select_mode(synthetic_transients(), "turn_on")

    splits = list(loho_splits(examples))

    assert [house for house, _, _ in splits] == ["H1", "H2", "H3"]
    for house, train, test in splits:
        assert set(test["house"]) == {house}
        assert house not in set(train["house"])
        assert len(train) + len(test) == len(examples)


def test_appliance_missing_from_training_houses_is_not_tested():
    examples = select_mode(synthetic_transients(), "turn_on")
    oven = examples[examples["house"] == "H3"].head(2).assign(appliance="Oven")
    examples = pd.concat([examples, oven], ignore_index=True)

    test_h3 = next(test for house, _, test in loho_splits(examples) if house == "H3")

    assert "Oven" not in set(test_h3["appliance"])


def test_fold_features_do_not_depend_on_the_held_out_house():
    examples = select_mode(synthetic_transients(), "turn_on")
    train, test = examples[examples["house"] != "H3"], examples[examples["house"] == "H3"]
    # Make the test house look completely different. If it leaked into the
    # prototypes or the scaler, the training features would change.
    corrupted = test.assign(waveform=[w * 50 for w in test["waveform"]], pre_level=5000.0)

    fold = fold_features(train, test, APPLIANCES)
    fold_corrupted = fold_features(train, corrupted, APPLIANCES)

    np.testing.assert_array_equal(fold["X_train"], fold_corrupted["X_train"])
    np.testing.assert_array_equal(fold["scaler"].mean_, fold_corrupted["scaler"].mean_)
    assert fold["scaler"].n_samples_seen_ == len(train)
    for appliance, prototype in fold["prototypes"].items():
        assert any(prototype is w for w in train["waveform"])
        assert not any(prototype is w for w in test["waveform"])


def test_full_evaluation_never_trains_on_the_test_house(monkeypatch):
    seen = []
    real_fold_features = evaluation.fold_features

    def recording_fold_features(train, test, appliances):
        seen.append((set(train["house"]), set(test["house"])))
        return real_fold_features(train, test, appliances)

    monkeypatch.setattr(evaluation, "fold_features", recording_fold_features)
    run_loho_evaluation(synthetic_transients(), APPLIANCES, seeds=(0,))

    assert len(seen) == 9   # 3 modes x 3 held-out houses; features don't depend on the seed
    for train_houses, test_houses in seen:
        assert len(test_houses) == 1
        assert train_houses.isdisjoint(test_houses)


def test_step_size_baseline_only_sees_the_step():
    examples = select_mode(synthetic_transients(), "turn_on")
    train, test = examples[examples["house"] != "H1"], examples[examples["house"] == "H1"]
    fold = fold_features(train, test, APPLIANCES)
    baseline = next(m for m in make_models(0) if m.name == "Random Forest, step size only")

    estimator, _ = fit_and_predict(baseline, fold, train["appliance"].to_numpy())

    assert estimator.n_features_in_ == 1
    assert len(fold["feature_names"]) > 1


def test_held_out_importance_finds_the_informative_feature():
    rng = np.random.default_rng(0)
    x = np.c_[np.r_[np.zeros(10), np.ones(10)], rng.normal(size=20)]   # feature 0 decides, feature 1 is noise
    y = np.array(["a"] * 10 + ["b"] * 10)
    forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(x, y)

    importance = held_out_importance(forest, x, y, seed=0)

    assert importance.shape == (2,)
    assert importance[0] > importance[1] + 0.2


def test_results_over_seeds_on_separable_data():
    transients = synthetic_transients()

    metrics, fold_metrics, predictions, importance = run_loho_evaluation(transients, APPLIANCES, seeds=(0, 1))

    for df, table in [(metrics, METRICS), (fold_metrics, FOLD_METRICS),
                      (predictions, PREDICTIONS), (importance, FEATURE_IMPORTANCE)]:
        validate(df, table)
    assert set(metrics["mode"]) == {"turn_on", "turn_off", "combined"}
    assert set(predictions["seed"]) == {0, 1}

    turn_on = metrics[metrics["mode"] == "turn_on"].set_index("model")
    assert turn_on.loc[MAIN_MODEL, "accuracy_mean"] == 1.0
    assert turn_on.loc[MAIN_MODEL, "n_test"] == (transients["transition"] == "on").sum()
    assert turn_on.loc["Majority class", "accuracy_mean"] == 0.5   # two balanced classes
    # k-NN doesn't use the seed, so its spread over seeds is exactly zero.
    assert turn_on.loc["k-NN (k=3)", "accuracy_std"] == 0
    assert (metrics["n_seeds"] == 2).all() and (metrics["n_folds"] == 3).all()
