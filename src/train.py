"""Train and tune three models with cross-validation on the TRAINING set only.

For each model we tune exactly one hyperparameter with GridSearchCV, using
GroupKFold so that every volunteer's data stays inside a single fold.
The best model is chosen by mean CV accuracy (never by test accuracy) and
saved with joblib.

Run from the project root:
    python -m src.train
"""
import json

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import PROJECT_ROOT, load_split

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42  # fixed seed so results are reproducible
N_FOLDS = 5


def build_candidates() -> dict:
    """Each entry: (estimator, {param_name: [values to try]}).

    Parameter names inside a Pipeline use the form "<step name>__<param>",
    e.g. "clf__C" means "the C parameter of the step called clf".
    """
    return {
        # Logistic regression is sensitive to feature scale, so we put a
        # StandardScaler in front of it. Because the scaler is part of the
        # Pipeline, GridSearchCV re-fits it on each training fold only, so the
        # validation fold never leaks into the scaling statistics.
        # Tuned: C = inverse regularization strength. Small C = simpler model
        # (strong penalty on large weights), large C = more flexible model.
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        # Trees split on thresholds, so feature scale doesn't matter: no scaler.
        # Tuned: max_depth. Shallow trees underfit; very deep trees memorize
        # the training people (overfit). None = grow until leaves are pure.
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, 10, 15, None]},
        ),
        # Tuned: n_estimators = number of trees. More trees average out the
        # noise of individual trees; gains flatten after a point while cost
        # keeps growing. n_jobs=1 here because GridSearchCV already runs the
        # folds in parallel (avoids over-subscribing the CPU).
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"n_estimators": [25, 50, 100, 200, 400]},
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # Only the training split is loaded here. The test set is not touched
    # until evaluate.py, after every choice has been made.
    X_train, y_train, subjects_train = load_split("train")

    # GroupKFold: rows from the same subject always land in the same fold,
    # so each validation fold contains only people the model hasn't seen.
    cv = GroupKFold(n_splits=N_FOLDS)

    summary = []      # one row per model: best param + CV score
    all_results = []  # every param value tried, for the README tuning table

    for name, (estimator, param_grid) in build_candidates().items():
        print(f"\n=== Tuning {name}: {param_grid} ===")
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,      # use all CPU cores
            refit=True,     # after CV, re-train on ALL training data with best value
        )
        # groups= is how GroupKFold learns which rows belong to which person.
        search.fit(X_train, y_train, groups=subjects_train)

        # Print the full tuning curve so you can see under/overfitting.
        results = pd.DataFrame(search.cv_results_)
        param_col = f"param_{next(iter(param_grid))}"
        curve = results[[param_col, "mean_test_score", "std_test_score"]].copy()
        curve.columns = ["value", "mean_cv_accuracy", "std_cv_accuracy"]
        print(curve.to_string(index=False, float_format="%.4f"))

        best_param = search.best_params_
        print(f"-> best {best_param}, mean CV accuracy {search.best_score_:.4f}")

        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
        summary.append({
            "model": name,
            "best_params": json.dumps(best_param),
            "mean_cv_accuracy": round(search.best_score_, 4),
            "std_cv_accuracy": round(results.loc[search.best_index_, "std_test_score"], 4),
        })
        curve.insert(0, "model", name)
        curve.insert(1, "param", next(iter(param_grid)))
        all_results.append(curve)

    summary_df = pd.DataFrame(summary).sort_values("mean_cv_accuracy", ascending=False)
    summary_df.to_csv(REPORTS_DIR / "cv_summary.csv", index=False)
    pd.concat(all_results).to_csv(REPORTS_DIR / "cv_tuning_curves.csv", index=False)

    # Pick the winner by CV accuracy only.
    best = summary_df.iloc[0]
    best_model = joblib.load(MODELS_DIR / f"{best['model']}.joblib")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    # Small metadata file so the app and README know what was saved and with
    # which scikit-learn version (joblib files can break across versions).
    metadata = {
        "best_model": best["model"],
        "best_params": json.loads(best["best_params"]),
        "mean_cv_accuracy": float(best["mean_cv_accuracy"]),
        "cv": f"GroupKFold(n_splits={N_FOLDS}) grouped by subject",
        "sklearn_version": sklearn.__version__,
    }
    (MODELS_DIR / "best_model_meta.json").write_text(json.dumps(metadata, indent=2))

    print("\n=== Summary (sorted by mean CV accuracy) ===")
    print(summary_df.to_string(index=False))
    print(f"\nBest model: {best['model']} -> saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
