"""Load the UCI HAR dataset, keeping the original train/test split.

Usage:
    from src.data_loader import load_split
    X_train, y_train, subjects_train = load_split("train")
"""
from pathlib import Path

import pandas as pd

# Project root = the folder that contains src/. Building paths from this file's
# location means the code works no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "UCI HAR Dataset"


def load_feature_names() -> list:
    """Read the 561 feature names and make them unique.

    features.txt contains 42 names that appear more than once (the
    bandsEnergy features). pandas needs unique column names, so repeats get
    a suffix: the first copy keeps its name, later ones become name_dup1,
    name_dup2, and so on.
    """
    features = pd.read_csv(
        DATA_DIR / "features.txt", sep=r"\s+", header=None, names=["idx", "name"]
    )
    # cumcount numbers each repeat of a name: 0 for the first, 1 for the second...
    dup_number = features.groupby("name").cumcount()
    unique_names = [
        name if n == 0 else f"{name}_dup{n}"
        for name, n in zip(features["name"], dup_number)
    ]
    return unique_names


def load_activity_labels() -> dict:
    """Return a mapping like {1: 'WALKING', 2: 'WALKING_UPSTAIRS', ...}."""
    labels = pd.read_csv(
        DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"]
    )
    return dict(zip(labels["id"], labels["activity"]))


def load_split(split: str):
    """Load one split ("train" or "test").

    Returns:
        X        DataFrame of shape (n_samples, 561) with named columns
        y        Series of activity names (e.g. "WALKING")
        subjects Series of volunteer IDs (1-30); needed for GroupKFold
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    folder = DATA_DIR / split
    # The files are space-separated with no header row; sep=r"\s+" means
    # "one or more whitespace characters" (some rows have leading spaces).
    X = pd.read_csv(folder / f"X_{split}.txt", sep=r"\s+", header=None)
    X.columns = load_feature_names()

    y_ids = pd.read_csv(folder / f"y_{split}.txt", header=None).iloc[:, 0]
    y = y_ids.map(load_activity_labels()).rename("activity")

    subjects = pd.read_csv(folder / f"subject_{split}.txt", header=None).iloc[:, 0]
    subjects = subjects.rename("subject")

    return X, y, subjects


if __name__ == "__main__":
    # Quick self-check: run `python -m src.data_loader` from the project root.
    for split in ("train", "test"):
        X, y, subjects = load_split(split)
        print(f"{split}: X={X.shape}, y={y.shape}, "
              f"subjects={sorted(subjects.unique().tolist())}")
