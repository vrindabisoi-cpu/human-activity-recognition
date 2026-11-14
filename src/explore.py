"""Step 2 of the assignment: check missing values and class balance.

Run from the project root:
    python -m src.explore
"""
import pandas as pd

from src.data_loader import PROJECT_ROOT, load_split

REPORTS_DIR = PROJECT_ROOT / "reports"


def missing_value_report(X: pd.DataFrame, y: pd.Series, name: str) -> None:
    """Print how many cells are missing (NaN) in the features and labels."""
    n_missing_X = int(X.isna().sum().sum())   # count per column, then total
    n_missing_y = int(y.isna().sum())          # a label that failed to map would be NaN
    print(f"[{name}] missing feature values: {n_missing_X} / {X.size:,} cells")
    print(f"[{name}] missing labels:         {n_missing_y}")


def class_balance(y: pd.Series) -> pd.DataFrame:
    """Return the count and percentage of each activity."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({
        "count": counts,
        "percent": (counts / counts.sum() * 100).round(1),
    })


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    X_train, y_train, _ = load_split("train")
    X_test, y_test, _ = load_split("test")

    print("=== Missing values ===")
    missing_value_report(X_train, y_train, "train")
    missing_value_report(X_test, y_test, "test")

    print("\n=== Class balance ===")
    balance = pd.concat(
        {"train": class_balance(y_train), "test": class_balance(y_test)}, axis=1
    )
    print(balance)

    # Imbalance ratio = largest class / smallest class. Near 1 means balanced.
    for name, y in (("train", y_train), ("test", y_test)):
        counts = y.value_counts()
        print(f"[{name}] imbalance ratio (max/min): {counts.max() / counts.min():.2f}")

    balance.to_csv(REPORTS_DIR / "class_balance.csv")
    print(f"\nSaved: {REPORTS_DIR / 'class_balance.csv'}")


if __name__ == "__main__":
    main()
