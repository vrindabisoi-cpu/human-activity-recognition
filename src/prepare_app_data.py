"""Save the test set as one compact file the deployed app can load.

The raw UCI folder is large and not committed to GitHub, so the app reads
app_data/test_set.parquet instead. Columns: the 561 features, plus
"activity" (true label) and "subject" (volunteer ID).

Run from the project root:
    python -m src.prepare_app_data
"""
from src.data_loader import PROJECT_ROOT, load_split

OUT_DIR = PROJECT_ROOT / "app_data"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    X, y, subjects = load_split("test")
    df = X.assign(activity=y.values, subject=subjects.values)
    out = OUT_DIR / "test_set.parquet"
    df.to_parquet(out, compression="zstd", index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
