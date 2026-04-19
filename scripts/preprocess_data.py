from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "banking77_data"
OUTPUT_DIR = PROJECT_DIR / "sample_data"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
CATEGORY_FILE = DATA_DIR / "category.json"

LOG_FILE = OUTPUT_DIR / "cleaning_log.txt"

RANDOM_STATE = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================
# LOAD CATEGORY
# =============================
def load_categories(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing category file: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # category.json dạng: { "0": "label_name", ... }
    return set(data.values())


# =============================
# LOAD CSV
# =============================
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    expected_cols = {"text", "category"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing_cols)}")

    return df.loc[:, ["text", "category"]].copy()


# =============================
# CLEAN DATA
# =============================
def clean_data(df: pd.DataFrame, valid_categories: set[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    stats = {
        "original_rows": len(df),
        "removed_empty_rows": 0,
        "removed_duplicates": 0,
        "removed_invalid_category": 0,
        "final_rows": 0,
    }

    duplicate_rows: list[dict[str, str]] = []
    invalid_category_rows: list[dict[str, str]] = []

    cleaned = df.copy()

    # Normalize
    cleaned["text"] = cleaned["text"].astype("string").str.strip()
    cleaned["category"] = cleaned["category"].astype("string").str.strip()
    cleaned["text"] = cleaned["text"].str.replace('"', "", regex=False)
    cleaned["category"] = cleaned["category"].str.replace('"', "", regex=False)

    # Remove empty
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["text", "category"])
    cleaned = cleaned[(cleaned["text"] != "") & (cleaned["category"] != "")]
    stats["removed_empty_rows"] = before - len(cleaned)

    # Remove invalid category
    before = len(cleaned)
    invalid_mask = ~cleaned["category"].isin(valid_categories)
    invalid_category_rows = cleaned.loc[invalid_mask, ["text", "category"]].to_dict("records")
    cleaned = cleaned[~invalid_mask]
    stats["removed_invalid_category"] = before - len(cleaned)

    # Remove duplicates
    before = len(cleaned)
    duplicate_mask = cleaned.duplicated(subset=["text", "category"], keep="first")
    duplicate_rows = cleaned.loc[duplicate_mask, ["text", "category"]].to_dict("records")
    cleaned = cleaned.drop_duplicates(subset=["text", "category"], keep="first")
    stats["removed_duplicates"] = before - len(cleaned)

    cleaned = cleaned.reset_index(drop=True)
    stats["final_rows"] = len(cleaned)

    stats["duplicate_rows"] = duplicate_rows
    stats["invalid_category_rows"] = invalid_category_rows

    return cleaned, stats


# =============================
# SPLIT
# =============================
def can_stratify(df: pd.DataFrame, test_size: float) -> bool:
    class_counts = df["category"].value_counts()
    if class_counts.empty:
        return False

    min_count = class_counts.min()
    n_classes = class_counts.shape[0]
    split_size = int(round(len(df) * test_size))
    return min_count >= 2 and split_size >= n_classes


def split_data(df: pd.DataFrame):
    temp_ratio = 1 - TRAIN_RATIO
    stratify_first = df["category"] if can_stratify(df, temp_ratio) else None
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_first,
    )

    stratify_second = temp_df["category"] if can_stratify(temp_df, TEST_RATIO / temp_ratio) else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / temp_ratio,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_second,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =============================
# WRITE OUTPUT
# =============================
def write_outputs(train_df, val_df, test_df, stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")

    log_lines = [
        "Preprocessing summary",
        f"Source train rows: {stats['source_train_rows']}",
        f"Source test rows: {stats['source_test_rows']}",
        f"Combined rows: {stats['original_rows']}",
        f"Removed empty rows: {stats['removed_empty_rows']}",
        f"Removed invalid category rows: {stats['removed_invalid_category']}",
        f"Removed duplicate rows: {stats['removed_duplicates']}",
        f"Final rows: {stats['final_rows']}",
        "",
        "Split sizes",
        f"Train: {len(train_df)}",
        f"Val: {len(val_df)}",
        f"Test: {len(test_df)}",
        "",
        "Invalid category rows",
    ]

    if stats["invalid_category_rows"]:
        for i, row in enumerate(stats["invalid_category_rows"], 1):
            log_lines.append(f"{i}. text={row['text']!r} | category={row['category']!r}")
    else:
        log_lines.append("None")

    log_lines.append("\nDuplicate rows")
    if stats["duplicate_rows"]:
        for i, row in enumerate(stats["duplicate_rows"], 1):
            log_lines.append(f"{i}. text={row['text']!r} | category={row['category']!r}")
    else:
        log_lines.append("None")

    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")


# =============================
# MAIN
# =============================
def main():
    valid_categories = load_categories(CATEGORY_FILE)

    train_source = load_data(TRAIN_FILE)
    test_source = load_data(TEST_FILE)

    combined = pd.concat([train_source, test_source], ignore_index=True)
    cleaned, stats = clean_data(combined, valid_categories)

    stats["source_train_rows"] = len(train_source)
    stats["source_test_rows"] = len(test_source)

    train_df, val_df, test_df = split_data(cleaned)
    write_outputs(train_df, val_df, test_df, stats)

    print(f"Saved cleaned dataset to: {OUTPUT_DIR}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
