import csv
import json
from pathlib import Path

import requests

# =============================
# CONFIG
# =============================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "banking77_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TRAIN_JSON = DATA_DIR / "train.json"
TEST_JSON = DATA_DIR / "test.json"
CATEGORY_JSON = DATA_DIR / "category.json"


# =============================
# DOWNLOAD
# =============================
def download_file(url, path):
    print(f"Downloading: {url}")
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Saved: {path}")
    else:
        raise Exception(f"Failed to download {url}")


# =============================
# LOAD CSV
# =============================
def load_csv(path):
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            text, label = row
            data.append({"text": text, "label": label})
    return data


# =============================
# MAIN PIPELINE
# =============================
def main():
    # 1. Download
    download_file(TRAIN_URL, TRAIN_PATH)
    download_file(TEST_URL, TEST_PATH)

    # 2. Load data
    train_data = load_csv(TRAIN_PATH)
    test_data = load_csv(TEST_PATH)

    # 3. Build label set
    all_labels = sorted(list(set([x["label"] for x in train_data + test_data])))

    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # 4. Save category.json
    with open(CATEGORY_JSON, "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2, ensure_ascii=False)

    print(f"Saved: {CATEGORY_JSON}")

    # 5. Convert label -> id
    for item in train_data:
        item["label_id"] = label2id[item["label"]]

    for item in test_data:
        item["label_id"] = label2id[item["label"]]

    # 6. Save JSON (clean format)
    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print("✅ DONE!")
    print(f"- Train samples: {len(train_data)}")
    print(f"- Test samples: {len(test_data)}")
    print(f"- Num labels: {len(all_labels)}")


if __name__ == "__main__":
    main()
