# Load AG News, Banking77, etc
"""Dataset loading for the JEV benchmark.

Every loader returns {"train": DataFrame, "test": DataFrame} with two columns:
``text`` (str) and ``label`` (str label name), so all models see the same format.
"""

import os
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

NEWSGROUPS_URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"

# Hugging Face IDs to try in order, in case a dataset gets moved or renamed
HF_DATASETS = {
    "ag_news": ["fancyzhx/ag_news", "ag_news"],
    "banking77": ["PolyAI/banking77", "legacy-datasets/banking77"],
}


# --------------------------------------------------------------------------- #
# 20 Newsgroups (loaded from the folder structure, per the course tip)
# --------------------------------------------------------------------------- #
def download_20newsgroups(raw_dir=RAW_DIR):
    """Download and extract 20news-bydate into raw_dir if it isn't there yet."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / "20news-bydate-train").exists():
        return
    archive = raw_dir / "20news-bydate.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(NEWSGROUPS_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(raw_dir)


def load_newsgroups_split(main_dir):
    """Walk main_dir/<category>/<file> into a DataFrame (listdir/listfiles/datalist)."""
    # Read the main directory's contents: the list of 20 category folders
    listdir = sorted(
        d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))
    )

    datalist = []
    for dirname in listdir:
        # Read the category folder's contents: the list of document files
        listfiles = sorted(os.listdir(os.path.join(main_dir, dirname)))
        for filename in listfiles:
            path = os.path.join(main_dir, dirname, filename)
            with open(path, encoding="latin1", errors="ignore") as f:
                text = f.read()
            datalist.append((dirname, filename, text))

    df = pd.DataFrame(datalist, columns=["label", "filename", "text"])
    return df[["text", "label"]]


def load_20newsgroups(raw_dir=RAW_DIR, download=True):
    raw_dir = Path(raw_dir)
    if download:
        download_20newsgroups(raw_dir)
    return {
        "train": load_newsgroups_split(raw_dir / "20news-bydate-train"),
        "test": load_newsgroups_split(raw_dir / "20news-bydate-test"),
    }


# --------------------------------------------------------------------------- #
# Hugging Face datasets
# --------------------------------------------------------------------------- #
def load_hf(names, text_col="text", raw_dir=RAW_DIR):
    """Load a Hugging Face dataset, converting integer labels to label names."""
    from datasets import load_dataset

    last_err = None
    for name in names:
        try:
            ds = load_dataset(name, cache_dir=str(Path(raw_dir) / "hf"))
            break
        except Exception as e:  # try the next ID
            last_err = e
    else:
        raise last_err

    label_names = ds["train"].features["label"].names
    out = {}
    for split in ["train", "test"]:
        df = ds[split].to_pandas()
        out[split] = pd.DataFrame(
            {
                "text": df[text_col].astype(str),
                "label": [label_names[i] for i in df["label"]],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Everything at once
# --------------------------------------------------------------------------- #
def load_all(raw_dir=RAW_DIR):
    """Return {dataset_name: {"train": df, "test": df}} for every dataset."""
    data = {"20newsgroups": load_20newsgroups(raw_dir)}
    for name, ids in HF_DATASETS.items():
        data[name] = load_hf(ids, raw_dir=raw_dir)
    return data


def get_labels(data, name):
    """Sorted label names for a dataset, taken from its training split."""
    return sorted(data[name]["train"]["label"].unique())


def summarize(data):
    """One row per dataset: train/test sizes and number of classes."""
    rows = [
        {
            "dataset": name,
            "train": len(s["train"]),
            "test": len(s["test"]),
            "classes": s["train"]["label"].nunique(),
        }
        for name, s in data.items()
    ]
    return pd.DataFrame(rows).set_index("dataset")