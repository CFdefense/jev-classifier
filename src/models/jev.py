"""TypeSafe Jev wrapper for the benchmark.

``run_jev`` writes one CSV row per test document as it goes, so a crash, rate
limit or sleeping laptop never loses work: rerun it and it resumes.
"""

import csv
import json
import os
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "results" / "predictions"

DEFAULT_INSTRUCTIONS = "Which category does this text belong to?"
DEFAULT_MAX_CHARS = 8000  # apply the same truncation to every model's input


def load_env(path=REPO_ROOT / ".env"):
    """Read KEY=value lines from .env into os.environ (no extra dependency)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_client():
    """Create a TypeSafe client, reading TYPESAFE_API_KEY from .env if needed."""
    from typesafe_sdk import TypeSafeClient

    load_env()
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("Set TYPESAFE_API_KEY in .env (see .env.example)")
    return TypeSafeClient()


def jev_classify(
    client,
    text,
    labels,
    instructions=DEFAULT_INSTRUCTIONS,
    max_chars=DEFAULT_MAX_CHARS,
):
    """Classify one text. Returns (pred, {label: prob}, latency_s, usage)."""
    from typesafe_sdk import Choice

    t0 = time.perf_counter()
    resp = client.system_one(
        text[:max_chars],
        {
            "label": Choice(
                instructions=instructions,
                criteria={lab: None for lab in labels},
            )
        },
    )
    latency = time.perf_counter() - t0
    ans = resp.choices["label"]
    return ans.choice, dict(ans.probabilities), latency, getattr(resp, "usage", None)


def run_jev(
    client,
    name,
    test_df,
    labels,
    n=None,
    max_chars=DEFAULT_MAX_CHARS,
    out_dir=PRED_DIR,
    seed=42,
    log_every=250,
):
    """Run Jev over a test set, appending each prediction to a CSV (resumable).

    n: if given, run on a random sample of n documents (quick trial run).
    Returns the full predictions DataFrame.
    """
    if n:
        test_df = test_df.sample(n=n, random_state=seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jev_{name}{f'_n{n}' if n else ''}.csv"

    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(prev.loc[prev["error"].isna(), "idx"])  # retry failed rows
        print(f"{name}: resuming, {len(done)} already done")

    new_file = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "idx", "true", "pred", "max_prob", "probs_json", "latency_s",
                    "input_tokens", "output_tokens", "error",
                ]
            )
        for i, (idx, row) in enumerate(test_df.iterrows()):
            if idx in done:
                continue
            try:
                pred, probs, lat, usage = jev_classify(
                    client, row["text"], labels, max_chars=max_chars
                )
                writer.writerow(
                    [
                        idx, row["label"], pred, max(probs.values()),
                        json.dumps(probs), round(lat, 4),
                        getattr(usage, "input_tokens", None),
                        getattr(usage, "output_tokens", None), "",
                    ]
                )
            except Exception as e:
                writer.writerow([idx, row["label"], "", "", "", "", "", "", repr(e)])
            f.flush()
            if log_every and i % log_every == 0:
                print(f"{name}: {i}/{len(test_df)}")

    preds = load_predictions(out_path)
    print(f"{name}: saved {len(preds)} rows -> {out_path}")
    return preds


def load_predictions(path):
    """Read a predictions CSV, keeping the latest successful row per document."""
    df = pd.read_csv(path)
    ok = df[df["error"].isna()].drop_duplicates("idx", keep="last")
    failed = df[~df["idx"].isin(ok["idx"])].drop_duplicates("idx", keep="last")
    return pd.concat([ok, failed]).sort_values("idx").reset_index(drop=True)


def quick_summary(preds):
    """Accuracy, error count and latency for a predictions DataFrame."""
    ok = preds[preds["error"].isna()]
    return {
        "n": len(preds),
        "errors": int(preds["error"].notna().sum()),
        "accuracy": round(float((ok["true"] == ok["pred"]).mean()), 4),
        "p50_latency_s": round(float(ok["latency_s"].median()), 4),
        "p95_latency_s": round(float(ok["latency_s"].quantile(0.95)), 4),
        "input_tokens": int(ok["input_tokens"].sum()),
        "output_tokens": int(ok["output_tokens"].sum()),
    }