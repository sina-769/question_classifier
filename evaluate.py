#!/usr/bin/env python3
"""Evaluate the classifier: frozen DistilBERT [CLS] states + logistic regression.

Reproduces the notebook's method as a plain script so the result can be
verified without Jupyter:

  1. tokenize with distilbert-base-uncased
  2. extract frozen [CLS] hidden states (768-dim)
  3. fit LogisticRegression on the train split, score the test split

Reports accuracy, per-class precision/recall/F1 with bootstrap 95% CIs,
recall per source (the health questions within SQuAD are the real test
for topic vs. source style) and how the decision threshold trades health
recall against precision.

Also reports the notebook-style variant (fit on train+test, score test)
for comparison with the original 2022 numbers.

Run:  python evaluate.py            # needs data/*.csv (run build_data.py first)
      python evaluate.py --skip-full
"""

import argparse
import csv
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from transformers import AutoModel, AutoTokenizer

MODEL_CKPT = "distilbert-base-uncased"


def extract_hidden_states(texts, tokenizer, model, device, batch_size=32, max_length=64):
    """Return np.ndarray (N, 768) of frozen [CLS] hidden states."""
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state[:, 0]  # [CLS] token
        out.append(hs.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            done = i + len(batch)
            print(f"  encoded {done}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    return np.vstack(out)


def load_split(path):
    rows = list(csv.DictReader(open(path)))
    return [r["prompt"] for r in rows], np.array([int(r["completion"]) for r in rows])


def load_sources(labels_path, split):
    """Source of each row in data/classifier_<split>.csv (build_data.py keeps labels.csv order)."""
    return np.array([r["source"].split(":")[0] for r in csv.DictReader(open(labels_path))
                     if r["split"] == split])


def health_prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return p[0], r[0], f[0]


def report(name, y_true, y_pred):
    print(f"\n== {name} ==")
    print("accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("confusion matrix (rows: true other/health):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["other", "health"], digits=3))


def bootstrap_ci(y_true, y_pred, n=2000, seed=0):
    """Percentile 95% CIs for accuracy and health precision/recall/F1."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        stats.append((accuracy_score(y_true[i], y_pred[i]), *health_prf(y_true[i], y_pred[i])))
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
    point = (accuracy_score(y_true, y_pred), *health_prf(y_true, y_pred))
    print(f"bootstrap 95% CIs ({n} resamples of the test split):")
    for k, name in enumerate(["accuracy", "health precision", "health recall", "health F1"]):
        print(f"  {name:17s} {point[k]:.3f}  [{lo[k]:.3f}, {hi[k]:.3f}]")


def by_source(y_true, y_pred, sources):
    print("\nper source (share classified correctly):")
    for src in sorted(set(sources)):
        for label, lname in [(0, "other"), (1, "health")]:
            m = (sources == src) & (y_true == label)
            if m.any():
                print(f"  {src:7s} {lname:6s} {(y_pred[m] == label).sum():3d}/{m.sum():3d}"
                      f"  ({(y_pred[m] == label).mean():.3f})")


def thresholds(y_true, p_health):
    print(f"\nthreshold on p(health) (ROC-AUC {roc_auc_score(y_true, p_health):.3f};"
          " chosen on the test split, so optimistic):")
    for t in (0.5, 0.3, 0.2, 0.1):
        pred = (p_health >= t).astype(int)
        p, r, _ = health_prf(y_true, pred)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        print(f"  t={t:.1f}  health recall {r:.3f}  precision {p:.3f}  FN={fn}  FP={fp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--model", default=MODEL_CKPT,
                    help="checkpoint name or local path (default: %(default)s)")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip the notebook-style fit on train+test")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    X_tr_text, y_tr = load_split(f"{args.data}/classifier_train.csv")
    X_te_text, y_te = load_split(f"{args.data}/classifier_test.csv")
    src_te = load_sources(args.labels, "test")
    assert len(src_te) == len(y_te), "labels.csv does not match data/ — rerun build_data.py"
    print(f"train: {len(X_tr_text)}  test: {len(X_te_text)}")

    print("\nEncoding train ...")
    X_tr = extract_hidden_states(X_tr_text, tokenizer, model, device)
    print("Encoding test ...")
    X_te = extract_hidden_states(X_te_text, tokenizer, model, device)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    report("Fit on train, score on test (held-out)", y_te, y_pred)
    bootstrap_ci(y_te, y_pred)
    by_source(y_te, y_pred, src_te)
    thresholds(y_te, clf.predict_proba(X_te)[:, 1])

    if not args.skip_full:
        X_full = np.vstack([X_tr, X_te])
        y_full = np.concatenate([y_tr, y_te])
        clf_full = LogisticRegression(max_iter=5000)
        clf_full.fit(X_full, y_full)
        acc = accuracy_score(y_te, clf_full.predict(X_te))
        print(f"\nNotebook-style fit on train+test, scored on test: {round(acc, 4)} "
              "(optimistic — test was part of the fit)")


if __name__ == "__main__":
    main()
