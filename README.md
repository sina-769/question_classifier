# Question Classifier

A lightweight text classifier that decides whether a question is about a
health(care) topic. Built on a method from
[*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
(O'Reilly): DistilBERT's frozen hidden states + a logistic-regression head —
a full training run that trains only 769 parameters (the encoder's 66M stay
frozen), runs on CPU, ~98% accuracy on the held-out split.

In 2026 the repository was reworked for licensing compliance: the raw
question texts were removed and replaced by a reproducible build pipeline
(see below for what changed and why).

## Context

This classifier was part of a team portfolio project during my
data-science training at Data Science Retreat, Berlin (January–April
2022), built to get hands-on with transformer models. I led the project:
I designed and built the classifier on my own — concept, labeling,
implementation — did most of the remaining work, and set the direction
for the team's technical tasks.

The pipeline routed incoming questions: health questions were flagged so
that the answers of a downstream language model could be checked
separately. Alongside the classifier, we fine-tuned the smallest GPT-2,
connected a GPT-3 model via the OpenAI API, experimented with GPT-Neo and
GPT-J, and ran systematic tests of how well these models answered health
questions, to get a feel for where the technology stood.

The health labels are expert labels: I labeled the questions as a
pharmacist, based on the kinds of health questions that come up in
community pharmacy practice.

**Looking back from 2026:** a system that answers medical questions would
today likely fall under the EU Medical Device Regulation, so one would
rather use the classifier the other way round: let the language model
handle only non-medical questions and route health questions to a human.
That shifts the critical metric from overall accuracy to recall on health
questions. Also, nearly all health questions come from one forum
(r/AskDocs) and nearly all others from Wikipedia-based SQuAD, so part of
the ~98% may reflect source style rather than topic; the 63 health
questions within SQuAD are the real test for that (see [Results](#results):
75% recall on the 12 of them in the test split).

## How it works

1. `build_data.py` downloads the upstream datasets, resolves `labels.csv`
   and writes `data/classifier_train.csv`, `data/classifier_test.csv` and
   `data/df_full.csv` (train+test combined; the classifier is fitted on `df_full`).
2. `train_classifier.ipynb` encodes the questions with DistilBERT
   (`distilbert-base-uncased`), extracts the `[CLS]` hidden states,
   projects them with UMAP for visualization, and fits a logistic
   regression on top.

The 768-dimensional hidden states, projected to 2D with UMAP (cosine metric)
— the two classes fall apart cleanly, which is why 769 trained parameters
are enough:

![UMAP projection of the frozen DistilBERT [CLS] hidden states on the
training split. Left: non-health questions (blue density), right: health
questions (red density). The two densities occupy distinct regions of the
2D embedding.](docs/umap_hidden_states.png)

```bash
python build_data.py        # downloads sources, writes data/*.csv
python evaluate.py           # trains + scores the classifier (CPU, ~5 min)
python train_classifier.ipynb  # or open in Colab/Jupyter
```

## Results

Held-out test split (455 questions: 181 health, 274 other), classifier
fitted on the train split only, default threshold 0.5. Output of
`evaluate.py`; 95% confidence intervals from 2,000 bootstrap resamples of
the test split.

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| health | 0.978 [0.953, 0.995] | 0.967 [0.938, 0.989] | 0.972 [0.953, 0.988] | 181 |
| other | 0.978 | 0.985 | 0.982 | 274 |
| macro avg | 0.978 | 0.976 | 0.977 | 455 |

Accuracy 0.978 [0.963, 0.991], ROC-AUC 0.998. Confusion matrix: 6 health
questions missed, 4 other questions flagged as health. With 455 test
questions the intervals are about ±2–3 points wide, so the gap to the
98.0% of 2022 is not a real difference.

**Per source.** Health questions from AskDocs are recognized at 98%
(166/169), but the health questions within SQuAD only at 75% (9/12). The
SQuAD subset is small, but it supports the concern raised above: part
of the overall score comes from source style (Reddit title vs. Wikipedia
question), not topic.

**Threshold.** If health questions are to be routed to a human, recall on
health is the metric that matters. Lowering the threshold on p(health)
from 0.5 to 0.2 raises health recall to 0.994 (1 miss instead of 6) at
precision 0.957 (8 false alarms instead of 4). The threshold was read off
the test split itself, so these numbers are optimistic; a real system
would pick it on a separate validation split.

Several of the 10 errors are borderline labels rather than model mistakes
(e.g. *"What condition was attributed to binge drinking in a 2004 study?"*,
labeled other).

## Data sources & licenses

The question texts are **not** stored in this repository. Instead,
`labels.csv` stores one row per labeled question — `source`, `source_id`,
`label` (1 = health, 0 = other), `split` — and `build_data.py` pulls the
texts from the upstream datasets, which retain their own licenses:

| Source | What was used | License |
|---|---|---|
| [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al.) | 1,355 questions (1,292 non-health, 63 health) | CC BY-SA 4.0 |
| [AskDocs](https://github.com/ju-resplande/askD) (Gomes) | 871 questions (health, from r/AskDocs titles) | GPL-3.0 |

`labels.csv` itself (the labeling decisions) is my own work and is part of
this repository under the Apache-2.0 license. If you redistribute the
*reconstructed datasets* (the CSVs produced by `build_data.py`), the
upstream licenses above apply to the question texts.

The AskDocs questions are titles of public Reddit posts (r/AskDocs). This
repository contains no Reddit content and no trained model weights, and it
is a non-commercial learning project. Reddit's
[Public Content Policy](https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy)
allows non-commercial uses "such as learning and community" and asks
commercial users to contact Reddit first. If you rebuild the data for
commercial use, check Reddit's terms yourself.

### Why the texts are not in the repo

An earlier version of this repository shipped the combined CSVs directly.
A provenance review found that ~41% of those rows could not be traced to
a licensable source with certainty, so they were dropped; the remaining
2,226 labeled rows are reproduced deterministically from the two sources
above. Of these, 2,014 form the train/test splits (1,559 / 455); 212 are
marked `split=unused` and are only written with `--include-unused`.
This keeps the repo clean for portfolio use without silently redistributing
third-party forum content.

The cleanup cost nothing in accuracy: the reduced dataset scores **97.8%**
on the held-out test split with the notebook's method (frozen DistilBERT
+ logistic regression), vs. 98.0% for the original 2022 dataset with ~70%
more rows — the classifier was already in its saturation region.
`evaluate.py` reproduces this number without Jupyter, together with the
metrics under [Results](#results).

## Repository layout

```
train_classifier.ipynb   # end-to-end: encode -> features -> logistic regression
build_data.py            # dataset build from upstream sources
evaluate.py              # same method as a script: train + score (P/R/F1, CIs, per source), no Jupyter
visualize.py             # UMAP 2D projection of the hidden states -> docs/*.png
labels.csv               # labeling decisions (source, source_id, label, split)
data/                    # generated output (git-ignored)
docs/                    # generated figures (committed)
```

## Credits & license

- The training-notebook method is adapted from the book
  [*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
  (O'Reilly), Apache-2.0 — see NOTICE.
- Dataset licenses: SQuAD v1.1 CC BY-SA 4.0; AskDocs GPL-3.0.
- Everything else in this repository (code, labels, documentation):
  Apache-2.0 (see LICENSE), © 2022–2026 Sina Rampe.
