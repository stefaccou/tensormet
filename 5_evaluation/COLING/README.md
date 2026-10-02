# COLING evaluation

Everything needed to reproduce the paper's evaluation: the downstream tasks, the word-intrusion judge and
the human study.

## Where the downstream tasks come from

The tasks are the suite that the sparse / interpretable embedding papers share. It goes back to Faruqui et
al. (2015). The code is SPINE's (Subramanian et al., 2018); POLAR (Mathew et al., 2020) reuses it and adds
two tasks. `downstream_sweep.py` ports each script unchanged: same features, classifier grids and selection rule.

| Task | First used in this suite | Script ported from | Data read from |
|---|---|---|---|
| WordSim-353 | Faruqui et al. (2015) | SPINE | SPINE |
| TREC question classification | Faruqui et al. (2015) | SPINE | POLAR |
| 20 Newsgroups (computer, religion, sports) | Faruqui et al. (2015) | SPINE | POLAR |
| NP bracketing | Faruqui et al. (2015) | SPINE | POLAR |
| sentiment | Faruqui et al. (2015) | POLAR | POLAR |
| discriminative attributes | Mathew et al. (2020) | POLAR | POLAR |
| word analogy (not reported) | Mathew et al. (2020) | POLAR | gensim |

The SPINE repository ships no task data, so the data files are those in POLAR's repository. Two points of
protocol: NP bracketing is the mean over 10 folds, as in POLAR (SPINE scores two folds); embeddings are
scored as they are, whereas POLAR standardises every dimension first.

## Setup

1. Install `tensormet` (the repository root: `pip install -e .`).
2. Set `DATA_DIR` to the data directory (environment variable `SCRATCH_DATA` or `DATA`).
3. Clone the three upstream repositories into `third_party/`: see `third_party/README.md`.

Run every command from this folder (`5_evaluation/COLING`).

## What is read from `DATA_DIR`

| What | Where |
|---|---|
| our decompositions (model files, checkpoints, logs) | `tensors/<n>-gram-raw-bos-eos-fineweb-en_1B/decomposition/` |
| GloVe 2024, 100d and 300d | `corpora/wiki_giga_2024_{100,300}_*.txt` |
| the SPINE release (SPINE and SPOWV vectors) | `corpora/spine/` |
| the NNSE release (`depDocNNSE{300,1000}.tab.zip`) | `corpora/nnse/` |

word2vec (gensim-data) and the judge (Hugging Face) are downloaded on first use.

## Rebuilding the tables from the stored scores

The scores of every run are in this folder (`downstream_results/`, `judge_results/`, `judge_ablation_results/`,
`human_eval_results/`), so the notebooks give the paper's numbers without running a sweep:

| Notebook | Gives |
|---|---|
| `Coling_eval_summary.ipynb` | every result, test and figure |
| `Coling_tables.ipynb` | the paper's LaTeX tables, written to `tables/` |
| `human_evaluation.ipynb` | the human study and the comparison of the judges with the annotators |
| `Coling_examples.ipynb` | example dimensions |

The summary and tables notebooks also read the training logs in `DATA_DIR` (training curves, seconds per
iteration).

## Running the evaluation again

A sweep skips every run that already has a record, so `--dry-run` on the stored results reports nothing to do.
To score everything again, add `--no-resume` (downstream tasks) or `--rejudge` (judge).

```bash
python downstream_sweep.py --dry-run         # print the plan, fit nothing
python downstream_sweep.py                   # our runs at their last checkpoint + GloVe / word2vec
python downstream_sweep.py --series best     # our runs at the state training kept (appendix B)
python downstream_sweep.py --series same     # runs differing in ss_frac, at a common checkpoint
python method_baselines.py                   # the interpretable baselines: POLAR, SPINE, SPOWV, NNSE, SINr
python judge_sweep.py                        # the word-intrusion judge, one per GPU
python judge_test.py                         # the comparison of judge models (uses human_eval_results/)
```

The classifiers of the downstream tasks are unseeded, as in the original scripts. A new run therefore differs
from the stored scores by about 0.01 (up to 0.04 on the newsgroup tasks).

## Files

| File | Role |
|---|---|
| `eval_utils.py` | the model table (`MODELS`) and model loading |
| `glove_baseline.py` | reading GloVe and word2vec |
| `downstream_sweep.py` | the downstream tasks, ported from the upstream scripts |
| `method_baselines.py` | the same tasks on the interpretable baselines |
| `compare.py` | reads the downstream scores into tables |
| `judge_sweep.py` | the word-intrusion judge over every model |
| `judge_ablation.py`, `judge_test.py` | which judge model and prompt agree best with the annotators |
| `coling_eval.py` | the tests and figures of the summary notebook |
