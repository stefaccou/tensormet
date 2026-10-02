"""POLAR downstream evaluation of tensormet models.

Scores every model of the model table (eval_utils.MODELS) and the GloVe / word2vec
baselines on the downstream tasks of POLAR (Mathew et al. 2020). The task code is a port
of POLAR's own scripts (third_party/POLAR/Downstream Task/): same features,
same classifier grids, same selection rules. WordSim-353 is added from SPINE
(third_party/spine/code/evaluation/intrinsic/evaluate_wordSim.py), ported the same way.

Two words used throughout:
    run   one (model, variant) pair, e.g. ('tt_4g_r100_scSoftPlus_ss0.025', 'raw')
    job   one unit of work of a run: one classifier fit, or the word-analogy evaluation

Outputs, all in polar_results/:
    tensormet_eval.jsonl      one record per finished run, appended when the run finishes
    sweep_<stamp>.json        manifest: settings, versions, models loaded / missing, planned runs, status
    sweep_<stamp>.log         everything this script printed
    fit_times_<stamp>.log     one start and one end line per classifier fit

On ampere:
    screen -S polar
    conda activate ccl
    cd 5_evaluation/COLING
    python polar_sweep.py --dry-run          # load the models, print the plan, fit nothing
    python polar_sweep.py                    # detach: Ctrl-a d; reattach: screen -r polar

A rerun skips the runs that already have a record (same model file, same seed), so a sweep
stopped with Ctrl-c continues where it stopped. A finished run that lacks an untrained task
(e.g. wordsim353, added later) gets it scored on the rerun, without refitting:
    python polar_sweep.py --backfill-only    # only that, then stop

--series picks which state of every run is loaded (see eval_utils):
    python polar_sweep.py                    # 'latest': the last checkpoint             -> sweep_<stamp>.*
    python polar_sweep.py --series best      # the model file: the best-semantic state   -> best_<stamp>.*
    python polar_sweep.py --series same      # the runs that differ only in --same-factor (default
                                             # ss_frac), each at the last checkpoint they all have;
                                             # our runs only                             -> same_<stamp>.*
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import platform
import signal
import sys
import time
import warnings
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path

import gensim
import numpy as np
from gensim.test.utils import datapath
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from eval_utils import (BASE, BASELINE_VOCABS, FIXED, GLOVE_WIDTHS, MODELS,
                        describe_models, load_models, load_same)

COLING_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = COLING_DIR / 'third_party'  # the upstream repositories, cloned by hand (third_party/README.md)
TASK_DIR = THIRD_PARTY_DIR / 'POLAR/Downstream Task'
WORDSIM_PATH = THIRD_PARTY_DIR / 'spine/code/evaluation/intrinsic/word_sim.tab'
RESULTS_DIR = COLING_DIR / 'polar_results'
RESULTS_PATH = RESULTS_DIR / 'tensormet_eval.jsonl'


# --- Settings -----------------------------------------------------------------
# --series -> (state of a run that eval_utils loads, prefix of the manifest and log files)
SERIES = {
    'latest': ('latest', 'sweep'),
    'best': ('model_file', 'best'),
    'same': ('same', 'same'),
}
# What --same-factor can be: the model table's columns and the settings they vary
SAME_FACTORS = sorted({'decomposition', 'ngram', *(key for column in MODELS.values() for key in column)})

N_JOBS = 100
SVC_CACHE_MB = 1000     # libsvm kernel cache (sklearn default: 200). Speed only: the fitted model is identical
ANALOGY_VOCAB = 300000  # gensim's restrict_vocab default


# --- Variants: how an embedding is rescaled before the tasks ---------------------
def _as_is(E):
    return E


def _unit_mean_norm(E):
    """Divide by one global scalar so that the mean row norm is 1.
    Removes the arbitrary scale of a Tucker factor and keeps the geometry."""
    return E / np.linalg.norm(E, axis=1).mean()


VARIANT_FNS = {'raw': _as_is, 'scaled': _unit_mean_norm}
# Default of --variants. Old records also hold a 'std' variant (per-dimension z-score), which is
# no longer run: nearly the same scores, much slower.
VARIANTS = ['raw']


def make_embeddings(models, variants):
    """The embedding matrix of every run.

    A run whose matrix has non-finite values (e.g. NaN factors) is left out here, because
    sklearn would otherwise raise in the middle of the sweep.
    Returns ({run: matrix}, {run: why it was left out})."""
    embeddings, skipped = {}, {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for name, model in models.items():
            for variant in variants:
                E = VARIANT_FNS[variant](model['E_raw'])
                n_bad_dims = int((~np.isfinite(E)).any(axis=0).sum())
                if n_bad_dims > 0:
                    skipped[(name, variant)] = f'{n_bad_dims} of {E.shape[1]} dims non-finite'
                else:
                    embeddings[(name, variant)] = E
    return embeddings, skipped


# --- Classifier grids, copied from the scripts in Downstream Task/ ----------------
def grid_sentence():
    """classify_sentiment.py and TREC/classify_task.py (identical lists)."""
    return [
        SVC(kernel='linear', C=0.025, class_weight='balanced'),
        SVC(kernel='linear', C=0.1, class_weight='balanced'),
        SVC(kernel='linear', C=5, class_weight='balanced'),
        SVC(kernel='linear', C=10, class_weight='balanced'),
        SVC(kernel='linear', C=50, class_weight='balanced'),
        SVC(kernel='linear', C=100, class_weight='balanced'),
        SVC(kernel='linear', C=500, class_weight='balanced'),
        SVC(kernel='linear', C=1000, class_weight='balanced'),
        SVC(kernel='linear', C=0.25, class_weight='balanced'),
        SVC(gamma=2, C=0.1, class_weight='balanced'),
        SVC(gamma=2, C=0.25, class_weight='balanced'),
        SVC(C=0.1, class_weight='balanced'),
        SVC(C=5, class_weight='balanced'),
        SVC(C=10, class_weight='balanced'),
        SVC(C=50, class_weight='balanced'),
        SVC(C=100, class_weight='balanced'),
        SVC(C=500, class_weight='balanced'),
        SVC(C=1000, class_weight='balanced'),
        SVC(class_weight='balanced'),
        MLPClassifier(alpha=1),
        GaussianNB(),
        RandomForestClassifier(),
        LogisticRegression(class_weight='balanced'),
        LogisticRegression(class_weight='balanced', C=.025),
        LogisticRegression(class_weight='balanced', C=0.1),
        LogisticRegression(class_weight='balanced', C=5),
        LogisticRegression(class_weight='balanced', C=10),
        LogisticRegression(class_weight='balanced', C=50),
        LogisticRegression(class_weight='balanced', C=100),
        LogisticRegression(class_weight='balanced', C=500),
    ]


def grid_newsgroups():
    """newsgroups/classify.py with num_classes=2: the first 22 entries of grid_sentence."""
    return grid_sentence()[:22]


def grid_np():
    """np_bracketing/classify_bracketing.py."""
    return [
        SVC(kernel='linear', C=0.025),
        SVC(kernel='linear', C=0.1),
        SVC(kernel='linear', C=1.0),
        SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        RandomForestClassifier(max_depth=5, n_estimators=50, max_features=10),
        MLPClassifier(alpha=1),
        RandomForestClassifier(n_estimators=20, max_features=10),
    ]


def prepare_classifier(clf, random_state=None, cache_mb=SVC_CACHE_MB):
    """Seed the classifier (if `random_state` is given) and enlarge the SVC kernel cache.
    Neither changes the grid."""
    if random_state is not None and 'random_state' in clf.get_params():
        clf.set_params(random_state=random_state)
    if isinstance(clf, SVC):
        clf.set_params(cache_size=cache_mb)
    return clf


# --- Tasks --------------------------------------------------------------------
# Data paths are relative to TASK_DIR; '{}' is filled with e.g. 'train_X'.

# Tasks that report the test score of the classifier with the best validation score:
# task -> (data path, classifier grid)
SELECT_BY_VAL = {
    'sentiment': ('sentiment/data/sentiment_{}.p', grid_sentence),
    'TREC': ('TREC/data/qa_{}.pickle', grid_sentence),
    'news_computer': ('newsgroups/data/news_computer_{}.p', grid_newsgroups),
    'news_religion': ('newsgroups/data/news_religion_{}.p', grid_newsgroups),
    'news_sports': ('newsgroups/data/news_sports_{}.p', grid_newsgroups),
}
NP_PATH = 'np_bracketing/data/npbracketing_{}{}.pickle'  # filled with e.g. ('train_X', fold)
NP_FOLDS = 10
DISCRIM_PATH = 'Discrim_Attr/data/discrim_attr_{}.p'
SPLITS = ('train', 'val', 'test')

# The score columns of a record, in table order
COLUMNS = ['word_analogy', 'sentiment', 'TREC', 'discrim_attr',
           'news_computer', 'news_religion', 'news_sports', 'np_bracketing', 'wordsim353']
# Tasks that run as jobs in the workers
TRAINED_TASKS = ('word_analogy', *SELECT_BY_VAL, 'np_bracketing')
# Cosine-only tasks: no classifier, always scored, in the main process
UNTRAINED_TASKS = ('discrim_attr', 'wordsim353')

# The trained tasks a variant runs (default: all of them). Rescaling only matters where a
# classifier with a fixed C sees the features: the cosine-based tasks give the same score as
# 'raw', and the large-C SVCs on TREC / sentiment are what makes 'scaled' slow.
# A task that a variant skips is None in the record.
VARIANT_TASKS = {'scaled': ('news_computer', 'news_religion', 'news_sports')}


def tasks_for(variant):
    """The trained tasks that `variant` runs."""
    return VARIANT_TASKS.get(variant, TRAINED_TASKS)


def jobs_per_run(variant):
    """How many jobs one run of `variant` has (the number that run_jobs yields)."""
    tasks = tasks_for(variant)
    n_jobs = 0
    if 'word_analogy' in tasks:
        n_jobs += 1
    for task, (_, grid) in SELECT_BY_VAL.items():
        if task in tasks:
            n_jobs += len(grid())
    if 'np_bracketing' in tasks:
        n_jobs += NP_FOLDS * len(grid_np())
    return n_jobs


# --- Task data and features -----------------------------------------------------
@lru_cache(maxsize=None)
def load_pickle(rel_path):
    with open(TASK_DIR / rel_path, 'rb') as f:
        return pickle.load(f)


def mean_feats(sentence, vectors, dim):
    """getFeats of the sentiment / TREC / newsgroups scripts: mean over the known lowercased words."""
    ret = np.zeros(dim)
    cnt = 0
    for word in sentence:
        if word.lower() in vectors:
            ret += vectors[word.lower()]
            cnt += 1
    if cnt > 0:
        ret /= cnt
    return ret


def concat_feats(phrase, vectors, dim):
    """getFeats of classify_bracketing.py: the word vectors concatenated, zeros for an unknown word."""
    ret = []
    for word in phrase:
        if word in vectors:
            ret.extend([v for v in vectors[word]])
        else:
            ret.extend([v for v in np.zeros(dim)])
    return np.array(ret)


def load_splits(path_template, feats, fold=None):
    """Features and labels of the train / val / test splits, as the scripts' loading loop builds them.
    An empty split gives [] for both. `fold` is only used by NP bracketing.
    Returns (X, y), each a list with one entry per split."""
    fold_args = () if fold is None else (fold,)
    X, y = [], []
    for split in SPLITS:
        texts = load_pickle(path_template.format(f'{split}_X', *fold_args))
        if len(texts) > 0:
            labels = load_pickle(path_template.format(f'{split}_y', *fold_args))
            X.append(np.array([feats(text) for text in texts]))
            y.append(np.array(labels))
        else:
            X.append([])
            y.append([])
    return X, y


# --- Jobs: run in the workers ---------------------------------------------------
def _log_fit(log_path, event, key, clf, seconds=''):
    """Append one tab-separated line to the fit-times log:
    time, event ('start' / 'end'), pid, model|variant|task|..., classifier, seconds."""
    if log_path is None:
        return
    run, *rest = key
    key_text = '|'.join(str(part) for part in (*run, *rest))
    clf_text = ' '.join(repr(clf).split())  # the repr on one line
    fields = [f'{time.time():.1f}', event, str(os.getpid()), key_text, clf_text, seconds]
    with open(log_path, 'a') as f:
        f.write('\t'.join(fields) + '\n')


def fit_and_score(key, clf, X, y, log_path=None):
    """One classifier job (the scripts' trainAndTest): fit on train, score on val (if there is
    one) and on test. Returns (key, val score or None, test score)."""
    warnings.simplefilter('ignore', ConvergenceWarning)
    _log_fit(log_path, 'start', key, clf)
    start = time.perf_counter()
    clf.fit(X[0], y[0])
    val_score = clf.score(X[1], y[1]) if len(X[1]) > 0 else None
    test_score = clf.score(X[2], y[2])
    _log_fit(log_path, 'end', key, clf, f'{time.perf_counter() - start:.1f}')
    return key, val_score, test_score


def analogy_job(key, words, E):
    """The word-analogy job, as in POLAR's main.ipynb: gensim's evaluate_word_analogies with its
    defaults (only the first 300000 words are used, questions with an unknown word are skipped).
    Returns (key, accuracy, number of questions scored)."""
    kv = gensim.models.KeyedVectors(vector_size=E.shape[1])
    kv.add_vectors(words, E)
    accuracy, sections = kv.evaluate_word_analogies(datapath('questions-words.txt'))
    total = sections[-1]
    return key, accuracy, len(total['correct']) + len(total['incorrect'])


def run_jobs(run, E, words, untrained, log_path=None, random_state=None, cache_mb=SVC_CACHE_MB):
    """Yields the jobs of one run, for joblib.

    A generator, so the features of a task are only built when joblib is about to dispatch its
    jobs. The untrained tasks need no job: they are scored here and stored in `untrained[run]`."""
    _, variant = run
    tasks = tasks_for(variant)
    vectors = {word: E[i] for i, word in enumerate(words)}
    dim = E.shape[1]
    mean = partial(mean_feats, vectors=vectors, dim=dim)
    concat = partial(concat_feats, vectors=vectors, dim=dim)
    prepare = partial(prepare_classifier, random_state=random_state, cache_mb=cache_mb)

    untrained[run] = untrained_scores(vectors)

    if 'word_analogy' in tasks:
        # gensim only reads the first ANALOGY_VOCAB rows, so only those are sent to the worker
        yield delayed(analogy_job)((run, 'analogy'), words[:ANALOGY_VOCAB], E[:ANALOGY_VOCAB])

    for task, (path, grid) in SELECT_BY_VAL.items():
        if task not in tasks:
            continue
        X, y = load_splits(path, mean)
        for k, clf in enumerate(grid()):
            yield delayed(fit_and_score)((run, task, k), prepare(clf), X, y, log_path)

    if 'np_bracketing' in tasks:
        for fold in range(NP_FOLDS):
            X, y = load_splits(NP_PATH, concat, fold)
            for k, clf in enumerate(grid_np()):
                yield delayed(fit_and_score)((run, 'np_bracketing', fold, k), prepare(clf), X, y, log_path)


# --- Untrained tasks: cosine only -----------------------------------------------
def sim(e1, e2):
    """Cosine similarity, written as in both upstream scripts."""
    return np.sum(e1 * e2) / (np.sqrt(np.sum(e1 * e1)) * np.sqrt(np.sum(e2 * e2)))


def discrim_attr_accuracy(vectors):
    """Discriminative attributes (classify_discrim_attr_TASK.py, main2), no training: for a triple
    (word1, word2, attribute), predict 1 if the attribute is closer to word1 than to word2.
    Triples with an unknown word are skipped. Returns (accuracy, number of triples scored)."""
    triples = load_pickle(DISCRIM_PATH.format('test_X'))
    labels = load_pickle(DISCRIM_PATH.format('test_y'))
    y_true, y_pred = [], []
    for i, triple in enumerate(triples):
        word1, word2, attribute = triple[0], triple[1], triple[2]
        if word1 not in vectors or word2 not in vectors or attribute not in vectors:
            continue
        closer_to_word1 = sim(vectors[word1], vectors[attribute]) > sim(vectors[word2], vectors[attribute])
        y_true.append(int(labels[i]))
        y_pred.append(1 if closer_to_word1 else 0)
    accuracy = accuracy_score(y_true, y_pred) if y_true else float('nan')
    return accuracy, len(y_true)


@lru_cache(maxsize=None)
def wordsim_data():
    """WordSim-353 as evaluate_wordSim.py reads it (loadTestData): (word pairs, human scores)."""
    lines = WORDSIM_PATH.read_text().splitlines()[1:]  # the first line is the header
    rows = [line.strip().split('\t') for line in lines]
    pairs = tuple((row[0], row[1]) for row in rows)
    human_scores = tuple(float(row[2]) for row in rows)
    return pairs, human_scores


def wordsim_rho(vectors):
    """WordSim-353 (SPINE's evaluate_wordSim.py): Spearman rho between cosine and human score, over
    the pairs with both words known (case-sensitive lookup). Returns (rho, number of pairs scored)."""
    pairs, human_scores = wordsim_data()
    predicted, gold = [], []
    for (word1, word2), human_score in zip(pairs, human_scores):
        if word1 in vectors and word2 in vectors:
            predicted.append(sim(vectors[word1], vectors[word2]))
            gold.append(human_score)
    if len(predicted) < 2:
        return float('nan'), len(predicted)
    return float(spearmanr(predicted, gold)[0]), len(predicted)


def untrained_scores(vectors, tasks=UNTRAINED_TASKS):
    """{task: (score, coverage)} for the tasks that need no classifier."""
    scores = {}
    if 'discrim_attr' in tasks:
        accuracy, n_scored = discrim_attr_accuracy(vectors)
        n_triples = len(load_pickle(DISCRIM_PATH.format('test_X')))
        scores['discrim_attr'] = (accuracy, {'triples_scored': n_scored / n_triples})
    if 'wordsim353' in tasks:
        rho, n_scored = wordsim_rho(vectors)
        n_pairs = len(wordsim_data()[0])
        scores['wordsim353'] = (rho, {'pairs_scored': n_scored / n_pairs})
    return scores


# --- Scoring a finished run -----------------------------------------------------
@lru_cache(maxsize=None)
def n_questions():
    """Number of questions in gensim's questions-words.txt (section headers start with ': ')."""
    with open(datapath('questions-words.txt')) as f:
        return sum(1 for line in f if not line.startswith(': ') and len(line.split()) == 4)


def select_by_val(outputs, run, task, n_clf):
    """The scripts' selection rule: the test score of the first classifier that reaches the highest
    validation score. Returns (test score, index of that classifier); (0.0, None) if no classifier
    has a validation score above 0."""
    best_val, best_test, best_k = 0.0, 0.0, None
    for k in range(n_clf):
        val, test = outputs[(run, task, k)]
        if val is not None and val > best_val:
            best_val, best_test, best_k = val, test, k
    return best_test, best_k


@lru_cache(maxsize=None)
def text_coverage(vocab):
    """How much of the trained tasks' texts a vocabulary covers: {task: {measure: fraction}}.
    `vocab` is a frozenset, so models that share a vocabulary share the cached result."""
    coverage = {}
    for task, (path, _) in SELECT_BY_VAL.items():
        texts = [text for split in SPLITS for text in load_pickle(path.format(f'{split}_X'))]
        known = [[word.lower() in vocab for word in text] for text in texts]
        coverage[task] = {
            'token_coverage': float(np.mean([is_known for text in known for is_known in text])),
            'texts_without_known_word': float(np.mean([not any(text) for text in known])),
        }
    # NP bracketing looks words up as they are; train + test of fold 0 is the whole dataset
    phrases = load_pickle(NP_PATH.format('train_X', 0)) + load_pickle(NP_PATH.format('test_X', 0))
    known = [[word in vocab for word in phrase] for phrase in phrases]
    coverage['np_bracketing'] = {
        'token_coverage': float(np.mean([is_known for phrase in known for is_known in phrase])),
        'phrases_fully_known': float(np.mean([all(phrase) for phrase in known])),
    }
    return coverage


def finish_run(run, outputs, untrained, models, stamp, random_state, extra=None):
    """Turns the job outputs of a finished run into its record, appends the record to
    RESULTS_PATH and returns it."""
    name, variant = run
    model = models[name]
    tasks = tasks_for(variant)
    scores = {}    # column -> score; None if the variant skips the task
    selected = {}  # task -> the classifier that was selected on validation

    # Word analogy
    if 'word_analogy' in tasks:
        scores['word_analogy'], n_analogy_scored = outputs[(run, 'analogy')]
    else:
        scores['word_analogy'], n_analogy_scored = None, None

    # Untrained tasks (already scored in run_jobs)
    for task, (score, _) in untrained[run].items():
        scores[task] = score

    # Tasks selected on validation
    for task, (_, grid) in SELECT_BY_VAL.items():
        if task not in tasks:
            scores[task] = selected[task] = None
            continue
        classifiers = grid()
        scores[task], best_k = select_by_val(outputs, run, task, len(classifiers))
        selected[task] = None if best_k is None else repr(classifiers[best_k])

    # NP bracketing: classify_bracketing.py prints the best test score over its grid,
    # and main.ipynb averages that over the folds
    np_folds = None
    scores['np_bracketing'] = None
    if 'np_bracketing' in tasks:
        np_folds = []
        for fold in range(NP_FOLDS):
            test_scores = [outputs[(run, 'np_bracketing', fold, k)][1] for k in range(len(grid_np()))]
            np_folds.append(float(max([0.0] + test_scores)))
        scores['np_bracketing'] = np.mean(np_folds)

    scores = {column: None if scores[column] is None else float(scores[column]) for column in COLUMNS}

    if n_analogy_scored is None:
        analogy_coverage = None
    else:
        analogy_coverage = n_analogy_scored / n_questions()
    coverage = {
        'word_analogy': {'questions_scored': analogy_coverage},
        **{task: task_coverage for task, (_, task_coverage) in untrained[run].items()},
        **text_coverage(frozenset(model['words'])),
    }

    record = {
        'time': stamp,
        'model': name,
        'model_path': model['path'],
        'config': model['config'],
        'variant': variant,
        'tasks': [task for task in COLUMNS if task in UNTRAINED_TASKS or task in tasks],
        'scores': scores,
        'np_bracketing_folds': np_folds,
        'selected_classifier': selected,
        'coverage': coverage,
        'random_state': random_state,
        # which state was loaded and how far the run got (None for a baseline)
        'checkpoint': model.get('checkpoint'),
        **(extra or {}),
    }
    append_record(record)
    return record


def append_record(record):
    with open(RESULTS_PATH, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def missing_untrained(record):
    """The UNTRAINED_TASKS that a record has no score for (it was written before they were added)."""
    return [task for task in UNTRAINED_TASKS if task not in record['scores']]


def backfill_record(record, E, words, tasks, stamp):
    """Appends a copy of `record` with `tasks` scored on E, and returns it.

    The copy keeps the record's 'time': finished_records then prefers it over the old line
    (same time, later in the file), and --resume-since still treats it like the old one."""
    vectors = {word: E[i] for i, word in enumerate(words)}
    new = json.loads(json.dumps(record))  # deep copy
    new.setdefault('coverage', {})
    for task, (score, task_coverage) in untrained_scores(vectors, tasks).items():
        new['scores'][task] = float(score)
        new['coverage'][task] = task_coverage
    # a record from before the 'tasks' field: the tasks it has a score for
    old_tasks = record.get('tasks', [task for task, score in record['scores'].items() if score is not None])
    new['tasks'] = [task for task in COLUMNS if task in old_tasks or task in tasks]
    new['backfilled'] = {**record.get('backfilled', {}), **{task: stamp for task in tasks}}
    append_record(new)
    return new


# --- Reading results back (for resuming, and for the notebooks) --------------------
def _modified_after(path, stamp):
    """True if the file at `path` changed after the ISO time `stamp` (False if either is unreadable)."""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime) > datetime.fromisoformat(stamp)
    except (OSError, ValueError, TypeError):
        return False


def finished_records(model_paths, random_state=None, since=None, results_path=RESULTS_PATH):
    """The record to reuse for each run: {(model, variant): its newest valid record}.

    A record is valid if
      - it was scored on the file the model is loaded from now (`model_paths[model]`),
      - with the same classifier seed (`random_state`),
      - not before `since` (an ISO time), when that is given,
      - and that file was not overwritten afterwards (the model file of a run still training).
    Lines without these fields (written by the first notebook version) are never reused."""
    newest = {}
    results_path = Path(results_path)
    if not results_path.exists():
        return newest
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not {'model', 'model_path', 'variant', 'time'} <= record.keys():
            continue
        if model_paths.get(record['model']) != record['model_path']:
            continue
        if record.get('random_state') != random_state:
            continue
        if since is not None and record['time'] < since:
            continue
        if _modified_after(record['model_path'], record['time']):
            continue
        run = (record['model'], record['variant'])
        # '>=': of two records with the same time the later line wins (a backfilled copy)
        if run not in newest or record['time'] >= newest[run]['time']:
            newest[run] = record
    return newest


def load_manifest(path=None, results_dir=RESULTS_DIR, prefix='sweep'):
    """A manifest, with its runs as tuples. Default: the newest `prefix`_*.json
    (prefix 'methods' for method_baselines.py)."""
    if path is None:
        found = sorted(Path(results_dir).glob(f'{prefix}_*.json'))
        if not found:
            raise FileNotFoundError(f'no {prefix}_*.json in {results_dir}; run the script first')
        path = found[-1]
    manifest = json.loads(Path(path).read_text())
    manifest['path'] = str(path)
    for key in ('runs', 'todo'):
        manifest[key] = [tuple(run) for run in manifest.get(key, [])]
    return manifest


def results_tables(manifest):
    """(scores, coverage) DataFrames for the manifest's runs that have a record.
    A task without a score (skipped by the variant, or not yet backfilled) is NaN."""
    import pandas as pd
    if 'models' not in manifest:  # the sweep failed, or is still loading its models
        return pd.DataFrame(columns=COLUMNS), pd.DataFrame()
    model_paths = {name: model['path'] for name, model in manifest['models'].items()}
    records = finished_records(model_paths, manifest['random_state'], manifest.get('resume_since'))
    runs = [run for run in manifest['runs'] if run in records]
    if not runs:
        return pd.DataFrame(columns=COLUMNS), pd.DataFrame()

    scores = pd.DataFrame({run: records[run]['scores'] for run in runs}).T
    scores = scores.reindex(columns=COLUMNS).astype(float)

    # coverage depends on the model, not on the variant: one row per model
    coverage_rows = {}
    for name, variant in runs:
        record_coverage = records[(name, variant)]['coverage']
        coverage_rows[name] = {(task, measure): value
                               for task in COLUMNS
                               for measure, value in record_coverage.get(task, {}).items()}
    coverage = pd.DataFrame(coverage_rows).T
    return scores, coverage


def checkpoint_table(manifest):
    """Per model, which state was loaded and how far the run got (eval_utils.checkpoint_info):
    the loaded state and its iteration, the best-semantic iteration (what the model file holds)
    and where that number comes from, the last checkpoint, the iteration cap, the run's status,
    and the primary semantic score at the best and at the last check.
    best_iteration (runs.jsonl) and fitness_best_iteration (fitness log) should agree for a finished run."""
    import pandas as pd
    rows = {}
    for name, model in manifest.get('models', {}).items():
        if model.get('checkpoint'):
            rows[name] = model['checkpoint']
    columns = ['source', 'iteration', 'best_iteration', 'best_from', 'last_checkpoint', 'max_iters', 'status',
               'sem_key', 'sem_best', 'sem_last', 'last_check', 'fitness_best_iteration', 'finished']
    return pd.DataFrame(rows).T.reindex(columns=columns)


def _log_lines(path):
    """The fields of each complete line of a fit-times log. A line that a worker is still
    writing is skipped."""
    with open(path) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if line.endswith('\n') and len(fields) == 6:
                yield fields


def fit_times(paths):
    """One row per finished fit in the given fit_times_*.log files."""
    import pandas as pd
    rows = []
    for path in paths:
        for _, event, _, key, clf, seconds in _log_lines(path):
            if event == 'end':
                model, variant, task, *_ = key.split('|')
                rows.append(dict(log=Path(path).name, model=model, variant=variant, task=task,
                                 clf=clf, seconds=float(seconds)))
    return pd.DataFrame(rows, columns=['log', 'model', 'variant', 'task', 'clf', 'seconds'])


def in_flight(path):
    """The fits that started but have not finished, longest-running first."""
    import pandas as pd
    started = {}
    for start_time, event, pid, key, clf, _ in _log_lines(path):
        if event == 'start':
            started[key] = (float(start_time), pid, clf)
        else:
            started.pop(key, None)
    now = time.time()
    rows = [dict(key=key, pid=pid, clf=clf, running_s=round(now - start_time))
            for key, (start_time, pid, clf) in started.items()]
    table = pd.DataFrame(rows, columns=['key', 'pid', 'clf', 'running_s'])
    return table.sort_values('running_s', ascending=False)


# --- Script -------------------------------------------------------------------
class _Tee:
    """Writes to several streams at once (the terminal and the sweep log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def _versions():
    import joblib
    import scipy
    import sklearn
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, gensim=gensim.__version__, joblib=joblib.__version__)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def build_parser(description='POLAR downstream evaluation of tensormet models.', table=True):
    """The options that every sweep has. `table` adds the ones that pick models from the model table."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--variants', nargs='+', default=VARIANTS, choices=sorted(VARIANT_FNS))
    p.add_argument('--only', nargs='+', default=None, metavar='PATTERN',
                   help="model names to run (shell-style patterns), e.g. 'tt_4g_r100_*' 'glove_*'")
    if table:
        p.add_argument('--ngrams', nargs='+', type=int, default=None,
                       help='table columns of these orders (default: all)')
        p.add_argument('--no-glove', action='store_true', help='skip the GloVe baselines')
        p.add_argument('--glove-widths', nargs='+', type=int, default=list(GLOVE_WIDTHS),
                       help='widths of the GloVe baselines glove_<w> / glove_nmf_<w> (default: %(default)s)')
        p.add_argument('--no-w2v', action='store_true', help='skip the word2vec baselines')
        p.add_argument('--baseline-vocabs', nargs='+', default=list(BASELINE_VOCABS), choices=BASELINE_VOCABS,
                       help="ours: the best model's words; own: the embedding's whole vocabulary (*_full)")
        p.add_argument('--series', default='latest', choices=sorted(SERIES),
                       help="latest: each run's latest checkpoint; best: the model file (best-semantic state); "
                            "same: the runs differing only in --same-factor, at their common checkpoint")
        p.add_argument('--same-factor', default='ss_frac', choices=SAME_FACTORS,
                       help='series same: the setting its matched runs differ in (default: %(default)s)')
    p.add_argument('--n-jobs', type=int, default=N_JOBS)
    p.add_argument('--random-state', type=int, default=None,
                   help='seed for MLP/RandomForest (the scripts leave them unseeded)')
    p.add_argument('--svc-cache-mb', type=int, default=SVC_CACHE_MB)
    p.add_argument('--no-resume', action='store_true', help='recompute runs already in the results log')
    p.add_argument('--resume-since', default=None, metavar='ISO_TIME',
                   help="only reuse records written after this time, e.g. 2026-09-23T17:00")
    p.add_argument('--progress-every', type=float, default=300, metavar='SECONDS')
    p.add_argument('--dry-run', action='store_true', help='load models and print the plan; fit nothing')
    p.add_argument('--backfill-only', action='store_true',
                   help='score finished runs missing an untrained task (e.g. wordsim353); fit nothing')
    return p


def _table_models(args):
    """The models of the model table that `args` asks for: (models, missing)."""
    if args.series == 'same':  # our runs only: a baseline has no checkpoints
        return load_same(args.same_factor, args.ngrams, only=args.only)
    return load_models(args.ngrams, only=args.only,
                       glove_widths=() if args.no_glove else args.glove_widths,
                       w2v=not args.no_w2v, baseline_vocabs=args.baseline_vocabs,
                       checkpoint=SERIES[args.series][0])


def _table_spec(args):
    """The manifest fields that describe which models were asked for."""
    spec = dict(
        series=args.series,
        checkpoint=SERIES[args.series][0],
        ngrams=args.ngrams,
        base=BASE,
        table={f'{decomposition}_{ngram}g': column for (decomposition, ngram), column in MODELS.items()},
        fixed=FIXED,
        glove_widths=[] if args.no_glove else args.glove_widths,
        w2v=not args.no_w2v,
        baseline_vocabs=args.baseline_vocabs,
    )
    if args.series == 'same':  # no baselines are loaded
        spec.update(same_factor=args.same_factor, glove_widths=[], w2v=False, baseline_vocabs=[])
    return spec


def main(argv=None, parser=None, load=_table_models, spec=_table_spec, prefix=None):
    """Run the sweep; returns the exit code.

    Another set of models can reuse it (see method_baselines.py) by passing its own `parser`,
    `load(args) -> (models, missing)`, `spec(args) -> manifest fields` and file `prefix`
    (default: the prefix of --series, see SERIES)."""
    args = (parser or build_parser()).parse_args(argv)
    for path in (TASK_DIR, WORDSIM_PATH):
        if not path.exists():
            sys.exit(f'{path} not found: clone the upstream repositories first, see {THIRD_PARTY_DIR / "README.md"}')
    if prefix is None:
        prefix = SERIES[args.series][1]
    stamp = datetime.now().isoformat(timespec='seconds')
    tag = stamp.replace(':', '')  # the stamp as used in file names

    # The files of this sweep
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / f'{prefix}_{tag}.json'
    sweep_log_path = RESULTS_DIR / f'{prefix}_{tag}.log'
    if prefix == 'sweep':
        fit_log_path = RESULTS_DIR / f'fit_times_{tag}.log'
    else:
        fit_log_path = RESULTS_DIR / f'fit_times_{prefix}_{tag}.log'

    # Everything printed also goes to the sweep log.
    # A kill or a closed terminal ends the sweep the way Ctrl-c does, so the manifest is updated.
    sweep_log = open(sweep_log_path, 'a', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, sweep_log)
    sys.stderr = _Tee(sys.__stderr__, sweep_log)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    if hasattr(signal, 'SIGHUP'):  # not on Windows
        signal.signal(signal.SIGHUP, _raise_interrupt)

    n_workers = min(args.n_jobs, os.cpu_count() or 1)
    manifest = dict(stamp=stamp, status='loading', argv=sys.argv, host=platform.node(), pid=os.getpid(),
                    variants=args.variants, only=args.only, **spec(args),
                    n_jobs=n_workers, random_state=args.random_state, svc_cache_mb=args.svc_cache_mb,
                    resume=not args.no_resume, resume_since=args.resume_since,
                    results_path=str(RESULTS_PATH), fit_times_log=str(fit_log_path), versions=_versions())

    def save_manifest(**updates):
        manifest.update(updates)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    start = time.perf_counter()

    def hours_elapsed():
        return (time.perf_counter() - start) / 3600

    save_manifest()
    print(f'[{stamp}] {prefix}: manifest {manifest_path}')
    n_runs_done = 0
    try:
        # 1. Load the models and build one embedding per run
        models, missing = load(args)
        describe_models(models, missing)
        embeddings, skipped = make_embeddings(models, args.variants)
        for run, reason in skipped.items():
            print('  skipped', run, reason)

        # 2. Resume: a run that already has a record is not fitted again
        if args.no_resume:
            records = {}
        else:
            model_paths = {name: model['path'] for name, model in models.items()}
            records = finished_records(model_paths, args.random_state, args.resume_since)

        # 3. Backfill: score the untrained tasks that a finished run's record does not have yet
        backfill = {}  # run -> tasks to add
        for run in embeddings:
            if run in records and missing_untrained(records[run]):
                backfill[run] = missing_untrained(records[run])
        if backfill:
            backfill_tasks = sorted({task for tasks in backfill.values() for task in tasks})
            print(f'\n{len(backfill)} finished runs lack an untrained task ({backfill_tasks}): '
                  'scored without refitting' + (' (not in a dry run)' if args.dry_run else ''))
            if not args.dry_run:
                for run, tasks in backfill.items():
                    name, variant = run
                    record = backfill_record(records[run], embeddings[run], models[name]['words'], tasks, stamp)
                    new_scores = '  '.join(f'{task}={record["scores"][task]:.3f}' for task in tasks)
                    print(f'  backfilled {name:40s} {variant:6s} {new_scores}')

        # 4. Plan: the runs still to fit, and how many jobs each has
        todo = [run for run in embeddings if run not in records]
        pending = {run: jobs_per_run(run[1]) for run in todo}  # run -> jobs not yet finished
        total_jobs = sum(pending.values())
        print(f'\n{len(embeddings)} runs ({len(models)} models x {args.variants}); '
              f'{len(embeddings) - len(todo)} already in {RESULTS_PATH.name}, {len(todo)} to go: '
              f'{total_jobs} jobs on {n_workers} workers')
        for name, variant in todo:
            print(f'  todo  {name:40s} {variant}')

        if args.dry_run:
            status = 'dry-run'
        elif args.backfill_only:
            status = 'backfill-only'
        else:
            status = 'running'
        save_manifest(
            status=status,
            backfilled=[] if args.dry_run else list(backfill),
            models={name: dict(path=model['path'], config=model['config'], n_words=len(model['words']),
                               dim=int(model['E_raw'].shape[1]), checkpoint=model.get('checkpoint'))
                    for name, model in models.items()},
            missing=missing,
            skipped={f'{name}|{variant}': reason for (name, variant), reason in skipped.items()},
            runs=list(embeddings),
            todo=todo,
            total_jobs=total_jobs,
            variant_tasks={variant: list(tasks_for(variant)) for variant in args.variants},
            jobs_per_run={variant: jobs_per_run(variant) for variant in args.variants},
        )
        if args.dry_run:
            print('nothing fitted (dry run)')
            return 0
        if args.backfill_only:
            print('nothing fitted (backfill only)')
            return 0
        if not todo:
            print('nothing fitted')
            save_manifest(status='finished', runs_finished=0, hours=round(hours_elapsed(), 3),
                          finished_at=datetime.now().isoformat(timespec='seconds'))
            return 0

        # 5. Fit: all jobs of all runs go to one pool; a run is scored when its last job is done
        outputs = {}    # job key -> what the job returned
        untrained = {}  # run -> the scores of its untrained tasks (filled by run_jobs)
        record_extra = {'sweep': manifest_path.name, 'series': manifest.get('series'),
                        'sklearn': manifest['versions']['sklearn']}
        jobs = itertools.chain.from_iterable(
            run_jobs(run, embeddings[run], models[run[0]]['words'], untrained, str(fit_log_path),
                     args.random_state, args.svc_cache_mb)
            for run in todo)
        try:  # results as soon as a job finishes
            parallel = Parallel(n_jobs=n_workers, batch_size=1, return_as='generator_unordered')
        except (TypeError, ValueError):  # joblib < 1.4: results only when every job is done
            parallel = Parallel(n_jobs=n_workers, batch_size=1, verbose=5)

        n_jobs_done = 0
        last_progress = time.perf_counter()
        for key, *result in parallel(jobs):
            run = key[0]
            outputs[key] = result
            n_jobs_done += 1
            pending[run] -= 1
            if pending[run] == 0:
                record = finish_run(run, outputs, untrained, models, stamp, args.random_state, record_extra)
                n_runs_done += 1
                scores = record['scores']
                scores_text = '  '.join(f'{column}={scores[column]:.3f}'
                                        for column in COLUMNS if scores[column] is not None)
                print(f'[{datetime.now():%H:%M:%S}] run {n_runs_done}/{len(todo)} done: '
                      f'{run[0]} {run[1]}  {scores_text}')
                save_manifest(runs_finished=n_runs_done)
            if time.perf_counter() - last_progress >= args.progress_every:
                last_progress = time.perf_counter()
                print(f'[{datetime.now():%H:%M:%S}] {n_jobs_done}/{total_jobs} jobs, '
                      f'{n_runs_done}/{len(todo)} runs, {hours_elapsed():.1f} h elapsed')

        print(f'done in {hours_elapsed():.2f} h; {n_runs_done} runs appended to {RESULTS_PATH}')
        save_manifest(status='finished', runs_finished=n_runs_done, hours=round(hours_elapsed(), 3),
                      finished_at=datetime.now().isoformat(timespec='seconds'))
        return 0
    except KeyboardInterrupt as e:
        print(f'\ninterrupted ({str(e) or "Ctrl-c"}); {n_runs_done} runs finished this session are saved')
        save_manifest(status='interrupted', runs_finished=n_runs_done, hours=round(hours_elapsed(), 3))
        return 130
    except Exception as e:
        save_manifest(status='failed', error=repr(e))
        raise


if __name__ == '__main__':
    sys.exit(main())
