"""Fetch and cache the Master Metaphor List's source/target concept words.

The Master Metaphor List (Lakoff, Espenson & Schwartz) enumerates the source
and target concepts behind common English metaphors (e.g. "container",
"journey", "vulnerability"). Those words are exactly the ones we care about
having a coherent nearest-neighbour structure, so they're used as
`query_words` for `DimConsistencyJudge.score_similarity_consistency` (see
`judge_eval.evaluate_similarity_run`) rather than an arbitrary vocab sample.

Network access + parsing only happens in `fetch_concepts`; importing this
module is free, and `load_concepts` reuses the cached pickle once one exists.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from tensormet.utils import DATA_DIR

MML_URLS = [
    "https://www.lang.osaka-u.ac.jp/~sugimoto/MasterMetaphorList/sources/index.html",
    "https://www.lang.osaka-u.ac.jp/~sugimoto/MasterMetaphorList/targets/index.html"
]
DEFAULT_SAVE_PATH = DATA_DIR / "corpora" / "mml.pkl"


def fetch_concepts(urls: list[str] = MML_URLS) -> list[str]:
    """Scrape the Master Metaphor List's index page for its concept words.

    Each <li> entry is a link to a page named like "Love_Is_A_Journey.html" or
    "Anger(1).html"; splitting on '.', '_', '(', ')', '-' and lowercasing turns
    those filenames into individual concept words. The first <li> is the page's
    own title (not a concept), so it's skipped.
    """

    concepts = set()
    for url in urls:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        for el in soup.find_all("li")[1:]:
            t = el.text.strip()
            parts = re.split(r"[\._\(\)\-]+", t.lower().replace(".html", ""))
            concepts.update(parts)
    concepts.discard("")  # stray empty splits (e.g. trailing "-" or "()")
    return sorted(concepts)


def save_concepts(concepts: list[str], save_path: Path = DEFAULT_SAVE_PATH) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(concepts, f)
    return save_path


def load_concepts(save_path: Path = DEFAULT_SAVE_PATH, refresh: bool = False) -> list[str]:
    """Load the cached concept list, scraping and caching it on first use.

    Pass `refresh=True` to re-scrape the source page even if a cached pickle
    already exists (e.g. after the Master Metaphor List site is updated).
    """
    save_path = Path(save_path)
    if refresh or not save_path.exists():
        concepts = fetch_concepts()
        save_concepts(concepts, save_path)
        return concepts
    with open(save_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    concepts = load_concepts(refresh=True)
    print(f"saved {len(concepts)} concepts to {DEFAULT_SAVE_PATH}")
    print(concepts)
