import logging
import re
from typing import List

from . import config


def setup_logging():
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_word_re = re.compile(r"[A-Za-z0-9']+")
_stop = {
    # minimal English stopword set
    "a","an","the","and","or","but","if","then","else","of","in","on","for","to","with","by","from","at","as","is","are","was","were","be","been","being"
}


def tokenize(text: str) -> List[str]:
    # Lowercase, extract words, drop stopwords
    toks = [m.group(0).lower() for m in _word_re.finditer(text)]
    return [t for t in toks if t not in _stop]

