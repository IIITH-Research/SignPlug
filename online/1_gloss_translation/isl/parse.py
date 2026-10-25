"""Stanza pre-processing (paper section 3.1, pipeline 1)."""
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Token:
    id: int
    text: str
    lemma: str
    upos: str
    head: int          # 0 = root
    deprel: str
    feats: dict = field(default_factory=dict)
    gloss: str = ""    # filled during post-processing
    mwe: bool = False  # True if this node is a merged multi-word expression
    parts: list = None # original tokens of a merged MWE

    def __repr__(self):
        return f"{self.id}:{self.text}/{self.upos}/{self.deprel}->{self.head}"


@lru_cache(maxsize=1)
def _pipeline():
    import stanza
    return stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                           verbose=False, download_method=None)


def _feats(s):
    if not s:
        return {}
    return dict(kv.split("=", 1) for kv in s.split("|"))


def parse(text):
    """Return a list of sentences, each a list of Token (one per syntactic word)."""
    doc = _pipeline()(text)
    sentences = []
    for sent in doc.sentences:
        sentences.append([
            Token(id=w.id, text=w.text, lemma=(w.lemma or w.text), upos=w.upos,
                  head=w.head, deprel=w.deprel, feats=_feats(w.feats))
            for w in sent.words
        ])
    return sentences
