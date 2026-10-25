"""Multi-word expression detection (paper section 3.1, pipeline 2).

The paper uses the jMWE index with NLTK's MWETokenizer. jMWE's index is built from WordNet,
so we use WordNet's multi-word lemmas (look_up_to, thank_you, talk_about, rolling_stock, ...)
as the MWE lexicon and match them over the parsed sentence's lemmas. Multi-word glosses from
the sign dataset (Madhya_Pradesh, How_Many, ...) can be added as trusted entries.
"""
from dataclasses import replace
from functools import lru_cache

WORDNET_MAX_LEN = 4
CONTENT_START = {"VERB", "NOUN", "PROPN", "ADJ", "ADV", "INTJ"}


@lru_cache(maxsize=1)
def lexicon():
    from nltk.corpus import wordnet as wn
    entries = set()
    for name in wn.all_lemma_names():
        if "_" in name:
            parts = tuple(name.lower().split("_"))
            if 2 <= len(parts) <= WORDNET_MAX_LEN and all(parts):
                entries.add(parts)
    return frozenset(entries)


@lru_cache(maxsize=4)
def _index(trusted):
    """first word -> [(entry, trusted)], longest entries first."""
    idx = {}
    for e in lexicon():
        idx.setdefault(e[0], []).append((e, e in trusted))
    for e in trusted:
        if e not in lexicon():
            idx.setdefault(e[0], []).append((e, True))
    for v in idx.values():
        v.sort(key=lambda x: (-len(x[0]), not x[1]))
    return idx


def _forms(tok):
    return {tok.lemma.lower(), tok.text.lower()}


def _matches(span, entry):
    return all(w in _forms(t) for t, w in zip(span, entry))


def _span_head(span, trusted):
    """Return the span's head if the span behaves as one syntactic unit, else None.

    Exactly one token may attach outside the span (the span head). A preposition attaching
    outside is allowed only when it is stranded ("what he is talking about") or follows a verb
    particle ("look up to his father"). This stops "go with me" from becoming GO_WITH.
    """
    ids = {t.id for t in span}
    # WordNet ADJ+NOUN / NUM+NOUN pairs ("blue_book") are left to rules R5/R6.
    # Dataset entries (Chief_Minister) are known signs, so they are always merged.
    if not trusted and any(t.head in ids and t.deprel in {"amod", "nummod"} for t in span):
        return None
    has_particle = any(t.deprel == "compound:prt" for t in span)
    outside = []
    for t in span:
        if t.head in ids:
            continue
        if t is not span[0] and (t.upos == "PUNCT" or
                                 (t.upos == "ADP" and (t.head < t.id or has_particle))):
            continue
        outside.append(t)
    if trusted and outside:
        # Known sign names only need to be contiguous; this also survives misparses such as
        # "Madhya" and "Pradesh" both attached to the verb. English phrase heads are final.
        return outside[-1]
    return outside[0] if len(outside) == 1 else None


def merge_mwes(tokens, trusted=frozenset()):
    """Merge MWE spans into single nodes. Returns (new_tokens, list_of_found_mwes).
    Dataset (trusted) entries are matched first, so "over and over again" is not
    broken up by the WordNet entry "knock over"."""
    idx = _index(trusted)
    tokens = list(tokens)
    found = []
    for trusted_pass in (True, False):
        i = 0
        while i < len(tokens):
            first = tokens[i]
            cands = sorted({c for f in _forms(first) for c in idx.get(f, [])},
                           key=lambda x: (-len(x[0]), not x[1]))
            for entry, is_trusted in cands:
                if trusted_pass and not is_trusted:
                    continue
                if len(entry) > len(tokens) - i or (not is_trusted and first.upos not in CONTENT_START):
                    continue
                span = tokens[i:i + len(entry)]
                if any(t.mwe for t in span):
                    continue
                head = _matches(span, entry) and _span_head(span, is_trusted)
                if head:
                    tokens = _collapse(tokens, span, entry, head)
                    found.append("_".join(entry).upper())
                    break
            i += 1
    return tokens, found


def _collapse(tokens, span, entry, head):
    ids = {t.id for t in span}
    head.parts = [replace(t) for t in span]      # kept so an unsigned MWE can be split again
    head.text = " ".join(t.text for t in span)
    head.gloss = "_".join(entry).upper()
    head.mwe = True
    removed = ids - {head.id}
    out = []
    for t in tokens:
        if t.id in removed:
            continue
        if t.head in removed:
            t.head = head.id
        out.append(t)
    return out
