"""Post-processing (paper section 3.3): questions, negation, stopwords, lemmatization."""

WH_WORDS = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how"}
NEGATIONS = {"not": "NOT", "n't": "NOT", "never": "NEVER", "no": "NO"}

# Paper's stopword list is not published; this one follows its examples (articles, copulas
# and tense auxiliaries dropped; prepositions and meaningful modals kept).
ARTICLES = {"a", "an", "the"}
DROPPED_AUX = {"be", "do", "have", "will", "shall", "would"}
DROPPED_MARKERS = {"to", "that"}
# Relations on the path to the root that mean a wh-word sits in an embedded clause.
EMBEDDING_RELS = {"advcl", "acl", "acl:relcl", "ccomp", "csubj"}


def _is_wh(t):
    return t.text.lower().split()[0] in WH_WORDS      # also MWEs like HOW_MANY


def _is_negation(t):
    return t.text.lower() in NEGATIONS or (t.lemma.lower() == "not" and t.upos == "PART")


def _in_main_clause(t, by_id):
    while t.head in by_id:
        if t.deprel in EMBEDDING_RELS:
            return False
        t = by_id[t.head]
    return t.deprel not in EMBEDDING_RELS


def is_question(tokens):
    """Ends with '?', starts with a main-clause wh-word, or starts with an inverted auxiliary."""
    words = [t for t in tokens if t.upos != "PUNCT"]
    if not words:
        return False
    if tokens[-1].text == "?":
        return True
    by_id = {t.id: t for t in tokens}
    first = words[0]
    if _is_wh(first) and first.feats.get("PronType") != "Rel" and _in_main_clause(first, by_id):
        return True
    if first.upos == "AUX":
        return any(t.deprel.startswith("nsubj") and t.id > first.id for t in tokens)
    return False


def _is_stopword(t):
    lemma = t.lemma.lower()
    if t.upos == "PUNCT":
        return True
    if t.upos == "DET" and lemma in ARTICLES:
        return True
    if (t.upos == "AUX" or t.deprel == "cop") and lemma in DROPPED_AUX:
        return True
    if lemma in DROPPED_MARKERS and (t.upos in {"PART", "SCONJ"} or t.deprel == "mark"):
        return True
    if t.feats.get("PronType") == "Rel":      # relativizers: which / that / who in relative clauses
        return True
    return False


def gloss_of(t):
    if t.mwe:
        return t.gloss
    if t.text.lower() in NEGATIONS:
        return NEGATIONS[t.text.lower()]
    if t.upos == "PRON" and t.feats.get("Poss") == "Yes":
        return t.text.upper()                  # MY / YOUR / HIS, not the lemma I / YOU / HE
    if t.upos == "PROPN":
        return t.text.upper()
    return t.lemma.upper()


def postprocess(ordered, original):
    """ordered: tokens after grammar rules; original: tokens in sentence order.

    Returns (glosses, info) where info records what each step did.
    """
    by_id = {t.id: t for t in original}
    question = is_question(original)
    body, negs, whs, dropped = [], [], [], []
    for t in ordered:
        if _is_negation(t):
            negs.append(t)
        elif _is_wh(t):
            if question and t.feats.get("PronType") != "Rel" and _in_main_clause(t, by_id):
                whs.append(t)
            else:
                dropped.append(t)              # wh-word used as a conjunction/relativizer
        elif _is_stopword(t):
            dropped.append(t)
        else:
            body.append(t)
    kept = body + negs + whs
    glosses = [gloss_of(t) for t in kept]
    info = {
        "question": question,
        "negation": [gloss_of(t) for t in negs],
        "wh": [gloss_of(t) for t in whs],
        "dropped": [t.text for t in dropped],
        "tokens": kept,
    }
    return glosses, info
