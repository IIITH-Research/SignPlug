"""ISL sign lexicon built from the dataset's gloss folder names (sign_annotation/raw).

Each output gloss is resolved to a sign folder:
  1. meaning-labeled sign (e.g. Scheme_(Govt_Scheme)) if the word's POS and context words match
  2. plain sign for the word (variants like Train_(Sign_2) count as plain; either works)
  3. synonym-group sign (e.g. Argue_Dispute_Quarrel for "quarrel"), only in the same meaning
  4. otherwise fingerspelled: fs-WORD
True multi-word glosses (Thank_You, Madhya_Pradesh) are added to the MWE lexicon so the paper's
MWE rule merges them into one gloss.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).with_name("lexicon_config.json")
SIGN_VARIANT = re.compile(r"^Sign_\d+$", re.I)


@dataclass(frozen=True)
class Sign:
    folder: str
    gloss: str
    kind: str = "sign"          # sign | sense | synonym
    upos: tuple = ()            # allowed POS (empty = any)
    cues: tuple = ()            # context words, at least one must occur (empty = none needed)

    def fits(self, tok, context):
        if self.upos and not (_pos_of(tok) & set(self.upos)):
            return False
        return not self.cues or bool(context & set(self.cues))


NOMINAL_ROLES = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:tmod", "obl:npmod"}


def _pos_of(tok):
    """POS tag, plus NOUN when the word fills a noun role (Stanza tags "Wait a second" ADJ)."""
    return {tok.upos, "NOUN"} if tok.deprel in NOMINAL_ROLES else {tok.upos}


def parse_folder(folder):
    """'Bahujan_Samaj_Party_(BSP)' -> (['Bahujan','Samaj','Party'], ['BSP']).
    Variant / regional labels are dropped; ALL-CAPS labels and trailing parts become aliases."""
    name, aliases = folder, []
    while True:
        m = re.search(r"_?\(([^()]*)\)$", name)
        if not m:
            break
        label = m.group(1)
        if label.isupper() and not SIGN_VARIANT.match(label):
            aliases.append(label)
        name = name[:m.start()]
    name = re.sub(r"_Sign_\d+$", "", name)
    name = name.replace("(s)", "")
    parts = [p for p in name.split("_") if p]
    while len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) > 1:
        aliases.append(parts.pop())
    return parts, aliases


class Lexicon:
    def __init__(self, folders, config=CONFIG):
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        cues = {k: tuple(v) for k, v in cfg["cue_sets"].items()}
        self.index = {}             # key tuple of lowercase words -> [Sign]
        excluded = set(cfg.get("exclude", {}).get("folders", []))
        self.folders = [f for f in folders if f not in excluded]

        def add(words, sign):
            key = tuple(words.lower().split()) if isinstance(words, str) else tuple(w.lower() for w in words)
            self.index.setdefault(key, []).append(sign)
            if len(key) == 1:
                for base in _base_forms(key[0]):
                    if base != key[0]:
                        self.index.setdefault((base,), []).append(sign)

        for folder in self.folders:
            if folder in cfg["senses"]:
                s = cfg["senses"][folder]
                add(s["word"], Sign(folder, s["word"].upper(), "sense", tuple(s.get("upos", ())), cues.get(s.get("cues"), ())))
            elif folder in cfg["synonym_groups"]:
                members = cfg["synonym_groups"][folder]
                first = members[0] if isinstance(members[0], str) else members[0]["word"]
                gloss = first.upper().replace(" ", "_")
                for m in members:
                    m = {"word": m} if isinstance(m, str) else m
                    add(m["word"], Sign(folder, gloss, "synonym", tuple(m.get("upos", ())), cues.get(m.get("cues"), ())))
            elif folder in cfg.get("label_only", {}):
                forms = cfg["label_only"][folder]
                sign = Sign(folder, forms[0].upper().replace(" ", "_"))
                for form in forms:
                    add(form, sign)
            elif folder in cfg["single_tokens"]:
                s = cfg["single_tokens"][folder]
                for form in s["forms"]:
                    add(form, Sign(folder, s["gloss"]))
            else:
                parts, aliases = parse_folder(folder)
                sign = Sign(folder, "_".join(parts).upper())
                add(parts, sign)
                for a in aliases:
                    add(a, sign)

        for folder, extra in cfg.get("extra_forms", {}).items():
            if folder not in self.folders:
                continue
            if isinstance(extra, dict):         # misspelt folder name: corrected gloss
                sign, words = Sign(folder, extra["gloss"]), extra["forms"]
            else:
                sign, words = Sign(folder, "_".join(parse_folder(folder)[0]).upper()), extra
            for w in words:
                if isinstance(w, dict):         # label word used only in one meaning
                    add(w["word"], Sign(folder, sign.gloss, "sense", tuple(w.get("upos", ())),
                                        cues.get(w.get("cues"), ())))
                else:
                    add(w, sign)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls(l.strip() for l in lines if l.strip())

    def mwe_entries(self):
        """Multi-word keys, handed to the MWE detector as trusted entries."""
        return frozenset(k for k in self.index if len(k) > 1)

    def resolve(self, tok, context):
        """Return the Sign for a translated token, or None if it must be fingerspelled."""
        if tok.mwe:
            keys = [tuple(tok.gloss.lower().split("_"))]
        else:
            keys = [(tok.lemma.lower(),), (tok.text.lower(),)]
        # The word itself is not evidence for its own meaning.
        context = context - {tok.lemma.lower(), tok.text.lower()}
        cands = []
        for k in keys:
            for s in self.index.get(k, []):
                if s not in cands:
                    cands.append(s)
        for kind in ("sense", "sign", "synonym"):
            for s in cands:
                if s.kind == kind and s.fits(tok, context):
                    return s
        return None


def _base_forms(word):
    """Dictionary forms of a folder word, so 'Numbers' also matches the lemma 'number'.
    Short function words are left alone ('its' must not become 'it')."""
    if len(word) <= 3:
        return set()
    from nltk.corpus import wordnet as wn
    return {b for pos in ("n", "v", "a", "r") if (b := wn.morphy(word, pos))}


def sentence_context(tokens):
    return {t.lemma.lower() for t in tokens} | {t.text.lower() for t in tokens}
