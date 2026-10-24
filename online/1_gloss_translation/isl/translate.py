"""English -> ISL gloss translation (Ghosh & Mamidi, ICON 2022).

Usage:
    python -m isl.translate "He has a blue book" [--trace] [--glosses isl_glosses.txt]

With a gloss list (the dataset's sign folder names), every gloss is resolved to a sign folder;
words with no matching sign are fingerspelled as fs-WORD.
"""
import argparse
from pathlib import Path

from .lexicon import Lexicon, sentence_context
from .mwe import merge_mwes
from .parse import parse
from .postprocess import gloss_of, postprocess
from .rules import linearize

DEFAULT_GLOSSES = Path(__file__).resolve().parent.parent / "isl_glosses.txt"


def translate_detailed(text, lexicon=None):
    """Run the full pipeline; returns a list of per-sentence stage dicts."""
    results = []
    for tokens in parse(text):
        stage = {"parse": [(t.id, t.text, t.lemma, t.upos, t.head, t.deprel) for t in tokens]}
        merged, mwes = merge_mwes(tokens, lexicon.mwe_entries() if lexicon else frozenset())
        stage["mwe"] = mwes
        ordered = linearize(merged)
        stage["after_rules"] = [t.gloss if t.mwe else t.text for t in ordered]
        glosses, info = postprocess(ordered, merged)
        stage["post"] = info
        stage["signs"] = []
        if lexicon:
            glosses = _to_signs(info["tokens"], glosses, lexicon, sentence_context(tokens), stage["signs"])
        stage["glosses"] = glosses
        results.append(stage)
    return results


def _to_signs(tokens, glosses, lexicon, context, log):
    """Resolve each gloss to a sign folder; unsigned words are fingerspelled (fs-WORD).
    A WordNet MWE with no sign (COME_BACK) is split back into its words, which are resolved one by one."""
    out = []
    for tok, g in zip(tokens, glosses):
        sign = lexicon.resolve(tok, context)
        if sign is None and tok.mwe and tok.parts:
            parts = [p for p in tok.parts if p.upos != "PUNCT"]
            out += _to_signs(parts, [gloss_of(p) for p in parts], lexicon, context, log)
            continue
        out.append(sign.gloss if sign else f"fs-{g}")
        log.append((out[-1], sign.folder if sign else None))
    return out


def translate(text, lexicon=None):
    return [g for s in translate_detailed(text, lexicon) for g in s["glosses"]]


def _print_trace(stage):
    print("  parse:")
    for row in stage["parse"]:
        print("    {:>2} {:<12} {:<12} {:<6} {:>2} {}".format(*row))
    print("  MWEs:        ", stage["mwe"] or "-")
    print("  after rules: ", " ".join(stage["after_rules"]))
    p = stage["post"]
    print("  question:    ", p["question"], "| wh:", p["wh"] or "-", "| neg:", p["negation"] or "-")
    print("  dropped:     ", p["dropped"] or "-")
    if stage["signs"]:
        print("  signs:       ", ", ".join(f"{g}={f or 'fingerspell'}" for g, f in stage["signs"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="+")
    ap.add_argument("--trace", action="store_true", help="print every pipeline stage")
    ap.add_argument("--glosses", default=str(DEFAULT_GLOSSES),
                    help="sign folder list; resolves glosses to signs (skipped if file missing)")
    ap.add_argument("--no-glosses", action="store_true", help="paper rules only, no sign lookup")
    args = ap.parse_args()
    lexicon = None if args.no_glosses else Lexicon.load(args.glosses)
    for sentence in args.sentence:
        stages = translate_detailed(sentence, lexicon)
        print(f"{sentence}  ->  {' '.join(g for s in stages for g in s['glosses'])}")
        if args.trace:
            for s in stages:
                _print_trace(s)


if __name__ == "__main__":
    main()
