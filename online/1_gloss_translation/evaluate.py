"""Evaluate the translator on testcases.json.

Usage: python evaluate.py [--cases testcases.json] [--glosses isl_glosses.txt]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from isl.lexicon import Lexicon
from isl.translate import translate

HERE = Path(__file__).resolve().parent


def wer(ref, hyp):
    """Word error rate over gloss tokens (Levenshtein distance / reference length)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(len(ref), 1)


def run(cases, lexicon=None):
    rows = []
    for c in cases:
        pred = translate(c["sentence"], lexicon)
        ref = c["expected"].split()
        rows.append({**c, "predicted": " ".join(pred), "ok": pred == ref, "wer": wer(ref, pred)})
    return rows


def _acc(rows):
    return f"{sum(r['ok'] for r in rows)}/{len(rows)} ({100 * sum(r['ok'] for r in rows) / len(rows):.1f}%)" if rows else "-"


def report(rows):
    w = max(len(r["sentence"]) for r in rows)
    print(f"{'id':<4} {'sentence':<{w}}  {'expected':<28} {'predicted':<28} ok")
    print("-" * (w + 66))
    for r in rows:
        mark = "OK" if r["ok"] else ("LIMIT" if r.get("known_limitation") else "FAIL")
        print(f"{r['id']:<4} {r['sentence']:<{w}}  {r['expected']:<28} {r['predicted']:<28} {mark}")

    main = [r for r in rows if not r.get("known_limitation")]
    limits = [r for r in rows if r.get("known_limitation")]
    print("\nExact match")
    print(f"  all (excl. known limitations): {_acc(main)}")
    print(f"  paper:                         {_acc([r for r in main if r['source'] == 'paper'])}")
    print(f"  constructed:                   {_acc([r for r in main if r['source'] == 'constructed'])}")
    print(f"  known limitations:             {_acc(limits)}")
    print(f"Mean gloss WER: {sum(r['wer'] for r in main) / len(main):.3f}")

    by_rule = defaultdict(list)
    for r in main:
        by_rule[r["rule"]].append(r)
    print("\nPer rule")
    for rule in sorted(by_rule):
        print(f"  {rule:<8} {_acc(by_rule[rule])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "testcases.json"))
    ap.add_argument("--glosses", default=None, help="resolve glosses to signs using this folder list")
    args = ap.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report(run(cases, Lexicon.load(args.glosses) if args.glosses else None))


if __name__ == "__main__":
    main()
