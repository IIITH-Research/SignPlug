"""Grammar transfer rules (paper section 3.2).

The dependency tree is linearized recursively. For each head, its children are placed left or
right according to the paper's rules; any relation the paper does not mention keeps its
original side of the head.

  R1  VERB -> NOUN(obj)          [NOUN][VERB]        all nominals go left of the verb (SOV)
  R2  VERB -> AUX(aux)           [AUX][VERB]
  R3  VERB -> ADV(advmod)        [VERB][ADV]
  R4  VERB -> VERB(any)          [VERB][VERB]        clauses follow, in original order
  R5  NOUN -> ADJ(amod)          [NOUN][ADJ]
  R6  NOUN -> NUM(nummod)        [NOUN][NUM]
  R7  NOUN -> ANY(acl:relcl)     [NOUN][CLAUSE]
"""
from collections import defaultdict

VERBAL_HEADS = {"VERB", "AUX"}

# R1. ccomp is a clause acting as the object ("I know [what he is talking about]"), so like
# any object it goes left of the verb: paper example -> I HE TALK_ABOUT KNOW.
NOMINAL_RELS = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:tmod", "obl:npmod",
                "obl:agent", "expl", "ccomp"}
AUX_RELS = {"aux", "aux:pass", "cop"}                                          # R2
CLAUSE_RELS = {"xcomp", "advcl", "conj", "parataxis", "csubj"}                 # R4


def _verb_slot(child):
    rel = child.deprel
    if rel in NOMINAL_RELS:
        return "nominal"
    if rel in AUX_RELS:
        return "aux"
    if rel == "advmod":
        return "adv"
    if rel in CLAUSE_RELS or child.upos == "VERB":
        return "clause"
    return None


def _noun_slot(child):
    rel = child.deprel
    if rel == "amod":
        return "amod"                       # R5
    if rel == "nummod":
        return "nummod"                     # R6
    if rel.startswith("acl"):
        return "acl"                        # R7
    if rel == "case":
        return "case"                       # postposition: "with me" -> I WITH
    return None


VERB_LEFT = ["nominal", "aux"]
VERB_RIGHT = ["adv", None, "clause"]        # None = unclassified, originally on the right
NOUN_RIGHT = ["amod", "nummod", None, "acl", "case"]


def linearize(tokens):
    """Return the tokens reordered by the transfer rules."""
    by_id = {t.id: t for t in tokens}
    children = defaultdict(list)
    roots = []
    for t in tokens:
        if t.head == 0 or t.head not in by_id:
            roots.append(t)
        else:
            children[t.head].append(t)

    def visit(head):
        kids = sorted(children[head.id], key=lambda t: t.id)
        verbal = head.upos in VERBAL_HEADS
        slot_of = _verb_slot if verbal else _noun_slot
        right_order = VERB_RIGHT if verbal else NOUN_RIGHT
        left_order = VERB_LEFT if verbal else []

        groups = defaultdict(list)
        orig_left = []
        for k in kids:
            slot = slot_of(k)
            if slot is None:
                if k.id < head.id:
                    orig_left.append(k)
                else:
                    groups[None].append(k)
            else:
                groups[slot].append(k)

        out = []
        for k in orig_left:
            out += visit(k)
        for slot in left_order:
            for k in groups[slot]:
                out += visit(k)
        out.append(head)
        for slot in right_order:
            for k in groups[slot]:
                out += visit(k)
        return out

    result = []
    for r in sorted(roots, key=lambda t: t.id):
        result += visit(r)
    return result
