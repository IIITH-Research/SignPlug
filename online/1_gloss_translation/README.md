# 1. Gloss translation (demo translator)

The paper obtains gloss sequences from an LLM; any LLM can be used for that step. For the live demo we
use this rule-based **demo translator**, which implements the grammar transfer rules of

> Abhigyan Ghosh and Radhika Mamidi. 2022. *English To Indian Sign Language: Rule-Based Translation System
> Along With Multi-Word Expressions and Synonym Substitution.* In Proceedings of the 19th International
> Conference on Natural Language Processing (ICON), 123–127.

| Module | Paper section |
|---|---|
| `isl/parse.py` | Stanza tokenization, POS, lemmas, dependency parse (§3.1) |
| `isl/mwe.py` | multi-word expressions from WordNet multi-word lemmas (§3.1) |
| `isl/rules.py` | reordering rules R1–R7 (§3.2) |
| `isl/postprocess.py` | wh-words and negation to the end, stopwords, lemmas (§3.3) |
| `isl/lexicon.py` + `lexicon_config.json` | maps glosses to sign folders: synonyms, meaning-labelled signs, variants; unknown words are fingerspelled |

```bash
pip install -r requirements.txt
python -c "import stanza, nltk; stanza.download('en'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python -m isl.translate "He is coming home" --trace --no-glosses
python evaluate.py                # test cases from the paper's examples + our own
python -m pytest tests -q
```

`isl.translate` maps glosses to signs when given a list of sign folder names (`--glosses`, one per line).
Without it, it prints the plain glosses from the rules.
