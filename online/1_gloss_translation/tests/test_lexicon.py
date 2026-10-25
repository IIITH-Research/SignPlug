"""Sign lookup against the dataset's gloss folders (isl_glosses.txt, copied from sign_annotation/raw)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isl.lexicon import Lexicon, parse_folder  # noqa: E402
from isl.translate import translate_detailed  # noqa: E402

LEX = Lexicon.load(ROOT / "isl_glosses.txt")
pytestmark = pytest.mark.skipif(LEX is None, reason="isl_glosses.txt not present")


def signs(sentence):
    return [pair for s in translate_detailed(sentence, LEX) for pair in s["signs"]]


def folder_for(sentence, gloss):
    return dict(signs(sentence)).get(gloss)


@pytest.mark.parametrize("folder, parts, aliases", [
    ("Train_(Sign_2)", ["Train"], []),
    ("Area_Sign_2", ["Area"], []),
    ("Bahujan_Samaj_Party_(BSP)", ["Bahujan", "Samaj", "Party"], ["BSP"]),
    ("Scheduled_Caste(s)_(SC)", ["Scheduled", "Caste"], ["SC"]),
    ("All_India_Trinamool_Congress_AITC_TMC", ["All", "India", "Trinamool", "Congress"], ["TMC", "AITC"]),
    ("Numbers_(Mumbai_Maharashtra)", ["Numbers"], []),
])
def test_parse_folder(folder, parts, aliases):
    assert parse_folder(folder) == (parts, aliases)


# --- meaning-labeled signs: used only when POS and context words match, else fingerspell/plain sign
def test_sense_matches_context():
    assert folder_for("The government launched a new scheme for farmers", "SCHEME") == "Scheme_(Govt_Scheme)"


def test_sense_mismatch_is_fingerspelled():
    pairs = dict(signs("He made a scheme to cheat his friend"))
    assert pairs.get("fs-SCHEME", "missing") is None and "SCHEME" not in pairs


def test_sense_falls_back_to_plain_sign():
    assert folder_for("I like mangoes", "LIKE") == "Like"
    assert folder_for("I liked your photo on Instagram", "LIKE") == "Like_(Social_Media)_(Sign_2)"


def test_cricket_bat():
    assert folder_for("He hit the ball with the bat in the cricket match", "BAT") == "Bat_(Cricket)"


# --- synonym groups: same meaning only
def test_synonym_group():
    assert folder_for("They quarrel every day", "ARGUE") == "Argue_Dispute_Quarrel"


def test_multiword_synonym_member():
    assert folder_for("The chief minister announced a general strike", "BANDH") == "Bandh_General_strike"


# --- variants: either works
def test_variant():
    assert folder_for("I tried", "TRY") == "Try_(Sign_2)"


# --- true multi-word glosses: merged by the MWE rule
@pytest.mark.parametrize("sentence, gloss, folder", [
    ("Thank you", "THANK_YOU", "Thank_You"),
    ("I live in Madhya Pradesh", "MADHYA_PRADESH", "Madhya_Pradesh"),
    ("The chief minister announced a general strike", "CHIEF_MINISTER", "Chief_Minister"),
])
def test_dataset_mwe(sentence, gloss, folder):
    assert folder_for(sentence, gloss) == folder


def test_wh_mwe_moves_to_end():
    out = [g for s in translate_detailed("How many books do you have?", LEX) for g in s["glosses"]]
    assert out[-1] == "HOW_MANY"


def test_unsigned_wordnet_mwe_is_split():
    out = [g for s in translate_detailed("He will come back", LEX) for g in s["glosses"]]
    assert "fs-COME_BACK" not in out


# --- names that look multi-word but are one sign
@pytest.mark.parametrize("sentence, gloss, folder", [
    ("Put a band-aid on the cut", "BAND-AID", "Band-Aid"),
    ("I have 2 brothers", "TWO", "2_Two"),
])
def test_single_token_signs(sentence, gloss, folder):
    assert folder_for(sentence, gloss) == folder


# --- signs whose video shows more words than the folder name (pics/<folder>)
def test_not_uses_no_sign():
    assert folder_for("He is not a doctor", "NO") == "no"


@pytest.mark.parametrize("sentence, gloss, folder", [
    ("We went on a trip", "TRAVEL", "travel"),
    ("She bought new trousers", "PANTS", "pants"),
    ("They begin work", "START", "start"),
    ("I need a baby cot", "BABY_COT", "Baby_Bot"),
    ("He knocks over and over again", "AGAIN_AND_AGAIN", "Again_and_again"),
])
def test_extra_forms(sentence, gloss, folder):
    assert folder_for(sentence, gloss) == folder


# --- folder name and video label both count; a label qualifier only blocks a different meaning
@pytest.mark.parametrize("sentence, gloss, folder", [
    ("My younger brother plays", "YOUNGER_BROTHER", "brother"),
    ("He is very brave", "STRONG", "strong"),
    ("He is very strong", "STRONG", "strong"),
    ("It is raining", "RAIN", "rain"),
    ("Wait a second", "SECOND", "Second"),
    ("My phone charger is broken", "CHARGER", "charger"),
    ("I need your advice", "ADVICE", "advice"),
    ("I advise you to rest", "ADVICE", "advice"),
    ("Show me the photo", "PHOTO", "photo"),
    ("They shoot a photo", "PHOTO", "photo"),
    ("Let us take a break", "PAUSE", "pause"),
    ("Do not break the glass", "BREAK", "break"),
    ("When will you come?", "WHEN", "When"),
])
def test_label_meaning_matches(sentence, gloss, folder):
    assert folder_for(sentence, gloss) == folder


@pytest.mark.parametrize("sentence, fingerspelled", [
    ("My elder brother plays", "fs-BROTHER"),       # elder brother sign is not in the dataset
    ("She came second in the race", "fs-SECOND"),   # ordinal, label is "second (time)"
    ("This is an old car", "fs-OLD"),               # label is "old (person)"
    ("The police shoot the thief", "fs-SHOOT"),     # photo's "shoot" is the camera meaning
])
def test_label_meaning_mismatch_is_fingerspelled(sentence, fingerspelled):
    pairs = dict(signs(sentence))
    assert fingerspelled in pairs and pairs[fingerspelled] is None


def test_german_clips_excluded():
    assert "APRIL" not in LEX.folders
    assert folder_for("My birthday is in April", "APRIL") == "April"
