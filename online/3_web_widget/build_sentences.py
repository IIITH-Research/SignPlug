"""Build the demo's sentence library (static/sentences.json) from candidate sentences.

Each candidate is run through the same pipeline as the live demo (translator + keyframe lookup).
A sentence is kept if at most one of its glosses has to be fingerspelled (and every letter of that
word has a sign); all other glosses must be real signs. A fingerspelled function word (TO, IN, OF, ...)
does not count as interesting, so such sentences are dropped. Up to TARGET[topic] are kept per topic.

Run whenever glosses change:  python build_sentences.py
"""
import json
import os

from server import LEXICON, STATIC_DIR, plan_signing, sync_keyframe_links   # also puts isl/ on sys.path
from isl.translate import translate_detailed

TARGET = {
    "Greetings": 10, "Family": 12, "Food & drink": 14, "Feelings": 12, "School & work": 12,
    "Places & travel": 12, "Time & weather": 10, "Questions": 10, "Health": 8,
}
FUNCTION_WORDS = {"TO", "IN", "ON", "AT", "BY", "OF", "FOR", "FROM", "WITH", "AND", "OR", "IT", "THE", "A"}

CANDIDATES = {
    "Greetings": [
        "Hello, my name is Ravi", "Good morning", "Good evening", "Good night",
        "Thank you, I am happy", "Hello, I am happy to see you", "Please come to my home",
        "Nice to meet you", "Please sit", "Thank you for your help", "Welcome to my home",
        "Hello friend, come with me", "Please wait", "Please help me",
        "Good afternoon", "Hello, my friend", "See you soon", "Thank you, my friend", "Please come again",
    ],
    "Family": [
        "My mother cooks rice every morning", "My father goes to the office", "My younger brother plays cricket",
        "My younger sister sings", "My grandfather tells stories", "I love my family",
        "My aunt lives in the village", "The baby is sleeping", "My son goes to school",
        "Her husband is a doctor", "His wife is a teacher", "The child is hungry",
        "My family is big", "The girl dances well", "The boy runs fast", "My father drinks tea",
        "We eat together", "My mother is happy today",
        "My grandfather is old", "My aunt is a teacher", "My husband cooks food", "My son likes school",
        "My mother loves me", "The child plays", "My father reads the book",
    ],
    "Food & drink": [
        "I drink coffee", "I want tea", "She drinks milk every morning", "I like mangoes",
        "He eats an apple", "We eat rice and fish", "The bread is fresh", "I am hungry",
        "I am thirsty", "The tea is hot", "The milk is cold", "I do not eat meat",
        "My mother cooks food", "Bananas are sweet", "I want an egg", "Fruits are good for health",
        "Vegetables are good", "Give me a cup of tea", "The food is sweet", "We go to a restaurant",
        "Put sugar in the tea", "Do you want coffee?",
        "The fish is fresh", "I like oranges", "My mother makes tea", "He drinks cold milk", "The rice is hot",
        "I want sweet mangoes",
    ],
    "Feelings": [
        "I am happy", "He is angry", "I am tired", "She is excited", "I am bored",
        "He feels lonely", "I am proud of you", "She is surprised", "I am confused",
        "Do not be afraid", "Stay calm", "He is anxious", "Why are you angry?",
        "I am happy to see you", "They laugh together", "The baby cries", "She smiles",
        "He is afraid", "She is lonely", "We are excited", "My friend is angry",
    ],
    "School & work": [
        "I go to school", "The teacher teaches us", "The student reads a book", "I write a letter",
        "We study together", "I want to learn sign language", "She works in an office",
        "He has a new job", "I use a computer", "Send me an email", "I have a question",
        "The class starts now", "I read books in the library", "The teacher answers my question",
        "I work every day", "He forgot his book",
        "The teacher is kind", "I like my school", "The student asks a question", "She writes a book",
        "We learn together",
    ],
    "Places & travel": [
        "I am going home", "We go to the market", "He goes to the hospital", "She lives in a village",
        "The city is big", "I like the beach", "We visit the temple", "The forest is dark",
        "They climb the mountain", "I travel by train", "We go by bus", "He drives a car",
        "The aeroplane flies in the sky", "I go to the airport", "The hotel is near",
        "The road is long", "I love India", "We walk in the garden", "The lake is beautiful",
        "The hospital is near", "The market is far", "We visit the market", "The village is beautiful",
        "I like the city", "We visit the village",
    ],
    "Time & weather": [
        "It is raining today", "The sun is hot", "It is cold today", "Snow falls in the mountains",
        "The wind is strong", "I will come tomorrow", "I saw him yesterday", "I wake early in the morning",
        "Come now", "See you tomorrow", "It is a hot day", "The sky is blue", "Clouds are in the sky",
        "I sleep at night",
        "The rain is heavy", "Today is cold", "Tomorrow is a holiday", "The night is dark",
        "The morning is cold", "The rain is coming",
    ],
    "Questions": [
        "What is your name?", "Where do you live?", "Who is your friend?", "Why are you late?",
        "When will you come?", "How many brothers do you have?", "Where is the hospital?",
        "What do you want?", "Who is your teacher?", "Where are you going?", "What is the time?",
        "Why are you tired?", "Do you like tea?",
    ],
    "Health": [
        "I am sick", "I need a doctor", "Call the doctor", "Go to the hospital", "My head hurts",
        "Take rest", "Sleep early", "Health is wealth", "The doctor helps the patient",
        "I feel weak", "Drink milk for good health",
        "The doctor is kind", "I am tired and sick", "Eat fruits every day", "The patient sleeps",
        "Wash your hands",
    ],
}


def check(sentence, available):
    stages = translate_detailed(sentence, LEXICON)
    items, _ = plan_signing(stages, available)
    spelled = [i for i in items if i["kind"] == "fingerspell"]
    gloss = " ".join(i["gloss"] for i in items)
    if len(spelled) > 1:
        return None, gloss, "fingerspells " + ", ".join(i["gloss"] for i in spelled)
    if spelled and (spelled[0]["missing"] or not spelled[0]["letters"]):
        return None, gloss, f"cannot fingerspell {spelled[0]['gloss']}"
    if spelled and spelled[0]["gloss"] in FUNCTION_WORDS:
        return None, gloss, f"fingerspells function word {spelled[0]['gloss']}"
    return {"sentence": sentence, "gloss": gloss, "fingerspelled": spelled[0]["gloss"] if spelled else None}, gloss, None


def main():
    available = sync_keyframe_links()
    library, rejected = [], []
    for topic, sentences in CANDIDATES.items():
        kept = 0
        for s in sentences:
            if kept >= TARGET[topic]:
                break
            entry, gloss, reason = check(s, available)
            if entry:
                library.append({"topic": topic, **entry})
                kept += 1
            else:
                rejected.append((topic, s, gloss, reason))
        print(f"{topic:16s} kept {kept}/{TARGET[topic]}")

    path = os.path.join(STATIC_DIR, "sentences.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=1)
    n_spelled = sum(1 for e in library if e["fingerspelled"])
    print(f"\n{len(library)} sentences -> {path}  ({n_spelled} with one fingerspelled word)")
    print("\nrejected:")
    for topic, s, gloss, reason in rejected:
        print(f"  [{topic}] {s!r} -> {gloss}  ({reason})")


if __name__ == "__main__":
    main()
