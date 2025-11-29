#!/usr/bin/env python3
"""
Generate synthetic training data for the intent classifier (unique-ish examples).

Usage:
    python generate_synthetic_intent_data.py

Output:
    - intent_training_synth.csv
"""

import csv
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "intent_training_synth.csv"

random.seed(42)

diseases = [
    "blast",
    "brown spot",
    "hispa",
    "dead heart",
    "tungro",
    "this disease",
    "this problem",
]

symptoms = [
    "small brown spots on the leaves",
    "white narrow lines along the veins",
    "central tillers drying and pulling out easily",
    "yellow orange leaves and stunted plants",
    "pale yellow angular patches with downy growth",
    "burnt leaf tips drying back",
]

weathers = [
    "it is very rainy and humid these days",
    "weather is dry and hot",
    "lot of rain and cloudy days",
    "very dry spell with cracked soil",
]

stages = [
    "nursery stage",
    "tillering stage",
    "vegetative stage",
    "booting stage",
    "near harvest",
]

# ---------------------------------------------------------------------------
# Helper for adding optional context
# ---------------------------------------------------------------------------

def optional_context():
    """Randomly add weather/stage fragments (or nothing)."""
    parts = []

    if random.random() < 0.6:  # sometimes include weather
        parts.append(random.choice(weathers))

    if random.random() < 0.6:  # sometimes include stage
        parts.append(f"at {random.choice(stages)}")

    if not parts:
        return ""

    return " " + " and ".join(parts)


# ---------------------------------------------------------------------------
# Generators for each label – with uniqueness
# ---------------------------------------------------------------------------

def gen_symptom_description(num=200):
    seen = set()
    rows = []
    attempts = 0
    max_attempts = num * 20

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        s = random.choice(symptoms)
        ctx = optional_context()
        text = s + ctx
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "SYMPTOM_DESCRIPTION"))

    print(f"[SYMPTOM_DESCRIPTION] Generated {len(rows)} unique examples.")
    return rows


def gen_ask_diagnosis(num=200):
    patterns = [
        "What disease is this?",
        "Can you tell me what disease this could be?",
        "Which disease is causing these symptoms?",
        "What problem is affecting my paddy leaves?",
        "Can you identify the disease from these symptoms?",
        "Is this {disease}?",
        "Do you think this is {disease}?",
        "What disease does this look like?",
        "Which disease?",
        "Diagnosis?",
        "මේක මොන ව්‍යාධියද?",
        "මේ ප්‍රශ්නය මොකක් ද?",
    ]
    seen = set()
    rows = []
    attempts = 0
    max_attempts = num * 20

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        pat = random.choice(patterns)
        if "{disease}" in pat:
            d = random.choice(diseases)
            base = pat.format(disease=d)
        else:
            base = pat

        # Sometimes mix in a symptom fragment to mimic real queries
        tail = ""
        if random.random() < 0.4:
            tail = " " + random.choice(symptoms)

        text = base + tail
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_DIAGNOSIS"))

    print(f"[ASK_DIAGNOSIS] Generated {len(rows)} unique examples.")
    return rows


def gen_ask_treatment(num=200):
    patterns = [
        "How do I treat this disease?",
        "What should I do to control this problem?",
        "How can I control this on my field?",
        "What pesticide should I spray for this?",
        "Is there any chemical to control this disease?",
        "Tell me the best way to treat it now.",
        "I need treatment recommendation for this.",
        "What can I do immediately in the field to reduce damage?",
        "How do I treat this, it is spreading fast.",
        "Which spray should I use to control this quickly?",
        "Any organic way to control this problem?",
        "Can I manage this only with cultural practices?",
        "Control measures?",
        "Treatment?",
        "Spray?",
        "Chemical control options?",
        "Give me immediate treatment options.",
        "Quick fix for this disease?",
        "මට මේක කොහොමද හරි කරගන්නෙ?",
        "මට spray කරන්න තියෙන්නේ මොකක්ද?",
    ]
    seen = set()
    rows = []
    attempts = 0
    max_attempts = num * 20

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        base = random.choice(patterns)
        ctx = optional_context()
        text = base + ctx
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_TREATMENT"))

    print(f"[ASK_TREATMENT] Generated {len(rows)} unique examples.")
    return rows


def gen_ask_prevention(num=200):
    patterns = [
        "How can I prevent this next season?",
        "What should I do to avoid this disease in future?",
        "Next crop, how do I make sure this doesn’t come again?",
        "How to stop this disease from coming back every year?",
        "What preventive measures should I take?",
        "Any long term way to prevent this problem?",
        "How to protect seedlings from getting this disease?",
        "How to avoid this problem when the weather is very wet?",
        "If this is {disease}, how to stop it spreading next crop?",
        "What can I change in fertiliser and water management to avoid this?",
        "Seed selection steps to prevent this disease?",
        "Prevention tips?",
        "Preventions?",
        "Preventive plan for next season please.",
        "Preventive measures before planting?",
        "මීලඟ කන්නයට මේක නැතිවෙන්න කොහොමද වගා කරගන්නෙ?",
        "ක්‍රමවත් ආරක්ෂණ පියවර මොනවද?",
    ]
    seen = set()
    rows = []
    attempts = 0
    max_attempts = num * 20

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        pat = random.choice(patterns)
        if "{disease}" in pat:
            d = random.choice(diseases)
            base = pat.format(disease=d)
        else:
            base = pat
        ctx = optional_context()
        text = base + ctx
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_PREVENTION"))

    print(f"[ASK_PREVENTION] Generated {len(rows)} unique examples.")
    return rows


def gen_ask_cause(num=200):
    patterns = [
        "What causes this disease?",
        "Is this caused by a fungus or an insect?",
        "Is this a virus problem?",
        "Why is this happening to my field?",
        "Why are these brown spots appearing on the leaves?",
        "Is it because of too much nitrogen fertiliser?",
        "Is poor drainage causing this disease?",
        "Could the seed be the cause of this problem?",
        "Does rainy weather cause this to spread fast?",
        "Is this problem due to insects scraping the leaves?",
        "Is this disease coming from the previous crop residues?",
        "Could grassy weeds on the bund be harbouring the pest?",
        "Does continuous rice without rotation cause this disease?",
        "Is poor soil fertility the main reason for this problem?",
        "Explain the main cause and favourable conditions.",
        "Is climate change making this disease worse?",
        "Why does it always get worse when it rains a lot?",
        "Main cause?",
        "Reason for this disease?",
        "Cause?",
        "Pathogen type?",
        "Is it seed-borne or soil-borne?",
        "මේකට හේතුව වෙන්නේ කුමන ජීවියා ද?",
    ]
    seen = set()
    rows = []
    attempts = 0
    max_attempts = num * 20

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        base = random.choice(patterns)
        ctx = optional_context()
        text = base + ctx
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_CAUSE"))

    print(f"[ASK_CAUSE] Generated {len(rows)} unique examples.")
    return rows


def gen_other(num=200):
    patterns = [
        "Hi",
        "Hello",
        "Good morning",
        "Thank you",
        "Thanks, that was helpful",
        "Ok",
        "👍",
        "Are you an AI chatbot?",
        "Explain briefly.",
        "Can you summarise it?",
        "Just testing the assistant.",
        "Ok I understood.",
        "That explanation was clear.",
        "Please continue.",
        "Stop here, enough.",
        "Not sure what to ask.",
        "I'm only testing your responses.",
        "හරි, තේරුණා.",
        "ලැබුනා බොහොම ස්තුතියි.",
        "Thank you for the advice.",
        "I will ask more later.",
        "Good bye.",
        "Bye.",
        "Great, thanks.",
        "The weather is mostly rainy and humid these days.",
        "It has been very dry and hot for weeks.",
        "Field is under standing water most of the time.",
        "We are using alternate wetting and drying irrigation.",
        "Nursery stage now, plants are very small.",
        "Crop is at tillering stage.",
        "Near harvest, grains are almost mature.",
        "Plants are in vegetative stage with many tillers.",
    ]
    seen = set()
    rows = []
    attempts = 0
    max_attempts = num * 20

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        base = random.choice(patterns)

        # Sometimes append a tiny benign tail to get more variety
        tail = ""
        if random.random() < 0.3:
            tail = " 🙂"
        if random.random() < 0.3:
            tail = " 👍"

        text = base + tail
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "OTHER"))

    print(f"[OTHER] Generated {len(rows)} unique examples.")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = []
    rows += gen_symptom_description(200)
    rows += gen_ask_diagnosis(200)
    rows += gen_ask_treatment(200)
    rows += gen_ask_prevention(200)
    rows += gen_ask_cause(200)
    rows += gen_other(200)

    print(f"Total synthetic examples: {len(rows)}")

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Written synthetic data to {OUT_PATH}")


if __name__ == "__main__":
    main()
