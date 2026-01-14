#!/usr/bin/env python3
"""
Generate synthetic training data for the intent classifier (bilingual EN/SI, unique-ish examples).

Usage:
    python generate_synthetic_intent_data.py
    python generate_synthetic_intent_data.py --per_label 600 --langs en si mix --out intent_training_synth.csv

Output:
    - intent_training_synth.csv (default)
"""

import argparse
import csv
from pathlib import Path
import random
import re

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_label", type=int, default=400, help="Target examples per label (default: 400)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--langs",
        nargs="+",
        default=["en", "si", "mix"],
        choices=["en", "si", "mix"],
        help="Languages to generate: en, si, mix (default: en si mix)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="intent_training_synth.csv",
        help="Output CSV filename (default: intent_training_synth.csv)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Domain vocab (EN + SI)
# ---------------------------------------------------------------------------

diseases_en = [
    "blast",
    "brown spot",
    "hispa",
    "dead heart",
    "tungro",
    "this disease",
    "this problem",
    "this issue",
]

# Common Sinhala farmer/extension phrasing (mix transliteration + native)
diseases_si = [
    "බ්ලස්ට් රෝගය",
    "වී බ්ලස්ට්",
    "දුඹුරු ලප රෝගය",
    "දම් ලප රෝගය",
    "හීස්පා කීට පීඩාව",
    "හිස්පා",
    "මැරුණු හද (Dead heart)",
    "මැරුණු හද ලක්ෂණය",
    "ටන්ග්රෝ රෝගය",
    "ටන්ග්රෝ වෛරස් රෝගය",
    "මේ රෝගය",
    "මේ ප්‍රශ්නය",
    "මේ ලෙඩේ",
]

symptoms_en = [
    "small brown spots on the leaves",
    "white narrow lines along the veins",
    "central tillers drying and pulling out easily",
    "yellow orange leaves and stunted plants",
    "burnt leaf tips drying back",
    "spindle-shaped lesions with grey center and brown margin",
    "leaf looks scorched in patches",
    "silvery scraped leaves and mines",
]

symptoms_si = [
    "කොළ මත කුඩා දුඹුරු ලප තියෙනවා",
    "කොළ නහරට සමාන්තරව සුදු පැහැ ලීනියර් රේඛා පේනවා",
    "මැද කොළ/මධ්‍ය තිල්ලර් එක වියළිලා පහසුවෙන් ඇදලා එළියට එන්නෙ",
    "කොළ කහ-තැඹිලි වෙලා පැළ කුඩායි (stunted)",
    "කොළ අගයන් ද烧 වගේ වියළී යනවා",
    "දිගැති (spindle) තුවාල වල මැද අළු පැහැයි, වටේ දුඹුරු සීමාවක් තියෙනවා",
    "ක්ෂේත්‍රයේ කොටස් කිහිපයක් කලු/ද烧 වගේ පේනවා",
    "කොළ මතුපිට සීරීලා රිදී/සුදු වගේ, ඇතුළෙන් මයින් වගේ ලක්ෂණ තියෙනවා",
]

weathers_en = [
    "it is very rainy and humid these days",
    "weather is dry and hot",
    "lot of rain and cloudy days",
    "very dry spell with cracked soil",
    "morning dew is heavy and leaves stay wet",
]

weathers_si = [
    "මේ දවස්වල වැසි වැඩි, ආර්ද්‍රතාවත් වැඩියි",
    "කාලගුණය වියළි සහ උණුසුම්",
    "වලාකුළු වැඩි, වැසිත් ඉවරයක් නැහැ",
    "දිග වියළි කාලයක්, පස ප裂 වෙලා",
    "උදේ තෙමීම වැඩි, කොළ දිගටම තෙත්මයි",
]

stages_en = [
    "nursery stage",
    "tillering stage",
    "vegetative stage",
    "booting stage",
    "heading stage",
    "near harvest",
]

stages_si = [
    "මඩුව අවස්ථාවෙ (nursery stage)",
    "ටිලරින්ග් අවස්ථාවෙ",
    "වැඩිවෙන (vegetative) අවස්ථාවෙ",
    "බූටින්ග් අවස්ථාවෙ",
    "හිඩින්ග් අවස්ථාවෙ",
    "කප්පාදු ආසන්නයේ",
]

# ---------------------------------------------------------------------------
# Lightweight EN->SI conversion (template/dictionary based)
# This is intentionally simple but creates lots of Sinhala-like variants.
# ---------------------------------------------------------------------------

REPLACEMENTS = [
    (r"\bWhat disease is this\?\b", "මේක මොන රෝගයක්ද?"),
    (r"\bDiagnosis\?\b", "රෝගය හඳුනාගන්න පුළුවන්ද?"),
    (r"\bWhich disease\?\b", "මොන රෝගයද?"),
    (r"\bHow do I treat\b", "මට ප්‍රතිකාර කරන්නේ කොහොමද"),
    (r"\bHow can I prevent\b", "මීලඟ කන්නයට වැළැක්වෙන්නේ කොහොමද"),
    (r"\bWhat causes\b", "මේකට හේතුව මොකක්ද"),
    (r"\bpaddy\b", "වී"),
    (r"\bfield\b", "ක්ෂේත්‍රය"),
    (r"\bleaves\b", "කොළ"),
    (r"\bdisease\b", "රෝගය"),
    (r"\bproblem\b", "ප්‍රශ්නය"),
    (r"\bspray\b", "spray කරන්න"),
    (r"\bfungus\b", "දිලීර (fungus)"),
    (r"\binsect\b", "කීට (insect)"),
    (r"\bvirus\b", "වෛරස් (virus)"),
]

def to_sinhalaish(text: str) -> str:
    out = text
    for pat, rep in REPLACEMENTS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    # Add Sinhala question particle sometimes
    if not out.endswith("?") and random.random() < 0.25:
        out = out.strip() + " ද?"
    return out

def pick_disease(lang: str) -> str:
    if lang == "en":
        return random.choice(diseases_en)
    if lang == "si":
        return random.choice(diseases_si)
    # mix
    return random.choice(diseases_si if random.random() < 0.6 else diseases_en)

def pick_symptom(lang: str) -> str:
    if lang == "en":
        return random.choice(symptoms_en)
    if lang == "si":
        return random.choice(symptoms_si)
    return random.choice(symptoms_si if random.random() < 0.6 else symptoms_en)

def pick_weather(lang: str) -> str:
    if lang == "en":
        return random.choice(weathers_en)
    if lang == "si":
        return random.choice(weathers_si)
    return random.choice(weathers_si if random.random() < 0.6 else weathers_en)

def pick_stage(lang: str) -> str:
    if lang == "en":
        return random.choice(stages_en)
    if lang == "si":
        return random.choice(stages_si)
    return random.choice(stages_si if random.random() < 0.6 else stages_en)

def optional_context(lang: str) -> str:
    parts = []
    if random.random() < 0.65:
        parts.append(pick_weather(lang))
    if random.random() < 0.65:
        if lang == "si":
            parts.append(f"{pick_stage(lang)}")
        else:
            parts.append(f"at {pick_stage(lang)}")
    if not parts:
        return ""
    joiner = " සහ " if lang == "si" else " and "
    return " " + joiner.join(parts)

def maybe_mix_code_switch(text: str) -> str:
    """Small realistic code-switching for Sri Lankan users."""
    if random.random() < 0.25:
        text += random.choice([" pls", " please", " ikmanin", " urgent", " ASAP", " bro"])
    if random.random() < 0.15:
        text = text.replace("රෝගය", "ලෙඩේ")
    return text


# ---------------------------------------------------------------------------
# Pattern banks (EN + SI) per intent
# ---------------------------------------------------------------------------

ASK_DIAGNOSIS_EN = [
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
]

ASK_DIAGNOSIS_SI = [
    "මේක මොන රෝගයක්ද?",
    "මේ ලක්ෂණ වලට හේතුව මොන රෝගයක්ද?",
    "වී කොළේ මෙහෙම වෙන්නේ මොන ලෙඩෙන්ද?",
    "මෙක {disease}ද?",
    "{disease} වගේද පේන්නේ?",
    "රෝගය හඳුනාගන්න පුළුවන්ද?",
    "මේ ලෙඩේ නම කියන්න පුළුවන්ද?",
    "මේ දේ මොකක්ද? රෝගයක්ද කීටයක්ද?",
]

ASK_TREATMENT_EN = [
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
]

ASK_TREATMENT_SI = [
    "මට මේක ප්‍රතිකාර කරගන්නේ කොහොමද?",
    "මේ ප්‍රශ්නය control කරන්න මොනවද කරන්නෙ?",
    "මගේ ක්ෂේත්‍රයේ මේක නවත්තන්න හොඳ ක්‍රම මොනවද?",
    "spray කරන්න තියෙන්නේ මොකක්ද?",
    "රසායනික පාලනයක් තියෙනවද?",
    "දැනටමත් වේගයෙන් පැතිරෙනවා—ඉක්මනින් කරන දේ කියන්න.",
    "කාබනික/නීම් වගේ ක්‍රම තියෙනවද?",
    "වගා ක්‍රම (cultural) වලින්ම පාලනය කරගන්න පුළුවන්ද?",
    "ඉක්මන් විසඳුමක් දීලා උදව් කරන්න.",
    "මෙයට හොඳ ප්‍රතිකාර නිර්දේශයක් දෙන්න.",
]

ASK_PREVENTION_EN = [
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
    "Preventive plan for next season please.",
    "Preventive measures before planting?",
]

ASK_PREVENTION_SI = [
    "මීලඟ කන්නයට මේක නැතිවෙන්න කොහොමද වගා කරගන්නේ?",
    "ඉදිරියේදී මේ රෝගය එන්නේ නැති වෙන්න මොනවාද කරන්න ඕනේ?",
    "හැම අවුරුද්දෙම එන මේ ලෙඩේ නැවැත්තවෙන්නේ කොහොමද?",
    "වගා කිරීමට පෙර වැළැක්වීමේ පියවර මොනවද?",
    "මඩුවේ බීජ පැළ ආරක්ෂා කරගන්න ක්‍රම කියන්න.",
    "වැසි වැඩි වෙද්දී මේ ප්‍රශ්නය එන්නේ නැති වෙන්න කරන්න ඕනේ දේ මොනවද?",
    "මෙක {disease} නම් මීලඟ වගාවට spread වීම නවත්තන්නේ කොහොමද?",
    "පොහොර/ජල කළමනාකරණයෙන් වෙනස් කරන්න පුළුවන් දේ මොනවද?",
    "බීජ තේරීම/බීජ ප්‍රතිකාර ගැන පියවර කියන්න.",
]

ASK_CAUSE_EN = [
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
    "Main cause?",
    "Cause?",
    "Pathogen type?",
    "Is it seed-borne or soil-borne?",
]

ASK_CAUSE_SI = [
    "මේකට හේතුව මොකක්ද?",
    "දිලීර රෝගයක්ද නැත්තම් කීට පීඩාවක්ද?",
    "වෛරස් ප්‍රශ්නයක්ද?",
    "මගේ ක්ෂේත්‍රයේ මෙහෙම වෙන්නේ ඇයි?",
    "කොළේ දුඹුරු ලප එන්නේ ඇයි?",
    "නයිට්‍රජන් (යූරියා) වැඩි නිසාද?",
    "ජල නිකාසය අඩු නිසාද මේ රෝගය එන්නේ?",
    "බීජයෙන්ම (seed-borne) එනවද?",
    "වැසි/ආර්ද්‍ර කාලගුණෙන්ම වේගයෙන් පැතිරෙනවද?",
    "කොළ සීරීම කීටයකින්ද?",
    "පැරණි ඉතිරි/පඳුරු ඉතිරි වලින් (residue) එනවද?",
    "බැඳුම්වල තණකොළ වලින් කීටය රැඳී ඉන්නවද?",
    "අඛණ්ඩව වී වගා කරන එක නිසාද?",
    "ප්‍රධාන හේතුව සහ හොඳට පැතිරෙන්න උදව් කරන තත්ත්ව කියන්න.",
]

OTHER_EN = [
    "Hi",
    "Hello",
    "Good morning",
    "Thank you",
    "Thanks, that was helpful",
    "Ok",
    "Are you an AI chatbot?",
    "Explain briefly.",
    "Can you summarise it?",
    "Just testing the assistant.",
    "Ok I understood.",
    "Please continue.",
    "Stop here, enough.",
    "Not sure what to ask.",
    "I'm only testing your responses.",
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
]

OTHER_SI = [
    "හෙලෝ",
    "ආයුබෝවන්",
    "සුභ උදෑසනක්",
    "ස්තුතියි",
    "බොහොම ස්තුතියි, උදව් වුනා",
    "හරි",
    "ඔයා AI chatbot එකක්ද?",
    "කෙටියෙන් විස්තර කරන්න.",
    "සාරාංශයක් දාන්න පුළුවන්ද?",
    "මම ටෙස්ට් කරනවා විතරයි.",
    "හරි තේරුණා.",
    "ඊළඟට කියන්න.",
    "ඇති, මෙතනින් නවත්තන්න.",
    "මට දැන් මොකක් අහන්නද කියලා නෑ.",
    "මම රිප්ලයි ටෙස්ට් කරනවා.",
    "ගුඩ්බයි",
    "බයි",
    "හොඳයි, ස්තුතියි",
    "මේ දවස්වල වැසි වැඩි, ආර්ද්‍රතාවත් වැඩියි.",
    "දිගටම වියළි කාලයක් උණුසුම්.",
    "ක්ෂේත්‍රයේ ජලය රැඳී ඉන්නවා වැඩියි.",
    "අපි AWD irrigation කරනවා.",
    "දැනට මඩුව අවස්ථාව, පැළ පොඩි.",
    "දැනට ටිලරින්ග් අවස්ථාව.",
    "කප්පාදු ආසන්නයේ, බීජ පරණ වෙමින් තියෙනවා.",
]

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gen_symptom_description(num: int, langs):
    seen, rows = set(), []
    attempts, max_attempts = 0, num * 30
    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)
        s = pick_symptom(lang)
        ctx = optional_context(lang)
        text = (s + ctx).strip()
        if lang in ("si", "mix") and random.random() < 0.35:
            text = maybe_mix_code_switch(text)
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "SYMPTOM_DESCRIPTION"))
    print(f"[SYMPTOM_DESCRIPTION] Generated {len(rows)} unique examples.")
    return rows

def gen_ask_diagnosis(num: int, langs):
    seen, rows = set(), []
    attempts, max_attempts = 0, num * 30
    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)

        if lang == "en":
            pat = random.choice(ASK_DIAGNOSIS_EN)
        elif lang == "si":
            # mix native Sinhala + converted English
            pat = random.choice(ASK_DIAGNOSIS_SI + [to_sinhalaish(x) for x in ASK_DIAGNOSIS_EN])
        else:  # mix
            pat = random.choice(ASK_DIAGNOSIS_EN + ASK_DIAGNOSIS_SI + [to_sinhalaish(x) for x in ASK_DIAGNOSIS_EN])

        if "{disease}" in pat:
            base = pat.format(disease=pick_disease(lang))
        else:
            base = pat

        tail = ""
        if random.random() < 0.45:
            tail = " " + pick_symptom(lang)

        text = (base + tail).strip()
        if lang in ("si", "mix") and random.random() < 0.35:
            text = maybe_mix_code_switch(text)

        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_DIAGNOSIS"))

    print(f"[ASK_DIAGNOSIS] Generated {len(rows)} unique examples.")
    return rows

def gen_ask_treatment(num: int, langs):
    seen, rows = set(), []
    attempts, max_attempts = 0, num * 30
    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)

        if lang == "en":
            base = random.choice(ASK_TREATMENT_EN)
        elif lang == "si":
            base = random.choice(ASK_TREATMENT_SI + [to_sinhalaish(x) for x in ASK_TREATMENT_EN])
        else:
            base = random.choice(ASK_TREATMENT_EN + ASK_TREATMENT_SI + [to_sinhalaish(x) for x in ASK_TREATMENT_EN])

        ctx = optional_context(lang)
        text = (base + ctx).strip()

        if lang in ("si", "mix") and random.random() < 0.35:
            text = maybe_mix_code_switch(text)

        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_TREATMENT"))

    print(f"[ASK_TREATMENT] Generated {len(rows)} unique examples.")
    return rows

def gen_ask_prevention(num: int, langs):
    seen, rows = set(), []
    attempts, max_attempts = 0, num * 30
    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)

        if lang == "en":
            pat = random.choice(ASK_PREVENTION_EN)
        elif lang == "si":
            pat = random.choice(ASK_PREVENTION_SI + [to_sinhalaish(x) for x in ASK_PREVENTION_EN])
        else:
            pat = random.choice(ASK_PREVENTION_EN + ASK_PREVENTION_SI + [to_sinhalaish(x) for x in ASK_PREVENTION_EN])

        if "{disease}" in pat:
            base = pat.format(disease=pick_disease(lang))
        else:
            base = pat

        ctx = optional_context(lang)
        text = (base + ctx).strip()

        if lang in ("si", "mix") and random.random() < 0.35:
            text = maybe_mix_code_switch(text)

        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_PREVENTION"))

    print(f"[ASK_PREVENTION] Generated {len(rows)} unique examples.")
    return rows

def gen_ask_cause(num: int, langs):
    seen, rows = set(), []
    attempts, max_attempts = 0, num * 30
    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)

        if lang == "en":
            base = random.choice(ASK_CAUSE_EN)
        elif lang == "si":
            base = random.choice(ASK_CAUSE_SI + [to_sinhalaish(x) for x in ASK_CAUSE_EN])
        else:
            base = random.choice(ASK_CAUSE_EN + ASK_CAUSE_SI + [to_sinhalaish(x) for x in ASK_CAUSE_EN])

        ctx = optional_context(lang)
        text = (base + ctx).strip()

        if lang in ("si", "mix") and random.random() < 0.35:
            text = maybe_mix_code_switch(text)

        if text in seen:
            continue
        seen.add(text)
        rows.append((text, "ASK_CAUSE"))

    print(f"[ASK_CAUSE] Generated {len(rows)} unique examples.")
    return rows

def gen_other(num: int, langs):
    seen, rows = set(), []
    attempts, max_attempts = 0, num * 30

    while len(rows) < num and attempts < max_attempts:
        attempts += 1
        lang = random.choice(langs)

        if lang == "en":
            base = random.choice(OTHER_EN)
        elif lang == "si":
            base = random.choice(OTHER_SI + [to_sinhalaish(x) for x in OTHER_EN])
        else:
            base = random.choice(OTHER_EN + OTHER_SI + [to_sinhalaish(x) for x in OTHER_EN])

        # Sometimes add tiny benign tail for variety
        tail = ""
        if random.random() < 0.25:
            tail += random.choice([" 🙂", " 👍", " ok", " හරි", " pls"])
        text = (base + tail).strip()

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
    args = parse_args()
    random.seed(args.seed)

    out_path = BASE_DIR / args.out
    langs = args.langs
    n = args.per_label

    rows = []
    rows += gen_symptom_description(n, langs)
    rows += gen_ask_diagnosis(n, langs)
    rows += gen_ask_treatment(n, langs)
    rows += gen_ask_prevention(n, langs)
    rows += gen_ask_cause(n, langs)
    rows += gen_other(n, langs)

    random.shuffle(rows)
    print(f"Total synthetic examples: {len(rows)}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Written synthetic data to {out_path}")


if __name__ == "__main__":
    main()
