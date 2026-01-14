#!/usr/bin/env python3
"""
Paddy Disease Chatbot API (FastAPI + Intent (EN ML / SI heuristic) + Image CNN)

Expected files in the SAME folder (chatbot directory):
- symptoms_causes.csv
- treatments_scenarios.csv
- symptoms_causes_si.csv
- treatments_scenarios_si.csv
- intent_classifier.joblib
- best_model.keras

Run:
    uvicorn chatbot_api:app --reload --port 5000
"""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from tensorflow.keras.models import load_model as load_keras_model


# -----------------------------------------------------------------------------
# Paths / helpers
# -----------------------------------------------------------------------------

def get_base_dir() -> Path:
    return Path(__file__).resolve().parent


Language = Literal["en", "si"]


# -----------------------------------------------------------------------------
# Data models (knowledge)
# -----------------------------------------------------------------------------

@dataclass
class DiseaseInfo:
    name: str
    short_overview: str
    leaf_symptoms_brief: str
    leaf_symptoms_detailed: str
    plant_symptoms_detailed: str
    primary_cause_type: str
    cause_summary: str
    conditions_favouring: str
    key_leaf_symptom_keywords: List[str]


@dataclass
class TreatmentOption:
    disease_name: str
    scenario_label: str
    crop_stage: str
    weather_condition: str
    water_condition: str
    recommendation_type: str
    recommendation_text: str
    key_context_keywords: List[str]


# -----------------------------------------------------------------------------
# Enums / intents
# -----------------------------------------------------------------------------

class Intent(Enum):
    SYMPTOMS = auto()
    MANAGEMENT = auto()
    PREVENTION = auto()
    CAUSE = auto()
    GENERAL = auto()


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


INTENT_LABELS = {
    "SYMPTOM_DESCRIPTION",
    "ASK_DIAGNOSIS",
    "ASK_TREATMENT",
    "ASK_PREVENTION",
    "ASK_CAUSE",
    "OTHER",
}


# -----------------------------------------------------------------------------
# API models (Pydantic)
# -----------------------------------------------------------------------------

class HistoryItem(BaseModel):
    role: ChatRole
    content: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    disease_name: Optional[str] = None
    intent: Optional[str] = None
    raw_intent_label: Optional[str] = None
    awaiting_refinement: bool = False
    used_cnn_prediction: bool = False
    debug: Dict[str, object] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Simple in-memory session store
# -----------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    current_disease: Optional[str] = None
    awaiting_refinement: bool = False
    last_intent: Optional[Intent] = None
    last_intent_label: Optional[str] = None
    used_cnn_prediction: bool = False
    last_guess_score: float = 0.0


class SessionStore:
    def __init__(self) -> None:
        self._store: Dict[str, SessionState] = {}

    def _new_session_id(self) -> str:
        return "sess_" + secrets.token_hex(8)

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        if session_id and session_id in self._store:
            return self._store[session_id]
        new_id = session_id or self._new_session_id()
        state = SessionState(session_id=new_id)
        self._store[new_id] = state
        return state

    def update(self, state: SessionState) -> None:
        self._store[state.session_id] = state


# -----------------------------------------------------------------------------
# NLP helpers
# -----------------------------------------------------------------------------

STOPWORDS_EN = {
    "the", "and", "or", "a", "an", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she", "them", "him", "her",
    "my", "your", "our", "their",
    "from", "at", "by", "as", "about", "around",
    "very", "really", "just", "like",
}

# Minimal Sinhala fillers (optional; safe to keep small)
STOPWORDS_SI = {
    "සහ", "හෝ", "එය", "මෙය", "එම", "ඔබ", "මම", "අපි", "ඔවුන්",
    "ද", "දක්වා", "මත", "තුළ", "පිළිබඳ", "වැනි", "නම්", "නැත",
}


def preprocess_text(text: str, lang: Language) -> List[str]:
    text = text.lower()

    if lang == "en":
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = [t.strip() for t in text.split() if t.strip()]
        tokens = [t for t in tokens if t not in STOPWORDS_EN]
        return tokens

    # Sinhala: keep Unicode letters/numbers, split on whitespace/punct
    text = re.sub(r"[^\w\u0D80-\u0DFF]+", " ", text, flags=re.UNICODE)
    tokens = [t.strip() for t in text.split() if t.strip()]
    tokens = [t for t in tokens if t not in STOPWORDS_SI]
    return tokens


def guess_disease_from_symptoms(
    kb: Dict[str, DiseaseInfo],
    symptom_text: str,
    lang: Language,
) -> Tuple[Optional[str], float, Dict[str, List[str]]]:
    user_tokens = set(preprocess_text(symptom_text, lang))
    if not user_tokens:
        return None, 0.0, {}

    best_name: Optional[str] = None
    best_score = 0.0
    matches_per: Dict[str, List[str]] = {}

    for name, info in kb.items():
        kw_tokens = set()
        for kw in info.key_leaf_symptom_keywords:
            kw_tokens.update(preprocess_text(kw, lang))
        desc_tokens = set(preprocess_text(info.leaf_symptoms_detailed, lang))

        common_kw = kw_tokens.intersection(user_tokens)
        common_desc = desc_tokens.intersection(user_tokens)

        score = 2 * len(common_kw) + 1 * len(common_desc)
        matches_per[name] = list(common_kw.union(common_desc))

        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None or best_score < 1.0:
        return None, 0.0, matches_per

    return best_name, best_score, matches_per


def extract_context_from_text(text: str) -> Tuple[str, str, str]:
    lower = text.lower()

    stage = "general"
    if any(w in lower for w in ["nursery", "seedling", "tray", "මඩුව", "බීජ පැළ"]):
        stage = "nursery"
    elif any(w in lower for w in ["tillering", "tiller", "vegetative", "ටිලරී", "වර්ධන"]):
        stage = "vegetative"
    elif any(w in lower for w in ["booting", "heading", "panicle", "flowering", "බූටින්ග්", "හිඩින්ග්", "පැනිකල"]):
        stage = "booting_heading"
    elif any(w in lower for w in ["grain filling", "ripening", "harvest", "mature", "ධාන්‍ය පිරවීම", "කප්පාදු"]):
        stage = "reproductive"

    weather = "normal"
    if any(w in lower for w in ["rain", "raining", "showers", "wet", "humid", "fog", "cloudy", "dew", "monsoon",
                                "වැසි", "තෙත්", "ආර්ද්‍ර", "වලාකුළු", "ගංවතුර"]):
        weather = "rainy_humid"
    if any(w in lower for w in ["dry", "no rain", "drought", "hot and dry", "cracked soil",
                                "වියළි", "වැසි නැත", "නියඟ", "පස ප裂"]):
        weather = "dry_drought"

    water = "any"
    if any(w in lower for w in ["standing water", "flooded", "waterlogged", "ponded",
                                "ජලය රැඳී", "ජලයෙන් පිරි", "ගංවතුර", "වතුරෙන් පිරි"]):
        water = "flooded"
    if any(w in lower for w in ["no water", "dry soil", "cracked", "drained",
                                "ජලය නැත", "වියළි පස", "පස ප裂", "නිකාසය"]):
        water = "dry"
    if any(w in lower for w in ["alternate wetting", "awd", "intermittent",
                                "awd", "විකල්ප තෙත් කිරීම"]):
        water = "awd"

    return stage, weather, water


# -----------------------------------------------------------------------------
# Intent classifier loading + Sinhala heuristic
# -----------------------------------------------------------------------------

def load_intent_model():
    path = get_base_dir() / "intent_classifier.joblib"
    if not path.exists():
        print("[WARN] intent_classifier.joblib not found - falling back to GENERAL.")
        return None
    print(f"[INFO] Loading intent classifier from {path}")
    return joblib.load(path)


INTENT_MODEL = load_intent_model()


def map_label_to_intent(label: str) -> Intent:
    if label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        return Intent.SYMPTOMS
    if label == "ASK_TREATMENT":
        return Intent.MANAGEMENT
    if label == "ASK_PREVENTION":
        return Intent.PREVENTION
    if label == "ASK_CAUSE":
        return Intent.CAUSE
    return Intent.GENERAL


def detect_intent(message: str, lang: Language) -> Tuple[Intent, str]:
    """
    EN: ML intent classifier (your joblib)
    SI: heuristic keyword detection (because the ML model is likely EN-trained)
    """
    msg = (message or "").strip()
    if not msg:
        return Intent.GENERAL, "OTHER"

    if lang == "si":
        m = msg.lower()

        # treatment / management
        if any(k in m for k in ["ප්‍රතිකාර", "කළමනාකරණ", "පාලනය", "මර්දනය", "ආරක්ෂා", "කියන්න"]):
            return Intent.MANAGEMENT, "ASK_TREATMENT"

        # prevention
        if any(k in m for k in ["වැළැක්ව", "වළක්ව", "ඉදිරි වගාව", "ඊළඟ වාරය", "මීළඟ"]):
            return Intent.PREVENTION, "ASK_PREVENTION"

        # cause
        if any(k in m for k in ["හේතුව", "කොහොමද ඇතිවෙන්නේ", "ඇයි", "කාරණා", "වෛරස", "ෆංගස්", "කීට"]):
            return Intent.CAUSE, "ASK_CAUSE"

        # symptoms / diagnosis
        if any(k in m for k in ["ලක්ෂණ", "රෝගය මොකක්ද", "මොකක්ද මේ", "හඳුනා", "කොළ", "ලප", "කහ", "අළු", "ඕවල්", "දිගු"]):
            return Intent.SYMPTOMS, "ASK_DIAGNOSIS"

        return Intent.GENERAL, "OTHER"

    # English ML path
    if INTENT_MODEL is None:
        return Intent.GENERAL, "OTHER"

    label = INTENT_MODEL.predict([msg])[0]
    if label not in INTENT_LABELS:
        label = "OTHER"
    return map_label_to_intent(label), label


# -----------------------------------------------------------------------------
# Knowledge loading (CSV) - language aware
# -----------------------------------------------------------------------------

def load_symptoms_causes_csv(filename: str) -> Dict[str, DiseaseInfo]:
    path = get_base_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found next to chatbot_api.py")

    kb: Dict[str, DiseaseInfo] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["disease_name"].strip()
            kw_raw = row.get("key_leaf_symptom_keywords") or ""
            kw_list = [k.strip() for k in kw_raw.split(";") if k.strip()]
            kb[name] = DiseaseInfo(
                name=name,
                short_overview=(row.get("short_overview") or "").strip(),
                leaf_symptoms_brief=(row.get("leaf_symptoms_brief") or "").strip(),
                leaf_symptoms_detailed=(row.get("leaf_symptoms_detailed") or "").strip(),
                plant_symptoms_detailed=(row.get("plant_symptoms_detailed") or "").strip(),
                primary_cause_type=(row.get("primary_cause_type") or "").strip(),
                cause_summary=(row.get("cause_summary") or "").strip(),
                conditions_favouring=(row.get("conditions_favouring") or "").strip(),
                key_leaf_symptom_keywords=kw_list,
            )
    return kb


def load_treatments_csv(filename: str) -> Dict[str, List[TreatmentOption]]:
    path = get_base_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found next to chatbot_api.py")

    disease_to_options: Dict[str, List[TreatmentOption]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["disease_name"].strip()
            kw_raw = row.get("key_context_keywords") or ""
            kw_list = [k.strip() for k in kw_raw.split(";") if k.strip()]
            opt = TreatmentOption(
                disease_name=name,
                scenario_label=(row.get("scenario_label") or "").strip(),
                crop_stage=(row.get("crop_stage") or "").strip(),
                weather_condition=(row.get("weather_condition") or "").strip(),
                water_condition=(row.get("water_condition") or "").strip(),
                recommendation_type=(row.get("recommendation_type") or "").strip(),
                recommendation_text=(row.get("recommendation_text") or "").strip(),
                key_context_keywords=kw_list,
            )
            disease_to_options.setdefault(name, []).append(opt)
    return disease_to_options


# Load both languages at import time
KB_BY_LANG: Dict[Language, Dict[str, DiseaseInfo]] = {
    "en": load_symptoms_causes_csv("symptoms_causes.csv"),
    "si": load_symptoms_causes_csv("symptoms_causes_si.csv"),
}

TREATMENTS_BY_LANG: Dict[Language, Dict[str, List[TreatmentOption]]] = {
    "en": load_treatments_csv("treatments_scenarios.csv"),
    "si": load_treatments_csv("treatments_scenarios_si.csv"),
}


# -----------------------------------------------------------------------------
# Image model (best_model.keras)
# -----------------------------------------------------------------------------

IMG_SIZE = 224
CLASS_NAMES = ["normal", "blast", "brown_spot", "hispa", "dead_heart", "tungro"]

MODEL_PATH = get_base_dir() / "best_model.keras"

try:
    if MODEL_PATH.exists():
        print(f"[INFO] Loading image model from {MODEL_PATH}")
        IMAGE_MODEL = load_keras_model(MODEL_PATH)
    else:
        print(f"[WARN] best_model.keras not found at {MODEL_PATH}. Image predictions disabled.")
        IMAGE_MODEL = None
except Exception as e:
    print(f"[WARN] Failed to load best_model.keras: {e}")
    IMAGE_MODEL = None


def preprocess_image_bytes(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return arr


def run_cnn_on_images(
    upload_files: List[UploadFile],
    debug: Dict[str, object],
    confidence_threshold: float = 0.6,
) -> Tuple[Optional[str], float, bool]:
    if IMAGE_MODEL is None or not upload_files:
        return None, 0.0, False

    upload_files = upload_files[:5]
    images_arr: List[np.ndarray] = []
    filenames: List[str] = []

    for uf in upload_files:
        data = uf.file.read()
        if not data:
            continue
        try:
            arr = preprocess_image_bytes(data)
            images_arr.append(arr)
            filenames.append(uf.filename or "image")
        except Exception as e:
            debug.setdefault("image_errors", []).append({"filename": uf.filename, "error": str(e)})

    if not images_arr:
        return None, 0.0, False

    batch = np.stack(images_arr, axis=0)
    probs_batch = IMAGE_MODEL.predict(batch).astype(float)

    top_classes: List[str] = []
    top_confidences: List[float] = []
    per_image_debug = []

    for i, probs in enumerate(probs_batch):
        top_idx = int(np.argmax(probs))
        top_disease = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])

        top_classes.append(top_disease)
        top_confidences.append(top_conf)

        per_image_debug.append({
            "filename": filenames[i],
            "top_class": top_disease,
            "top_confidence": top_conf,
            "probs": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        })

    debug["image_predictions"] = per_image_debug

    high_conf_classes = [cls for cls, conf in zip(top_classes, top_confidences) if conf >= confidence_threshold]
    if not high_conf_classes:
        return None, 0.0, False

    unique_classes = set(high_conf_classes)
    if len(unique_classes) > 1:
        return None, 0.0, True

    consensus_class = next(iter(unique_classes))
    confs = [conf for cls, conf in zip(top_classes, top_confidences) if cls == consensus_class]
    avg_conf = float(sum(confs) / max(len(confs), 1))

    return consensus_class, avg_conf, False


# -----------------------------------------------------------------------------
# Localized phrasing (EN/SI)
# -----------------------------------------------------------------------------

UI = {
    "en": {
        "images_unclear": (
            "Based on the images alone, I cannot clearly match them to a single disease. "
            "Some images give mixed or low-confidence signals.\n\n"
            "Please also describe the leaf symptoms in words, for example:\n"
            "- Colour and pattern of spots or streaks on leaves\n"
            "- Which part of the plant is most affected\n"
            "- Any stunting, yellowing, or dead tillers\n"
            "- Recent weather (very wet / humid, or very dry)\n\n"
            "With that description plus the images, I can give a more reliable diagnosis."
        ),
        "no_match": (
            "I could not confidently match the current images or description to a specific disease "
            "in my knowledge base (normal, blast, brown_spot, hispa, dead_heart, tungro).\n\n"
            "It might be a nutrient problem, a different pest/disease, or symptoms not "
            "fully covered here. You can try describing:\n"
            "- Colour and pattern of spots or streaks on leaves\n"
            "- Whether plants are stunted or tillering poorly\n"
            "- Any insects you see on leaves or stems\n"
            "- Recent weather (very wet / humid, or very dry)\n"
        ),
        "header_symptom": "From the symptoms or question you provided, this most likely matches **{disease}**.\n\n",
        "header_images": "From the images you uploaded, this most likely matches **{disease}**.\n\n",
        "header_both": "Based on both your description and the images, this most likely matches **{disease}**.\n\n",
        "healthy_keep": (
            "The crop appears healthy based on the current diagnosis.\n\n"
            "To keep it that way, focus on good agronomy:\n"
            "- Use recommended varieties and certified seed.\n"
            "- Maintain balanced fertilisation and good land levelling.\n"
            "- Keep fields and bunds weed-free and irrigate on time.\n"
            "- Monitor regularly so that any pest or disease is caught early.\n\n"
            "No disease-specific pesticide is needed when the crop is healthy."
        ),
        "overview_title": "Disease: {disease}",
        "overview": "Overview:\n{txt}",
        "leaf": "Symptoms on leaves:\n{txt}",
        "plant": "Whole-plant effects:\n{txt}",
        "cause": "Cause:\n{txt}",
        "favour": "Conditions favouring outbreaks:\n{txt}",
        "cause_block": "Cause of {disease}:\n{cause}\n\nConditions that favour this problem:\n{fav}",
        "treat_overview_intro": "For {disease}, here is a high-level overview of management options under different situations:\n",
        "treat_refine_prompt": (
            "To tailor these recommendations more precisely, please describe:\n"
            "- The current weather (mostly rainy and humid, very dry, or normal)\n"
            "- The crop stage (nursery, vegetative/tillering, booting–heading, or near harvest)\n"
            "- How water is managed (standing water, AWD, or quite dry)\n"
        ),
        "refined_intro": "Given the conditions you described (stage: {stage}, weather: {weather}, water: {water}), the most relevant options for {disease} are:\n",
        "refined_footer": (
            "\nAlways cross-check these steps with your local Department of Agriculture "
            "or extension officer, and strictly follow product labels for any pesticides "
            "or seed treatments."
        ),
        "no_specific_scenario": (
            "For {disease} under the described conditions, I don't have a very specific scenario stored. "
            "You can still consider the general cultural, variety and IPM steps already listed, "
            "and follow local extension advice for any chemical use."
        ),
        "general_hints": (
            "\n\nYou can ask follow-up questions such as:\n"
            "- How do I treat this disease?\n"
            "- How can I prevent it next season?\n"
            "- What exactly causes this problem?"
        ),
    },
    "si": {
        "images_unclear": (
            "රූප පමණක් භාවිතා කරලා එකම රෝගයකට පැහැදිලිව ගැළපෙන්න තරම් විශ්වාසයක් නැහැ. "
            "රූප කිහිපයකින් මිශ්‍ර/අඩු විශ්වාස ප්‍රතිඵල ලැබෙනවා.\n\n"
            "කරුණාකර ලක්ෂණ වචන වලින්ත් විස්තර කරන්න:\n"
            "- කොළ මත ලප/රේඛා වල වර්ණය සහ ආකෘතිය\n"
            "- ශාකයේ කුමන කොටස වැඩියෙන් බලපෑමට ලක්වෙලාද\n"
            "- පැළ කුඩා වීම, කහ වීම, මැරුණු ටිලරී වගේ ලක්ෂණ තිබේද\n"
            "- මෑත කාලගුණය (වැසි/ආර්ද්‍ර, හෝ වියළි)\n\n"
            "එසේ කළාම රූප + විස්තර දෙකම මත වඩා විශ්වාසදායක නිගමනයක් දෙන්න පුළුවන්."
        ),
        "no_match": (
            "දැනට ඇති රූප හෝ විස්තර මත, මගේ දත්ත ගබඩාවේ රෝගයක් (normal, blast, brown_spot, hispa, dead_heart, tungro) "
            "සමඟ විශ්වාසදායක ලෙස ගැළපීමට නොහැකි විය.\n\n"
            "පෝෂණ ගැටලුවක්, වෙනත් පළිබෝධයක්/රෝගයක්, හෝ මෙහි ආවරණය නොවූ ලක්ෂණයක් විය හැක. කරුණාකර මෙවැනි දේවල් කියන්න:\n"
            "- කොළ මත ලප/රේඛා වල වර්ණය සහ ආකෘතිය\n"
            "- පැළ කුඩා වීම හෝ ටිලරී අඩුවීම තිබේද\n"
            "- කොළ/කඳේ කීටයන් දකින්නේද\n"
            "- මෑත කාලගුණය (වැසි/ආර්ද්‍ර හෝ වියළි)\n"
        ),
        "header_symptom": "ඔබ ලබාදුන් ලක්ෂණ/ප්‍රශ්නය මත, වැඩි ඉඩකඩ ඇත්තේ **{disease}** වේ.\n\n",
        "header_images": "ඔබ උඩුගත කළ රූප මත, වැඩි ඉඩකඩ ඇත්තේ **{disease}** වේ.\n\n",
        "header_both": "ඔබගේ විස්තරය සහ රූප දෙකම මත, වැඩි ඉඩකඩ ඇත්තේ **{disease}** වේ.\n\n",
        "healthy_keep": (
            "දැනට ලැබුණු නිගමනය අනුව වගාව සුව පෙනෙයි.\n\n"
            "එය එසේම තබා ගැනීමට:\n"
            "- නිර්දේශිත වර්ග සහ සහතික කළ බීජ භාවිතා කරන්න.\n"
            "- සම්මත පොෂණය සහ හොඳ මට්ටම් කිරීම පවත්වා ගන්න.\n"
            "- බැඳුම්/ක්ෂේත්‍රය පිරිසිදු කර කාලෝචිත ජල සැපයුම පවත්වා ගන්න.\n"
            "- නිරන්තර නිරීක්ෂණය කරන්න.\n\n"
            "වගාව සුව නම් රෝගයට විශේෂිත රසායනිකයක් අවශ්‍ය නැහැ."
        ),
        "overview_title": "රෝගය: {disease}",
        "overview": "සාරාංශය:\n{txt}",
        "leaf": "කොළ ලක්ෂණ:\n{txt}",
        "plant": "ශාකයේ සමස්ථ බලපෑම:\n{txt}",
        "cause": "හේතුව:\n{txt}",
        "favour": "රෝගය වැඩි වීමට හේතු වන කොන්දේසි:\n{txt}",
        "cause_block": "{disease} හේතුව:\n{cause}\n\nරෝගය වැඩි වීමට හේතු වන කොන්දේසි:\n{fav}",
        "treat_overview_intro": "{disease} සඳහා, තත්ව අනුව කළමනාකරණ මාර්ගෝපදේශ සාරාංශයක් මෙන්න:\n",
        "treat_refine_prompt": (
            "තවත් නිවැරදිව නිර්දේශ තෝරා දීමට කරුණාකර කියන්න:\n"
            "- කාලගුණය (වැසි/ආර්ද්‍ර, වියළි, හෝ සාමාන්‍ය)\n"
            "- වගා අදියර (මඩුව, වර්ධන/ටිලරී, බූටින්ග්–හිඩින්ග්, හෝ කප්පාදු ආසන්නය)\n"
            "- ජල කළමනාකරණය (ජලය රැඳී තිබේද, AWD ද, හෝ වියළි ද)\n"
        ),
        "refined_intro": "ඔබ පැවසූ තත්වය අනුව (අදියර: {stage}, කාලගුණය: {weather}, ජලය: {water}) {disease} සඳහා වැදගත් නිර්දේශ:\n",
        "refined_footer": (
            "\nදේශීය කෘෂිකර්ම උපදේශක/කෘෂි දෙපාර්තමේන්තු උපදෙස්ද සමඟ සනාථ කරගෙන, "
            "ඕනෑම රසායනික භාවිතයකදී ලේබල් උපදෙස් දැඩිව අනුගමනය කරන්න."
        ),
        "no_specific_scenario": (
            "ඔබ පැවසූ තත්වයට ගැළපෙන විශේෂ සෙනාරියෝ එකක් දැනට දත්ත ගබඩාවේ නැහැ. "
            "එහෙත් සාමාන්‍ය සංස්කෘතික/බීජ/ IPM ක්‍රියාමාර්ග අනුගමනය කර, "
            "රසායනික භාවිතය සඳහා දේශීය උපදෙස් අනුගමනය කරන්න."
        ),
        "general_hints": (
            "\n\nඔබට අසන්න පුළුවන් තවත් ප්‍රශ්න:\n"
            "- මේක ප්‍රතිකාර කරන්නේ කොහොමද?\n"
            "- ඉදිරි වගාවට වැළැක්වීම කොහොමද?\n"
            "- හේතුව මොකක්ද?"
        ),
    },
}


def disease_disp(lang: Language, disease_key: str) -> str:
    # Prefer Sinhala/English content from KB itself for nice labels if you later decide to store display names.
    # For now keep key -> readable
    return disease_key.replace("_", " ")


# -----------------------------------------------------------------------------
# Treatment formatting
# -----------------------------------------------------------------------------

def format_disease_explanation(info: DiseaseInfo, lang: Language) -> str:
    u = UI[lang]
    parts: List[str] = []
    parts.append(u["overview_title"].format(disease=disease_disp(lang, info.name)))

    if info.short_overview:
        parts.append("\n" + u["overview"].format(txt=info.short_overview))
    leaf_text = " ".join([t for t in [info.leaf_symptoms_brief, info.leaf_symptoms_detailed] if t]).strip()
    if leaf_text:
        parts.append("\n" + u["leaf"].format(txt=leaf_text))
    if info.plant_symptoms_detailed:
        parts.append("\n" + u["plant"].format(txt=info.plant_symptoms_detailed))
    if info.cause_summary:
        parts.append("\n" + u["cause"].format(txt=info.cause_summary))
    if info.conditions_favouring:
        parts.append("\n" + u["favour"].format(txt=info.conditions_favouring))

    return "\n".join(parts)


def format_treatment_overview(
    disease_name: str,
    options: List[TreatmentOption],
    lang: Language,
) -> str:
    u = UI[lang]
    if not options:
        return "I do not yet have detailed treatment options stored for this disease." if lang == "en" else \
               "මෙම රෝගයට අදාළ විස්තරාත්මක කළමනාකරණ විකල්ප තවමත් දත්ත ගබඩාවේ සම්පූර්ණ නොවේ."

    name_disp = disease_disp(lang, disease_name)
    parts: List[str] = [u["treat_overview_intro"].format(disease=name_disp)]

    by_type: Dict[str, List[TreatmentOption]] = {}
    for opt in options:
        by_type.setdefault(opt.recommendation_type, []).append(opt)

    if lang == "en":
        type_titles = {
            "cultural": "Cultural / field management:",
            "variety_seed": "Variety and seed management:",
            "nutrient_water": "Nutrient and water management:",
            "chemical_biological": "Chemical / biological options (need-based):",
            "monitoring": "Monitoring and decision-making:",
            "ipm": "Integrated pest management (IPM) guidelines:",
        }
    else:
        type_titles = {
            "cultural": "සංස්කෘතික / ක්ෂේත්‍ර කළමනාකරණය:",
            "variety_seed": "වර්ග හා බීජ කළමනාකරණය:",
            "nutrient_water": "පෝෂණ සහ ජල කළමනාකරණය:",
            "chemical_biological": "රසායනික / ජෛවික විකල්ප (අවශ්‍යතාව මත):",
            "monitoring": "නිරීක්ෂණය සහ තීරණ ගැනීම:",
            "ipm": "IPM මාර්ගෝපදේශ:",
        }

    for rec_type, opts in by_type.items():
        parts.append(type_titles.get(rec_type, rec_type + ":"))
        for opt in opts:
            parts.append(f"  - {opt.scenario_label}: {opt.recommendation_text}")
        parts.append("")

    parts.append(u["treat_refine_prompt"])
    return "\n".join(parts)


def filter_treatments_for_context(
    options: List[TreatmentOption],
    crop_stage: str,
    weather_condition: str,
    water_condition: str,
) -> List[TreatmentOption]:
    results: List[TreatmentOption] = []
    for opt in options:
        if opt.crop_stage not in ("any", "general", crop_stage):
            continue
        if opt.weather_condition not in ("any", weather_condition):
            continue
        if opt.water_condition not in ("any", water_condition):
            continue
        results.append(opt)
    return results


def format_refined_treatments(
    disease_name: str,
    options: List[TreatmentOption],
    crop_stage: str,
    weather_condition: str,
    water_condition: str,
    lang: Language,
) -> str:
    u = UI[lang]
    name_disp = disease_disp(lang, disease_name)

    if not options:
        return u["no_specific_scenario"].format(disease=name_disp)

    parts: List[str] = []
    parts.append(u["refined_intro"].format(
        stage=crop_stage, weather=weather_condition, water=water_condition, disease=name_disp
    ))

    for opt in options:
        parts.append(f"- {opt.scenario_label} [{opt.recommendation_type}]: {opt.recommendation_text}")

    parts.append(u["refined_footer"])
    return "\n".join(parts)


def refine_treatments_for_message(
    treatments_map: Dict[str, List[TreatmentOption]],
    disease_name: str,
    message: str,
    lang: Language,
) -> str:
    stage, weather, water = extract_context_from_text(message)
    all_opts = treatments_map.get(disease_name, [])
    relevant = filter_treatments_for_context(all_opts, stage, weather, water)
    return format_refined_treatments(disease_name, relevant, stage, weather, water, lang)


def answer_question(
    kb: Dict[str, DiseaseInfo],
    treatments_map: Dict[str, List[TreatmentOption]],
    disease_name: str,
    message: str,
    intent: Intent,
    lang: Language,
) -> Tuple[str, bool]:
    u = UI[lang]
    disease_name = disease_name.strip()

    if disease_name not in kb:
        return (
            (f"I don't have specific information for '{disease_name}' yet."
             if lang == "en"
             else f"'{disease_name}' සඳහා විශේෂිත තොරතුරු දැනට දත්ත ගබඩාවේ නැහැ."),
            False,
        )

    info = kb[disease_name]

    if disease_name == "normal":
        if intent in (Intent.MANAGEMENT, Intent.PREVENTION):
            return u["healthy_keep"], False
        return format_disease_explanation(info, lang), False

    if intent == Intent.SYMPTOMS:
        return format_disease_explanation(info, lang), False

    if intent == Intent.CAUSE:
        txt = u["cause_block"].format(
            disease=disease_disp(lang, info.name),
            cause=info.cause_summary,
            fav=info.conditions_favouring,
        )
        return txt, False

    if intent == Intent.PREVENTION:
        options = treatments_map.get(disease_name, [])
        overview = format_treatment_overview(disease_name, options, lang)
        if lang == "en":
            txt = ("For prevention in future seasons, you generally use the same set of measures "
                   "as for management, but applied earlier and more systematically.\n\n" + overview)
        else:
            txt = ("ඉදිරි වගාව සඳහා වැළැක්වීමේදී, කළමනාකරණයට සමාන ක්‍රියාමාර්ග "
                   "මුල් අවස්ථාවේ සිටම ක්‍රමානුකූලව භාවිතා කිරීම වැදගත්.\n\n" + overview)
        return txt, True

    if intent == Intent.MANAGEMENT:
        options = treatments_map.get(disease_name, [])
        overview = format_treatment_overview(disease_name, options, lang)
        return overview, True

    txt = format_disease_explanation(info, lang) + u["general_hints"]
    return txt, False


# -----------------------------------------------------------------------------
# History-based disease inference (language-aware)
# -----------------------------------------------------------------------------

def infer_disease_from_history(
    kb: Dict[str, DiseaseInfo],
    history: List[HistoryItem],
    lang: Language,
    debug: Dict[str, object],
) -> Optional[str]:
    if not history:
        return None

    best_disease = None
    best_score = 0.0
    used_message = None

    for item in reversed(history):
        if item.role != ChatRole.user:
            continue

        _, raw_label = detect_intent(item.content, lang)
        if raw_label not in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
            continue

        guessed, score, matches = guess_disease_from_symptoms(kb, item.content, lang)
        if guessed and score > best_score:
            best_disease = guessed
            best_score = score
            used_message = {
                "text": item.content,
                "raw_label": raw_label,
                "score": score,
                "overlap_tokens": matches.get(guessed, []),
            }

    if best_disease:
        debug["history_disease_inferred"] = {
            "disease": best_disease,
            "score": best_score,
            "from_message": used_message,
        }
    return best_disease


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="Paddy Disease Chatbot API (EN/SI + images)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS = SessionStore()


@app.get("/api/health")
async def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "intentModelLoaded": INTENT_MODEL is not None,
        "imageModelLoaded": IMAGE_MODEL is not None,
        "diseasesLoaded_en": list(KB_BY_LANG["en"].keys()),
        "diseasesLoaded_si": list(KB_BY_LANG["si"].keys()),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    session_id: Optional[str] = Form(None),
    message: str = Form(""),
    history: str = Form("[]"),
    language: Language = Form("en"),
    images: List[UploadFile] = File(default_factory=list),
) -> ChatResponse:
    lang: Language = language if language in ("en", "si") else "en"
    kb = KB_BY_LANG[lang]
    treatments = TREATMENTS_BY_LANG[lang]
    u = UI[lang]

    debug: Dict[str, object] = {"lang": lang}
    history_items: List[HistoryItem] = []

    # Parse history JSON
    try:
        raw_list = json.loads(history or "[]")
        if isinstance(raw_list, list):
            for item in raw_list:
                try:
                    history_items.append(HistoryItem(**item))
                except Exception as e:
                    debug.setdefault("history_parse_errors", []).append({"item": item, "error": str(e)})
    except Exception as e:
        debug["history_json_error"] = str(e)

    state = SESSIONS.get_or_create(session_id)
    used_cnn = False

    # Refinement mode: treat message as context
    if state.awaiting_refinement and state.current_disease and message.strip():
        refined = refine_treatments_for_message(treatments, state.current_disease, message, lang)
        state.awaiting_refinement = False
        SESSIONS.update(state)
        return ChatResponse(
            session_id=state.session_id,
            reply=refined,
            disease_name=state.current_disease,
            intent=state.last_intent.name if state.last_intent else None,
            raw_intent_label=state.last_intent_label,
            awaiting_refinement=False,
            used_cnn_prediction=state.used_cnn_prediction,
            debug=debug,
        )

    # 0) CNN on images
    cnn_disease: Optional[str] = None
    cnn_conf: float = 0.0
    cnn_conflict: bool = False

    if images:
        cnn_disease, cnn_conf, cnn_conflict = run_cnn_on_images(images, debug)
        debug["cnn_disease"] = cnn_disease
        debug["cnn_confidence"] = cnn_conf
        debug["cnn_conflict"] = cnn_conflict

        # Images-only but unclear/conflicting => ask for symptom description in selected language
        if (cnn_conflict or cnn_disease is None) and not message.strip():
            return ChatResponse(
                session_id=state.session_id,
                reply=u["images_unclear"],
                disease_name=None,
                intent=None,
                raw_intent_label="OTHER",
                awaiting_refinement=False,
                used_cnn_prediction=False,
                debug=debug,
            )

    # 1) Intent detection (lang-aware)
    intent, raw_label = detect_intent(message, lang)
    debug["raw_intent_label"] = raw_label
    debug["intent"] = intent.name

    # 2) Decide disease: CNN -> session -> history -> text
    disease: Optional[str] = None

    if cnn_disease and not cnn_conflict and cnn_disease in kb:
        disease = cnn_disease
        used_cnn = True
        debug["cnn_prediction_used"] = True
    elif cnn_conflict:
        debug["cnn_prediction_ignored"] = "conflicting high-confidence classes"

    if disease is None and state.current_disease:
        disease = state.current_disease
        debug["session_disease_reused"] = disease

    if disease is None and history_items:
        inferred = infer_disease_from_history(kb, history_items, lang, debug)
        if inferred:
            disease = inferred
            debug["history_disease_used"] = inferred

    if disease is None and raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        guessed, score, matches = guess_disease_from_symptoms(kb, message, lang)
        if guessed:
            disease = guessed
            state.last_guess_score = score
            debug["symptom_guess_current_message"] = {
                "disease": guessed,
                "score": score,
                "overlap_tokens": matches.get(guessed, []),
            }

    if disease is None:
        guessed, score, matches = guess_disease_from_symptoms(kb, message, lang)
        if guessed:
            disease = guessed
            state.last_guess_score = score
            debug["symptom_guess_fallback"] = {
                "disease": guessed,
                "score": score,
                "overlap_tokens": matches.get(guessed, []),
            }

    # 3) Still no disease => localized fail-safe
    if disease is None:
        return ChatResponse(
            session_id=state.session_id,
            reply=u["no_match"],
            disease_name=None,
            intent=intent.name,
            raw_intent_label=raw_label,
            awaiting_refinement=False,
            used_cnn_prediction=False,
            debug=debug,
        )

    # Store in session
    state.current_disease = disease

    # 4) Build answer
    header = ""
    disp = disease_disp(lang, disease)

    if used_cnn and not message.strip():
        header = u["header_images"].format(disease=disp)
    elif raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        header = u["header_symptom"].format(disease=disp)
    elif used_cnn and message.strip():
        header = u["header_both"].format(disease=disp)

    reply_body, needs_refinement = answer_question(kb, treatments, disease, message, intent, lang)
    reply = header + reply_body

    state.last_intent = intent
    state.last_intent_label = raw_label
    state.awaiting_refinement = needs_refinement
    state.used_cnn_prediction = used_cnn
    SESSIONS.update(state)

    return ChatResponse(
        session_id=state.session_id,
        reply=reply,
        disease_name=disease,
        intent=intent.name,
        raw_intent_label=raw_label,
        awaiting_refinement=needs_refinement,
        used_cnn_prediction=used_cnn,
        debug=debug,
    )

frontend_dir = get_base_dir() / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
