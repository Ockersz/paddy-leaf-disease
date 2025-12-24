#!/usr/bin/env python3
"""
Paddy Disease Chatbot API (FastAPI + ML Intent Classifier + Image CNN)

Expected files in the SAME folder (chatbot directory):
- symptoms_causes.csv
- treatments_scenarios.csv
- intent_classifier.joblib
- best_model.keras        <-- image classifier

Run (example):
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
from typing import Dict, List, Optional, Tuple

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
    SYMPTOMS = auto()      # describing / asking what it looks like
    MANAGEMENT = auto()    # treatment / control
    PREVENTION = auto()    # prevent / next season
    CAUSE = auto()         # what causes this
    GENERAL = auto()       # catch all / chit-chat / misc


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


# Raw labels from the ML classifier
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


class CNNPrediction(BaseModel):
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    """Kept for reference – actual endpoint uses multipart form instead."""
    session_id: Optional[str] = None
    message: str
    history: List[HistoryItem] = Field(default_factory=list)
    cnn_prediction: Optional[CNNPrediction] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    disease_name: Optional[str] = None
    intent: Optional[str] = None            # high-level intent name
    raw_intent_label: Optional[str] = None  # ML label (SYMPTOM_DESCRIPTION, ASK_TREATMENT, etc.)
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

STOPWORDS = {
    "the", "and", "or", "a", "an", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she", "them", "him", "her",
    "my", "your", "our", "their",
    "from", "at", "by", "as", "about", "around",
    "very", "really", "just", "like",
}


def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def guess_disease_from_symptoms(
    kb: Dict[str, DiseaseInfo],
    symptom_text: str,
) -> Tuple[Optional[str], float, Dict[str, List[str]]]:
    """
    Use token overlap between user symptom text and stored
    leaf symptom keywords + detailed description.
    """
    user_tokens = set(preprocess_text(symptom_text))
    if not user_tokens:
        return None, 0.0, {}

    best_name: Optional[str] = None
    best_score = 0.0
    matches_per: Dict[str, List[str]] = {}

    for name, info in kb.items():
        kw_tokens = set()
        for kw in info.key_leaf_symptom_keywords:
            kw_tokens.update(preprocess_text(kw))
        desc_tokens = set(preprocess_text(info.leaf_symptoms_detailed))

        common_kw = kw_tokens.intersection(user_tokens)
        common_desc = desc_tokens.intersection(user_tokens)

        # Weighted: keyword matches count double
        score = 2 * len(common_kw) + 1 * len(common_desc)
        matches_per[name] = list(common_kw.union(common_desc))

        if score > best_score:
            best_score = score
            best_name = name

    MIN_SCORE = 1.0
    if best_name is None or best_score < MIN_SCORE:
        return None, 0.0, matches_per

    return best_name, best_score, matches_per


def extract_context_from_text(
    text: str,
) -> Tuple[str, str, str]:
    """
    Roughly extract (crop_stage, weather_condition, water_condition)
    from a free-text description.
    """
    lower = text.lower()

    # crop stage
    stage = "general"
    if any(w in lower for w in ["nursery", "seedling", "tray"]):
        stage = "nursery"
    elif any(w in lower for w in ["tillering", "tiller", "vegetative"]):
        stage = "vegetative"
    elif any(w in lower for w in ["booting", "heading", "panicle", "flowering"]):
        stage = "booting_heading"
    elif any(w in lower for w in ["grain filling", "ripening", "harvest", "mature"]):
        stage = "reproductive"

    # weather
    weather = "normal"
    if any(w in lower for w in ["rain", "raining", "showers", "wet", "humid", "fog", "cloudy", "dew", "monsoon"]):
        weather = "rainy_humid"
    if any(w in lower for w in ["dry", "no rain", "drought", "hot and dry", "cracked soil"]):
        weather = "dry_drought"

    # water
    water = "any"
    if any(w in lower for w in ["standing water", "flooded", "waterlogged", "ponded"]):
        water = "flooded"
    if any(w in lower for w in ["no water", "dry soil", "cracked", "drained"]):
        water = "dry"
    if any(w in lower for w in ["alternate wetting", "awd", "intermittent"]):
        water = "awd"

    return stage, weather, water


# -----------------------------------------------------------------------------
# Intent classifier loading
# -----------------------------------------------------------------------------

def load_intent_model():
    path = get_base_dir() / "intent_classifier.joblib"
    if not path.exists():
        print("[WARN] intent_classifier.joblib not found - falling back to naive GENERAL intent.")
        return None
    print(f"[INFO] Loading intent classifier from {path}")
    return joblib.load(path)


INTENT_MODEL = load_intent_model()


def map_label_to_intent(label: str) -> Intent:
    """
    Map the ML label to our high-level Intent enum.
    """
    if label == "SYMPTOM_DESCRIPTION":
        return Intent.SYMPTOMS
    if label == "ASK_DIAGNOSIS":
        # Asking disease name is effectively a 'what disease / symptoms' question
        return Intent.SYMPTOMS
    if label == "ASK_TREATMENT":
        return Intent.MANAGEMENT
    if label == "ASK_PREVENTION":
        return Intent.PREVENTION
    if label == "ASK_CAUSE":
        return Intent.CAUSE
    # OTHER or unknown
    return Intent.GENERAL


def detect_intent_ml(message: str) -> Tuple[Intent, str]:
    """
    Use the ML model to predict intent.
    Returns:
        (high_level_intent, raw_label)
    """
    if not message.strip():
        # No text: treat as GENERAL / OTHER
        return Intent.GENERAL, "OTHER"

    if INTENT_MODEL is None:
        return Intent.GENERAL, "OTHER"

    label = INTENT_MODEL.predict([message])[0]
    if label not in INTENT_LABELS:
        label = "OTHER"

    intent = map_label_to_intent(label)
    return intent, label


# -----------------------------------------------------------------------------
# Knowledge loading (CSV)
# -----------------------------------------------------------------------------

def load_symptoms_causes() -> Dict[str, DiseaseInfo]:
    path = get_base_dir() / "symptoms_causes.csv"
    if not path.exists():
        raise FileNotFoundError("symptoms_causes.csv not found next to chatbot_api.py")

    kb: Dict[str, DiseaseInfo] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["disease_name"].strip()
            kw_raw = row.get("key_leaf_symptom_keywords") or ""
            kw_list = [k.strip().lower() for k in kw_raw.split(";") if k.strip()]
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


def load_treatments() -> Dict[str, List[TreatmentOption]]:
    path = get_base_dir() / "treatments_scenarios.csv"
    if not path.exists():
        raise FileNotFoundError("treatments_scenarios.csv not found next to chatbot_api.py")

    disease_to_options: Dict[str, List[TreatmentOption]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["disease_name"].strip()
            kw_raw = row.get("key_context_keywords") or ""
            kw_list = [k.strip().lower() for k in kw_raw.split(";") if k.strip()]
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


# -----------------------------------------------------------------------------
# Image model (best_model.keras)
# -----------------------------------------------------------------------------

IMG_SIZE = 224

BASE_DIR = get_base_dir()
MODEL_PATH = BASE_DIR / "best_model.keras"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"

# -------------------------------------------------------------------------
# Load class names (10-class list) from JSON saved during training
# -------------------------------------------------------------------------
try:
    if CLASS_INDICES_PATH.exists():
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES: List[str] = json.load(f)
        print(f"[INFO] Loaded {len(CLASS_NAMES)} class names from {CLASS_INDICES_PATH}")
    else:
        print(f"[WARN] class_indices.json not found at {CLASS_INDICES_PATH}. "
              f"Image predictions disabled.")
        CLASS_NAMES = []
except Exception as e:
    print(f"[WARN] Failed to load class_indices.json: {e}")
    CLASS_NAMES = []

# -------------------------------------------------------------------------
# Load Keras model
# -------------------------------------------------------------------------
try:
    if MODEL_PATH.exists() and CLASS_NAMES:
        print(f"[INFO] Loading image model from {MODEL_PATH}")
        IMAGE_MODEL = load_keras_model(MODEL_PATH, compile=False)

        # Sanity check: model output size vs class list length
        num_outputs = IMAGE_MODEL.output_shape[-1]
        if num_outputs != len(CLASS_NAMES):
            print(
                f"[WARN] Model outputs {num_outputs} units, "
                f"but {len(CLASS_NAMES)} class names are loaded. "
                f"Disabling image predictions to avoid misaligned labels."
            )
            IMAGE_MODEL = None
    else:
        if not MODEL_PATH.exists():
            print(f"[WARN] best_model.keras not found at {MODEL_PATH}. "
                  f"Image predictions disabled.")
        IMAGE_MODEL = None
except Exception as e:
    print(f"[WARN] Failed to load best_model.keras: {e}")
    IMAGE_MODEL = None


def preprocess_image_bytes(data: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes to a float32 array of shape (H, W, 3),
    using the SAME preprocessing as during training.

    Important: NO /255.0 here because the training pipeline fed raw
    0–255 images into the model (EfficientNet's internal preprocessing
    handles the rest).
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32")  # 0–255 range, no scaling
    return arr  # shape (H, W, 3)


def run_cnn_on_images(
    upload_files: List[UploadFile],
    debug: Dict[str, object],
    confidence_threshold: float = 0.6,
) -> Tuple[Optional[str], float, bool]:
    """
    Run the CNN on up to 5 images, return:
      (cnn_disease, cnn_confidence, conflict_flag)

    - cnn_disease: best class if confident and consistent, else None
    - cnn_confidence: average top confidence for that class
    - conflict_flag: True if high-confidence images disagree
    """
    # If model or class names are not available, skip
    if IMAGE_MODEL is None or not CLASS_NAMES or not upload_files:
        return None, 0.0, False

    # Cap at 5 images as per frontend
    upload_files = upload_files[:5]

    images_arr: List[np.ndarray] = []
    filenames: List[str] = []

    for uf in upload_files:
        data = uf.file.read()  # synchronous is fine inside FastAPI path
        if not data:
            continue
        try:
            arr = preprocess_image_bytes(data)
            images_arr.append(arr)
            filenames.append(uf.filename or "image")
        except Exception as e:
            # Skip problematic images but log in debug
            debug.setdefault("image_errors", []).append(
                {"filename": uf.filename, "error": str(e)}
            )

    if not images_arr:
        return None, 0.0, False

    # Shape: (N, H, W, 3) with float32 0–255, same as training
    batch = np.stack(images_arr, axis=0).astype("float32")

    probs_batch = IMAGE_MODEL.predict(batch)
    probs_batch = probs_batch.astype(float)

    top_classes: List[str] = []
    top_confidences: List[float] = []

    per_image_debug = []

    for i, probs in enumerate(probs_batch):
        top_idx = int(np.argmax(probs))
        top_disease = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])

        top_classes.append(top_disease)
        top_confidences.append(top_conf)

        per_image_debug.append(
            {
                "filename": filenames[i],
                "top_class": top_disease,
                "top_confidence": top_conf,
                "probs": {
                    name: float(p)
                    for name, p in zip(CLASS_NAMES, probs)
                },
            }
        )

    debug["image_predictions"] = per_image_debug

    # Consider only reasonably confident predictions
    high_conf_classes = [
        cls for cls, conf in zip(top_classes, top_confidences)
        if conf >= confidence_threshold
    ]

    if not high_conf_classes:
        # All images are low confidence
        return None, 0.0, False

    unique_classes = set(high_conf_classes)
    if len(unique_classes) > 1:
        # Conflicting diseases across images
        return None, 0.0, True

    # All confident predictions agree on the same disease
    consensus_class = next(iter(unique_classes))
    avg_conf = float(
        sum(conf for cls, conf in zip(top_classes, top_confidences)
            if cls == consensus_class)
        / max(
            sum(1 for cls in high_conf_classes if cls == consensus_class),
            1,
        )
    )

    return consensus_class, avg_conf, False


# -----------------------------------------------------------------------------
# Core answer / treatment formatting
# -----------------------------------------------------------------------------

def format_disease_explanation(info: DiseaseInfo) -> str:
    parts: List[str] = []
    parts.append("Disease: %s" % info.name.replace("_", " "))
    if info.short_overview:
        parts.append("\nOverview:\n%s" % info.short_overview)
    if info.leaf_symptoms_brief or info.leaf_symptoms_detailed:
        leaf_text = info.leaf_symptoms_brief
        if info.leaf_symptoms_detailed:
            if leaf_text:
                leaf_text += " " + info.leaf_symptoms_detailed
            else:
                leaf_text = info.leaf_symptoms_detailed
        parts.append("\nSymptoms on leaves:\n%s" % leaf_text)
    if info.plant_symptoms_detailed:
        parts.append("\nWhole-plant effects:\n%s" % info.plant_symptoms_detailed)
    if info.cause_summary:
        parts.append("\nCause:\n%s" % info.cause_summary)
    if info.conditions_favouring:
        parts.append(
            "\nConditions favouring outbreaks:\n%s" % info.conditions_favouring
        )
    return "\n".join(parts)


def format_treatment_overview(
    disease_name: str,
    options: List[TreatmentOption],
) -> str:
    if not options:
        return "I do not yet have detailed treatment options stored for this disease."

    name_disp = disease_name.replace("_", " ")
    parts: List[str] = []
    parts.append(
        "For %s, here is a high-level overview of management options under different situations:\n"
        % name_disp
    )

    by_type: Dict[str, List[TreatmentOption]] = {}
    for opt in options:
        by_type.setdefault(opt.recommendation_type, []).append(opt)

    type_titles = {
        "cultural": "Cultural / field management:",
        "variety_seed": "Variety and seed management:",
        "nutrient_water": "Nutrient and water management:",
        "chemical_biological": "Chemical / biological options (need-based):",
        "monitoring": "Monitoring and decision-making:",
        "ipm": "Integrated pest management (IPM) guidelines:",
    }

    for rec_type, opts in by_type.items():
        title = type_titles.get(rec_type, rec_type.capitalize() + ":")
        parts.append(title)
        for opt in opts:
            parts.append("  - %s: %s" % (opt.scenario_label, opt.recommendation_text))
        parts.append("")

    parts.append(
        "To tailor these recommendations more precisely, please describe:\n"
        "- The current weather (for example: mostly rainy and humid, or very dry, or normal),\n"
        "- The crop stage (nursery, vegetative/tillering, booting–heading, or near harvest), and\n"
        "- How water is usually managed (standing water, alternate wetting and drying, or quite dry).\n"
        "Based on that, I can highlight the most relevant options."
    )

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
) -> str:
    name_disp = disease_name.replace("_", " ")
    if not options:
        return (
            "For %s under the described conditions, I don't have a very specific "
            "scenario stored. You can still consider the general cultural, variety "
            "and IPM steps already listed, and follow local extension advice for any "
            "chemical use." % name_disp
        )

    parts: List[str] = []
    parts.append(
        "Given the conditions you described (stage: %s, weather: %s, water: %s), the most relevant options for %s are:\n"
        % (crop_stage, weather_condition, water_condition, name_disp)
    )

    for opt in options:
        parts.append(
            "- %s [%s]: %s"
            % (opt.scenario_label, opt.recommendation_type, opt.recommendation_text)
        )

    parts.append(
        "\nAlways cross-check these steps with your local Department of Agriculture "
        "or extension officer, and strictly follow product labels for any pesticides "
        "or seed treatments."
    )

    return "\n".join(parts)


def answer_question(
    kb: Dict[str, DiseaseInfo],
    treatments_map: Dict[str, List[TreatmentOption]],
    disease_name: str,
    message: str,
    intent: Intent,
) -> Tuple[str, bool]:
    """
    Core brain once we know:
    - which disease_name,
    - which high-level intent.
    Returns:
        reply_text, needs_refinement_flag
    """
    disease_name = disease_name.strip()
    if disease_name not in kb:
        return (
            "I don't have specific information for '%s' yet. Please check if the disease name matches the knowledge base."
            % disease_name,
            False,
        )

    info = kb[disease_name]

    # Healthy crop special case
    if disease_name == "normal":
        if intent in (Intent.MANAGEMENT, Intent.PREVENTION):
            txt = (
                "The crop appears healthy based on the current diagnosis.\n\n"
                "To keep it that way, focus on good agronomy:\n"
                "- Use recommended varieties and certified seed.\n"
                "- Maintain balanced fertilisation and good land levelling.\n"
                "- Keep fields and bunds weed-free and irrigate on time.\n"
                "- Monitor regularly so that any pest or disease is caught early.\n\n"
                "No disease-specific pesticide is needed when the crop is healthy."
            )
            return txt, False

        txt = format_disease_explanation(info)
        return txt, False

    # SYMPTOMS: describe what it looks like (and implied diagnosis)
    if intent == Intent.SYMPTOMS:
        txt = format_disease_explanation(info)
        return txt, False

    # CAUSE: focus on cause + conditions
    if intent == Intent.CAUSE:
        txt = (
            "Cause of %s:\n%s\n\nConditions that favour this problem:\n%s"
            % (info.name.replace("_", " "), info.cause_summary, info.conditions_favouring)
        )
        return txt, False

    # PREVENTION: same tools as management, framed for next season
    if intent == Intent.PREVENTION:
        options = treatments_map.get(disease_name, [])
        overview = format_treatment_overview(disease_name, options)
        txt = (
            "For prevention in future seasons, you generally use the same set of measures "
            "as for management, but applied earlier and more systematically.\n\n%s"
            % overview
        )
        return txt, True  # ask for conditions to refine further

    # MANAGEMENT / TREATMENT
    if intent == Intent.MANAGEMENT:
        options = treatments_map.get(disease_name, [])
        overview = format_treatment_overview(disease_name, options)
        return overview, True

    # GENERAL / fallback: give explanation + hints
    txt = format_disease_explanation(info)
    txt += (
        "\n\nYou can ask follow-up questions such as:\n"
        "- How do I treat this disease?\n"
        "- How can I prevent it next season?\n"
        "- What exactly causes this problem?"
    )
    return txt, False


def refine_treatments_for_message(
    treatments_map: Dict[str, List[TreatmentOption]],
    disease_name: str,
    message: str,
) -> str:
    stage, weather, water = extract_context_from_text(message)
    all_opts = treatments_map.get(disease_name, [])
    relevant = filter_treatments_for_context(all_opts, stage, weather, water)
    return format_refined_treatments(disease_name, relevant, stage, weather, water)


# -----------------------------------------------------------------------------
# History-based disease inference
# -----------------------------------------------------------------------------

def infer_disease_from_history(
    kb: Dict[str, DiseaseInfo],
    history: List[HistoryItem],
    debug: Dict[str, object],
) -> Optional[str]:
    """
    Look through previous user messages in history and try to recover a disease
    based on the last SYMPTOM_DESCRIPTION / ASK_DIAGNOSIS-like messages.

    This makes follow-ups like "How to treat it?" work even if session_id
    wasn't reused by the frontend.
    """
    if not history:
        return None

    best_disease = None
    best_score = 0.0
    used_message = None

    # Walk from latest to oldest
    for item in reversed(history):
        if item.role != ChatRole.user:
            continue

        intent, raw_label = detect_intent_ml(item.content)

        if raw_label not in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
            continue

        guessed, score, matches = guess_disease_from_symptoms(kb, item.content)
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


def history_suggests_treatment_or_prevention(
    history: List[HistoryItem],
) -> bool:
    """
    Look backwards through user history.
    If the last *relevant* user message was ASK_TREATMENT or ASK_PREVENTION,
    we treat the current context-only message as a refinement.

    We stop when we hit a clear symptom/diagnosis message, because that usually
    means a new case.
    """
    for item in reversed(history):
        if item.role != ChatRole.user:
            continue

        intent, raw_label = detect_intent_ml(item.content)

        # If the last relevant question was about treatment/prevention,
        # we want to refine for that.
        if raw_label in ("ASK_TREATMENT", "ASK_PREVENTION"):
            return True

        # If we reach a symptom/diagnosis before any treatment/prevention,
        # consider that a "new case" and stop.
        if raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
            return False

    return False


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="Paddy Disease Chatbot API (intent classifier + history + images)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load knowledge at import time
KB: Dict[str, DiseaseInfo] = load_symptoms_causes()
TREATMENTS: Dict[str, List[TreatmentOption]] = load_treatments()
SESSIONS = SessionStore()


@app.get("/api/health")
async def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "diseasesLoaded": list(KB.keys()),
        "intentModelLoaded": INTENT_MODEL is not None,
        "imageModelLoaded": IMAGE_MODEL is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    session_id: Optional[str] = Form(None),
    message: str = Form(""),
    history: str = Form("[]"),
    images: List[UploadFile] = File(default_factory=list),
) -> ChatResponse:
    """
    Main chat endpoint.

    Expects multipart/form-data from the React frontend:
      - session_id: string (optional)
      - message: string (may be empty if images only)
      - history: JSON string of [{role, content}, ...]
      - images: 0-5 image files (field name "images")
    """
    # Parse history JSON
    history_items: List[HistoryItem] = []
    debug: Dict[str, object] = {}

    try:
        if history:
            raw_list = json.loads(history)
            if isinstance(raw_list, list):
                for item in raw_list:
                    try:
                        history_items.append(HistoryItem(**item))
                    except Exception as e:
                        debug.setdefault("history_parse_errors", []).append(
                            {"item": item, "error": str(e)}
                        )
    except Exception as e:
        debug["history_json_error"] = str(e)

    state = SESSIONS.get_or_create(session_id)
    used_cnn = False

    # If we are in "refinement" mode, this message is treated as weather/stage context
    if state.awaiting_refinement and state.current_disease and message.strip():
        refined = refine_treatments_for_message(TREATMENTS, state.current_disease, message)
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

    # -----------------------
    # 0) CNN on images (if any)
    # -----------------------
    cnn_disease: Optional[str] = None
    cnn_conf: float = 0.0
    cnn_conflict: bool = False

    if images:
        cnn_disease, cnn_conf, cnn_conflict = run_cnn_on_images(images, debug)
        debug["cnn_disease"] = cnn_disease
        debug["cnn_confidence"] = cnn_conf
        debug["cnn_conflict"] = cnn_conflict

        # If only images and they're unclear/conflicting -> ask for text description
        if (cnn_conflict or cnn_disease is None) and not message.strip():
            reply = (
                "Based on the images alone, I cannot clearly match them to a single disease. "
                "Some images give mixed or low-confidence signals.\n\n"
                "Please also describe the leaf symptoms in words, for example:\n"
                "- Colour and pattern of spots or streaks on leaves\n"
                "- Which part of the plant is most affected\n"
                "- Any stunting, yellowing, or dead tillers\n"
                "- Recent weather (very wet / humid, or very dry)\n\n"
                "With that description plus the images, I can give a more reliable diagnosis."
            )
            return ChatResponse(
                session_id=state.session_id,
                reply=reply,
                disease_name=None,
                intent=None,
                raw_intent_label="OTHER",
                awaiting_refinement=False,
                used_cnn_prediction=False,
                debug=debug,
            )

    # -----------------------
    # 1) Predict intent from text (ML)
    # -----------------------
    intent, raw_label = detect_intent_ml(message)
    debug["raw_intent_label"] = raw_label
    debug["intent"] = intent.name

    # -----------------------
    # 2) Decide disease (ordering: CNN -> session -> history -> text)
    # -----------------------
    disease: Optional[str] = None

    # 2.1 CNN suggestion first if confident and non-conflicting
    if cnn_disease and not cnn_conflict and cnn_disease in KB:
        disease = cnn_disease
        used_cnn = True
        debug["cnn_prediction_used"] = True
    elif cnn_conflict:
        # we already handled pure image-conflict above; here we have text as well
        debug["cnn_prediction_ignored"] = "conflicting high-confidence classes"

    # 2.2 If we already have a disease in this session, reuse it
    if disease is None and state.current_disease:
        disease = state.current_disease
        debug["session_disease_reused"] = disease

    # 2.3 Try to infer disease from HISTORY (previous user messages)
    if disease is None and history_items:
        inferred = infer_disease_from_history(KB, history_items, debug)
        if inferred:
            disease = inferred
            debug["history_disease_used"] = inferred

    # 2.4 If STILL no disease, and the CURRENT message looks like
    #     symptoms / diagnosis question, guess from THIS message
    if disease is None and raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        guessed, score, matches = guess_disease_from_symptoms(KB, message)
        if guessed:
            disease = guessed
            state.last_guess_score = score
            debug["symptom_guess_current_message"] = {
                "disease": guessed,
                "score": score,
                "overlap_tokens": matches.get(guessed, []),
            }

    # 2.5 Absolute last resort – try guessing from this message anyway
    if disease is None:
        guessed, score, matches = guess_disease_from_symptoms(KB, message)
        if guessed:
            disease = guessed
            state.last_guess_score = score
            debug["symptom_guess_fallback"] = {
                "disease": guessed,
                "score": score,
                "overlap_tokens": matches.get(guessed, []),
            }

    # -----------------------
    # 3) If still no disease – generic fail-safe reply
    # -----------------------
    if disease is None:
        reply = (
            "I could not confidently match the current images or description to a specific disease "
            "in my knowledge base (normal, blast, brown_spot, hispa, dead_heart, tungro).\n\n"
            "It might be a nutrient problem, a different pest/disease, or symptoms not "
            "fully covered here. You can try describing:\n"
            "- Colour and pattern of spots or streaks on leaves\n"
            "- Whether plants are stunted or tillering poorly\n"
            "- Any insects you see on leaves or stems\n"
            "- Recent weather (very wet / humid, or very dry)\n"
        )
        return ChatResponse(
            session_id=state.session_id,
            reply=reply,
            disease_name=None,
            intent=intent.name,
            raw_intent_label=raw_label,
            awaiting_refinement=False,
            used_cnn_prediction=False,
            debug=debug,
        )

    # Store disease in session
    state.current_disease = disease

    # -----------------------
    # 4) Build answer text
    # -----------------------
    header = ""

    if used_cnn and not message.strip():
        # Images-only case with clear CNN diagnosis
        header = (
            f"From the images you uploaded, this most likely matches "
            f"**{disease.replace('_', ' ')}**.\n\n"
        )
    elif raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        header = (
            f"From the symptoms or question you provided, this most likely matches "
            f"**{disease.replace('_', ' ')}**.\n\n"
        )
    elif used_cnn and message.strip():
        header = (
            f"Based on both your description and the images, this most likely matches "
            f"**{disease.replace('_', ' ')}**.\n\n"
        )

    reply_body, needs_refinement = answer_question(
        KB, TREATMENTS, disease, message, intent
    )

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
    app.mount(
        "/",
        StaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )
