#!/usr/bin/env python3
"""
Paddy Disease Chatbot API (FastAPI + ML Intent Classifier with history fallback)

Expected files in the SAME folder:
- symptoms_causes.csv
- treatments_scenarios.csv
- intent_classifier.joblib

Run:
    uvicorn chatbot_api:app --reload
"""

from __future__ import annotations

import csv
import re
import secrets
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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

def get_base_dir() -> Path:
    return Path(__file__).resolve().parent


def load_intent_model():
    path = get_base_dir() / "intent_classifier.joblib"
    if not path.exists():
        print("[WARN] intent_classifier.joblib not found – falling back to naive GENERAL intent.")
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
            "I don’t have specific information for '%s' yet. Please check if the disease name matches the knowledge base."
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
# History-based disease inference (NEW)
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
    history: List[HistoryItem]
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

app = FastAPI(title="Paddy Disease Chatbot API (intent classifier + history)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
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
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    state = SESSIONS.get_or_create(req.session_id)
    debug: Dict[str, object] = {}
    used_cnn = False

    # If we are in "refinement" mode, this message is treated as weather/stage context
    if state.awaiting_refinement and state.current_disease:
        refined = refine_treatments_for_message(TREATMENTS, state.current_disease, req.message)
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
    # 1) Predict intent from text (ML)
    # -----------------------
    intent, raw_label = detect_intent_ml(req.message)
    debug["raw_intent_label"] = raw_label
    debug["intent"] = intent.name

    # -----------------------
    # 2) Decide disease (NEW ORDERING)
    # -----------------------
    disease: Optional[str] = None

    # 2.1 CNN prediction first (if supplied & confident)
    if req.cnn_prediction and req.cnn_prediction.confidence >= 0.6:
        pred_name = req.cnn_prediction.disease_name.strip()
        if pred_name in KB:
            disease = pred_name
            used_cnn = True
            debug["cnn_prediction_used"] = True
            debug["cnn_confidence"] = req.cnn_prediction.confidence
        else:
            debug["cnn_prediction_ignored"] = f"Unknown disease '{pred_name}'"

    # 2.2 If we already have a disease in this session, reuse it
    if disease is None and state.current_disease:
        disease = state.current_disease
        debug["session_disease_reused"] = disease

    # 2.3 Try to infer disease from HISTORY (previous user messages)
    if disease is None and req.history:
        inferred = infer_disease_from_history(KB, req.history, debug)
        if inferred:
            disease = inferred
            debug["history_disease_used"] = inferred

    # 2.4 If STILL no disease, and the CURRENT message looks like
    #     symptoms / diagnosis question, guess from THIS message
    if disease is None and raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        guessed, score, matches = guess_disease_from_symptoms(KB, req.message)
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
        guessed, score, matches = guess_disease_from_symptoms(KB, req.message)
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
            "I could not confidently match those symptoms or questions to a specific disease "
            "in my knowledge base (normal, blast, brown_spot, hispa, dead_heart, tungro).\n\n"
            "It might be a nutrient problem, a different pest/disease, or symptoms not "
            "fully covered here. You can try describing:\n"
            "- colour and pattern of spots or streaks on leaves,\n"
            "- whether plants are stunted or tillering poorly,\n"
            "- and any insects or unusual weather recently."
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
    if raw_label in ("SYMPTOM_DESCRIPTION", "ASK_DIAGNOSIS"):
        header = (
            f"From the symptoms or question you provided, this most likely matches "
            f"**{disease.replace('_', ' ')}**.\n\n"
        )

    reply_body, needs_refinement = answer_question(
        KB, TREATMENTS, disease, req.message, intent
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
