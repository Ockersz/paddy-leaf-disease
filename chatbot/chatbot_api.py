#!/usr/bin/env python3
"""
Single-file Paddy Disease Chatbot API (FastAPI)

Expected files in the SAME folder:
- symptoms_causes.csv
- treatments_scenarios.csv

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
    SYMPTOMS = auto()
    MANAGEMENT = auto()
    PREVENTION = auto()
    CAUSE = auto()
    GENERAL = auto()


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


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
    intent: Optional[str] = None
    awaiting_refinement: bool = False
    used_cnn_prediction: bool = False
    debug: Dict[str, object] = Field(default_factory=dict)
    suggested_next_questions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Simple in-memory session store
# -----------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    current_disease: Optional[str] = None
    awaiting_refinement: bool = False
    last_intent: Optional[Intent] = None
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


SYMPTOM_TRIGGERS = [
    "symptom", "symptoms", "sign", "signs",
    "what does it look", "how does it look",
    "appearance", "looks like", "identify", "identify this",
    "spot", "spots", "lesion", "lesions", "patches",
]

MANAGEMENT_TRIGGERS = [
    "treat", "treatment", "manage", "management", "control",
    "spray", "spraying", "medicine", "pesticide", "fungicide",
    "insecticide", "what should i do", "how do i get rid",
    "how to get rid", "how can i get rid",
]

PREVENTION_TRIGGERS = [
    "prevent", "prevention", "avoid getting", "avoid this",
    "next time", "future season", "how not to get",
]

CAUSE_TRIGGERS = [
    "cause", "caused by", "what causes", "pathogen",
    "virus", "bacteria", "fungus", "insect", "bug",
]

DIAGNOSIS_TRIGGERS = [
    "what disease", "which disease",
    "what is this disease", "name of this disease",
    "what is this problem", "what problem is this",
    "what is this", "what could this be",
]


def detect_intent(question: Optional[str]) -> Intent:
    if not question:
        return Intent.GENERAL
    q = question.lower()
    if any(t in q for t in MANAGEMENT_TRIGGERS):
        return Intent.MANAGEMENT
    if any(t in q for t in PREVENTION_TRIGGERS):
        return Intent.PREVENTION
    if any(t in q for t in CAUSE_TRIGGERS):
        return Intent.CAUSE
    if any(t in q for t in SYMPTOM_TRIGGERS):
        return Intent.SYMPTOMS
    return Intent.GENERAL


def guess_disease_from_symptoms(
    kb: Dict[str, DiseaseInfo],
    symptom_text: str,
) -> Tuple[Optional[str], float, Dict[str, List[str]]]:
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


def build_context_text(history: List[HistoryItem], current_message: str) -> str:
    user_texts = [h.content for h in history if h.role == ChatRole.user]
    return " ".join(user_texts + [current_message]).strip()


def get_previous_user_message(history: List[HistoryItem]) -> Optional[str]:
    for h in reversed(history):
        if h.role == ChatRole.user:
            return h.content
    return None


# -----------------------------------------------------------------------------
# Knowledge loading (CSV)
# -----------------------------------------------------------------------------

def get_base_dir() -> Path:
    return Path(__file__).resolve().parent


def load_symptoms_causes() -> Dict[str, DiseaseInfo]:
    path = get_base_dir() / "symptoms_causes.csv"
    if not path.exists():
        raise FileNotFoundError(f"symptoms_causes.csv not found at {path}")

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
        raise FileNotFoundError(f"treatments_scenarios.csv not found at {path}")

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


def build_suggested_questions(
    last_intent: Optional[Intent],
    awaiting_refinement: bool,
    disease_name: Optional[str],
) -> List[str]:
    display_name = disease_name.replace("_", " ") if disease_name else "this problem"

    if awaiting_refinement:
        return [
            "It’s rainy and humid, plants are at tillering with standing water.",
            "It’s been hot and dry, soil is cracked and plants are at vegetative stage.",
            "Weather is normal, field is at booting–heading with alternate wetting and drying.",
        ]

    if last_intent in (Intent.MANAGEMENT, Intent.PREVENTION):
        return [
            "Can you summarise the most important 2–3 actions?",
            f"How can I prevent {display_name} in the next season?",
            "Is there an organic or low-chemical option?",
        ]

    if last_intent == Intent.SYMPTOMS:
        return [
            f"How do I treat {display_name}?",
            f"What causes {display_name}?",
            "How can I tell this apart from nutrient deficiency?",
        ]

    if last_intent == Intent.CAUSE:
        return [
            f"How do I control {display_name} now?",
            f"How can I reduce the risk of {display_name} coming back?",
        ]

    return [
        "What treatments do you recommend for my situation?",
        "How can I prevent this in future seasons?",
        "What exactly causes this problem?",
    ]


def answer_question(
    kb: Dict[str, DiseaseInfo],
    treatments_map: Dict[str, List[TreatmentOption]],
    disease_name: str,
    message: str,
    disease_source: str = "unknown",
) -> Tuple[str, Intent, bool]:
    disease_name = disease_name.strip()
    if disease_name not in kb:
        return (
            "I don’t have specific information for '%s' yet. Please check if the disease name matches the knowledge base."
            % disease_name,
            Intent.GENERAL,
            False,
        )

    info = kb[disease_name]
    intent = detect_intent(message)
    lower_msg = message.lower()
    diagnosis_mode = any(t in lower_msg for t in DIAGNOSIS_TRIGGERS)

    from_cnn = disease_source == "cnn"
    from_symptoms = disease_source == "symptoms"
    from_session = disease_source == "session"

    if intent == Intent.GENERAL:
        user_tokens = set(preprocess_text(message))
        disease_tokens = set()
        for kw in info.key_leaf_symptom_keywords:
            disease_tokens.update(preprocess_text(kw))
        disease_tokens.update(preprocess_text(info.leaf_symptoms_detailed))
        if disease_tokens.intersection(user_tokens):
            intent = Intent.SYMPTOMS

    if disease_name == "normal":
        if intent in (Intent.MANAGEMENT, Intent.PREVENTION):
            lead = "The crop appears healthy based on the current diagnosis."
            if from_cnn:
                lead = "From the uploaded image(s), the plants are being classified as **healthy**."
            elif from_symptoms:
                lead = "From the symptoms you described, there are no clear signs of major leaf disease."

            txt = (
                f"{lead}\n\n"
                "To keep it that way, focus on good agronomy:\n"
                "- Use recommended varieties and certified seed.\n"
                "- Maintain balanced fertilisation and good land levelling.\n"
                "- Keep fields and bunds weed-free and irrigate on time.\n"
                "- Monitor regularly so that any pest or disease is caught early.\n\n"
                "No disease-specific pesticide is needed when the crop is healthy."
            )
            return txt, intent, False

        txt = format_disease_explanation(info)
        return txt, intent, False

    display_name = info.name.replace("_", " ")

    if intent == Intent.SYMPTOMS or (intent == Intent.GENERAL and diagnosis_mode):
        lead_parts: List[str] = []

        if from_cnn:
            lead_parts.append(
                "Based on the uploaded image(s), this most likely matches **%s**."
                % display_name
            )
        elif from_symptoms:
            lead_parts.append(
                "From the symptoms you described, this most likely matches **%s**."
                % display_name
            )
        elif from_session:
            lead_parts.append(
                "Continuing from your earlier messages, we are still talking about **%s**."
                % display_name
            )
        else:
            lead_parts.append("This looks consistent with **%s**." % display_name)

        explanation = format_disease_explanation(info)
        tail = (
            "\n\nYou can ask follow-up questions such as:\n"
            "- How do I treat or control this disease?\n"
            "- How can I prevent it next season?\n"
            "- What exactly causes this problem?"
        )

        return " ".join(lead_parts) + "\n\n" + explanation + tail, intent, False

    if intent == Intent.CAUSE:
        lead = "Here is what typically causes **%s**:" % display_name
        if from_cnn:
            lead = (
                "From the uploaded image(s), this appears to be **%s**. Here is what typically causes it:"
                % display_name
            )
        elif from_symptoms:
            lead = (
                "From your symptom description, this appears to be **%s**. Here is what typically causes it:"
                % display_name
            )

        txt = (
            f"{lead}\n\n"
            f"{info.cause_summary}\n\n"
            f"Conditions that favour this problem:\n{info.conditions_favouring}"
        )
        return txt, intent, False

    if intent in (Intent.MANAGEMENT, Intent.PREVENTION):
        options = treatments_map.get(disease_name, [])
        overview = format_treatment_overview(disease_name, options)

        lead_parts: List[str] = []
        if from_cnn:
            lead_parts.append(
                "Based on the image analysis, the disease is likely **%s**."
                % display_name
            )
        elif from_symptoms:
            lead_parts.append(
                "From the symptoms you described, the disease appears to be **%s**."
                % display_name
            )
        elif from_session:
            lead_parts.append(
                "We are still dealing with **%s** from your previous description."
                % display_name
            )
        else:
            lead_parts.append("This appears to be **%s**." % display_name)

        if intent == Intent.PREVENTION:
            lead_parts.append(
                "Here are management and prevention options to reduce current damage "
                "and lower the risk in future seasons."
            )
        else:
            lead_parts.append(
                "Here are the main management options you can consider."
            )

        txt = " ".join(lead_parts) + "\n\n" + overview
        return txt, intent, True

    explanation = format_disease_explanation(info)
    lead = "Here is a summary for **%s**." % display_name
    if from_cnn:
        lead = (
            "Based on the uploaded image(s), this has been classified as **%s**."
            % display_name
        )
    elif from_symptoms:
        lead = (
            "From the symptoms you described, this most closely matches **%s**."
            % display_name
        )

    tail = (
        "\n\nYou can ask follow-up questions such as:\n"
        "- How do I treat this disease?\n"
        "- How can I prevent it next season?\n"
        "- What exactly causes this problem?"
    )

    return f"{lead}\n\n{explanation}{tail}", Intent.GENERAL, False


def refine_treatments_for_context(
    treatments_map: Dict[str, List[TreatmentOption]],
    disease_name: str,
    context_text: str,
) -> Tuple[str, str, str, str]:
    stage, weather, water = extract_context_from_text(context_text)
    all_opts = treatments_map.get(disease_name, [])
    relevant = filter_treatments_for_context(all_opts, stage, weather, water)

    header = (
        "Thanks, that helps.\n\n"
        "From what you said, I’m assuming:\n"
        f"- Stage: **{stage}**\n"
        f"- Weather: **{weather}**\n"
        f"- Water: **{water}**\n\n"
    )

    core = format_refined_treatments(disease_name, relevant, stage, weather, water)
    return header + core, stage, weather, water


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="Paddy Disease Chatbot API (single file)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KB: Dict[str, DiseaseInfo] = load_symptoms_causes()
TREATMENTS: Dict[str, List[TreatmentOption]] = load_treatments()
SESSIONS = SessionStore()


@app.get("/api/health")
async def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "diseasesLoaded": list(KB.keys()),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    state = SESSIONS.get_or_create(req.session_id)
    debug: Dict[str, object] = {}
    used_cnn = False
    disease_source = "unknown"

    # Build context and intents
    context_text = build_context_text(req.history, req.message)
    message_intent = detect_intent(req.message)
    previous_user_msg = get_previous_user_message(req.history)
    previous_user_intent = detect_intent(previous_user_msg)

    ctx_stage, ctx_weather, ctx_water = extract_context_from_text(context_text)
    has_rich_context = (
        ctx_stage != "general" or ctx_weather != "normal" or ctx_water != "any"
    )
    debug["extracted_context"] = {
        "stage": ctx_stage,
        "weather": ctx_weather,
        "water": ctx_water,
        "has_rich_context": has_rich_context,
    }
    debug["message_intent"] = message_intent.name
    debug["previous_user_intent"] = previous_user_intent.name if previous_user_intent else None

    # Disease selection
    disease: Optional[str] = None

    if req.cnn_prediction and req.cnn_prediction.confidence >= 0.6:
        pred_name = req.cnn_prediction.disease_name.strip()
        if pred_name in KB:
            disease = pred_name
            used_cnn = True
            disease_source = "cnn"
            debug["cnn_prediction_used"] = True
            debug["cnn_confidence"] = req.cnn_prediction.confidence
        else:
            debug["cnn_prediction_ignored"] = f"Unknown disease '{pred_name}'"

    if disease is None and state.current_disease:
        disease = state.current_disease
        disease_source = "session"
        debug["session_disease_reused"] = disease

    if disease is None:
        guessed, score, matches = guess_disease_from_symptoms(
            KB,
            context_text or req.message,
        )
        if guessed:
            disease = guessed
            disease_source = "symptoms"
            state.last_guess_score = score
            debug["symptom_guess"] = {
                "disease": guessed,
                "score": score,
                "overlap_tokens": matches.get(guessed, []),
            }

    if disease is None:
        reply = (
            "I could not confidently match those symptoms to a specific disease in my "
            "knowledge base (normal, blast, brown_spot, hispa, dead_heart, tungro).\n\n"
            "It might be a nutrient problem, a different pest/disease, or symptoms not "
            "fully covered here. You can try describing:\n"
            "- colour and pattern of spots or streaks on leaves,\n"
            "- whether plants are stunted or tillering poorly,\n"
            "- and any insects or unusual weather recently."
        )
        debug["disease_source"] = "none"
        suggested = [
            "Leaves have small brown spots with yellow halo.",
            "Leaves show white lesions with brown border, especially near the neck.",
            "Central tillers are drying and can be pulled out easily.",
        ]
        return ChatResponse(
            session_id=state.session_id,
            reply=reply,
            disease_name=None,
            intent=None,
            awaiting_refinement=False,
            used_cnn_prediction=False,
            debug=debug,
            suggested_next_questions=suggested,
        )

    debug["disease_source"] = disease_source
    state.current_disease = disease

    # -------------------------------------------------------------------------
    # Decide if this turn should be treated as refinement
    # -------------------------------------------------------------------------

    # 1) Session said we were waiting for refinement (user has already asked “how to treat / prevent”)
    should_refine = state.awaiting_refinement

    # 2) OR: user previously asked a management/prevention question, and now sends only weather/stage/water
    if (
        not should_refine
        and has_rich_context
        and previous_user_intent in (Intent.MANAGEMENT, Intent.PREVENTION)
        and message_intent == Intent.GENERAL
    ):
        should_refine = True
        debug["refinement_trigger"] = "history_based"

    # -------------------------------------------------------------------------
    # Refinement path
    # -------------------------------------------------------------------------
    if should_refine:
        refined, stg, wth, wat = refine_treatments_for_context(
            TREATMENTS,
            disease,
            context_text,
        )
        state.awaiting_refinement = False
        SESSIONS.update(state)

        debug["refined_for"] = {
            "stage": stg,
            "weather": wth,
            "water": wat,
        }

        suggested = build_suggested_questions(
            last_intent=state.last_intent,
            awaiting_refinement=False,
            disease_name=disease,
        )

        return ChatResponse(
            session_id=state.session_id,
            reply=refined,
            disease_name=disease,
            intent=state.last_intent.name if state.last_intent else None,
            awaiting_refinement=False,
            used_cnn_prediction=used_cnn,
            debug=debug,
            suggested_next_questions=suggested,
        )

    # -------------------------------------------------------------------------
    # One-shot direct targeted management/prevention (with context in same msg)
    # -------------------------------------------------------------------------
    if message_intent in (Intent.MANAGEMENT, Intent.PREVENTION) and has_rich_context:
        refined, stg, wth, wat = refine_treatments_for_context(
            TREATMENTS,
            disease,
            context_text,
        )
        state.last_intent = message_intent
        state.awaiting_refinement = False
        state.used_cnn_prediction = used_cnn
        SESSIONS.update(state)

        debug["direct_refinement"] = {
            "stage": stg,
            "weather": wth,
            "water": wat,
        }

        suggested = build_suggested_questions(
            last_intent=message_intent,
            awaiting_refinement=False,
            disease_name=disease,
        )

        return ChatResponse(
            session_id=state.session_id,
            reply=refined,
            disease_name=disease,
            intent=message_intent.name,
            awaiting_refinement=False,
            used_cnn_prediction=used_cnn,
            debug=debug,
            suggested_next_questions=suggested,
        )

    # -------------------------------------------------------------------------
    # Normal Q&A flow
    # -------------------------------------------------------------------------
    reply, intent, needs_refinement = answer_question(
        KB, TREATMENTS, disease, req.message, disease_source
    )

    state.last_intent = intent
    state.awaiting_refinement = needs_refinement
    state.used_cnn_prediction = used_cnn
    SESSIONS.update(state)

    suggested = build_suggested_questions(
        last_intent=intent,
        awaiting_refinement=needs_refinement,
        disease_name=disease,
    )

    return ChatResponse(
        session_id=state.session_id,
        reply=reply,
        disease_name=disease,
        intent=intent.name if intent else None,
        awaiting_refinement=needs_refinement,
        used_cnn_prediction=used_cnn,
        debug=debug,
        suggested_next_questions=suggested,
    )
