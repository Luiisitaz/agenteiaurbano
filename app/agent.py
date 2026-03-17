from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, TypedDict

import requests

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim

from .config import settings
from .database import SessionLocal
from .embeddings import embed_text
from .tools import (
    create_report,
    search_similar_reports,
    reports_near_location,
    get_reports_by_cedula,
    list_reports,
    get_report_by_id,
    update_report_location,
)
from .rag import rag_search

REPORT_TYPES = {
    "bache",
    "fuga_de_agua",
    "trafico_accidente",
    "luz_de_trafico_rota",
    "otros",
}

REPORT_TYPE_ALIASES = {
    "hueco": "bache",
    "bache": "bache",
    "fuga": "fuga_de_agua",
    "agua": "fuga_de_agua",
    "accidente": "trafico_accidente",
    "choque": "trafico_accidente",
    "trafico": "trafico_accidente",
    "colision": "trafico_accidente",
    "colisión": "trafico_accidente",
    "semaforo": "luz_de_trafico_rota",
    "semáforo": "luz_de_trafico_rota",
    "luz": "luz_de_trafico_rota",
}

CEDULA_FLEX_RE = re.compile(r"\b[\d-]{5,20}\b")
REPORT_ID_RE = re.compile(r"(?:reporte|id|numero|n(?:u|ú)mero)\s*#?\s*(\d+)", re.IGNORECASE)

PENDING_DUPLICATES: Dict[str, Dict[str, Any]] = {}
PENDING_REPORTS: Dict[str, Dict[str, Any]] = {}
PENDING_QUERY_REQUESTS: Dict[str, Dict[str, Any]] = {}
PENDING_LOCATION_CONFIRMATIONS: Dict[str, Dict[str, Any]] = {}
PENDING_UPDATES: Dict[str, Dict[str, Any]] = {}

LOGGER = logging.getLogger("urban-agent")


class AgentConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return super().format(record)

        parts = [self.formatTime(record, "%H:%M:%S")]
        event = payload.get("event")
        if event:
            parts.append(f"[{event}]")

        for key in ("node", "tool", "session_id", "intent", "query_scope", "next", "reason", "error"):
            value = payload.get(key)
            if value not in (None, "", []):
                parts.append(f"{key}={value}")

        if payload.get("report_id") is not None:
            parts.append(f"report_id={payload['report_id']}")
        if payload.get("top_id") is not None:
            parts.append(f"top_id={payload['top_id']}")
        if payload.get("similarity") is not None:
            parts.append(f"similarity={payload['similarity']:.3f}")
        if payload.get("count") is not None:
            parts.append(f"count={payload['count']}")
        if payload.get("candidate_count") is not None:
            parts.append(f"candidate_count={payload['candidate_count']}")

        details: list[str] = []
        for key in (
            "cedula",
            "report_type",
            "location_text",
            "latitude",
            "longitude",
            "missing",
            "top_candidate",
            "input",
            "output",
            "user_message",
            "response",
        ):
            value = payload.get(key)
            if value in (None, "", []):
                continue
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            details.append(f"{key}={rendered}")

        if details:
            parts.append("| " + " | ".join(details))

        return " ".join(parts)


def _setup_logger() -> None:
    if LOGGER.handlers:
        return
    level_name = (settings.log_level or "info").upper()
    level = getattr(logging, level_name, logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = AgentConsoleFormatter()
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


def _truncate(text: Optional[str], limit: int = 400) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _log_event(event: str, state: AgentState | None = None, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"event": event}
    if state:
        payload.update(
            {
                "session_id": state.get("session_id"),
                "intent": state.get("intent"),
                "query_scope": state.get("query_scope"),
                "cedula": state.get("cedula"),
                "report_type": state.get("report_type"),
                "location_text": _truncate(state.get("location_text")),
                "latitude": state.get("latitude"),
                "longitude": state.get("longitude"),
            }
        )
        if "user_message" in state:
            payload["user_message"] = _truncate(state.get("user_message"))
        if "response" in state:
            payload["response"] = _truncate(state.get("response"))
    if extra:
        payload.update(extra)
    try:
        LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def _log_tool_call(tool: str, input_data: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"event": "tool.call", "tool": tool}
    if input_data:
        payload["input"] = input_data
    try:
        LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def _log_tool_result(tool: str, output_data: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"event": "tool.result", "tool": tool}
    if output_data:
        payload["output"] = output_data
    try:
        LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def _trace_node(name: str, func):
    def wrapper(state: AgentState) -> AgentState:
        _log_event("node.enter", state, {"node": name})
        result = func(state)
        _log_event("node.exit", result if result is not None else state, {"node": name})
        return result

    return wrapper


_setup_logger()


def _normalize_report_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in REPORT_TYPES:
        return v
    return REPORT_TYPE_ALIASES.get(v)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _location_text_matches(query_text: str, candidate_text: str) -> bool:
    if not query_text or not candidate_text:
        return False
    query_norm = _normalize_text(_strip_location_noise(query_text) or query_text)
    candidate_norm = _normalize_text(candidate_text)
    if not query_norm or not candidate_norm:
        return False
    if query_norm in candidate_norm:
        return True
    stopwords = {
        "de",
        "del",
        "la",
        "el",
        "los",
        "las",
        "en",
        "por",
        "a",
        "al",
        "y",
        "o",
        "via",
        "calle",
        "avenida",
        "av",
        "ave",
        "panama",
        "provincia",
        "ciudad",
        "republica",
        "edificio",
        "piso",
        "sector",
        "barrio",
        "urbanizacion",
        "corregimiento",
        "ph",
    }
    query_tokens = [
        token
        for token in query_norm.split()
        if token and token not in stopwords and not token.isdigit() and len(token) >= 3
    ]
    if not query_tokens:
        return False
    candidate_tokens = {
        token
        for token in candidate_norm.split()
        if token and token not in stopwords and not token.isdigit() and len(token) >= 3
    }
    if not candidate_tokens:
        return False

    def token_matches(qt: str) -> bool:
        if qt in candidate_tokens:
            return True
        if len(qt) >= 4:
            for ct in candidate_tokens:
                if ct.startswith(qt) or qt.startswith(ct):
                    return True
        return False

    matched = [qt for qt in query_tokens if token_matches(qt)]
    if not matched:
        return False
    if any(len(qt) >= 5 for qt in matched):
        return True
    overlap = len(set(matched)) / len(set(query_tokens))
    if overlap >= 0.5:
        return True
    return SequenceMatcher(None, " ".join(query_tokens), " ".join(candidate_tokens)).ratio() >= 0.7


def _strip_location_noise(query: str) -> str:
    tokens = query.split()
    leading_noise = {"en", "la", "el", "los", "las", "por", "cerca", "de", "del", "frente"}
    trailing_noise = {
        "frente",
        "ahi",
        "ah",
        "alla",
        "aprox",
        "aproximado",
        "aproximadamente",
        "cerca",
    }
    while tokens and _normalize_text(tokens[0]) in leading_noise:
        tokens.pop(0)
    while tokens and _normalize_text(tokens[-1]) in trailing_noise:
        tokens.pop()
    return " ".join(tokens).strip()


def _looks_like_precise_address(query: str) -> bool:
    lowered = _normalize_text(query)
    address_hints = [
        "calle",
        "avenida",
        "ave",
        "av",
        "via",
        "v ia",
        "corregimiento",
        "barrio",
        "urbanizacion",
        "ph",
        "edificio",
        "torre",
        "local",
        "panama",
    ]
    has_number = any(char.isdigit() for char in query)
    return has_number or any(hint in lowered for hint in address_hints)


def _is_affirmative(text: str) -> bool:
    lowered = _normalize_text(text)
    patterns = [
        r"\bsi\b",
        r"\byes\b",
        r"\bcorrecto\b",
        r"\bexacto\b",
        r"\bconfirmo\b",
        r"\besa es\b",
        r"\bese es\b",
        r"\bes ahi\b",
        r"\bdale\b",
        r"\bok\b",
        r"\bokay\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_negative(text: str) -> bool:
    lowered = _normalize_text(text)
    patterns = [
        r"\bno\b",
        r"\bno es\b",
        r"\bnegativo\b",
        r"\bincorrecto\b",
        r"\bequivocado\b",
        r"\botra\b",
        r"\bninguna\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _extract_choice(text: str, max_value: int) -> Optional[int]:
    match = re.search(r"\b([1-9])\b", text)
    if not match:
        return None
    choice = int(match.group(1))
    if 1 <= choice <= max_value:
        return choice
    return None


def _format_place_candidate(candidate: Dict[str, Any]) -> str:
    name = candidate.get("name") or candidate.get("formatted_address") or "ubicacion"
    address = candidate.get("formatted_address") or "direccion no disponible"
    return f"{name}, {address}"


def _score_place_candidate(query: str, candidate: Dict[str, Any]) -> float:
    clean_query = _normalize_text(_strip_location_noise(query) or query)
    name = _normalize_text(candidate.get("name", ""))
    address = _normalize_text(candidate.get("formatted_address", ""))
    name_score = SequenceMatcher(None, clean_query, name).ratio() if clean_query and name else 0.0
    address_score = SequenceMatcher(None, clean_query, address).ratio() if clean_query and address else 0.0
    startswith_bonus = 0.15 if clean_query and name.startswith(clean_query) else 0.0
    contains_bonus = 0.08 if clean_query and clean_query in name else 0.0
    panama_bonus = 0.05 if "panama" in address else 0.0
    return max(name_score, address_score) + startswith_bonus + contains_bonus + panama_bonus


def _location_requires_confirmation(query: str, candidates: List[Dict[str, Any]]) -> bool:
    return False


def _build_location_confirmation_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    visible_candidates = candidates[:3]
    if len(visible_candidates) == 1:
        return (
            "Creo que la ubicacion es esta:\n\n"
            f"1. {_format_place_candidate(visible_candidates[0])}\n\n"
            'Si es correcta, responde "si".\n'
            "Si no, escribe una referencia mas precisa."
        )

    lines = ["Creo que la ubicacion podria ser una de estas opciones:", ""]
    for idx, candidate in enumerate(visible_candidates, start=1):
        lines.append(f"{idx}. {_format_place_candidate(candidate)}")
    lines.extend(
        [
            "",
            'Responde "si" si la opcion 1 es correcta.',
            "Si no, responde con el numero correcto o escribe una referencia mas precisa.",
        ]
    )
    return "\n".join(lines)


def _build_location_options_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    lines = ["Estas son las opciones que encontre para esa ubicacion:", ""]
    for idx, candidate in enumerate(candidates[:3], start=1):
        lines.append(f"{idx}. {_format_place_candidate(candidate)}")
    lines.extend(["", "Responde con el numero correcto o dame una referencia mas precisa."])
    return "\n".join(lines)


def _clone_state_without_response(state: AgentState) -> AgentState:
    data = dict(state)
    data.pop("response", None)
    return data


def _extract_cedula(text: str) -> Optional[str]:
    for match in CEDULA_FLEX_RE.finditer(text):
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 5:
            return token
    return None


def _is_plausible_cedula(value: Optional[str]) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    return len(digits) >= 5


def _extract_report_id(text: str) -> Optional[int]:
    if re.fullmatch(r"\s*#?\d+\s*", text):
        return int(re.sub(r"\D", "", text))
    match = REPORT_ID_RE.search(text)
    return int(match.group(1)) if match else None


def _looks_like_nearby_query(text: str) -> bool:
    lowered = _normalize_text(text)
    return bool(
        re.search(r"\b(cerca|cercanos|cercanas|alrededor)\b", lowered)
        and re.search(r"\b(reporte|reportes|incidente|incidentes)\b", lowered)
    )


def _looks_like_cedula_query(text: str) -> bool:
    lowered = _normalize_text(text)
    if "mis reportes" in lowered or "mi reporte" in lowered or "mis incidentes" in lowered:
        return True
    if "cedula" in lowered and any(word in lowered for word in ["mis", "ver", "consultar", "mostrar", "listar"]):
        return True
    return False


def _looks_like_listing_query(text: str) -> bool:
    lowered = _normalize_text(text)
    has_query_verb = bool(
        re.search(r"\b(ver|mostrar|listar|consultar|buscar|dime|menciona|mencionar|cuales|que)\b", lowered)
    )
    has_report_subject = bool(re.search(r"\b(reporte|reportes|incidente|incidentes)\b", lowered))
    if has_query_verb and has_report_subject:
        return True
    return any(
        phrase in lowered
        for phrase in [
            "quiero ver todo",
            "ver todo",
            "mostrar todo",
            "listar todo",
            "consultar todo",
            "quiero ver todos",
            "quiero ver todas",
            "todos los reportes",
            "todos los incidentes",
            "que reportes hay",
            "que incidentes hay",
            "dime que reportes hay",
            "dime que incidentes hay",
            "menciona reportes",
            "menciona incidentes",
        ]
    )


def _looks_like_create_report(text: str) -> bool:
    lowered = _normalize_text(text)
    if _looks_like_listing_query(text) or _looks_like_cedula_query(text) or _looks_like_nearby_query(text):
        return False
    explicit_create_phrases = [
        "quiero hacer un reporte",
        "quiero reportar",
        "hacer un reporte",
        "crear reporte",
        "crear un reporte",
        "registrar reporte",
        "reportar un",
        "reportar una",
        "denunciar",
    ]
    if any(phrase in lowered for phrase in explicit_create_phrases):
        return True
    incident_keywords = [
        "choque",
        "accidente",
        "bache",
        "hueco",
        "fuga",
        "semaforo",
        "luz de trafico",
        "colision",
    ]
    if any(phrase in lowered for phrase in ["hay un", "hay una", "ocurrio", "paso", "se rompio", "se dano"]):
        return any(keyword in lowered for keyword in incident_keywords)
    return bool(
        re.search(r"\b(reporta|reportar|reporte|crear|registrar|hacer)\b", lowered)
        and any(keyword in lowered for keyword in incident_keywords)
    )


def _looks_like_report_query(text: str) -> bool:
    lowered = _normalize_text(text)
    phrases = [
        "hay reportes",
        "hay reporte",
        "hay un reporte",
        "hay algun reporte",
        "hay algún reporte",
        "ver reportes",
        "consultar reportes",
        "lista de reportes",
        "listado de reportes",
        "reportes recientes",
        "reportes en",
        "reportes por",
        "reportes de",
        "incidentes en",
        "incidentes por",
        "incidentes de",
    ]
    if any(phrase in lowered for phrase in phrases):
        return True
    has_report_subject = bool(re.search(r"\b(reporte|reportes|incidente|incidentes)\b", lowered))
    has_query_verb = bool(re.search(r"\b(hay|ver|mostrar|listar|consultar|buscar|dime|menciona|cuales|que)\b", lowered))
    return has_report_subject and has_query_verb


def _looks_like_update_request(text: str) -> bool:
    lowered = _normalize_text(text)
    if _extract_report_id(text) is None:
        return False
    return bool(
        re.search(
            r"\b(corregir|corrige|actualizar|actualiza|editar|edita|cambiar|cambia|modificar|modifica|ajustar|ajusta)\b",
            lowered,
        )
        and re.search(r"\b(ubicacion|direccion|lugar|mapa)\b", lowered)
    )


def _looks_like_report_id_query(text: str) -> bool:
    report_id = _extract_report_id(text)
    if report_id is None:
        return False
    lowered = _normalize_text(text)
    if re.fullmatch(r"#?\d+", lowered):
        return True
    return bool(re.search(r"\b(reporte|consulta|detalle|ver|id|numero)\b", lowered))


def _message_has_incident_info(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "bache",
        "hueco",
        "fuga",
        "accidente",
        "choque",
        "colision",
        "colisión",
        "semaforo",
        "semáforo",
        "luz de trafico",
        "luz de tráfico",
        "calle",
        "avenida",
        "via",
        "vía",
        "frente a",
        "cerca de",
    ]
    return any(word in lowered for word in keywords)


def _is_greeting(text: str) -> bool:
    lowered = text.lower().strip()
    return any(word in lowered for word in ["hola", "buenas", "buenos dias", "buenas tardes", "buenas noches"])


def _build_missing_prompt_fallback(missing: list[str]) -> str:
    parts = []
    if "cedula" in missing:
        parts.append("tu cedula")
    if "report_type" in missing:
        parts.append("que paso")
    if "location" in missing:
        parts.append("la ubicacion o coordenadas")
    if len(parts) == 1:
        return f"Para crear el reporte necesito {parts[0]}. Me la compartes?"
    joined = ", ".join(parts[:-1]) + " y " + parts[-1]
    return f"Para crear el reporte necesito {joined}. Me lo compartes?"


def _build_missing_prompt(missing: list[str], context: str) -> str:
    fallback = _build_missing_prompt_fallback(missing)
    fields = []
    if "cedula" in missing:
        fields.append("cedula")
    if "report_type" in missing:
        fields.append("que paso")
    if "location" in missing:
        fields.append("ubicacion o coordenadas")

    system_prompt = (
        "Genera una pregunta corta en espanol para solicitar datos faltantes de un reporte. "
        "Debes pedir exactamente estos datos: "
        + "; ".join(fields)
        + ". No pidas datos extra. Maximo 2 oraciones."
    )
    try:
        llm = _get_response_llm()
        _log_tool_call("llm.response", {"context": _truncate(context), "mode": "missing_prompt"})
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)])
        content = (result.content or "").strip()
        if content:
            _log_tool_result("llm.response", {"content": _truncate(content)})
            return content
    except Exception:
        pass
    return fallback


def _normalize_query_scope(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = _normalize_text(value)
    aliases = {
        "all": "all",
        "todos": "all",
        "todo": "all",
        "todas": "all",
        "mine": "mine",
        "mios": "mine",
        "mias": "mine",
        "mio": "mine",
        "mia": "mine",
        "mis": "mine",
        "tuyos": "mine",
        "nearby": "nearby",
        "cercanos": "nearby",
        "cercanas": "nearby",
        "cerca": "nearby",
        "alrededor": "nearby",
    }
    return aliases.get(lowered)


def _means_all_report_types(text: str) -> bool:
    lowered = _normalize_text(text)
    return lowered in {"todo", "todos", "toda", "todas"} or any(
        phrase in lowered
        for phrase in [
            "todos los tipos",
            "todas las categorias",
            "todos los reportes",
            "todos los incidentes",
            "cualquier tipo",
            "sin filtro",
            "sin filtros",
        ]
    )


def _extract_query_scope(text: str) -> Optional[str]:
    lowered = _normalize_text(text)
    if re.fullmatch(r"(todos?|todas?)", lowered):
        return "all"
    if re.search(r"\b(cerca|cercanos|cercanas|alrededor)\b", lowered):
        return "nearby"
    if any(
        phrase in lowered
        for phrase in [
            "mis reportes",
            "mis incidentes",
            "solo los mios",
            "solo mios",
            "mios",
            "mias",
        ]
    ):
        return "mine"
    if any(
        phrase in lowered
        for phrase in [
            "quiero ver todo",
            "ver todo",
            "mostrar todo",
            "listar todo",
            "consultar todo",
            "todos los reportes",
            "todos los incidentes",
        ]
    ):
        return "all"
    return None


def _build_query_scope_prompt() -> str:
    return "Quieres ver todos los reportes, solo los tuyos o los cercanos a una ubicacion?"


def _build_query_missing_cedula_prompt() -> str:
    return "Para ver tus reportes, comparteme tu cedula."


def _build_query_missing_location_prompt() -> str:
    return "Para buscar reportes cercanos, comparteme la ubicacion o las coordenadas."


def _snapshot_query_state(state: AgentState) -> Dict[str, Any]:
    return {
        "intent": "query_reports",
        "query_scope": state.get("query_scope"),
        "cedula": state.get("cedula"),
        "report_type": state.get("report_type"),
        "location_text": state.get("location_text"),
        "latitude": state.get("latitude"),
        "longitude": state.get("longitude"),
        "radius_m": state.get("radius_m"),
    }


def _format_report_line(report: Dict[str, Any]) -> str:
    parts = [f"#{report['id']}", report["report_type"], "-", report["status"]]
    location_text = report.get("location_text")
    description = report.get("description")
    if location_text:
        parts.extend(["-", location_text])
    elif description:
        parts.extend(["-", description])
    return " ".join(str(part) for part in parts if part not in (None, ""))


def _format_cluster_line(cluster: Dict[str, Any]) -> str:
    count = int(cluster.get("count") or 0)
    report_label = "reporte" if count == 1 else "reportes"
    area = cluster.get("location_text") or "ubicacion sin referencia"
    return (
        f"{count} {report_label} de {cluster['report_type']} - "
        f"prioridad {cluster['priority']} - area {area}"
    )


def _build_location_refine_prompt(context: str) -> str:
    fallback = "No logre ubicar ese punto con precision. Puedes compartirme una calle, barrio o una referencia cercana?"
    system_prompt = (
        "Pide mas detalle de ubicacion porque no se pudo geolocalizar. "
        "No pidas cedula ni enumeres tipos de reporte. Maximo 2 oraciones, en espanol."
    )
    try:
        llm = _get_response_llm()
        _log_tool_call("llm.response", {"context": _truncate(context), "mode": "location_refine"})
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)])
        content = (result.content or "").strip()
        if content:
            _log_tool_result("llm.response", {"content": _truncate(content)})
            return content
    except Exception:
        pass
    return fallback


def _build_update_missing_prompt(missing: list[str], context: str) -> str:
    fallback_parts = []
    if "report_id" in missing:
        fallback_parts.append("el numero de reporte")
    if "cedula" in missing:
        fallback_parts.append("tu cedula")
    if "location" in missing:
        fallback_parts.append("la nueva ubicacion o coordenadas")
    if not fallback_parts:
        return "Necesito mas detalles para actualizar el reporte."
    if len(fallback_parts) == 1:
        fallback = f"Para actualizar el reporte necesito {fallback_parts[0]}. Me lo compartes?"
    else:
        fallback = "Para actualizar el reporte necesito " + ", ".join(fallback_parts[:-1]) + " y " + fallback_parts[-1] + "."

    fields = []
    if "report_id" in missing:
        fields.append("numero de reporte")
    if "cedula" in missing:
        fields.append("cedula")
    if "location" in missing:
        fields.append("nueva ubicacion o coordenadas")
    system_prompt = (
        "Genera una pregunta corta en espanol para solicitar datos faltantes de una actualizacion de reporte. "
        "Debes pedir exactamente estos datos: "
        + "; ".join(fields)
        + ". No pidas datos extra. Maximo 2 oraciones."
    )
    try:
        llm = _get_response_llm()
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)])
        content = (result.content or "").strip()
        if content:
            return content
    except Exception:
        pass
    return fallback


def _infer_report_type(text: str) -> Optional[str]:
    lowered = text.lower()
    keyword_map = [
        ("trafico_accidente", ["choque", "accidente", "colision", "colisión", "atropello", "colisiono"]),
        ("fuga_de_agua", ["fuga", "tuberia", "tubería", "agua", "goteo"]),
        ("bache", ["bache", "hueco", "hoyo", "crater", "cráter"]),
        ("luz_de_trafico_rota", ["semaforo", "semáforo", "luz de trafico", "luz de tráfico"]),
    ]
    for report_type, keywords in keyword_map:
        if any(word in lowered for word in keywords):
            return report_type
    return None


def _looks_like_location_text(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "calle",
        "avenida",
        "av.",
        "via",
        "vía",
        "ph",
        "sector",
        "barrio",
        "frente",
        "cerca",
        "cruce",
        "interseccion",
        "intersección",
        "carretera",
        "ruta",
    ]
    return any(word in lowered for word in keywords)


def _extract_location_heuristic(text: str) -> Optional[str]:
    patterns = [
        r"(?:\ben\b|\ben la\b|\ben el\b|\bpor\b|\bpor la\b|\bpor el\b|\bsobre\b|\bsobre la\b|\bsobre el\b)\s+([A-Za-z0-9#\-\s\.áéíóúüñ]+)",
        r"(?:\bfrente a\b|\bal frente de\b|\bfrente del\b|\bcerca de\b|\bcerca del\b|\bcerca de la\b|\bdetras de\b|\bdetrás de\b|\bjunto a\b)\s+([A-Za-z0-9#\-\s\.áéíóúüñ]+)",
        r"(?:\bubicacion\b|\bubicación\b|\bdireccion\b|\bdirección\b)\s*[:\-]\s*([A-Za-z0-9#\-\s\.áéíóúüñ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            loc = match.group(1).strip()
            loc = re.split(r"\b(cedula|cédula|mi cedula|mi cédula)\b", loc, flags=re.IGNORECASE)[0].strip()
            if len(loc) >= 3:
                return loc
    if _looks_like_location_text(text):
        return text.strip()
    return None


class ParseResult(BaseModel):
    intent: str = Field(
        description="create_report, query_reports, reports_near_location, get_reports_by_cedula, general_query"
    )
    query_scope: Optional[str] = Field(default=None, description="all, mine, nearby")
    cedula: Optional[str] = None
    report_type: Optional[str] = None
    description: Optional[str] = None
    location_text: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_m: Optional[int] = None


class AgentState(TypedDict, total=False):
    user_message: str
    session_id: str
    intent: str
    query_scope: Optional[str]
    cedula: Optional[str]
    report_type: Optional[str]
    description: Optional[str]
    location_text: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    radius_m: Optional[int]
    response: Optional[str]
    embedding: Optional[List[float]]
    skip_duplicate: Optional[bool]
    report_id: Optional[int]


SYSTEM_PROMPT = """
You are an assistant for urban incident reporting in Panama.
Extract structured data from the user message.
Valid intents: create_report, query_reports, general_query.
Valid query scopes: all, mine, nearby.
Valid report types: bache, fuga_de_agua, trafico_accidente, luz_de_trafico_rota, otros.
If a cedula is present, extract it. The cedula can be any reasonable combination of digits and hyphens.
If coordinates are present, extract them; otherwise leave latitude/longitude null.
If the user wants to view, list, show, consult, or search reports, use query_reports.
If the user wants reports near a place, use query_reports with query_scope=nearby.
If the user wants their own reports, use query_reports with query_scope=mine.
If the user wants all or recent reports, use query_reports with query_scope=all.
If the user is reporting a new incident, use create_report.
If unsure, use general_query.
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.chat_model, temperature=0)


def _get_response_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.chat_model, temperature=0.6)


def _parse_with_llm(message: str) -> ParseResult:
    llm = _get_llm().with_structured_output(ParseResult)
    _log_tool_call("llm.parse", {"message": _truncate(message)})
    result = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
    )
    _log_tool_result("llm.parse", result.model_dump())
    return result


def _free_chat_response(message: str) -> str:
    system_prompt = (
        "Eres un asistente ciudadano amable para reportes urbanos en Panama. "
        "Responde en espanol de forma natural y breve (1 a 3 oraciones). "
        "Si es apropiado, ofrece ayuda para crear reportes o consultar incidentes, "
        "pero no fuerces el tema si el usuario solo saluda o hace charla."
    )
    try:
        llm = _get_response_llm()
        _log_tool_call("llm.response", {"message": _truncate(message), "mode": "free_chat"})
        result = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=message)])
        content = (result.content or "").strip()
        if content:
            _log_tool_result("llm.response", {"content": _truncate(content)})
            return content
    except Exception:
        pass
    return "Hola. En que puedo ayudarte?"


def _geocode_with_nominatim(location_text: str) -> Optional[Dict[str, float]]:
    geolocator = Nominatim(user_agent=settings.geocoding_user_agent)
    _log_tool_call("geocode.nominatim", {"location_text": _truncate(location_text)})
    location = geolocator.geocode(location_text)
    if not location:
        _log_tool_result("geocode.nominatim", {"found": False})
        return None
    result = {"lat": float(location.latitude), "lon": float(location.longitude)}
    _log_tool_result("geocode.nominatim", {"found": True, **result})
    return result


def _build_google_places_queries(location_text: str) -> List[str]:
    raw_query = location_text.strip()
    clean_query = _strip_location_noise(raw_query)
    candidates = [raw_query]
    if clean_query and clean_query.lower() != raw_query.lower():
        candidates.append(clean_query)
    with_country = f"{clean_query or raw_query}, Panama"
    if with_country.lower() not in {candidate.lower() for candidate in candidates}:
        candidates.append(with_country)
    return candidates


def _google_places_request(
    *,
    tool_name: str,
    endpoint: str,
    params: Dict[str, Any],
    query: str,
    result_key: str,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        _log_tool_call(tool_name, {"location_text": _truncate(query)})
        resp = requests.get(endpoint, params=params, timeout=10)
    except requests.RequestException:
        return [], "REQUEST_FAILED"

    if resp.status_code != 200:
        return [], f"HTTP_{resp.status_code}"

    data = resp.json()
    status = data.get("status")
    if status not in {"OK", "ZERO_RESULTS"}:
        _log_tool_result(tool_name, {"status": status})
        return [], status
    return data.get(result_key) or [], None


def _search_google_places_candidates(location_text: str, limit: int = 3) -> tuple[List[Dict[str, Any]], Optional[str]]:
    if not settings.google_maps_api_key:
        return [], "MISSING_API_KEY"

    unique_candidates: Dict[str, Dict[str, Any]] = {}
    last_error: Optional[str] = None
    for query in _build_google_places_queries(location_text):
        find_place_params = {
            "input": query,
            "inputtype": "textquery",
            "fields": "formatted_address,name,geometry,place_id",
            "key": settings.google_maps_api_key,
            "language": settings.google_places_language,
        }
        results, error = _google_places_request(
            tool_name="geocode.google_find_place",
            endpoint="https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
            params=find_place_params,
            query=query,
            result_key="candidates",
        )
        if not results and error is None:
            text_search_params = {
                "query": query,
                "key": settings.google_maps_api_key,
                "region": settings.google_places_region,
                "language": settings.google_places_language,
            }
            results, error = _google_places_request(
                tool_name="geocode.google_text_search",
                endpoint="https://maps.googleapis.com/maps/api/place/textsearch/json",
                params=text_search_params,
                query=query,
                result_key="results",
            )
        if error:
            last_error = error
            continue

        for item in results:
            geometry = item.get("geometry", {}).get("location") or {}
            if "lat" not in geometry or "lng" not in geometry:
                continue
            candidate = {
                "place_id": item.get("place_id"),
                "name": item.get("name") or "",
                "formatted_address": item.get("formatted_address") or "",
                "latitude": float(geometry["lat"]),
                "longitude": float(geometry["lng"]),
            }
            candidate["match_score"] = _score_place_candidate(location_text, candidate)
            key = candidate.get("place_id") or f"{candidate['name']}|{candidate['formatted_address']}"
            previous = unique_candidates.get(key)
            if previous is None or candidate["match_score"] > previous["match_score"]:
                unique_candidates[key] = candidate

    candidates = sorted(unique_candidates.values(), key=lambda item: item["match_score"], reverse=True)[:limit]
    _log_tool_result(
        "geocode.google_places",
        {
            "count": len(candidates),
            "candidates": [
                {
                    "name": candidate["name"],
                    "formatted_address": candidate["formatted_address"],
                    "match_score": round(candidate["match_score"], 3),
                }
                for candidate in candidates
            ],
        },
    )
    return candidates, last_error if not candidates else None


def _resolve_location(location_text: str) -> Dict[str, Any]:
    if not settings.google_maps_api_key:
        return {"status": "error", "error": "MISSING_API_KEY"}
    candidates, error_code = _search_google_places_candidates(location_text)
    if candidates:
        top = candidates[0]
        return {
            "status": "resolved",
            "geo": {"lat": top["latitude"], "lon": top["longitude"]},
            "resolved_location_text": _format_place_candidate(top),
        }
    if error_code and error_code != "ZERO_RESULTS":
        return {"status": "error", "error": error_code}
    return {"status": "not_found"}


def _handle_pending_location_confirmation(
    state: AgentState, session_id: str, message: str
) -> Optional[AgentState]:
    pending_location = PENDING_LOCATION_CONFIRMATIONS.get(session_id)
    if not pending_location:
        return None

    candidates = pending_location.get("candidates", [])
    pending_state = _clone_state_without_response(pending_location.get("state", {}))
    original_query = pending_location.get("original_query") or pending_state.get("location_text") or ""

    choice = _extract_choice(message, len(candidates))
    if choice is not None:
        selected = candidates[choice - 1]
        PENDING_LOCATION_CONFIRMATIONS.pop(session_id, None)
        PENDING_REPORTS.pop(session_id, None)
        PENDING_QUERY_REQUESTS.pop(session_id, None)
        result: AgentState = {
            **pending_state,
            "session_id": session_id,
            "user_message": message,
            "location_text": _format_place_candidate(selected),
            "latitude": selected["latitude"],
            "longitude": selected["longitude"],
        }
        _log_event("location.choice", result, {"place_id": selected.get("place_id"), "option": choice})
        return result

    if _is_affirmative(message):
        selected = candidates[0]
        PENDING_LOCATION_CONFIRMATIONS.pop(session_id, None)
        PENDING_REPORTS.pop(session_id, None)
        PENDING_QUERY_REQUESTS.pop(session_id, None)
        result = {
            **pending_state,
            "session_id": session_id,
            "user_message": message,
            "location_text": _format_place_candidate(selected),
            "latitude": selected["latitude"],
            "longitude": selected["longitude"],
        }
        _log_event("location.confirmed", result, {"place_id": selected.get("place_id")})
        return result

    if _is_negative(message):
        result = {
            **pending_state,
            "session_id": session_id,
            "user_message": message,
            "response": _build_location_options_prompt(original_query, candidates),
        }
        _log_event("location.rejected", result)
        return result

    refined_query = f"{original_query}, {message}".strip(", ")
    PENDING_LOCATION_CONFIRMATIONS.pop(session_id, None)
    PENDING_REPORTS.pop(session_id, None)
    PENDING_QUERY_REQUESTS.pop(session_id, None)
    result = {
        **pending_state,
        "session_id": session_id,
        "user_message": message,
        "location_text": refined_query,
        "latitude": None,
        "longitude": None,
    }
    _log_event("location.refine", result)
    return result


def parse_node(state: AgentState) -> AgentState:
    message = state.get("user_message", "")
    session_id = state.get("session_id") or "default"
    _log_event("parse.start", state)

    pending = PENDING_DUPLICATES.get(session_id)
    if pending:
        if re.search(r"\b(unir|unirme|join|existente)\b", message, re.IGNORECASE):
            report_id = _extract_report_id(message) or pending.get("report_id")
            PENDING_DUPLICATES.pop(session_id, None)
            PENDING_REPORTS.pop(session_id, None)
            result = {
                "response": f"Listo. Te uni al reporte #{report_id}. Gracias por confirmar.",
                "intent": "respond_only",
            }
            _log_event("duplicate.join", {**state, **result})
            return result
        if re.search(r"\b(nuevo|crear|otro)\b", message, re.IGNORECASE):
            pending_parsed = pending.get("parsed", {})
            PENDING_DUPLICATES.pop(session_id, None)
            result = {**pending_parsed, "intent": "create_report", "skip_duplicate": True}
            _log_event("duplicate.create_new", {**state, **result})
            return result

    if re.search(r"\b(cancelar|cancelado|olvida|olvidalo|olv?dalo)\b", message, re.IGNORECASE):
        PENDING_REPORTS.pop(session_id, None)
        PENDING_QUERY_REQUESTS.pop(session_id, None)
        PENDING_LOCATION_CONFIRMATIONS.pop(session_id, None)
        PENDING_UPDATES.pop(session_id, None)
        result = {
            "response": "Listo, cancele el proceso actual. Si quieres iniciar otro, dime.",
            "intent": "respond_only",
        }
        _log_event("report.cancel", {**state, **result})
        return result

    pending_location_result = _handle_pending_location_confirmation(state, session_id, message)
    if pending_location_result is not None:
        return pending_location_result

    pending_report = PENDING_REPORTS.get(session_id, {})
    pending_query = PENDING_QUERY_REQUESTS.get(session_id, {})
    pending_update = PENDING_UPDATES.get(session_id, {})
    cedula_from_text = _extract_cedula(message)
    report_id_from_text = _extract_report_id(message)
    lowered = message.lower()
    normalized = _normalize_text(message)
    implicit_location_query = bool(
        _extract_location_heuristic(message)
        and re.search(r"\b(hay|dime|menciona|cuales|que)\b", normalized)
        and not _looks_like_create_report(message)
    )
    explicit_listing_query = _looks_like_listing_query(message)
    explicit_nearby_query = _looks_like_nearby_query(message)
    explicit_cedula_query = _looks_like_cedula_query(message)
    explicit_report_query = _looks_like_report_query(message)
    explicit_scope_query = _extract_query_scope(message)
    explicit_all_types_query = bool(cedula_from_text and _means_all_report_types(message))
    explicit_query_requested = bool(
        pending_query
        or explicit_listing_query
        or explicit_nearby_query
        or explicit_cedula_query
        or explicit_report_query
        or implicit_location_query
        or explicit_all_types_query
    )
    update_requested = _looks_like_update_request(message) or bool(pending_update)
    report_id_query = _looks_like_report_id_query(message)
    reportish = (
        _looks_like_create_report(message)
        or explicit_query_requested
        or _message_has_incident_info(message)
        or explicit_report_query
        or implicit_location_query
        or update_requested
        or report_id_query
    )
    if not pending_report and not pending_query and not cedula_from_text and not reportish:
        result = {"intent": "free_chat"}
        _log_event("parse.free_chat", {**state, **result})
        return result

    parsed = _parse_with_llm(message)
    data = parsed.model_dump()

    if update_requested:
        intent = "update_report"
        location_text = data.get("location_text") or _extract_location_heuristic(message) or pending_update.get("location_text")
        candidate_cedula = data.get("cedula") or cedula_from_text or pending_update.get("cedula")
        if candidate_cedula and not _is_plausible_cedula(candidate_cedula):
            candidate_cedula = None
        state_update: AgentState = {
            "intent": intent,
            "report_id": report_id_from_text or pending_update.get("report_id"),
            "cedula": candidate_cedula,
            "location_text": location_text,
            "latitude": data.get("latitude") if data.get("latitude") is not None else pending_update.get("latitude"),
            "longitude": data.get("longitude") if data.get("longitude") is not None else pending_update.get("longitude"),
        }
        missing: list[str] = []
        if not state_update.get("report_id"):
            missing.append("report_id")
        if not state_update.get("cedula"):
            missing.append("cedula")
        if not state_update.get("location_text") and (
            state_update.get("latitude") is None or state_update.get("longitude") is None
        ):
            missing.append("location")
        if missing:
            PENDING_UPDATES[session_id] = {
                "intent": "update_report",
                "report_id": state_update.get("report_id"),
                "cedula": state_update.get("cedula"),
                "location_text": state_update.get("location_text"),
                "latitude": state_update.get("latitude"),
                "longitude": state_update.get("longitude"),
            }
            state_update["response"] = _build_update_missing_prompt(missing, message)
            _log_event("update.missing_fields", state_update, {"missing": missing})
            return state_update
        PENDING_UPDATES.pop(session_id, None)
        _log_event("update.ready", state_update)
        return state_update

    if report_id_query:
        result: AgentState = {"intent": "get_report_by_id", "report_id": report_id_from_text}
        _log_event("parse.report_id", {**state, **result})
        return result

    intent = data.get("intent", "general_query")
    llm_query_scope = _normalize_query_scope(data.get("query_scope"))
    if intent in {"reports_near_location", "get_reports_by_cedula"}:
        intent = "query_reports"
    if explicit_query_requested or intent == "query_reports":
        intent = "query_reports"
    if intent == "query_reports" and not explicit_query_requested:
        if _looks_like_create_report(message) or _message_has_incident_info(message):
            intent = "create_report"
    if pending_report:
        intent = "create_report"
    elif intent == "general_query":
        if _looks_like_create_report(message) or (_message_has_incident_info(message) and not explicit_query_requested):
            intent = "create_report"

    query_scope = pending_query.get("query_scope") or explicit_scope_query
    if intent == "query_reports":
        if query_scope is None and llm_query_scope in {"mine", "nearby"}:
            query_scope = llm_query_scope
        elif query_scope is None and explicit_nearby_query:
            query_scope = "nearby"
        elif query_scope is None and explicit_cedula_query:
            query_scope = "mine"
        elif query_scope is None and cedula_from_text and (pending_query or explicit_query_requested):
            query_scope = "mine"
        elif query_scope is None and (
            report_type := _normalize_report_type(data.get("report_type")) or _infer_report_type(message)
        ):
            query_scope = "all"
        elif query_scope is None and _means_all_report_types(message):
            query_scope = "all"
    explicit_all_report_types = intent == "query_reports" and _means_all_report_types(message)
    report_type = None if explicit_all_report_types else (
        _normalize_report_type(data.get("report_type"))
        or _infer_report_type(message)
        or pending_query.get("report_type")
    )
    cedula = data.get("cedula") or cedula_from_text or pending_query.get("cedula")
    if intent == "create_report" and _message_has_incident_info(message):
        description = data.get("description") or message
    else:
        description = pending_report.get("description") or data.get("description")

    location_text = data.get("location_text") or _extract_location_heuristic(message)
    state_update: AgentState = {
        "intent": intent,
        "query_scope": query_scope if intent == "query_reports" else None,
        "cedula": cedula or pending_report.get("cedula"),
        "report_type": report_type or pending_report.get("report_type"),
        "description": description,
        "location_text": location_text or pending_query.get("location_text") or pending_report.get("location_text"),
        "latitude": (
            data.get("latitude")
            if data.get("latitude") is not None
            else pending_query.get("latitude", pending_report.get("latitude"))
        ),
        "longitude": (
            data.get("longitude")
            if data.get("longitude") is not None
            else pending_query.get("longitude", pending_report.get("longitude"))
        ),
        "radius_m": data.get("radius_m") or pending_query.get("radius_m"),
    }

    if state_update["intent"] == "create_report":
        PENDING_QUERY_REQUESTS.pop(session_id, None)
        missing: list[str] = []
        if not state_update.get("cedula"):
            missing.append("cedula")
        if not state_update.get("report_type"):
            missing.append("report_type")
        if not state_update.get("location_text") and (
            state_update.get("latitude") is None or state_update.get("longitude") is None
        ):
            missing.append("location")

        if missing:
            PENDING_REPORTS[session_id] = {
                "intent": "create_report",
                "cedula": state_update.get("cedula"),
                "report_type": state_update.get("report_type"),
                "description": state_update.get("description"),
                "location_text": state_update.get("location_text"),
                "latitude": state_update.get("latitude"),
                "longitude": state_update.get("longitude"),
            }
            state_update["response"] = _build_missing_prompt(missing, message)
            _log_event("parse.missing_fields", state_update, {"missing": missing})
            return state_update

        PENDING_REPORTS.pop(session_id, None)
        _log_event("parse.ready_create", state_update)
        return state_update

    if state_update["intent"] == "query_reports":
        PENDING_REPORTS.pop(session_id, None)
        if not state_update.get("query_scope") and state_update.get("location_text"):
            state_update["query_scope"] = "all"
        if not state_update.get("query_scope"):
            PENDING_QUERY_REQUESTS[session_id] = _snapshot_query_state(state_update)
            state_update["response"] = _build_query_scope_prompt()
            _log_event("query.await_scope", state_update)
            return state_update
        if state_update.get("query_scope") == "mine" and not state_update.get("cedula"):
            PENDING_QUERY_REQUESTS[session_id] = _snapshot_query_state(state_update)
            state_update["response"] = _build_query_missing_cedula_prompt()
            _log_event("query.missing_cedula", state_update)
            return state_update
        if state_update.get("query_scope") == "nearby" and not state_update.get("location_text") and (
            state_update.get("latitude") is None or state_update.get("longitude") is None
        ):
            PENDING_QUERY_REQUESTS[session_id] = _snapshot_query_state(state_update)
            state_update["response"] = _build_query_missing_location_prompt()
            _log_event("query.missing_location", state_update)
            return state_update
        PENDING_QUERY_REQUESTS.pop(session_id, None)
        _log_event("parse.ready_query", state_update)
        return state_update

    _log_event("parse.done", state_update)
    return state_update


def geocode_node(state: AgentState) -> AgentState:
    _log_event("geocode.start", state)
    if state.get("latitude") is not None and state.get("longitude") is not None:
        _log_event("geocode.skip", state, {"reason": "coords_present"})
        return state
    location_text = state.get("location_text")
    if not location_text:
        state["response"] = _build_missing_prompt(["location"], state.get("user_message", ""))
        _log_event("geocode.missing_location", state)
        return state
    resolution = _resolve_location(location_text)
    if resolution["status"] == "confirm":
        candidates = resolution.get("candidates") or []
        if candidates:
            top = candidates[0]
            state["latitude"] = top["latitude"]
            state["longitude"] = top["longitude"]
            state["location_text"] = _format_place_candidate(top)
            _log_event(
                "geocode.auto_select",
                state,
                {"top_candidate": state["location_text"], "candidate_count": len(candidates)},
            )
            return state
    if resolution["status"] != "resolved":
        error_code = resolution.get("error")
        if state.get("intent") == "create_report":
            session_id = state.get("session_id") or "default"
            PENDING_REPORTS[session_id] = {
                "intent": "create_report",
                "cedula": state.get("cedula"),
                "report_type": state.get("report_type"),
                "description": state.get("description"),
                "location_text": state.get("location_text"),
                "latitude": state.get("latitude"),
                "longitude": state.get("longitude"),
            }
        elif state.get("intent") == "update_report":
            session_id = state.get("session_id") or "default"
            PENDING_UPDATES[session_id] = {
                "intent": "update_report",
                "report_id": state.get("report_id"),
                "cedula": state.get("cedula"),
                "location_text": state.get("location_text"),
                "latitude": state.get("latitude"),
                "longitude": state.get("longitude"),
            }
        elif state.get("intent") == "query_reports":
            session_id = state.get("session_id") or "default"
            PENDING_QUERY_REQUESTS[session_id] = _snapshot_query_state(state)
        if error_code == "MISSING_API_KEY":
            state["response"] = (
                "Necesito configurar GOOGLE_MAPS_API_KEY para usar Google Places. "
                "Agrega la clave en el .env y reinicia el servidor."
            )
            _log_event("geocode.error", state, {"error": error_code})
            return state
        if error_code in {"REQUEST_DENIED", "INVALID_REQUEST"}:
            state["response"] = (
                "Google Places rechazo la solicitud. Verifica que la API key sea valida, "
                "tenga facturacion activa y que Places API este habilitada."
            )
            _log_event("geocode.error", state, {"error": error_code})
            return state
        if error_code == "OVER_QUERY_LIMIT":
            state["response"] = (
                "Se alcanzo el limite de consultas de Google Places. Intenta mas tarde."
            )
            _log_event("geocode.error", state, {"error": error_code})
            return state
        state["response"] = _build_location_refine_prompt(location_text)
        _log_event("geocode.not_found", state, {"error": error_code})
        return state
    geo = resolution["geo"]
    state["latitude"] = geo["lat"]
    state["longitude"] = geo["lon"]
    if resolution.get("resolved_location_text"):
        state["location_text"] = resolution["resolved_location_text"]
    _log_event("geocode.success", state)
    return state


def duplicate_check_node(state: AgentState) -> AgentState:
    _log_event("duplicate_check.start", state)
    description = state.get("description") or ""
    location_text = state.get("location_text") or ""
    report_type = state.get("report_type") or ""
    embedding_text = f"{report_type}. {description}. {location_text}".strip()
    embedding: Optional[List[float]] = None
    try:
        _log_tool_call("embed_text", {"text": _truncate(embedding_text)})
        embedding = embed_text(embedding_text)
        _log_tool_result("embed_text", {"dimensions": len(embedding)})
        state["embedding"] = embedding
    except Exception as exc:
        _log_tool_result("embed_text", {"error": str(exc)})
        state["embedding"] = None

    if embedding is None:
        state["response"] = "No pude validar duplicados ahora. Creare el reporte igualmente."
        _log_event("duplicate_check.skip", state, {"reason": "embedding_failed"})
        return state

    with SessionLocal() as db:
        _log_tool_call("search_similar_reports", {"limit": 3})
        similar = search_similar_reports(db, embedding, limit=3)
        _log_tool_result("search_similar_reports", {"count": len(similar)})

    if similar:
        top = similar[0]
        if top.get("similarity", 0) >= settings.duplicate_threshold:
            session_id = state.get("session_id") or "default"
            PENDING_DUPLICATES[session_id] = {
                "report_id": top.get("id"),
                "parsed": {
                    "intent": "create_report",
                    "cedula": state.get("cedula"),
                    "report_type": state.get("report_type"),
                    "description": state.get("description"),
                    "location_text": state.get("location_text"),
                    "latitude": state.get("latitude"),
                    "longitude": state.get("longitude"),
                },
            }
            state["response"] = (
                "Ya existe un reporte similar cercano:\n"
                f"Reporte #{top.get('id')} - {top.get('description')}\n"
                "¿Desea crear otro reporte o unirse al existente? "
                "Responde 'unirme al #ID' o 'crear nuevo'."
            )
            _log_event("duplicate_check.hit", state, {"top_id": top.get("id"), "similarity": top.get("similarity")})
            return state

    _log_event("duplicate_check.miss", state, {"similar_count": len(similar)})
    return state


def create_report_node(state: AgentState) -> AgentState:
    _log_event("create_report.start", state)
    description = state.get("description")
    if not description:
        report_type = state.get("report_type") or "reporte"
        location_text = state.get("location_text")
        if location_text:
            description = f"{report_type} en {location_text}"
        else:
            description = f"{report_type} reportado"
    if state.get("embedding") is None:
        report_type = state.get("report_type") or ""
        location_text = state.get("location_text") or ""
        embedding_text = f"{report_type}. {description}. {location_text}".strip()
        try:
            _log_tool_call("embed_text", {"text": _truncate(embedding_text)})
            embedding = embed_text(embedding_text)
            _log_tool_result("embed_text", {"dimensions": len(embedding)})
            state["embedding"] = embedding
        except Exception as exc:
            _log_tool_result("embed_text", {"error": str(exc)})
            state["embedding"] = None
    data = {
        "cedula": state.get("cedula"),
        "report_type": state.get("report_type"),
        "description": description,
        "location_text": state.get("location_text"),
        "latitude": state.get("latitude"),
        "longitude": state.get("longitude"),
    }
    with SessionLocal() as db:
        _log_tool_call("create_report", data)
        report = create_report(db, data, embedding=state.get("embedding"))
        _log_tool_result("create_report", {"id": report.id, "status": report.status})

    state["response"] = (
        f"Reporte creado: #{report.id} ({report.report_type}). "
        f"Estado: {report.status}. Gracias por ayudar."
    )
    _log_event("create_report.done", state, {"report_id": report.id})
    return state


def update_report_node(state: AgentState) -> AgentState:
    _log_event("update_report.start", state)
    report_id = state.get("report_id")
    cedula = state.get("cedula")
    if not report_id:
        state["response"] = "Indica el numero de reporte para actualizar la ubicacion."
        _log_event("update_report.missing_id", state)
        return state
    if not cedula:
        state["response"] = "Necesito tu cedula para validar la actualizacion."
        _log_event("update_report.missing_cedula", state)
        return state
    if state.get("latitude") is None or state.get("longitude") is None:
        state["response"] = "Necesito la ubicacion o coordenadas para actualizar el reporte."
        _log_event("update_report.missing_location", state)
        return state

    with SessionLocal() as db:
        _log_tool_call("get_report_by_id", {"report_id": report_id})
        report = get_report_by_id(db, report_id)
        if not report:
            state["response"] = f"No encontre el reporte #{report_id}."
            PENDING_UPDATES.pop(state.get("session_id") or "default", None)
            _log_event("update_report.not_found", state)
            return state
        if report.cedula != cedula:
            state["response"] = "La cedula no coincide con el reporte. Verifica e intenta de nuevo."
            PENDING_UPDATES.pop(state.get("session_id") or "default", None)
            _log_event("update_report.cedula_mismatch", state)
            return state
        _log_tool_call(
            "update_report_location",
            {
                "report_id": report_id,
                "location_text": state.get("location_text"),
                "lat": state.get("latitude"),
                "lon": state.get("longitude"),
            },
        )
        updated = update_report_location(
            db,
            report_id,
            location_text=state.get("location_text") or report.location_text,
            lat=state.get("latitude"),
            lon=state.get("longitude"),
        )
        if not updated:
            state["response"] = f"No pude actualizar el reporte #{report_id}."
            PENDING_UPDATES.pop(state.get("session_id") or "default", None)
            _log_event("update_report.failed", state)
            return state

    state["response"] = f"Reporte #{report_id} actualizado con la nueva ubicacion."
    PENDING_UPDATES.pop(state.get("session_id") or "default", None)
    _log_event("update_report.done", state, {"report_id": report_id})
    return state


def get_report_by_id_node(state: AgentState) -> AgentState:
    _log_event("get_report_by_id.start", state)
    report_id = state.get("report_id") or _extract_report_id(state.get("user_message", "") or "")
    if not report_id:
        state["response"] = "Indica el numero de reporte para consultar."
        _log_event("get_report_by_id.missing", state)
        return state
    with SessionLocal() as db:
        _log_tool_call("get_report_by_id", {"report_id": report_id})
        report = get_report_by_id(db, report_id)
        if not report:
            state["response"] = f"No encontre el reporte #{report_id}."
            _log_event("get_report_by_id.not_found", state)
            return state
    state["response"] = (
        f"#{report.id} {report.report_type} - {report.status}. "
        f"{report.description} ({report.location_text})"
    )
    _log_event("get_report_by_id.done", state, {"report_id": report_id})
    return state


def query_reports_node(state: AgentState) -> AgentState:
    _log_event("query_reports.start", state)
    scope = state.get("query_scope")
    report_type = state.get("report_type")
    if scope == "nearby" and (state.get("latitude") is None or state.get("longitude") is None):
        state["response"] = "Necesito una ubicacion para buscar reportes cercanos."
        _log_event("query_reports.missing_location", state)
        return state
    with SessionLocal() as db:
        if scope == "mine":
            cedula = state.get("cedula")
            _log_tool_call("get_reports_by_cedula", {"cedula": cedula, "report_type": report_type})
            results = get_reports_by_cedula(db, cedula, report_type=report_type)
            _log_tool_result("get_reports_by_cedula", {"count": len(results)})
            header = "Reportes encontrados:"
            empty_message = "No encontre reportes para tu cedula con esos filtros."
            lines = [f"#{r['id']} {r['report_type']} - {r['status']}" for r in results]
        elif scope == "nearby":
            lat = state.get("latitude")
            lon = state.get("longitude")
            radius = state.get("radius_m") or settings.nearby_radius_m
            _log_tool_call(
                "reports_near_location",
                {"lat": lat, "lon": lon, "radius_m": radius, "report_type": report_type},
            )
            results = reports_near_location(
                db,
                lat=lat,
                lon=lon,
                radius_m=radius,
                report_type=report_type,
            )
            _log_tool_result("reports_near_location", {"count": len(results), "radius_m": radius})
            if not results:
                expanded_radius = max(radius * 3, radius + 500)
                _log_tool_call(
                    "reports_near_location",
                    {"lat": lat, "lon": lon, "radius_m": expanded_radius, "report_type": report_type},
                )
                results = reports_near_location(
                    db,
                    lat=lat,
                    lon=lon,
                    radius_m=expanded_radius,
                    report_type=report_type,
                )
                _log_tool_result("reports_near_location", {"count": len(results), "radius_m": expanded_radius})
                if results:
                    header = f"Reportes cercanos (amplie el radio a {expanded_radius}m):"
                else:
                    header = "Reportes cercanos:"
            else:
                header = "Reportes cercanos:"
            empty_message = "No encontre reportes cercanos con esos filtros."
            if not results and state.get("location_text"):
                _log_tool_call("list_reports", {"report_type": report_type, "limit": settings.max_results})
                fallback = list_reports(db, report_type=report_type)
                _log_tool_result("list_reports", {"count": len(fallback)})
                results = [
                    r
                    for r in fallback
                    if _location_text_matches(state.get("location_text") or "", r.get("location_text") or "")
                ]
                if results:
                    header = "Reportes relacionados por ubicacion:"
            lines = [f"#{r['id']} {r['report_type']} - {r['description']}" for r in results]
        else:
            # Default: list or semantic search
            message = state.get("user_message", "")
            location_query = state.get("location_text")
            if location_query:
                _log_tool_call("list_reports", {"report_type": report_type, "limit": settings.max_results})
                fallback = list_reports(db, report_type=report_type)
                _log_tool_result("list_reports", {"count": len(fallback)})
                results = [
                    r
                    for r in fallback
                    if _location_text_matches(location_query, r.get("location_text") or "")
                ]
                if results:
                    header = "Reportes relacionados por ubicacion:"
                    empty_message = "No encontre reportes para esa ubicacion."
                    lines = [f"#{r['id']} {r['report_type']} - {r['description']}" for r in results]
                    state["response"] = header + "\n" + "\n".join(lines)
                    _log_event("query_reports.done", state, {"count": len(results)})
                    return state
            if _looks_like_listing_query(message) or _looks_like_report_query(message):
                _log_tool_call("list_reports", {"report_type": report_type, "limit": settings.max_results})
                results = list_reports(db, report_type=report_type)
                _log_tool_result("list_reports", {"count": len(results)})
                header = "Reportes recientes:"
                empty_message = "No encontre reportes recientes con esos filtros."
                lines = [f"#{r['id']} {r['report_type']} - {r['status']}" for r in results]
            else:
                _log_tool_call("rag_search", {"query": _truncate(message)})
                results = rag_search(message)
                _log_tool_result("rag_search", {"count": len(results)})
                header = "Reportes relacionados:"
                empty_message = "No encontre reportes relacionados."
                lines = [f"#{r['id']} {r['report_type']} - {r['description']}" for r in results]

    if not results:
        state["response"] = empty_message
        _log_event("query_reports.empty", state)
        return state

    state["response"] = header + "\n" + "\n".join(lines)
    _log_event("query_reports.done", state, {"count": len(results)})
    return state


def reports_near_location_node(state: AgentState) -> AgentState:
    _log_event("reports_near.start", state)
    if state.get("latitude") is None or state.get("longitude") is None:
        state["response"] = "Necesito latitud y longitud para buscar reportes cercanos."
        _log_event("reports_near.missing_coords", state)
        return state
    with SessionLocal() as db:
        _log_tool_call(
            "reports_near_location",
            {"lat": state.get("latitude"), "lon": state.get("longitude"), "radius_m": state.get("radius_m")},
        )
        results = reports_near_location(
            db,
            lat=state.get("latitude"),
            lon=state.get("longitude"),
            radius_m=state.get("radius_m"),
        )
        _log_tool_result("reports_near_location", {"count": len(results)})
    if not results:
        state["response"] = "No encontre reportes cercanos en ese radio."
        _log_event("reports_near.empty", state)
        return state
    lines = [
        f"#{r['id']} {r['report_type']} - {r['description']}"
        for r in results
    ]
    state["response"] = "Reportes cercanos:\n" + "\n".join(lines)
    _log_event("reports_near.done", state, {"count": len(results)})
    return state


def get_reports_by_cedula_node(state: AgentState) -> AgentState:
    _log_event("reports_by_cedula.start", state)
    cedula = state.get("cedula") or _extract_cedula(state.get("user_message", ""))
    if not cedula:
        state["response"] = "Indica la cedula para buscar tus reportes."
        _log_event("reports_by_cedula.missing", state)
        return state
    with SessionLocal() as db:
        _log_tool_call("get_reports_by_cedula", {"cedula": cedula})
        results = get_reports_by_cedula(db, cedula)
        _log_tool_result("get_reports_by_cedula", {"count": len(results)})
    if not results:
        state["response"] = f"No hay reportes para la cedula {cedula}."
        _log_event("reports_by_cedula.empty", state)
        return state
    lines = [
        f"#{r['id']} {r['report_type']} - {r['status']}"
        for r in results
    ]
    state["response"] = f"Reportes de {cedula}:\n" + "\n".join(lines)
    _log_event("reports_by_cedula.done", state, {"count": len(results)})
    return state


def general_query_node(state: AgentState) -> AgentState:
    _log_event("general_query.start", state)
    query = state.get("user_message", "")
    if not (
        _looks_like_report_query(query)
        or _looks_like_nearby_query(query)
        or _looks_like_cedula_query(query)
        or _message_has_incident_info(query)
    ):
        state["response"] = _free_chat_response(query)
        _log_event("general_query.free_chat", state)
        return state
    _log_tool_call("rag_search", {"query": _truncate(query)})
    results = rag_search(query)
    _log_tool_result("rag_search", {"count": len(results)})
    if not results:
        state["response"] = "No encontre reportes relacionados."
        _log_event("general_query.empty", state)
        return state
    lines = [
        f"#{r['id']} {r['report_type']} - {r['description']}"
        for r in results
    ]
    state["response"] = "Reportes relacionados:\n" + "\n".join(lines)
    _log_event("general_query.done", state, {"count": len(results)})
    return state


def free_chat_node(state: AgentState) -> AgentState:
    message = state.get("user_message", "")
    state["response"] = _free_chat_response(message)
    _log_event("free_chat.done", state)
    return state


def route_after_parse(state: AgentState) -> str:
    if state.get("response"):
        _log_event("route.after_parse", state, {"next": "respond_only"})
        return "respond_only"
    intent = state.get("intent", "general_query")
    if intent == "free_chat":
        _log_event("route.after_parse", state, {"next": "free_chat"})
        return "free_chat"
    if intent == "get_report_by_id":
        _log_event("route.after_parse", state, {"next": "get_report_by_id"})
        return "get_report_by_id"
    skip_duplicate = state.get("skip_duplicate") is True
    needs_geo = (
        intent == "create_report"
        or intent == "reports_near_location"
        or intent == "update_report"
        or (intent == "query_reports" and state.get("query_scope") == "nearby")
    ) and (
        state.get("latitude") is None or state.get("longitude") is None
    )
    if needs_geo:
        _log_event("route.after_parse", state, {"next": "geocode"})
        return "geocode"
    if intent == "update_report":
        _log_event("route.after_parse", state, {"next": "update_report"})
        return "update_report"
    if intent == "create_report":
        if skip_duplicate:
            _log_event("route.after_parse", state, {"next": "create_report"})
            return "create_report"
        _log_event("route.after_parse", state, {"next": "duplicate_check"})
        return "duplicate_check"
    if intent in {"query_reports", "reports_near_location", "get_reports_by_cedula"}:
        _log_event("route.after_parse", state, {"next": "query_reports"})
        return "query_reports"
    _log_event("route.after_parse", state, {"next": "general_query"})
    return "general_query"


def route_after_geocode(state: AgentState) -> str:
    if state.get("response"):
        _log_event("route.after_geocode", state, {"next": "respond_only"})
        return "respond_only"
    if state.get("intent") == "update_report":
        _log_event("route.after_geocode", state, {"next": "update_report"})
        return "update_report"
    if state.get("intent") == "create_report":
        if state.get("skip_duplicate") is True:
            _log_event("route.after_geocode", state, {"next": "create_report"})
            return "create_report"
        _log_event("route.after_geocode", state, {"next": "duplicate_check"})
        return "duplicate_check"
    _log_event("route.after_geocode", state, {"next": "query_reports"})
    return "query_reports"


def route_after_duplicate(state: AgentState) -> str:
    if state.get("response"):
        _log_event("route.after_duplicate", state, {"next": "respond_only"})
        return "respond_only"
    _log_event("route.after_duplicate", state, {"next": "create_report"})
    return "create_report"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse", _trace_node("parse", parse_node))
    graph.add_node("geocode", _trace_node("geocode", geocode_node))
    graph.add_node("duplicate_check", _trace_node("duplicate_check", duplicate_check_node))
    graph.add_node("create_report", _trace_node("create_report", create_report_node))
    graph.add_node("update_report", _trace_node("update_report", update_report_node))
    graph.add_node("get_report_by_id", _trace_node("get_report_by_id", get_report_by_id_node))
    graph.add_node("query_reports", _trace_node("query_reports", query_reports_node))
    graph.add_node("reports_near_location", _trace_node("reports_near_location", reports_near_location_node))
    graph.add_node("get_reports_by_cedula", _trace_node("get_reports_by_cedula", get_reports_by_cedula_node))
    graph.add_node("general_query", _trace_node("general_query", general_query_node))
    graph.add_node("free_chat", _trace_node("free_chat", free_chat_node))
    graph.add_node("respond_only", _trace_node("respond_only", lambda state: state))

    graph.set_entry_point("parse")

    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "respond_only": "respond_only",
            "free_chat": "free_chat",
            "geocode": "geocode",
            "create_report": "create_report",
            "update_report": "update_report",
            "get_report_by_id": "get_report_by_id",
            "duplicate_check": "duplicate_check",
            "query_reports": "query_reports",
            "reports_near_location": "reports_near_location",
            "get_reports_by_cedula": "get_reports_by_cedula",
            "general_query": "general_query",
        },
    )

    graph.add_conditional_edges(
        "geocode",
        route_after_geocode,
        {
            "respond_only": "respond_only",
            "update_report": "update_report",
            "create_report": "create_report",
            "duplicate_check": "duplicate_check",
            "query_reports": "query_reports",
            "reports_near_location": "reports_near_location",
        },
    )

    graph.add_conditional_edges(
        "duplicate_check",
        route_after_duplicate,
        {
            "respond_only": "respond_only",
            "create_report": "create_report",
        },
    )

    graph.add_edge("create_report", "respond_only")
    graph.add_edge("update_report", "respond_only")
    graph.add_edge("get_report_by_id", "respond_only")
    graph.add_edge("query_reports", "respond_only")
    graph.add_edge("reports_near_location", "respond_only")
    graph.add_edge("get_reports_by_cedula", "respond_only")
    graph.add_edge("general_query", "respond_only")
    graph.add_edge("free_chat", "respond_only")
    graph.add_edge("respond_only", END)

    return graph.compile()


AGENT = build_graph()


def run_agent(message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    state: AgentState = {"user_message": message, "session_id": session_id or "default"}
    _log_event("agent.run.start", state)
    result = AGENT.invoke(state)
    _log_event("agent.run.end", result)
    return {
        "response": result.get("response", ""),
        "data": {
            "intent": result.get("intent"),
            "cedula": result.get("cedula"),
        },
    }
