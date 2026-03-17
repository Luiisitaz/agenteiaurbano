from __future__ import annotations

PRIORITY_BY_REPORT_TYPE = {
    "trafico_accidente": "high",
    "fuga_de_agua": "medium",
    "bache": "low",
    "otros": "medium",
    "luz_de_trafico_rota": "medium",
}

PRIORITY_ORDER = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def priority_from_report_type(report_type: str | None) -> str:
    normalized = (report_type or "").strip().lower()
    return PRIORITY_BY_REPORT_TYPE.get(normalized, "medium")


def priority_case_sql(column_name: str = "report_type") -> str:
    return (
        f"CASE {column_name} "
        "WHEN 'trafico_accidente' THEN 'high' "
        "WHEN 'fuga_de_agua' THEN 'medium' "
        "WHEN 'bache' THEN 'low' "
        "ELSE 'medium' "
        "END"
    )


def priority_rank_sql(column_name: str = "priority") -> str:
    return (
        f"CASE {column_name} "
        "WHEN 'high' THEN 0 "
        "WHEN 'medium' THEN 1 "
        "WHEN 'low' THEN 2 "
        "ELSE 3 "
        "END"
    )
