from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import text
from pgvector.psycopg2.vector import Vector
from sqlalchemy.orm import Session

from .models import Report
from .config import settings
from .reporting import priority_from_report_type, priority_rank_sql


def create_report(db: Session, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> Report:
    report = Report(
        cedula=data["cedula"],
        report_type=data["report_type"],
        description=data["description"],
        location_text=data.get("location_text"),
        latitude=data["latitude"],
        longitude=data["longitude"],
        priority=data.get("priority") or priority_from_report_type(data.get("report_type")),
        status=data.get("status", "pendiente"),
        embedding=embedding,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def search_similar_reports(
    db: Session, embedding: List[float], limit: int = 3
) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT id, report_type, description, location_text, latitude, longitude,
               1 - (embedding <=> :embedding) AS similarity
        FROM reports
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> :embedding
        LIMIT :limit
        """
    )
    rows = db.execute(sql, {"embedding": Vector(embedding), "limit": limit}).mappings().all()
    return [dict(row) for row in rows]


def reports_near_location(
    db: Session,
    lat: float,
    lon: float,
    radius_m: int | None = None,
    limit: int | None = None,
    report_type: str | None = None,
) -> List[Dict[str, Any]]:
    radius = radius_m or settings.nearby_radius_m
    max_results = limit or settings.max_results
    sql = text(
        """
        SELECT id, report_type, priority, description, location_text, latitude, longitude, status, created_at
        FROM reports
        WHERE ST_DWithin(
            ST_MakePoint(longitude, latitude)::geography,
            ST_MakePoint(:lon, :lat)::geography,
            :radius
        )
          AND (:report_type IS NULL OR report_type = :report_type)
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    rows = db.execute(
        sql,
        {
            "lat": lat,
            "lon": lon,
            "radius": radius,
            "limit": max_results,
            "report_type": report_type,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def get_reports_by_cedula(
    db: Session, cedula: str, limit: int | None = None, report_type: str | None = None
) -> List[Dict[str, Any]]:
    max_results = limit or settings.max_results
    sql = text(
        """
        SELECT id, report_type, priority, status, location_text, description, created_at
        FROM reports
        WHERE cedula = :cedula
          AND (:report_type IS NULL OR report_type = :report_type)
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    rows = db.execute(
        sql,
        {"cedula": cedula, "limit": max_results, "report_type": report_type},
    ).mappings().all()
    return [dict(row) for row in rows]


def list_reports(
    db: Session,
    limit: int | None = None,
    report_type: str | None = None,
) -> List[Dict[str, Any]]:
    max_results = limit or settings.max_results
    sql = text(
        """
        SELECT id, report_type, priority, status, location_text, description, created_at
        FROM reports
        WHERE (:report_type IS NULL OR report_type = :report_type)
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    rows = db.execute(
        sql,
        {"limit": max_results, "report_type": report_type},
    ).mappings().all()
    return [dict(row) for row in rows]


def get_report_by_id(db: Session, report_id: int) -> Optional[Report]:
    return db.query(Report).filter(Report.id == report_id).first()


def update_report_location(
    db: Session,
    report_id: int,
    *,
    location_text: str,
    lat: float,
    lon: float,
) -> Optional[Report]:
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return None
    report.location_text = location_text
    report.latitude = lat
    report.longitude = lon
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_report_clusters(
    db: Session,
    *,
    scope: str = "all",
    cedula: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_m: int | None = None,
    report_type: str | None = None,
    limit: int | None = None,
    cluster_radius_m: int | None = None,
) -> List[Dict[str, Any]]:
    max_results = limit or settings.max_results
    search_radius = radius_m or settings.nearby_radius_m
    cluster_radius = cluster_radius_m or settings.cluster_radius_m
    priority_rank = priority_rank_sql("priority")
    sql = text(
        f"""
        WITH filtered AS (
            SELECT
                id,
                cedula,
                report_type,
                priority,
                description,
                location_text,
                latitude,
                longitude,
                status,
                created_at,
                ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) AS geom_4326,
                ST_Transform(ST_SetSRID(ST_MakePoint(longitude, latitude), 4326), 3857) AS geom_3857
            FROM reports
            WHERE (:report_type IS NULL OR report_type = :report_type)
              AND (:cedula IS NULL OR cedula = :cedula)
              AND (
                  :scope <> 'nearby'
                  OR ST_DWithin(
                      ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
                      ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                      :radius
                  )
              )
        ),
        clustered AS (
            SELECT
                *,
                ST_ClusterDBSCAN(geom_3857, eps := :cluster_radius, minpoints := 1)
                    OVER (PARTITION BY report_type) AS cluster_num
            FROM filtered
        ),
        cluster_summaries AS (
            SELECT
                report_type,
                priority,
                COALESCE(cluster_num, -1) AS cluster_num,
                COUNT(*)::integer AS count,
                ST_Y(ST_Transform(ST_Centroid(ST_Collect(geom_3857)), 4326)) AS lat,
                ST_X(ST_Transform(ST_Centroid(ST_Collect(geom_3857)), 4326)) AS lon,
                (ARRAY_AGG(location_text ORDER BY created_at DESC) FILTER (WHERE location_text IS NOT NULL))[1] AS location_text,
                ARRAY_AGG(id ORDER BY created_at DESC)::integer[] AS report_ids,
                MAX(created_at) AS latest_created_at,
                {priority_rank} AS priority_rank
            FROM clustered
            GROUP BY report_type, priority, COALESCE(cluster_num, -1)
        ),
        ranked AS (
            SELECT
                CONCAT(
                    'cluster-',
                    ROW_NUMBER() OVER (ORDER BY priority_rank, latest_created_at DESC, report_type, cluster_num)
                ) AS cluster_id,
                report_type,
                priority,
                count,
                lat,
                lon,
                location_text,
                report_ids,
                latest_created_at,
                priority_rank
            FROM cluster_summaries
        )
        SELECT
            cluster_id,
            report_type,
            priority,
            count,
            lat,
            lon,
            location_text,
            report_ids,
            latest_created_at
        FROM ranked
        ORDER BY priority_rank, latest_created_at DESC
        LIMIT :limit
        """
    )
    rows = db.execute(
        sql,
        {
            "scope": scope,
            "cedula": cedula,
            "lat": lat,
            "lon": lon,
            "radius": search_radius,
            "report_type": report_type,
            "limit": max_results,
            "cluster_radius": cluster_radius,
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def list_reports_for_map(db: Session, limit: int | None = None) -> List[Dict[str, Any]]:
    max_results = limit or settings.max_results
    sql = text(
        """
        SELECT id, latitude, longitude, report_type
        FROM reports
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    rows = db.execute(sql, {"limit": max_results}).mappings().all()
    return [
        {"id": row["id"], "lat": row["latitude"], "lon": row["longitude"], "type": row["report_type"]}
        for row in rows
    ]
