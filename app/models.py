from __future__ import annotations

from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from .database import Base
from .config import settings


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    cedula = Column(String(20), index=True, nullable=False)
    report_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    location_text = Column(Text)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    priority = Column(String, nullable=False, default="medium")
    status = Column(String, nullable=False, default="pendiente")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    embedding = Column(Vector(settings.embedding_dimensions))
