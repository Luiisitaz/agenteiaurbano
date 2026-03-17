from __future__ import annotations

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None


class ReportOut(BaseModel):
    id: int
    cedula: str
    report_type: str
    priority: str
    description: str
    location_text: Optional[str]
    latitude: float
    longitude: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReportMapItem(BaseModel):
    id: int
    lat: float
    lon: float
    type: str


class ReportListItem(BaseModel):
    id: int
    report_type: str
    priority: str
    status: str
    location_text: Optional[str]
    created_at: datetime


class ReportClusterItem(BaseModel):
    cluster_id: str
    report_type: str
    priority: str
    count: int
    lat: float
    lon: float
    location_text: Optional[str]
    report_ids: List[int]
    latest_created_at: datetime
