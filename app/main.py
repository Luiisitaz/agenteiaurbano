from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse, ReportOut, ReportMapItem, ReportClusterItem
from .database import get_db
from .tools import list_reports_for_map, get_report_by_id, get_report_clusters
from .agent import run_agent

app = FastAPI(title="Urban AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent
APP_STATIC_DIR = ROOT_DIR / "static" / "app"

app.mount("/assets/app", StaticFiles(directory=APP_STATIC_DIR), name="app-assets")


@app.get("/")
def root():
    return {"status": "Urban AI Agent running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message, req.session_id)
    return ChatResponse(**result)


@app.get("/reports/map", response_model=list[ReportMapItem])
def reports_map(db=Depends(get_db)):
    items = list_reports_for_map(db)
    return [ReportMapItem(**item) for item in items]


@app.get("/reports/clusters", response_model=list[ReportClusterItem])
def reports_clusters(
    scope: str = Query(default="all", pattern="^(all|mine|nearby)$"),
    cedula: str | None = None,
    report_type: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_m: int | None = None,
    limit: int | None = None,
    db=Depends(get_db),
):
    if scope == "mine" and not cedula:
        raise HTTPException(status_code=400, detail="cedula is required when scope=mine")
    if scope == "nearby" and (lat is None or lon is None):
        raise HTTPException(status_code=400, detail="lat and lon are required when scope=nearby")

    items = get_report_clusters(
        db,
        scope=scope,
        cedula=cedula if scope == "mine" else None,
        report_type=report_type,
        lat=lat if scope == "nearby" else None,
        lon=lon if scope == "nearby" else None,
        radius_m=radius_m if scope == "nearby" else None,
        limit=limit,
    )
    return [ReportClusterItem(**item) for item in items]


@app.get("/reports/{report_id}", response_model=ReportOut)
def report_detail(report_id: int, db=Depends(get_db)):
    report = get_report_by_id(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/map", response_class=HTMLResponse)
def map_page():
    html_path = ROOT_DIR / "static" / "map.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/app", response_class=HTMLResponse)
def app_page():
    html_path = APP_STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
