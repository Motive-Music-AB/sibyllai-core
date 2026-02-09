"""SibyllAI FastAPI Backend - Two-Phase Music Analysis API."""
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import uuid
import asyncio
from typing import Optional
import shutil
from concurrent.futures import ThreadPoolExecutor

# Add sibyllai_core and sibyllai_web to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))  # Add backend directory to path

from core.segmentation import segment_only
from core.analysis import analyze_segments
from core.library_index import (
    DEFAULT_DB_PATH,
    DEFAULT_HOP_RATIO,
    DEFAULT_WINDOW_SIZES,
    build_library_index,
    build_features_from_cue,
    build_vector_from_cue,
    get_library_info,
    load_meta_from_db,
    load_schema_from_db,
    normalize_key,
    match_cue_to_library,
)
from sibyllai_core.sibyl_format import load_project

app = FastAPI(
    title="SibyllAI API",
    version="1.0.0",
    description="Music analysis API with two-phase workflow"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded files and sessions
UPLOAD_DIR = Path("temp/uploads")
OUTPUT_DIR = Path("temp/outputs")
DEBUG_LOG_FILE = Path("temp/debug.log")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# WebSocket connections tracker
active_connections: dict[str, WebSocket] = {}

# Analysis status tracking (for progress updates)
analysis_status: dict[str, dict] = {}
library_status: dict[str, dict] = {}

# Thread pool for background analysis
executor = ThreadPoolExecutor(max_workers=2)


# Request/Response Models
class SegmentPreviewRequest(BaseModel):
    file_id: str
    music_thresh: float = 0.2
    min_gap: float = 5.0
    min_cue_length: float = 3.0
    silence_thresh: float = 0.01
    mix_type: str = "full_mix"  # "clean_mx" for music-only, "full_mix" for full film mix


class SegmentPreviewResponse(BaseModel):
    segments: list[tuple[float, float]]
    duration: float
    total_segments: int


class AnalyzeCuesRequest(BaseModel):
    file_id: str
    segments: list[tuple[float, float]]
    fps: int = 25
    threshold: float = 0.5


class AnalysisProgressUpdate(BaseModel):
    current: int
    total: int
    status: str
    progress_percent: float


class DebugLogEntry(BaseModel):
    type: str
    message: str
    timestamp: str


class CueUpdateRequest(BaseModel):
    """Request model for updating curated attributes of a cue."""
    instruments: Optional[list[str]] = None
    genres: Optional[list[str]] = None
    style: Optional[list[str]] = None
    moods: Optional[list[str]] = None


class CueReplacementRequest(BaseModel):
    track_path: str
    start: float
    end: float
    score: float
    window_size: float
    status: str = "matched"


class LibraryBuildRequest(BaseModel):
    folder: str
    window_sizes: Optional[list[float]] = None
    hop_ratio: float = DEFAULT_HOP_RATIO
    include_moods: bool = True
    reset: bool = True
    db_path: Optional[str] = None
    dedupe_simple: bool = True


class LibraryMatchRequest(BaseModel):
    session_id: str
    cue_id: str
    top_n: int = 3
    db_path: Optional[str] = None
    unique_tracks: bool = True


# Endpoints

@app.get("/")
async def root():
    """API health check."""
    return {
        "name": "SibyllAI API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/debug-logs")
async def post_debug_logs(logs: list[DebugLogEntry]):
    """
    Receive debug logs from frontend and append to debug.log file.
    Rotates log file if it exceeds 1000 lines to prevent unbounded growth.
    This allows Claude Code to read logs directly without screenshots.
    """
    try:
        # Rotate log if it gets too large (keep last 500 lines)
        if DEBUG_LOG_FILE.exists():
            with open(DEBUG_LOG_FILE, "r") as f:
                existing_lines = f.readlines()

            if len(existing_lines) > 1000:
                # Keep only last 500 lines
                with open(DEBUG_LOG_FILE, "w") as f:
                    f.writelines(existing_lines[-500:])

        # Append new logs
        with open(DEBUG_LOG_FILE, "a") as f:
            for log in logs:
                f.write(f"[{log.timestamp}] [{log.type.upper()}] {log.message}\n")

        return {"status": "ok", "logged": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write logs: {str(e)}")


@app.get("/api/debug-logs")
async def get_debug_logs():
    """Get recent debug logs (last 100 lines)."""
    try:
        if not DEBUG_LOG_FILE.exists():
            return {"logs": []}

        with open(DEBUG_LOG_FILE, "r") as f:
            lines = f.readlines()
            # Return last 100 lines
            return {"logs": lines[-100:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload audio/video file for analysis.
    Returns a file_id for subsequent operations.
    """
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_extension}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": file_path.stat().st_size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/segment-preview", response_model=SegmentPreviewResponse)
async def segment_preview(request: SegmentPreviewRequest):
    """
    Phase 1: Fast segmentation using YAMNet only.
    Returns detected segments for user review.
    """
    # Find uploaded file
    file_path = None
    for ext in [".wav", ".mp3", ".m4a", ".mp4", ".mov"]:
        candidate = UPLOAD_DIR / f"{request.file_id}{ext}"
        if candidate.exists():
            file_path = candidate
            break

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        segments, duration = segment_only(
            audio_path=file_path,
            music_thresh=request.music_thresh,
            min_gap=request.min_gap,
            min_cue_length=request.min_cue_length,
            silence_thresh=request.silence_thresh,
            mix_type=request.mix_type,
        )

        return SegmentPreviewResponse(
            segments=segments,
            duration=duration,
            total_segments=len(segments)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


def run_analysis_in_background(
    session_id: str,
    file_path: Path,
    segments: list,
    output_path: Path,
    fps: int,
    threshold: float
):
    """Run analysis in background thread and update status dict."""
    try:
        def progress_callback(current: int, total: int, status: str):
            # Update status dict (thread-safe for reads)
            analysis_status[session_id] = {
                "current": current,
                "total": total,
                "status": status,
                "progress_percent": (current / total) * 100 if total > 0 else 0,
                "complete": False,
                "error": None,
                "project": None
            }

        # Run analysis
        project = analyze_segments(
            audio_path=file_path,
            segments=segments,
            output_dir=output_path,
            fps=fps,
            thr=threshold,
            progress_callback=progress_callback
        )

        # Mark complete with project data
        analysis_status[session_id] = {
            "current": len(segments),
            "total": len(segments),
            "status": "Complete!",
            "progress_percent": 100,
            "complete": True,
            "error": None,
            "project": project
        }

    except Exception as e:
        # Mark as failed
        analysis_status[session_id] = {
            "current": 0,
            "total": len(segments),
            "status": f"Error: {str(e)}",
            "progress_percent": 0,
            "complete": True,
            "error": str(e),
            "project": None
        }


def run_library_build_in_background(job_id: str, request: LibraryBuildRequest, source_name: str | None = None, source_type: str | None = None):
    """Build library index in background thread and update status dict."""
    try:
        def progress_callback(update: dict):
            total = update.get("total_tracks", 0) or 0
            current = update.get("current_track", 0) or 0
            progress_percent = (current / total) * 100 if total > 0 else 0
            library_status[job_id] = {
                "status": "running",
                "progress_percent": progress_percent,
                "current": current,
                "total": total,
                "windows_indexed": update.get("windows_indexed", 0),
                "message": update.get("message", "Building index..."),
                "result": None,
                "error": None,
            }

        result = build_library_index(
            folder=Path(request.folder),
            db_path=Path(request.db_path) if request.db_path else DEFAULT_DB_PATH,
            window_sizes=request.window_sizes or DEFAULT_WINDOW_SIZES,
            hop_ratio=request.hop_ratio,
            include_moods=request.include_moods,
            reset=request.reset,
            progress_callback=progress_callback,
            source_name=source_name,
            source_type=source_type,
            dedupe_simple=request.dedupe_simple,
        )
        library_status[job_id] = {
            "status": "complete",
            "progress_percent": 100,
            "current": result.get("tracks_indexed", 0),
            "total": result.get("tracks_indexed", 0),
            "windows_indexed": result.get("windows_indexed", 0),
            "message": "Index build complete",
            "result": result,
            "error": None,
        }
    except Exception as e:
        library_status[job_id] = {
            "status": "error",
            "progress_percent": 0,
            "current": 0,
            "total": 0,
            "windows_indexed": 0,
            "message": "Index build failed",
            "result": None,
            "error": str(e),
        }


def _top_items(scores: dict[str, float], n: int = 3) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]]


def _build_match_reasons(cue_features: dict, match_features: dict) -> list[str]:
    reasons: list[str] = []

    # Tempo
    cue_bpm = cue_features.get("bpm")
    match_bpm = match_features.get("bpm")
    if isinstance(cue_bpm, (int, float)) and isinstance(match_bpm, (int, float)):
        diff = abs(cue_bpm - match_bpm)
        if diff <= 5:
            reasons.append("Similar tempo")
        elif diff <= 12:
            reasons.append("Close tempo")

    # Key
    cue_key = normalize_key(cue_features.get("key"))
    match_key = normalize_key(match_features.get("key"))
    if cue_key and match_key:
        if cue_key == match_key:
            reasons.append("Same key")
        elif cue_key[0] == match_key[0]:
            reasons.append("Parallel key")

    # Instruments overlap
    cue_inst = cue_features.get("instruments", {})
    match_inst = match_features.get("instruments", {})
    if cue_inst and match_inst:
        cue_top = set(_top_items(cue_inst, 3))
        match_top = set(_top_items(match_inst, 3))
        overlap = list(cue_top & match_top)
        if overlap:
            reasons.append(f"Instrument: {overlap[0]}")

    # CLAP genre/style overlap
    cue_clap = cue_features.get("clap", {}).get("genre", {})
    match_clap = match_features.get("clap", {}).get("genre", {})
    if cue_clap and match_clap:
        cue_top = _top_items(cue_clap, 1)
        match_top = _top_items(match_clap, 1)
        if cue_top and match_top and cue_top[0] == match_top[0]:
            reasons.append(f"Style: {cue_top[0]}")

    # Valence/arousal proximity
    cue_val = cue_features.get("valence")
    cue_aro = cue_features.get("arousal")
    match_val = match_features.get("valence")
    match_aro = match_features.get("arousal")
    if all(isinstance(v, (int, float)) for v in [cue_val, cue_aro, match_val, match_aro]):
        if abs(cue_val - match_val) <= 0.5 and abs(cue_aro - match_aro) <= 0.5:
            reasons.append("Similar mood")

    # Deduplicate and keep top 3
    out: list[str] = []
    for r in reasons:
        if r not in out:
            out.append(r)
        if len(out) >= 3:
            break
    return out


@app.post("/api/analyze-cues")
async def analyze_cues(request: AnalyzeCuesRequest):
    """
    Phase 2: Full analysis on confirmed segments.
    Runs expensive detectors (BPM, CLAP, mood, instruments, key).
    Returns immediately with session_id; analysis runs in background.
    """
    # Find uploaded file
    file_path = None
    for ext in [".wav", ".mp3", ".m4a", ".mp4", ".mov"]:
        candidate = UPLOAD_DIR / f"{request.file_id}{ext}"
        if candidate.exists():
            file_path = candidate
            break

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Create output directory for this analysis
    session_id = str(uuid.uuid4())
    output_path = OUTPUT_DIR / session_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize status
    analysis_status[session_id] = {
        "current": 0,
        "total": len(request.segments),
        "status": "Starting analysis...",
        "progress_percent": 0,
        "complete": False,
        "error": None,
        "project": None
    }

    # Start analysis in background thread
    executor.submit(
        run_analysis_in_background,
        session_id,
        file_path,
        request.segments,
        output_path,
        request.fps,
        request.threshold
    )

    # Return immediately with session_id
    return {
        "session_id": session_id,
        "total_segments": len(request.segments),
        "output_path": str(output_path)
    }


@app.get("/api/analysis-status/{session_id}")
async def get_analysis_status(session_id: str):
    """Get current analysis progress for a session."""
    if session_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Session not found")

    status = analysis_status[session_id]

    # If complete and has project, return full response
    if status["complete"] and status["project"]:
        return {
            "session_id": session_id,
            "complete": True,
            "progress_percent": 100,
            "status": "Complete!",
            "project": status["project"],
            "error": None
        }
    elif status["complete"] and status["error"]:
        return {
            "session_id": session_id,
            "complete": True,
            "progress_percent": 0,
            "status": status["status"],
            "project": None,
            "error": status["error"]
        }
    else:
        return {
            "session_id": session_id,
            "complete": False,
            "current": status["current"],
            "total": status["total"],
            "progress_percent": status["progress_percent"],
            "status": status["status"],
            "project": None,
            "error": None
        }


@app.get("/api/projects/{session_id}")
async def get_project(session_id: str):
    """Load a previously analyzed project."""
    project_path = OUTPUT_DIR / session_id / "project.sibyl.json"

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        project = load_project(project_path)
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {str(e)}")


@app.websocket("/ws/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time analysis progress updates.
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]


@app.put("/api/projects/{session_id}/cues/{cue_id}")
async def update_cue(session_id: str, cue_id: str, update: CueUpdateRequest):
    """
    Update curated attributes for a specific cue.
    Persists changes to the project.sibyl.json file.
    """
    import json

    project_path = OUTPUT_DIR / session_id / "project.sibyl.json"

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        # Load existing project
        with open(project_path, 'r') as f:
            project = json.load(f)

        # Find and update the cue
        cue_found = False
        for cue in project.get("cues", []):
            if cue.get("id") == cue_id:
                cue_found = True
                curated = cue.get("musical_profile", {}).get("curated", {})

                # Update only the fields that were provided
                if update.instruments is not None:
                    curated["instruments"] = update.instruments
                if update.genres is not None:
                    curated["genres"] = update.genres
                if update.style is not None:
                    curated["style"] = update.style
                if update.moods is not None:
                    curated["moods"] = update.moods

                # Write back to cue
                if "musical_profile" not in cue:
                    cue["musical_profile"] = {}
                cue["musical_profile"]["curated"] = curated
                break

        if not cue_found:
            raise HTTPException(status_code=404, detail=f"Cue {cue_id} not found")

        # Save back to disk
        with open(project_path, 'w') as f:
            json.dump(project, f, indent=2)

        return {"status": "updated", "cue_id": cue_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update cue: {str(e)}")


@app.put("/api/projects/{session_id}/cues/{cue_id}/replacement")
async def update_cue_replacement(session_id: str, cue_id: str, update: CueReplacementRequest):
    """
    Update replacement info for a specific cue.
    Persists project_context replacement and status to project.sibyl.json.
    """
    import json

    project_path = OUTPUT_DIR / session_id / "project.sibyl.json"

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        with open(project_path, 'r') as f:
            project = json.load(f)

        cue_found = False
        for cue in project.get("cues", []):
            if cue.get("id") == cue_id:
                cue_found = True
                project_context = cue.get("project_context", {})
                project_context["status"] = update.status
                project_context["replacement"] = {
                    "track_path": update.track_path,
                    "start": update.start,
                    "end": update.end,
                    "score": update.score,
                    "window_size": update.window_size,
                }
                cue["project_context"] = project_context
                break

        if not cue_found:
            raise HTTPException(status_code=404, detail=f"Cue {cue_id} not found")

        with open(project_path, 'w') as f:
            json.dump(project, f, indent=2)

        return {"status": "updated", "cue_id": cue_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update replacement: {str(e)}")


@app.post("/api/library/build")
async def build_library(request: LibraryBuildRequest):
    """
    Build or rebuild the local library index from a folder of audio files.
    Runs in a background thread and returns a job_id for polling.
    """
    folder = Path(request.folder)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder not found or not a directory")

    job_id = str(uuid.uuid4())
    library_status[job_id] = {
        "status": "running",
        "progress_percent": 0,
        "current": 0,
        "total": 0,
        "windows_indexed": 0,
        "message": "Starting index build",
        "result": None,
        "error": None,
    }

    executor.submit(
        run_library_build_in_background,
        job_id,
        request,
        Path(request.folder).name,
        "path",
    )

    return {"job_id": job_id, "status": "running"}


@app.post("/api/library/build-upload")
async def build_library_upload(
    files: list[UploadFile] = File(...),
    include_moods: bool = Form(True),
    reset: bool = Form(True),
    dedupe_simple: bool = Form(True),
):
    """
    Build library index by uploading a folder of audio files from the client.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    job_id = str(uuid.uuid4())
    upload_root = Path("temp/library_uploads") / job_id
    upload_root.mkdir(parents=True, exist_ok=True)

    library_status[job_id] = {
        "status": "running",
        "progress_percent": 0,
        "current": 0,
        "total": 0,
        "windows_indexed": 0,
        "message": "Uploading files...",
        "result": None,
        "error": None,
    }

    source_name = None
    for upload in files:
        rel_path = Path(upload.filename)
        if source_name is None and len(rel_path.parts) > 0:
            source_name = rel_path.parts[0]
        dest = upload_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

    request = LibraryBuildRequest(
        folder=str(upload_root),
        include_moods=include_moods,
        reset=reset,
        dedupe_simple=dedupe_simple,
    )

    executor.submit(
        run_library_build_in_background,
        job_id,
        request,
        source_name,
        "upload",
    )

    return {"job_id": job_id, "status": "running"}


@app.get("/api/library/status/{job_id}")
async def library_build_status(job_id: str):
    if job_id not in library_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return library_status[job_id]


@app.get("/api/library/info")
async def library_info(db_path: Optional[str] = None):
    """Return info about the current library index if it exists."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    return get_library_info(path)


@app.post("/api/library/match")
async def match_library(request: LibraryMatchRequest):
    """
    Match a cue from an analyzed project to the library index.
    Returns top N matches with in-track time ranges.
    """
    db_path = Path(request.db_path) if request.db_path else DEFAULT_DB_PATH
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Library index not found")

    project_path = OUTPUT_DIR / request.session_id / "project.sibyl.json"
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = load_project(project_path)
    cue = next((c for c in project.get("cues", []) if c.get("id") == request.cue_id), None)
    if not cue:
        raise HTTPException(status_code=404, detail="Cue not found")

    schema = load_schema_from_db(db_path)
    meta = load_meta_from_db(db_path)
    include_moods = bool(meta.get("include_moods", True))
    window_sizes = meta.get("window_sizes", DEFAULT_WINDOW_SIZES)
    cue_features = build_features_from_cue(cue)
    cue_vector = build_vector_from_cue(cue, schema, include_moods=include_moods)
    cue_length = float(cue.get("end", 0.0) - cue.get("start", 0.0))

    matches = match_cue_to_library(
        cue_vector=cue_vector,
        cue_length=cue_length,
        db_path=db_path,
        top_n=request.top_n,
        window_sizes=window_sizes,
        unique_tracks=request.unique_tracks,
        include_features=True,
    )

    for match in matches:
        features = match.pop("features", {})
        match["reasons"] = _build_match_reasons(cue_features, features)
        # Include key metadata for display
        match["bpm"] = features.get("bpm")
        match["key"] = features.get("key")
        # Get top instruments (sorted by confidence)
        instruments = features.get("instruments", {})
        if instruments:
            sorted_inst = sorted(instruments.items(), key=lambda x: x[1], reverse=True)[:5]
            match["instruments"] = [name for name, _ in sorted_inst]
        else:
            match["instruments"] = []
        # Get top genres
        genres = features.get("genres", {})
        if genres:
            sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
            match["genres"] = [name for name, _ in sorted_genres]
        else:
            match["genres"] = []

    return {
        "cue_id": request.cue_id,
        "cue_length": cue_length,
        "matches": matches,
    }


# Optional: Cleanup endpoint
@app.delete("/api/cleanup/{file_id}")
async def cleanup_files(file_id: str):
    """Remove uploaded file and associated outputs."""
    try:
        # Remove uploaded file
        for ext in [".wav", ".mp3", ".m4a", ".mp4", ".mov"]:
            file_path = UPLOAD_DIR / f"{file_id}{ext}"
            if file_path.exists():
                file_path.unlink()

        return {"status": "cleaned", "file_id": file_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
