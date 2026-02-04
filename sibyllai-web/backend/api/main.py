"""SibyllAI FastAPI Backend - Two-Phase Music Analysis API."""
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import uuid
import asyncio
from typing import Optional
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Add sibyllai_core and sibyllai_web to path
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))  # Add backend directory to path

from core.segmentation import segment_only
from core.analysis import analyze_segments
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

# Thread pool for background analysis
executor = ThreadPoolExecutor(max_workers=2)


# Request/Response Models
class SegmentPreviewRequest(BaseModel):
    file_id: str
    music_thresh: float = 0.2
    min_gap: float = 5.0
    min_cue_length: float = 3.0


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
            min_cue_length=request.min_cue_length
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
