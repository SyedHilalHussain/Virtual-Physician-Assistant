"""
FastAPI application entry point — MHP Cancer Pain Management Intake Agent.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api.patient_routes import router as patient_router
from api.doctor_routes import router as doctor_router
from api.voice_routes import router as voice_router

app = FastAPI(
    title="MHP Cancer Pain Management — Clinical Intake Agent",
    description="AI-powered intake agent for cancer pain management palliative care",
    version="3.0.0",  # v3: voice-first agent
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Include routers
app.include_router(patient_router, prefix="/api/patient", tags=["Patient"])
app.include_router(doctor_router, prefix="/api/doctor", tags=["Doctor"])
app.include_router(voice_router, tags=["Voice"])


@app.get("/")
async def root():
    """Login / signup page — entry point."""
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/patient")
async def patient_page():
    return FileResponse(os.path.join(frontend_path, "patient.html"))


@app.get("/doctor")
async def doctor_page():
    return FileResponse(os.path.join(frontend_path, "doctor.html"))
