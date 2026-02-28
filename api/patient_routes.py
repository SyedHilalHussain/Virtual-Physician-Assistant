"""
Patient-facing API endpoints — login, chat, intake session management.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db
from database.crud import (
    get_patient_by_id, get_patient_by_name, create_patient, get_all_patients,
    get_last_intake_session,
)
from agent.graph import start_session, process_message, complete_intake, send_urgent_alert

router = APIRouter()

# ──────────────────────── IN-MEMORY SESSION STORE ────────────────────────
# Maps patient_id -> AgentState
# In production, use Redis or similar for multi-worker support
active_sessions: dict[int, dict] = {}


# ──────────────────────── REQUEST MODELS ────────────────────────

class PatientLoginRequest(BaseModel):
    patient_id: Optional[int] = None
    name: Optional[str] = None


class PatientRegisterRequest(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    care_type: str = "palliative"


class ChatMessageRequest(BaseModel):
    patient_id: int
    message: str


class StartConsultationRequest(BaseModel):
    patient_id: int


# ──────────────────────── ENDPOINTS ────────────────────────

@router.get("/list")
def list_patients(db: Session = Depends(get_db)):
    """List all registered patients."""
    patients = get_all_patients(db)
    return [
        {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "gender": p.gender,
            "diagnosis": p.diagnosis,
        }
        for p in patients
    ]


@router.post("/register")
def register_patient(req: PatientRegisterRequest, db: Session = Depends(get_db)):
    """Register a new patient."""
    patient = create_patient(
        db,
        name=req.name,
        age=req.age,
        gender=req.gender,
        diagnosis=req.diagnosis,
        care_type=req.care_type,
    )
    return {
        "id": patient.id,
        "name": patient.name,
        "message": f"Patient {patient.name} registered successfully.",
    }


@router.post("/login")
def patient_login(req: PatientLoginRequest, db: Session = Depends(get_db)):
    """
    Patient logs in. Returns patient info + last visit summary.
    Does NOT start conversation yet — that happens on /start-consultation.
    """
    patient = None
    if req.patient_id:
        patient = get_patient_by_id(db, req.patient_id)
    elif req.name:
        patient = get_patient_by_name(db, req.name)

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found. Please register first.")

    # Get last session summary for the welcome screen
    last_session = get_last_intake_session(db, patient.id)
    last_visit_summary = None

    if last_session:
        last_visit_summary = {
            "session_date": last_session.session_date.isoformat() if last_session.session_date else None,
            "pain_score": last_session.pain_score,
            "pain_location": last_session.pain_location,
            "medication_compliance": last_session.medication_compliance,
            "emotional_state": last_session.emotional_state,
            "sleep_quality": last_session.sleep_quality,
        }

    return {
        "patient_id": patient.id,
        "patient_name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "diagnosis": patient.diagnosis,
        "is_returning": last_session is not None,
        "last_visit_summary": last_visit_summary,
    }


@router.post("/start-consultation")
def start_consultation(req: StartConsultationRequest):
    """
    Called when patient clicks 'Start Consultation'.
    Runs memory recall + generates initial greeting from Noor.
    """
    # Start agent session — runs Memory Recall Node
    state = start_session(req.patient_id)
    active_sessions[req.patient_id] = state

    # Generate initial greeting
    state, greeting = process_message(state, "[Patient has logged in. Greet them warmly.]")
    active_sessions[req.patient_id] = state

    return {
        "greeting": greeting,
    }


@router.post("/chat")
def patient_chat(req: ChatMessageRequest):
    """
    Process a patient's chat message through the intake agent.
    Returns the AI response. Fires urgent email IMMEDIATELY on distress detection.
    """
    if req.patient_id not in active_sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Please start a consultation first."
        )

    state = active_sessions[req.patient_id]

    # Run Intake Interview Node
    state, ai_response = process_message(state, req.message)
    active_sessions[req.patient_id] = state

    # ── URGENT EMAIL: Fire IMMEDIATELY if distress detected ──
    if state.get("is_urgent") and not state.get("_urgent_notified"):
        send_urgent_alert(state)
        state["_urgent_notified"] = True
        active_sessions[req.patient_id] = state

    result = {
        "response": ai_response,
        "is_complete": state.get("is_complete", False),
        "is_urgent": state.get("is_urgent", False),
    }

    # If intake is complete, run Storage + Notification nodes
    if state.get("is_complete"):
        state = complete_intake(state)
        active_sessions[req.patient_id] = state
        result["session_id"] = state.get("session_id")
        result["message"] = "Intake complete. Doctor has been notified."

    return result


@router.get("/session/{patient_id}")
def get_session_status(patient_id: int):
    """Check if a patient has an active session."""
    if patient_id in active_sessions:
        state = active_sessions[patient_id]
        return {
            "active": True,
            "is_complete": state.get("is_complete", False),
            "message_count": len(state.get("messages", [])),
        }
    return {"active": False}
