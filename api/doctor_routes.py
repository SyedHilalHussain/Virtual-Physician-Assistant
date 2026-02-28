"""
Doctor-facing API endpoints — portal, summaries, notifications, Ask Agent.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db
from database.crud import (
    get_all_doctors, get_doctor_by_id, get_doctor_notifications,
    get_intake_session_by_id, get_all_intake_sessions, get_patient_by_id,
    get_last_intake_session, get_all_patients,
)
from agent.graph import ask_agent_about_patient

router = APIRouter()


# ──────────────────────── REQUEST MODELS ────────────────────────

class DoctorLoginRequest(BaseModel):
    doctor_id: Optional[int] = None
    email: Optional[str] = None


class AskAgentRequest(BaseModel):
    patient_id: int
    question: str


# ──────────────────────── ENDPOINTS ────────────────────────

@router.get("/list")
def list_doctors(db: Session = Depends(get_db)):
    """List all registered doctors."""
    doctors = get_all_doctors(db)
    return [
        {
            "id": d.id,
            "name": d.name,
            "email": d.email,
            "specialty": d.specialty,
        }
        for d in doctors
    ]


@router.post("/login")
def doctor_login(req: DoctorLoginRequest, db: Session = Depends(get_db)):
    """Doctor login."""
    doctor = None
    if req.doctor_id:
        doctor = get_doctor_by_id(db, req.doctor_id)
    elif req.email:
        from database.models import Doctor
        doctor = db.query(Doctor).filter(Doctor.email == req.email).first()

    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found.")

    return {
        "doctor_id": doctor.id,
        "name": doctor.name,
        "email": doctor.email,
        "specialty": doctor.specialty,
    }


@router.get("/patients/{doctor_id}")
def doctor_patients(doctor_id: int, db: Session = Depends(get_db)):
    """Get all patients with their latest intake summary for the doctor dashboard."""
    doctor = get_doctor_by_id(db, doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found.")

    patients = get_all_patients(db)
    result = []

    for p in patients:
        last_session = get_last_intake_session(db, p.id)
        all_sessions = get_all_intake_sessions(db, p.id)

        patient_data = {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "gender": p.gender,
            "diagnosis": p.diagnosis,
            "total_visits": len(all_sessions),
            "last_visit": None,
        }

        if last_session:
            patient_data["last_visit"] = {
                "session_id": last_session.id,
                "session_date": last_session.session_date.isoformat() if last_session.session_date else None,
                "pain_score": last_session.pain_score,
                "pain_location": last_session.pain_location,
                "medication_compliance": last_session.medication_compliance,
                "emotional_state": last_session.emotional_state,
                "is_urgent": last_session.is_urgent,
                "urgent_reason": last_session.urgent_reason,
            }

        result.append(patient_data)

    return result


@router.get("/notifications/{doctor_id}")
def doctor_notifications(doctor_id: int, db: Session = Depends(get_db)):
    """Get all notifications for a doctor, most recent first."""
    doctor = get_doctor_by_id(db, doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found.")

    notifications = get_doctor_notifications(db, doctor_id)
    return [
        {
            "id": n.id,
            "patient_id": n.patient_id,
            "patient_name": get_patient_by_id(db, n.patient_id).name if get_patient_by_id(db, n.patient_id) else "Unknown",
            "session_id": n.session_id,
            "message": n.message,
            "sent_at": n.sent_at.isoformat() if n.sent_at else None,
            "status": n.status,
        }
        for n in notifications
    ]


@router.get("/summary/{session_id}")
def intake_summary(session_id: int, db: Session = Depends(get_db)):
    """Get the full structured summary for a specific intake session."""
    session = get_intake_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Intake session not found.")

    patient = get_patient_by_id(db, session.patient_id)

    # Get previous session for comparison
    all_sessions = get_all_intake_sessions(db, session.patient_id)
    previous = None
    for i, s in enumerate(all_sessions):
        if s.id == session_id and i + 1 < len(all_sessions):
            previous = all_sessions[i + 1]
            break

    result = {
        "session_id": session.id,
        "patient": {
            "id": patient.id if patient else None,
            "name": patient.name if patient else "Unknown",
            "age": patient.age if patient else None,
            "gender": patient.gender if patient else None,
            "diagnosis": patient.diagnosis if patient else None,
        },
        "session_date": session.session_date.isoformat() if session.session_date else None,
        "current": {
            "pain_score": session.pain_score,
            "pain_location": session.pain_location,
            "pain_type": session.pain_type,
            "pain_duration": session.pain_duration,
            "medication_compliance": session.medication_compliance,
            "medication_missed_reason": session.medication_missed_reason,
            "side_effects": session.side_effects,
            "sleep_quality": session.sleep_quality,
            "appetite": session.appetite,
            "emotional_state": session.emotional_state,
            "mobility": session.mobility,
            "new_symptoms": session.new_symptoms,
            "specific_concerns": session.specific_concerns,
            "home_support": session.home_support,
            "is_urgent": session.is_urgent,
            "urgent_reason": session.urgent_reason,
        },
        "previous": None,
        "transcript": session.raw_transcript,
    }

    if previous:
        result["previous"] = {
            "session_date": previous.session_date.isoformat() if previous.session_date else None,
            "pain_score": previous.pain_score,
            "pain_location": previous.pain_location,
            "pain_type": previous.pain_type,
            "medication_compliance": previous.medication_compliance,
            "side_effects": previous.side_effects,
            "sleep_quality": previous.sleep_quality,
            "appetite": previous.appetite,
            "emotional_state": previous.emotional_state,
            "mobility": previous.mobility,
        }

    return result


@router.get("/patient-history/{patient_id}")
def patient_history(patient_id: int, db: Session = Depends(get_db)):
    """Get all intake sessions for a patient."""
    patient = get_patient_by_id(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    sessions = get_all_intake_sessions(db, patient_id)
    return {
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "diagnosis": patient.diagnosis,
        },
        "sessions": [
            {
                "session_id": s.id,
                "session_date": s.session_date.isoformat() if s.session_date else None,
                "pain_score": s.pain_score,
                "pain_location": s.pain_location,
                "pain_type": s.pain_type,
                "pain_duration": s.pain_duration,
                "medication_compliance": s.medication_compliance,
                "medication_missed_reason": s.medication_missed_reason,
                "side_effects": s.side_effects,
                "sleep_quality": s.sleep_quality,
                "appetite": s.appetite,
                "emotional_state": s.emotional_state,
                "mobility": s.mobility,
                "new_symptoms": s.new_symptoms,
                "specific_concerns": s.specific_concerns,
                "home_support": s.home_support,
                "is_urgent": s.is_urgent,
                "urgent_reason": s.urgent_reason,
            }
            for s in sessions
        ],
    }


@router.post("/ask-agent")
def ask_agent(req: AskAgentRequest):
    """
    Doctor asks a question about a specific patient.
    The agent responds using that patient's full memory and intake history.
    """
    try:
        response = ask_agent_about_patient(req.patient_id, req.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
