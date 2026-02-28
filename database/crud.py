from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.models import Patient, IntakeSession, Doctor, Notification
from datetime import datetime, timezone
from typing import Optional


# ──────────────────────── PATIENT OPERATIONS ────────────────────────


def get_patient_by_id(db: Session, patient_id: int) -> Optional[Patient]:
    return db.query(Patient).filter(Patient.id == patient_id).first()


def get_patient_by_name(db: Session, name: str) -> Optional[Patient]:
    return db.query(Patient).filter(Patient.name.ilike(f"%{name}%")).first()


def create_patient(db: Session, name: str, age: int = None, gender: str = None,
                   diagnosis: str = None, care_type: str = "palliative") -> Patient:
    patient = Patient(
        name=name,
        age=age,
        gender=gender,
        diagnosis=diagnosis,
        care_type=care_type,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_all_patients(db: Session):
    return db.query(Patient).order_by(Patient.name).all()


# ──────────────────────── INTAKE SESSION OPERATIONS ────────────────────────


def get_last_intake_session(db: Session, patient_id: int) -> Optional[IntakeSession]:
    return (
        db.query(IntakeSession)
        .filter(IntakeSession.patient_id == patient_id)
        .order_by(desc(IntakeSession.session_date))
        .first()
    )


def get_all_intake_sessions(db: Session, patient_id: int):
    return (
        db.query(IntakeSession)
        .filter(IntakeSession.patient_id == patient_id)
        .order_by(desc(IntakeSession.session_date))
        .all()
    )


def create_intake_session(db: Session, patient_id: int, intake_data: dict) -> IntakeSession:
    session = IntakeSession(
        patient_id=patient_id,
        pain_score=intake_data.get("pain_score"),
        pain_location=intake_data.get("pain_location"),
        pain_type=intake_data.get("pain_type"),
        pain_duration=intake_data.get("pain_duration"),
        medication_compliance=intake_data.get("medication_compliance"),
        medication_missed_reason=intake_data.get("medication_missed_reason"),
        side_effects=intake_data.get("side_effects"),
        sleep_quality=intake_data.get("sleep_quality"),
        appetite=intake_data.get("appetite"),
        emotional_state=intake_data.get("emotional_state"),
        mobility=intake_data.get("mobility"),
        new_symptoms=intake_data.get("new_symptoms"),
        specific_concerns=intake_data.get("specific_concerns"),
        home_support=intake_data.get("home_support"),
        raw_transcript=intake_data.get("raw_transcript"),
        is_urgent=intake_data.get("is_urgent", False),
        urgent_reason=intake_data.get("urgent_reason"),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_intake_session_by_id(db: Session, session_id: int) -> Optional[IntakeSession]:
    return db.query(IntakeSession).filter(IntakeSession.id == session_id).first()


# ──────────────────────── DOCTOR OPERATIONS ────────────────────────


def get_doctor_by_id(db: Session, doctor_id: int) -> Optional[Doctor]:
    return db.query(Doctor).filter(Doctor.id == doctor_id).first()


def get_all_doctors(db: Session):
    return db.query(Doctor).all()


def create_doctor(db: Session, name: str, email: str,
                  specialty: str = "Cancer Pain Management") -> Doctor:
    doctor = Doctor(name=name, email=email, specialty=specialty)
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return doctor


def get_assigned_doctor(db: Session) -> Optional[Doctor]:
    """Get the first available doctor. In production, this would have assignment logic."""
    return db.query(Doctor).first()


# ──────────────────────── NOTIFICATION OPERATIONS ────────────────────────


def create_notification(db: Session, doctor_id: int, patient_id: int,
                        session_id: int, message: str,
                        status: str = "pending") -> Notification:
    notification = Notification(
        doctor_id=doctor_id,
        patient_id=patient_id,
        session_id=session_id,
        message=message,
        status=status,
        sent_at=datetime.now(timezone.utc),
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification


def update_notification_status(db: Session, notification_id: int, status: str):
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if notification:
        notification.status = status
        db.commit()
    return notification


def get_doctor_notifications(db: Session, doctor_id: int):
    return (
        db.query(Notification)
        .filter(Notification.doctor_id == doctor_id)
        .order_by(desc(Notification.sent_at))
        .all()
    )
