from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, Enum, Boolean,
    create_engine
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone
import enum

Base = declarative_base()


class CareType(str, enum.Enum):
    PALLIATIVE = "palliative"
    CURATIVE = "curative"
    SUPPORTIVE = "supportive"


class NotificationStatus(str, enum.Enum):
    SENT = "sent"
    FAILED = "failed"
    PENDING = "pending"


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)
    diagnosis = Column(Text, nullable=True)
    care_type = Column(String(50), default="palliative")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    intake_sessions = relationship("IntakeSession", back_populates="patient", order_by="IntakeSession.session_date.desc()")
    notifications = relationship("Notification", back_populates="patient")


class IntakeSession(Base):
    __tablename__ = "intake_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    session_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Clinical intake fields
    pain_score = Column(Float, nullable=True)
    pain_location = Column(Text, nullable=True)
    pain_type = Column(String(255), nullable=True)  # burning, stabbing, aching, etc.
    pain_duration = Column(Text, nullable=True)
    medication_compliance = Column(String(50), nullable=True)  # yes, partial, no
    medication_missed_reason = Column(Text, nullable=True)
    side_effects = Column(Text, nullable=True)
    sleep_quality = Column(Text, nullable=True)
    appetite = Column(Text, nullable=True)
    emotional_state = Column(Text, nullable=True)
    mobility = Column(Text, nullable=True)
    new_symptoms = Column(Text, nullable=True)
    specific_concerns = Column(Text, nullable=True)
    home_support = Column(Text, nullable=True)

    # Full conversation transcript
    raw_transcript = Column(Text, nullable=True)

    # Flags
    is_urgent = Column(Boolean, default=False)
    urgent_reason = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    patient = relationship("Patient", back_populates="intake_sessions")
    notifications = relationship("Notification", back_populates="session")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    specialty = Column(String(255), default="Cancer Pain Management")

    # Relationships
    notifications = relationship("Notification", back_populates="doctor")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("intake_sessions.id"), nullable=False)
    message = Column(Text, nullable=True)
    sent_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String(50), default="pending")

    # Relationships
    doctor = relationship("Doctor", back_populates="notifications")
    patient = relationship("Patient", back_populates="notifications")
    session = relationship("IntakeSession", back_populates="notifications")
