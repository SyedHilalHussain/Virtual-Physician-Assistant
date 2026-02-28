"""
LangGraph agent graph definition — orchestrates the 4 nodes.

This module provides:
  - start_session(patient_id) — initializes a session with memory recall
  - process_message(state, user_message) — runs the intake interview node
  - complete_intake(state) — runs storage + notification nodes
  - send_urgent_alert(state) — fires urgent email immediately mid-conversation
  - ask_agent_about_patient(patient_id, question) — doctor asks agent a question
"""
import json
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from agent.nodes import (
    AgentState,
    memory_recall_node,
    intake_interview_node,
    storage_node,
    notification_node,
    get_llm,
)
from agent.memory import get_managed_patient_history
from database import SessionLocal
from database.crud import (
    get_patient_by_id, get_last_intake_session, get_assigned_doctor,
    create_notification, create_intake_session,
)
from notifications.email import send_doctor_email


def start_session(patient_id: int) -> AgentState:
    """
    Initialize a new intake session for a patient.
    Runs the Memory Recall Node to load all context.
    Returns the initial agent state with memory loaded.
    """
    initial_state: AgentState = {
        "patient_id": patient_id,
        "patient_name": "",
        "memory_context": "",
        "previous_session": None,
        "messages": [],
        "intake_data": None,
        "session_id": None,
        "is_complete": False,
        "is_urgent": False,
        "urgent_reason": None,
    }

    # Node 1 — Memory Recall
    state = memory_recall_node(initial_state)
    return state


def process_message(state: AgentState, user_message: str) -> tuple[AgentState, str]:
    """
    Process a single patient message through the Intake Interview Node.
    Returns (updated_state, ai_response_text).
    """
    # Node 2 — Intake Interview
    updated_state = intake_interview_node(state, user_message)

    # Get the latest AI message to return
    ai_response = ""
    if updated_state["messages"]:
        last_msg = updated_state["messages"][-1]
        if last_msg["role"] == "ai":
            # Clean the INTAKE_COMPLETE marker from display
            ai_response = last_msg["content"].replace("INTAKE_COMPLETE", "").strip()

    return updated_state, ai_response


def complete_intake(state: AgentState) -> AgentState:
    """
    Run the post-interview pipeline:
    - Storage Node (save to PostgreSQL + ChromaDB)
    - Notification Node (email doctor + store notification)
    Returns the final state.
    """
    # Node 3 — Storage
    state = storage_node(state)

    # Node 4 — Notification
    state = notification_node(state)

    return state


def send_urgent_alert(state: AgentState) -> bool:
    """
    Fire an urgent email to the doctor IMMEDIATELY when distress is detected
    mid-conversation. Does NOT wait for intake completion.
    Returns True if email was sent.
    """
    db = SessionLocal()
    try:
        doctor = get_assigned_doctor(db)
        if not doctor:
            print("WARNING: No doctor found. Cannot send urgent alert.")
            return False

        patient_name = state.get("patient_name", "Unknown Patient")
        urgent_reason = state.get("urgent_reason", "Distress detected during intake")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Build the last few messages for context
        recent_messages = state.get("messages", [])[-6:]
        context_lines = []
        for msg in recent_messages:
            role = "Noor" if msg["role"] == "ai" else "Patient"
            context_lines.append(f"{role}: {msg['content'][:200]}")
        context_text = "\n".join(context_lines) if context_lines else "No conversation context available."

        subject = f"🚨 URGENT — Immediate Attention Required — {patient_name} — {now}"

        body = f"""Dear Dr. {doctor.name},

🚨 URGENT ALERT — IMMEDIATE ATTENTION REQUIRED

Patient {patient_name} has expressed signs of severe distress during their intake conversation with Noor.

REASON FOR ALERT:
  {urgent_reason}

RECENT CONVERSATION CONTEXT:
{context_text}

This alert was sent in REAL-TIME the moment distress was detected. The intake session may still be in progress.

Please take immediate action.

— Noor, MHP Clinical Intake Agent
   {now}
"""

        email_sent = send_doctor_email(
            to_email=doctor.email,
            subject=subject,
            body=body,
        )

        # Store a notification record (use session_id=0 as placeholder since intake is in progress)
        create_notification(
            db,
            doctor_id=doctor.id,
            patient_id=state["patient_id"],
            session_id=state.get("session_id") or 0,
            message=body,
            status="sent" if email_sent else "failed",
        )

        return email_sent
    finally:
        db.close()


def ask_agent_about_patient(patient_id: int, question: str) -> str:
    """
    Doctor asks the agent a question about a specific patient.
    Agent uses that patient's memory + intake history to respond.
    """
    llm = get_llm()
    db = SessionLocal()

    try:
        patient = get_patient_by_id(db, patient_id)
        if not patient:
            return "Patient not found."

        # Get structured last session
        last_session = get_last_intake_session(db, patient_id)
        session_info = ""
        if last_session:
            session_info = f"""
Latest Intake Session ({last_session.session_date}):
  Pain Score: {last_session.pain_score}/10
  Pain Location: {last_session.pain_location}
  Pain Type: {last_session.pain_type}
  Pain Duration: {last_session.pain_duration}
  Medication Compliance: {last_session.medication_compliance}
  Missed Reason: {last_session.medication_missed_reason or 'N/A'}
  Side Effects: {last_session.side_effects}
  Sleep Quality: {last_session.sleep_quality}
  Appetite: {last_session.appetite}
  Emotional State: {last_session.emotional_state}
  Mobility: {last_session.mobility}
  New Symptoms: {last_session.new_symptoms}
  Concerns: {last_session.specific_concerns}
  Home Support: {last_session.home_support}
  Urgent: {last_session.is_urgent} — {last_session.urgent_reason or ''}
"""
    finally:
        db.close()

    # Get semantic memory — context managed
    conversation_history = get_managed_patient_history(patient_id)

    system_prompt = f"""You are Noor, the clinical intake agent at MHP. A doctor is asking you a question about a patient.
Answer based ONLY on the patient data and conversation history provided below. Be concise, clinical, and accurate.

PATIENT: {patient.name}, Age: {patient.age}, Gender: {patient.gender}, Diagnosis: {patient.diagnosis}

{session_info}

CONVERSATION HISTORY:
{conversation_history if conversation_history else 'No previous conversations recorded.'}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    return response.content
