"""
LangGraph agent nodes — the 4 core processing nodes:
  1. Memory Recall Node
  2. Intake Interview Node
  3. Storage Node
  4. Notification Node
"""
import json
import re
from datetime import datetime, timezone
from typing import TypedDict, Annotated, Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import settings
from agent.prompts import INTAKE_AGENT_SYSTEM_PROMPT, EXTRACTION_PROMPT, SUMMARY_PROMPT
from agent.memory import get_full_patient_history, get_managed_patient_history, store_conversation_memory, recall_patient_memory
from database import SessionLocal
from database.crud import (
    get_patient_by_id, get_last_intake_session, create_intake_session,
    get_assigned_doctor, create_notification, update_notification_status,
)
from notifications.email import send_doctor_email, build_email_body


# ──────────────────────── LLM SETUP ────────────────────────

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.7,
    )


# ──────────────────────── STATE DEFINITION ────────────────────────

class AgentState(TypedDict):
    patient_id: int
    patient_name: str
    memory_context: str
    previous_session: dict | None
    messages: list  # Full conversation message history
    intake_data: dict | None
    session_id: int | None
    is_complete: bool
    is_urgent: bool
    urgent_reason: str | None


# ──────────────────────── NODE 1: MEMORY RECALL ────────────────────────

def memory_recall_node(state: AgentState) -> AgentState:
    """
    Triggered when patient logs in.
    - Queries ChromaDB for patient's full conversation history
    - Queries PostgreSQL for last intake session structured data
    - Builds a context summary for the agent
    """
    patient_id = state["patient_id"]
    db = SessionLocal()

    try:
        # Get patient info
        patient = get_patient_by_id(db, patient_id)
        patient_name = patient.name if patient else "Patient"

        # Get last structured intake session from PostgreSQL
        last_session = get_last_intake_session(db, patient_id)
        previous_session = None

        if last_session:
            previous_session = {
                "session_date": last_session.session_date.isoformat() if last_session.session_date else None,
                "pain_score": last_session.pain_score,
                "pain_location": last_session.pain_location,
                "pain_type": last_session.pain_type,
                "pain_duration": last_session.pain_duration,
                "medication_compliance": last_session.medication_compliance,
                "medication_missed_reason": last_session.medication_missed_reason,
                "side_effects": last_session.side_effects,
                "sleep_quality": last_session.sleep_quality,
                "appetite": last_session.appetite,
                "emotional_state": last_session.emotional_state,
                "mobility": last_session.mobility,
                "new_symptoms": last_session.new_symptoms,
                "specific_concerns": last_session.specific_concerns,
                "home_support": last_session.home_support,
            }

        # Get semantic memory from ChromaDB — CONTEXT MANAGED
        # Uses smart retrieval: full transcripts for last 2 visits,
        # summaries only for older visits. Prevents context explosion.
        conversation_history = get_managed_patient_history(patient_id)

        # Build context block for the system prompt
        memory_parts = []
        memory_parts.append(f"PATIENT NAME: {patient_name}")

        if patient:
            if patient.age:
                memory_parts.append(f"AGE: {patient.age}")
            if patient.gender:
                memory_parts.append(f"GENDER: {patient.gender}")
            if patient.diagnosis:
                memory_parts.append(f"DIAGNOSIS: {patient.diagnosis}")

        if previous_session:
            memory_parts.append(f"\nLAST VISIT ({previous_session['session_date']}):")
            memory_parts.append(f"  Pain Score: {previous_session['pain_score']}/10")
            memory_parts.append(f"  Pain Location: {previous_session['pain_location']}")
            memory_parts.append(f"  Pain Type: {previous_session['pain_type']}")
            memory_parts.append(f"  Medication Compliance: {previous_session['medication_compliance']}")
            memory_parts.append(f"  Side Effects: {previous_session['side_effects']}")
            memory_parts.append(f"  Sleep Quality: {previous_session['sleep_quality']}")
            memory_parts.append(f"  Appetite: {previous_session['appetite']}")
            memory_parts.append(f"  Emotional State: {previous_session['emotional_state']}")
            memory_parts.append(f"  Mobility: {previous_session['mobility']}")
            memory_parts.append(f"  Home Support: {previous_session['home_support']}")
        else:
            memory_parts.append("\nThis is the patient's FIRST VISIT. No previous records.")

        if conversation_history:
            memory_parts.append(f"\nPREVIOUS CONVERSATION HISTORY:\n{conversation_history}")

        memory_context = "\n".join(memory_parts)

    finally:
        db.close()

    return {
        **state,
        "patient_name": patient_name,
        "memory_context": memory_context,
        "previous_session": previous_session,
        "messages": state.get("messages", []),
        "is_complete": False,
        "is_urgent": False,
    }


# ──────────────────────── NODE 2: INTAKE INTERVIEW ────────────────────────

def intake_interview_node(state: AgentState, user_message: str) -> AgentState:
    """
    Conducts conversational intake interview using Gemini.
    Called repeatedly for each patient message.
    Returns updated state with new AI response.
    """
    llm = get_llm()

    # Build system prompt with memory context
    system_prompt = INTAKE_AGENT_SYSTEM_PROMPT.format(
        memory_context=state.get("memory_context", "")
    )

    # Build message list
    messages = [SystemMessage(content=system_prompt)]

    # Add conversation history
    for msg in state.get("messages", []):
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))

    # Add the new user message
    messages.append(HumanMessage(content=user_message))

    # Get AI response
    response = llm.invoke(messages)
    ai_text = response.content

    # Update message history
    updated_messages = state.get("messages", []).copy()
    updated_messages.append({"role": "human", "content": user_message})
    updated_messages.append({"role": "ai", "content": ai_text})

    # Check if intake is complete (agent signals with INTAKE_COMPLETE)
    is_complete = "INTAKE_COMPLETE" in ai_text

    # Check for urgency markers
    is_urgent = state.get("is_urgent", False)
    urgent_reason = state.get("urgent_reason", None)

    # Detect urgency in patient message
    urgency_keywords = ["suicid", "kill myself", "end my life", "want to die",
                        "self-harm", "self harm", "can't go on", "no point living"]
    lower_msg = user_message.lower()
    for keyword in urgency_keywords:
        if keyword in lower_msg:
            is_urgent = True
            urgent_reason = f"Patient expressed distress: '{user_message[:100]}'"
            break

    # Clean the INTAKE_COMPLETE marker from the displayed message
    display_text = ai_text.replace("INTAKE_COMPLETE", "").strip()

    return {
        **state,
        "messages": updated_messages,
        "is_complete": is_complete,
        "is_urgent": is_urgent,
        "urgent_reason": urgent_reason,
    }


# ──────────────────────── NODE 3: STORAGE NODE ────────────────────────

def storage_node(state: AgentState) -> AgentState:
    """
    When intake is complete:
    - Extracts structured data from conversation using Gemini
    - Saves structured intake data to PostgreSQL
    - Saves conversation to ChromaDB
    """
    llm = get_llm()

    # Build full transcript from messages
    transcript_parts = []
    for msg in state.get("messages", []):
        role = "Noor" if msg["role"] == "ai" else "Patient"
        transcript_parts.append(f"{role}: {msg['content']}")
    full_transcript = "\n\n".join(transcript_parts)

    # Extract structured data using Gemini
    extraction_prompt = EXTRACTION_PROMPT.format(transcript=full_transcript)
    response = llm.invoke([HumanMessage(content=extraction_prompt)])

    # Parse the JSON response
    try:
        # Clean potential markdown code fences
        raw_text = response.content.strip()
        raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
        raw_text = re.sub(r'\s*```$', '', raw_text)
        intake_data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback — save what we can
        intake_data = {
            "pain_score": None,
            "pain_location": None,
            "pain_type": None,
            "pain_duration": None,
            "medication_compliance": None,
            "medication_missed_reason": None,
            "side_effects": None,
            "sleep_quality": None,
            "appetite": None,
            "emotional_state": None,
            "mobility": None,
            "new_symptoms": None,
            "specific_concerns": None,
            "home_support": None,
            "is_urgent": state.get("is_urgent", False),
            "urgent_reason": state.get("urgent_reason"),
        }

    # Override urgency from state if detected
    if state.get("is_urgent"):
        intake_data["is_urgent"] = True
        intake_data["urgent_reason"] = state.get("urgent_reason", "")

    # Add transcript
    intake_data["raw_transcript"] = full_transcript

    # Save to PostgreSQL
    db = SessionLocal()
    try:
        session_record = create_intake_session(
            db, state["patient_id"], intake_data
        )
        session_id = session_record.id
    finally:
        db.close()

    # Save to ChromaDB
    summary = f"Pain: {intake_data.get('pain_score')}/10 at {intake_data.get('pain_location')}. " \
              f"Meds: {intake_data.get('medication_compliance')}. " \
              f"Emotional: {intake_data.get('emotional_state')}."

    store_conversation_memory(
        patient_id=state["patient_id"],
        session_id=session_id,
        conversation_text=full_transcript,
        summary=summary,
    )

    return {
        **state,
        "intake_data": intake_data,
        "session_id": session_id,
    }


# ──────────────────────── NODE 4: NOTIFICATION NODE ────────────────────────

def notification_node(state: AgentState) -> AgentState:
    """
    After storage:
    - Generates intelligent clinical summary using Gemini
    - Compares current vs previous session
    - Sends email to assigned doctor
    - Stores notification record
    """
    llm = get_llm()

    # Generate clinical summary
    current_str = json.dumps(state.get("intake_data", {}), indent=2, default=str)
    previous_str = json.dumps(state.get("previous_session"), indent=2, default=str) \
        if state.get("previous_session") else "No previous session data available."

    summary_prompt = SUMMARY_PROMPT.format(
        current_session=current_str,
        previous_session=previous_str,
    )

    response = llm.invoke([HumanMessage(content=summary_prompt)])
    clinical_summary = response.content

    # Get the assigned doctor
    db = SessionLocal()
    try:
        doctor = get_assigned_doctor(db)
        if not doctor:
            print("WARNING: No doctor found in database. Skipping notification.")
            return state

        # Build email body
        email_body = build_email_body(
            doctor_name=doctor.name,
            patient_name=state["patient_name"],
            intake_data=state.get("intake_data", {}),
            previous_session=state.get("previous_session"),
            clinical_summary=clinical_summary,
        )

        subject = (
            f"{'🚨 URGENT — ' if state.get('is_urgent') else ''}"
            f"Patient Intake Complete — {state['patient_name']} — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # Send email
        email_sent = send_doctor_email(
            to_email=doctor.email,
            subject=subject,
            body=email_body,
        )

        # Store notification record
        notification = create_notification(
            db,
            doctor_id=doctor.id,
            patient_id=state["patient_id"],
            session_id=state["session_id"],
            message=email_body,
            status="sent" if email_sent else "failed",
        )
    finally:
        db.close()

    return state
