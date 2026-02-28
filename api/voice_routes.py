"""
WebSocket endpoint for real-time voice conversations with Noor.

Uses Gemini 2.0 Live API for native speech-to-speech.
No separate STT/TTS pipeline — the model processes audio natively.

With input_audio_transcription + output_audio_transcription enabled,
Gemini also provides text transcripts of both sides of the conversation.
These transcripts power:
  - Live transcript display in the browser
  - Doctor reports / clinical summaries
  - Urgency detection (suicidal ideation, self-harm keywords)

Architecture:
  Browser mic (PCM 16kHz) ──WebSocket──► FastAPI ──► Gemini Live API
  Browser speakers ◄──WebSocket── FastAPI ◄── Gemini Live API (PCM 24kHz)
  Transcripts flow:  Gemini ──► FastAPI ──► Browser (for display)
                     Gemini ──► FastAPI ──► Storage Node (for doctor report)
"""
import asyncio
import json
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.voice_session import (
    create_live_config,
    get_genai_client,
    build_voice_system_prompt,
    LIVE_MODEL,
)
from agent.graph import start_session, complete_intake, send_urgent_alert

router = APIRouter()

# Completion signal — the exact phrase Noor says to end the session
COMPLETION_SIGNAL = "take care and we will see you at your next visit"

# Urgency keywords to detect in patient speech
URGENCY_KEYWORDS = [
    "suicid", "kill myself", "end my life", "want to die",
    "self-harm", "self harm", "can't go on", "no point living",
    "hurt myself", "don't want to live",
]


def _extract_transcription_text(content) -> str:
    """Safely extract text from a Gemini transcription Content object."""
    if not content:
        return ""
    parts = getattr(content, 'parts', None) or []
    texts = []
    for part in parts:
        t = getattr(part, 'text', None)
        if t:
            texts.append(t)
    return " ".join(texts)


@router.websocket("/ws/voice/{patient_id}")
async def voice_session(websocket: WebSocket, patient_id: int):
    """
    Real-time voice conversation WebSocket.

    Protocol (browser ↔ server messages):

    Browser → Server:
      - Binary:  raw PCM audio bytes (16-bit, 16kHz, mono)
      - JSON:    {"type": "text_message", "content": "..."}  (text mode fallback)
      - JSON:    {"type": "end_session"}

    Server → Browser:
      - Binary:  raw PCM audio bytes (16-bit, 24kHz, mono) — Noor's voice
      - JSON:    {"type": "transcript", "role": "ai"|"patient", "text": "..."}
      - JSON:    {"type": "status", "status": "connecting|ready|complete|saved|error", "message": "..."}
      - JSON:    {"type": "turn_complete"}
      - JSON:    {"type": "urgent"}
    """
    await websocket.accept()
    print(f"[VOICE] WebSocket connected — patient {patient_id}")

    # Session state
    transcript_parts = []    # ["Noor: ...", "Patient: ...", ...]
    is_complete = False
    is_urgent = False
    urgent_reason = None
    urgent_notified = False
    state = None
    user_ended = False       # True if user clicked "End Session"

    try:
        # ── Step 1: Memory Recall ──
        await websocket.send_json({
            "type": "status",
            "status": "connecting",
            "message": "Loading your medical history..."
        })

        state = start_session(patient_id)

        # ── Step 2: Build voice system prompt with patient context ──
        system_prompt = build_voice_system_prompt(state["memory_context"])
        config = create_live_config(system_prompt)
        client = get_genai_client()

        await websocket.send_json({
            "type": "status",
            "status": "ready",
            "message": "Connected to Noor"
        })

        # ── Step 3: Open Gemini Live session ──
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:

            # ── Step 4: Trigger Noor's greeting ──
            greeting_prompt = (
                f"Patient {state['patient_name']} has just connected. "
                "Greet them warmly and personally."
            )
            if state.get("previous_session"):
                greeting_prompt += (
                    " This is a returning patient — reference their previous visit "
                    "and ask how they have been."
                )
            else:
                greeting_prompt += (
                    " This is their first visit — introduce yourself warmly "
                    "and explain what you do."
                )

            await session.send(input=greeting_prompt, end_of_turn=True)

            # Stream greeting audio + transcription to browser
            greeting_text = ""
            async for response in session.receive():
                # Audio data
                if response.data:
                    await websocket.send_bytes(response.data)

                # Direct text (fallback — some models return this)
                if response.text:
                    greeting_text += response.text

                # Transcription events
                sc = getattr(response, 'server_content', None)
                if sc:
                    # Output transcription — Noor's speech as text
                    out_trans = getattr(sc, 'output_transcription', None)
                    if out_trans:
                        text = _extract_transcription_text(out_trans)
                        if text:
                            greeting_text += text
                            transcript_parts.append(f"Noor: {text}")
                            await websocket.send_json({
                                "type": "transcript",
                                "role": "ai",
                                "text": text
                            })

                    # Turn complete
                    turn_done = getattr(sc, 'turn_complete', False)
                    if turn_done:
                        await websocket.send_json({"type": "turn_complete"})
                        break

            # If we got greeting text from response.text but not from transcription
            if greeting_text and not any(p.startswith("Noor:") for p in transcript_parts):
                transcript_parts.append(f"Noor: {greeting_text}")
                await websocket.send_json({
                    "type": "transcript",
                    "role": "ai",
                    "text": greeting_text
                })

            # ── Step 5: Main conversation loop ──
            stop_event = asyncio.Event()

            async def browser_to_gemini():
                """Receive audio + text from browser, forward to Gemini."""
                nonlocal is_urgent, urgent_reason, user_ended

                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(
                            websocket.receive(), timeout=600  # 10 min timeout
                        )

                        if raw.get("type") == "websocket.disconnect":
                            stop_event.set()
                            break

                        # Binary = raw PCM audio from mic
                        if "bytes" in raw:
                            await session.send(
                                input={
                                    "data": raw["bytes"],
                                    "mime_type": "audio/pcm;rate=16000"
                                }
                            )

                        # Text = JSON messages
                        elif "text" in raw:
                            msg = json.loads(raw["text"])
                            msg_type = msg.get("type", "")

                            if msg_type == "user_transcript":
                                # Browser's Web Speech API transcript (fallback)
                                user_text = msg.get("text", "")
                                if user_text.strip():
                                    # Only add if we don't have Gemini's own transcription
                                    # (avoid duplicates)
                                    pass  # Server-side input_audio_transcription handles this

                            elif msg_type == "text_message":
                                # Text mode: patient typed a message
                                content = msg.get("content", "")
                                if content.strip():
                                    transcript_parts.append(f"Patient: {content}")
                                    await session.send(
                                        input=content, end_of_turn=True
                                    )

                                    # Check for urgency
                                    lower = content.lower()
                                    for kw in URGENCY_KEYWORDS:
                                        if kw in lower:
                                            is_urgent = True
                                            urgent_reason = f"Patient expressed distress: '{content[:100]}'"
                                            break

                            elif msg_type == "end_session":
                                user_ended = True
                                stop_event.set()
                                break

                    except asyncio.TimeoutError:
                        print(f"[VOICE] Timeout — patient {patient_id}")
                        stop_event.set()
                        break
                    except WebSocketDisconnect:
                        stop_event.set()
                        break
                    except Exception as e:
                        print(f"[VOICE] browser→gemini error: {e}")
                        if "disconnect" in str(e).lower():
                            stop_event.set()
                            break

            async def gemini_to_browser():
                """Receive audio + text + transcriptions from Gemini, forward to browser."""
                nonlocal is_complete, is_urgent, urgent_reason, urgent_notified

                while not stop_event.is_set():
                    try:
                        async for response in session.receive():
                            if stop_event.is_set():
                                break

                            # Forward audio to browser speakers
                            if response.data:
                                try:
                                    await websocket.send_bytes(response.data)
                                except Exception:
                                    stop_event.set()
                                    break

                            # Process direct text (fallback for some model configs)
                            if response.text:
                                text = response.text
                                transcript_parts.append(f"Noor: {text}")
                                try:
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "role": "ai",
                                        "text": text
                                    })
                                except Exception:
                                    stop_event.set()
                                    break

                                # Check completion
                                if COMPLETION_SIGNAL in text.lower():
                                    is_complete = True

                            # Process server_content events (transcription, turn_complete)
                            sc = getattr(response, 'server_content', None)
                            if sc:
                                # Output transcription — what Noor said (text)
                                out_trans = getattr(sc, 'output_transcription', None)
                                if out_trans:
                                    text = _extract_transcription_text(out_trans)
                                    if text:
                                        transcript_parts.append(f"Noor: {text}")
                                        try:
                                            await websocket.send_json({
                                                "type": "transcript",
                                                "role": "ai",
                                                "text": text
                                            })
                                        except Exception:
                                            stop_event.set()
                                            break

                                        # Check completion in Noor's speech
                                        if COMPLETION_SIGNAL in text.lower():
                                            is_complete = True

                                # Input transcription — what patient said (text)
                                in_trans = getattr(sc, 'input_transcription', None)
                                if in_trans:
                                    text = _extract_transcription_text(in_trans)
                                    if text:
                                        transcript_parts.append(f"Patient: {text}")
                                        try:
                                            await websocket.send_json({
                                                "type": "transcript",
                                                "role": "patient",
                                                "text": text
                                            })
                                        except Exception:
                                            stop_event.set()
                                            break

                                        # Urgency check on patient's speech
                                        lower = text.lower()
                                        for kw in URGENCY_KEYWORDS:
                                            if kw in lower:
                                                is_urgent = True
                                                urgent_reason = f"Patient expressed distress: '{text[:100]}'"
                                                break

                                # Turn complete
                                turn_done = getattr(sc, 'turn_complete', False)
                                if turn_done:
                                    try:
                                        await websocket.send_json({"type": "turn_complete"})
                                    except Exception:
                                        stop_event.set()
                                        break

                                    # Fire urgent alert immediately if needed
                                    if is_urgent and not urgent_notified and state:
                                        state["is_urgent"] = True
                                        state["urgent_reason"] = urgent_reason
                                        state["messages"] = _build_messages(transcript_parts)
                                        send_urgent_alert(state)
                                        urgent_notified = True
                                        try:
                                            await websocket.send_json({"type": "urgent"})
                                        except Exception:
                                            pass

                                    # If naturally complete, signal and stop
                                    if is_complete:
                                        try:
                                            await websocket.send_json({
                                                "type": "status",
                                                "status": "complete",
                                                "message": "Intake complete"
                                            })
                                        except Exception:
                                            pass
                                        stop_event.set()
                                        break

                    except Exception as e:
                        if stop_event.is_set():
                            break
                        err_str = str(e).lower()
                        if "disconnect" in err_str or "closed" in err_str:
                            stop_event.set()
                            break
                        print(f"[VOICE] gemini→browser error: {e}")
                        traceback.print_exc()
                        await asyncio.sleep(0.1)

            # Run both directions concurrently
            await asyncio.gather(
                browser_to_gemini(),
                gemini_to_browser(),
                return_exceptions=True
            )

        # ── Step 6: Post-session processing ──
        # ALWAYS finalize if we have any transcript — not just when intake is "complete"
        # This ensures doctor reports are created even for partial sessions
        if transcript_parts and state:
            print(f"[VOICE] Session ended for patient {patient_id}. "
                  f"Complete={is_complete}, Parts={len(transcript_parts)}. "
                  f"Running storage + notification...")

            state["messages"] = _build_messages(transcript_parts)
            state["is_complete"] = is_complete
            state["is_urgent"] = is_urgent
            state["urgent_reason"] = urgent_reason

            # Run Storage Node + Notification Node
            try:
                state = complete_intake(state)
                print(f"[VOICE] Session saved. session_id={state.get('session_id')}")
            except Exception as e:
                print(f"[VOICE] Error during finalization: {e}")
                traceback.print_exc()

            try:
                await websocket.send_json({
                    "type": "status",
                    "status": "saved",
                    "message": "Your information has been saved and your doctor has been notified.",
                    "session_id": state.get("session_id")
                })
            except Exception:
                pass  # WebSocket might already be closed
        else:
            print(f"[VOICE] No transcript captured for patient {patient_id}. Skipping finalization.")

    except WebSocketDisconnect:
        print(f"[VOICE] Patient {patient_id} disconnected")
        # Still try to save if we have data
        if transcript_parts and state:
            try:
                state["messages"] = _build_messages(transcript_parts)
                state["is_complete"] = is_complete
                state["is_urgent"] = is_urgent
                state["urgent_reason"] = urgent_reason
                state = complete_intake(state)
                print(f"[VOICE] Disconnected session saved. session_id={state.get('session_id')}")
            except Exception as e:
                print(f"[VOICE] Error saving disconnected session: {e}")
    except Exception as e:
        print(f"[VOICE] Session error for patient {patient_id}: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        print(f"[VOICE] Session ended — patient {patient_id}")


def _build_messages(transcript_parts: list) -> list:
    """Convert transcript_parts list into the messages format used by AgentState."""
    messages = []
    for part in transcript_parts:
        if part.startswith("Noor: "):
            messages.append({"role": "ai", "content": part[6:]})
        elif part.startswith("Patient: "):
            messages.append({"role": "human", "content": part[9:]})
    return messages
