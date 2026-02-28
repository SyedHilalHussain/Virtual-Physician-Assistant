"""
Gemini Live API voice session manager.

Handles real-time speech-to-speech conversations using Gemini 2.5 Flash Native Audio.
The Live API processes audio natively — no separate STT/TTS pipeline needed.

Transcription is enabled so that both patient and Noor speech are captured
as text for the doctor report, urgency detection, and clinical record.
"""
from google import genai
from google.genai import types

from config import settings
from agent.prompts import INTAKE_AGENT_SYSTEM_PROMPT


# ──────────────────────── CONFIGURATION ────────────────────────

# Gemini model that supports the Live API (native audio)
LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"

# Voice options: Puck, Charon, Kore, Fenrir, Aoede
# Kore = warm, clear female voice — ideal for clinical assistant persona
VOICE_NAME = "Kore"

# Audio format constants
INPUT_SAMPLE_RATE = 16000   # Browser sends 16kHz PCM
OUTPUT_SAMPLE_RATE = 24000  # Gemini outputs 24kHz PCM


# ──────────────────────── CLIENT ────────────────────────

_client = None

def get_genai_client() -> genai.Client:
    """Create and return a Gemini API client (singleton)."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


# ──────────────────────── VOICE SYSTEM PROMPT ────────────────────────

VOICE_BEHAVIOR_ADDON = """

VOICE CONVERSATION BEHAVIOR (CRITICAL — you are speaking aloud, not typing):
- You are having a real-time voice conversation. Speak naturally and conversationally.
- Keep sentences short to medium length. Do NOT give long monologues.
- Pause between thoughts naturally — you are speaking, not writing an essay.
- Use natural conversational markers: "I see...", "Alright...", "Let me note that down...", "Tell me more about that..."
- When the patient shares something emotional, pause and acknowledge it before continuing.
- Do NOT use bullet points, numbered lists, asterisks, markdown, or any text formatting — you are speaking.
- Do NOT use emojis or special characters.
- Do NOT say "INTAKE_COMPLETE" out loud. When you have collected all 14 required fields and you close the conversation warmly, end your final spoken message with the exact phrase "Take care and we will see you at your next visit." — this is a system signal to mark the intake as complete.
- Speak as if the patient is sitting right in front of you.
"""


def build_voice_system_prompt(memory_context: str) -> str:
    """
    Build the complete system prompt for a voice session.
    Takes the base Noor prompt, injects patient memory, and adds voice-specific behavior.
    """
    # Start with the base prompt and inject memory
    base_prompt = INTAKE_AGENT_SYSTEM_PROMPT.format(memory_context=memory_context)

    # Replace the text-based INTAKE_COMPLETE instruction with voice-appropriate one
    base_prompt = base_prompt.replace(
        'When you are done collecting ALL fields, you MUST include the exact phrase '
        '"INTAKE_COMPLETE" at the very end of your final message (after your warm closing). '
        'This is a system signal — the patient will not see it.',
        'When you are done collecting ALL fields, close warmly and end with the exact phrase: '
        '"Take care and we will see you at your next visit." — this signals session completion.'
    )

    return base_prompt + VOICE_BEHAVIOR_ADDON


# ──────────────────────── LIVE CONFIG ────────────────────────

def create_live_config(system_prompt: str) -> types.LiveConnectConfig:
    """
    Build the Gemini Live API configuration.

    Enables:
    - Audio response modality (Noor speaks back)
    - Input audio transcription (what patient says → text for doctor report)
    - Output audio transcription (what Noor says → text for doctor report)
    - Voice selection (Kore — warm female)
    """
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=system_prompt)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=VOICE_NAME
                )
            )
        ),
        # Enable transcription of both directions for clinical record
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
