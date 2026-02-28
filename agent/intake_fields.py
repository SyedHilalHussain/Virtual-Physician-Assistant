"""
Required intake fields definition for the Cancer Pain Management intake agent.
"""

REQUIRED_INTAKE_FIELDS = {
    "pain_score": {
        "description": "Current pain score on a scale of 0-10",
        "type": "number",
        "required": True,
    },
    "pain_location": {
        "description": "Where exactly the pain is located",
        "type": "string",
        "required": True,
    },
    "pain_type": {
        "description": "Type of pain — burning, stabbing, aching, constant, intermittent",
        "type": "string",
        "required": True,
    },
    "pain_duration": {
        "description": "How long this pain episode has lasted",
        "type": "string",
        "required": True,
    },
    "medication_compliance": {
        "description": "Whether patient took all prescribed medications since last visit",
        "type": "string",  # yes, partial, no
        "required": True,
    },
    "medication_missed_reason": {
        "description": "If medications were missed, why",
        "type": "string",
        "required": False,  # only if non-compliant
    },
    "side_effects": {
        "description": "Side effects experienced from current medications",
        "type": "string",
        "required": True,
    },
    "sleep_quality": {
        "description": "How sleep has been — hours, disruptions, quality",
        "type": "string",
        "required": True,
    },
    "appetite": {
        "description": "Appetite and nutrition status — eating normally, changes",
        "type": "string",
        "required": True,
    },
    "emotional_state": {
        "description": "Emotional wellbeing — anxiety, depression, fear, coping",
        "type": "string",
        "required": True,
    },
    "mobility": {
        "description": "Ability to move around, any new limitations",
        "type": "string",
        "required": True,
    },
    "new_symptoms": {
        "description": "Any new symptoms not previously reported",
        "type": "string",
        "required": True,
    },
    "specific_concerns": {
        "description": "Any specific concern the patient wants the doctor to know today",
        "type": "string",
        "required": True,
    },
    "home_support": {
        "description": "Support situation at home — managing alone, family helping",
        "type": "string",
        "required": True,
    },
}


def get_required_field_names() -> list[str]:
    """Return list of all required field names."""
    return [k for k, v in REQUIRED_INTAKE_FIELDS.items() if v["required"]]


def get_all_field_names() -> list[str]:
    """Return list of all field names."""
    return list(REQUIRED_INTAKE_FIELDS.keys())
