"""
SMTP email notification module for doctor alerts.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from config import settings


def send_doctor_email(to_email: str, subject: str, body: str) -> bool:
    """
    Send an email notification to the doctor via SMTP (Gmail).
    Returns True if sent successfully, False otherwise.
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.SMTP_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject

        # Plain text version
        msg.attach(MIMEText(body, "plain"))

        # HTML version (nicely formatted)
        html_body = body.replace("\n", "<br>")
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            {html_body}
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, "html"))

        # Connect and send
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
            server.send_message(msg)

        print(f"Email sent successfully to {to_email}")
        return True

    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        return False


def build_email_body(doctor_name: str, patient_name: str,
                     intake_data: dict, previous_session: dict | None,
                     clinical_summary: str) -> str:
    """
    Build the email notification body in the specified format.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Current values
    pain_score = intake_data.get("pain_score", "N/A")
    prev_pain = previous_session.get("pain_score", "N/A") if previous_session else "N/A"
    pain_location = intake_data.get("pain_location", "N/A")
    med_compliance = intake_data.get("medication_compliance", "N/A")
    emotional = intake_data.get("emotional_state", "N/A")
    new_concerns = intake_data.get("specific_concerns", "None reported")
    new_symptoms = intake_data.get("new_symptoms", "None reported")

    # Build changes section
    changes = []
    if previous_session:
        if intake_data.get("pain_score") is not None and previous_session.get("pain_score") is not None:
            diff = float(intake_data["pain_score"]) - float(previous_session["pain_score"])
            if diff > 0:
                changes.append(f"Pain score INCREASED from {previous_session['pain_score']}/10 to {intake_data['pain_score']}/10")
            elif diff < 0:
                changes.append(f"Pain score DECREASED from {previous_session['pain_score']}/10 to {intake_data['pain_score']}/10")
            else:
                changes.append(f"Pain score unchanged at {intake_data['pain_score']}/10")

        if intake_data.get("medication_compliance") != previous_session.get("medication_compliance"):
            changes.append(f"Medication compliance changed: {previous_session.get('medication_compliance', 'N/A')} → {intake_data.get('medication_compliance', 'N/A')}")

        if intake_data.get("sleep_quality") != previous_session.get("sleep_quality"):
            changes.append(f"Sleep quality update: {intake_data.get('sleep_quality', 'N/A')}")

        if intake_data.get("emotional_state") != previous_session.get("emotional_state"):
            changes.append(f"Emotional state update: {intake_data.get('emotional_state', 'N/A')}")
    else:
        changes.append("First visit — no previous data for comparison.")

    changes_text = "\n".join(f"  - {c}" for c in changes) if changes else "  No significant changes."

    # Build flagged items
    flags = []
    if intake_data.get("is_urgent"):
        flags.append(f"🚨 URGENT: {intake_data.get('urgent_reason', 'Patient flagged for urgent attention')}")
    if intake_data.get("pain_score") and float(intake_data["pain_score"]) >= 7:
        flags.append(f"⚠️ High pain score: {intake_data['pain_score']}/10")
    if intake_data.get("medication_compliance", "").lower() in ["no", "partial"]:
        flags.append(f"⚠️ Medication compliance: {intake_data['medication_compliance']}")
    if intake_data.get("new_symptoms") and intake_data["new_symptoms"].lower() not in ["none", "no", "null", "none reported"]:
        flags.append(f"⚠️ New symptoms reported: {intake_data['new_symptoms']}")

    flags_text = "\n".join(f"  {f}" for f in flags) if flags else "  No urgent items."

    body = f"""Dear Dr. {doctor_name},

Your patient {patient_name} has completed their pre-consultation intake.

SUMMARY:
  - Pain Score Today: {pain_score}/10  (Previous visit: {prev_pain}/10)
  - Pain Location: {pain_location}
  - Medication Compliance: {med_compliance}
  - Emotional State: {emotional}
  - New Concerns: {new_concerns}

CHANGES SINCE LAST VISIT:
{changes_text}

FLAGGED FOR ATTENTION:
{flags_text}

CLINICAL SUMMARY:
{clinical_summary}

Full clinical summary is available in your portal.

— Noor, MHP Clinical Intake Agent
   {now}
"""
    return body
