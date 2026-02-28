"""
ChromaDB memory module — stores and retrieves patient conversation history
for semantic recall across visits.

CONTEXT MANAGEMENT STRATEGY:
  - Last 2 visits → full conversation transcript (detailed context)
  - Older visits  → summary only (compressed context)
  - This prevents context window explosion for patients with many visits
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings
from datetime import datetime, timezone
from typing import Optional


# Persistent ChromaDB client
_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_PATH)
    return _client


def get_patient_collection(patient_id: int):
    """
    Each patient gets their own collection for total data isolation.
    """
    client = get_chroma_client()
    collection_name = f"patient_{patient_id}_memory"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"description": f"Conversation memory for patient {patient_id}"}
    )


def store_conversation_memory(patient_id: int, session_id: int,
                               conversation_text: str, summary: str = ""):
    """
    Store a full conversation transcript + summary into ChromaDB
    for this patient. Each session is stored as a separate document.
    """
    collection = get_patient_collection(patient_id)

    doc_id = f"session_{session_id}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # Store the conversation with metadata
    collection.upsert(
        ids=[doc_id],
        documents=[conversation_text],
        metadatas=[{
            "session_id": str(session_id),
            "patient_id": str(patient_id),
            "timestamp": timestamp,
            "summary": summary,
        }]
    )


def recall_patient_memory(patient_id: int, query: str = "",
                           n_results: int = 5) -> list[dict]:
    """
    Retrieve the most relevant past conversations for this patient.
    If no query is provided, returns the most recent conversations.
    """
    collection = get_patient_collection(patient_id)

    # Check if collection has any documents
    count = collection.count()
    if count == 0:
        return []

    n = min(n_results, count)

    if query:
        # Semantic search — find conversations most relevant to the query
        results = collection.query(
            query_texts=[query],
            n_results=n,
        )
    else:
        # Get all documents (most recent first based on metadata)
        results = collection.get(
            include=["documents", "metadatas"],
        )
        # Format to match query output structure
        return [
            {
                "document": doc,
                "metadata": meta,
            }
            for doc, meta in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            )
        ]

    # Format query results
    memories = []
    if results and results.get("documents"):
        for docs, metas in zip(results["documents"], results["metadatas"]):
            for doc, meta in zip(docs, metas):
                memories.append({
                    "document": doc,
                    "metadata": meta,
                })

    return memories


def get_full_patient_history(patient_id: int) -> str:
    """
    Get all stored conversation history for a patient as a single string.
    Used by the Memory Recall Node to build context.

    *** LEGACY — sends ALL transcripts. Use get_managed_patient_history() instead. ***
    """
    memories = recall_patient_memory(patient_id)
    if not memories:
        return ""

    # Sort by timestamp if available
    sorted_memories = sorted(
        memories,
        key=lambda m: m.get("metadata", {}).get("timestamp", ""),
    )

    history_parts = []
    for mem in sorted_memories:
        meta = mem.get("metadata", {})
        summary = meta.get("summary", "")
        timestamp = meta.get("timestamp", "Unknown date")

        header = f"--- Session on {timestamp} ---"
        if summary:
            header += f"\nSummary: {summary}"

        history_parts.append(f"{header}\n{mem.get('document', '')}")

    return "\n\n".join(history_parts)


# ──────────────────────── CONTEXT-MANAGED MEMORY ────────────────────────

def get_managed_patient_history(patient_id: int, full_recent: int = 2,
                                 max_old_summaries: int = 8) -> str:
    """
    Smart context management for patient history.

    Strategy:
      - Most recent `full_recent` visits → include FULL conversation transcript
      - Older visits → include ONLY the summary line (1-2 sentences each)
      - Cap old summaries at `max_old_summaries` to prevent unbounded growth

    Example for a patient with 12 visits:
      - Visits 11-12: full transcript (~3,000 tokens each)
      - Visits 3-10:  summary only (~50 tokens each)
      - Visits 1-2:   dropped (too old to be useful)
      - Total: ~6,400 tokens instead of ~36,000 tokens

    This keeps context focused and within budget while preserving
    the most important recent detail + long-term trend awareness.
    """
    memories = recall_patient_memory(patient_id)
    if not memories:
        return ""

    # Sort by timestamp — oldest first
    sorted_memories = sorted(
        memories,
        key=lambda m: m.get("metadata", {}).get("timestamp", ""),
    )

    total = len(sorted_memories)

    if total == 0:
        return ""

    # Split into old and recent
    if total <= full_recent:
        # Few visits — send everything in full
        recent_memories = sorted_memories
        old_memories = []
    else:
        recent_memories = sorted_memories[-full_recent:]   # last N visits — FULL
        old_memories = sorted_memories[:-full_recent]      # older visits — SUMMARY ONLY

    # Cap old summaries (keep the most recent old ones)
    if len(old_memories) > max_old_summaries:
        old_memories = old_memories[-max_old_summaries:]

    history_parts = []

    # ── Old visits: SUMMARY ONLY ──
    if old_memories:
        history_parts.append("OLDER VISIT SUMMARIES (condensed):")
        for mem in old_memories:
            meta = mem.get("metadata", {})
            summary = meta.get("summary", "")
            timestamp = meta.get("timestamp", "Unknown date")

            if summary:
                history_parts.append(f"  • {timestamp[:10]}: {summary}")
            else:
                # No summary stored — extract first 150 chars of transcript as fallback
                doc = mem.get("document", "")
                snippet = doc[:150].replace("\n", " ").strip()
                if snippet:
                    history_parts.append(f"  • {timestamp[:10]}: {snippet}...")

    # ── Recent visits: FULL TRANSCRIPT ──
    if recent_memories:
        history_parts.append("\nRECENT VISIT DETAILS (full context):")
        for mem in recent_memories:
            meta = mem.get("metadata", {})
            summary = meta.get("summary", "")
            timestamp = meta.get("timestamp", "Unknown date")

            header = f"--- Session on {timestamp} ---"
            if summary:
                header += f"\nSummary: {summary}"

            history_parts.append(f"{header}\n{mem.get('document', '')}")

    return "\n\n".join(history_parts)
