"""Record exact customer feedback/question excerpts with message-scoped retries.

The application binds the current message. The model supplies only question text,
never a session/message identifier. Intent and answerability still need the model
and business context; substring validation does not establish either meaning.
"""

from hashlib import sha256

from .leads import LeadSession, LeadValidationError
from .storage import EventConflictError, JsonlStore, StorageError


FEEDBACK_NOTICE = (
    "This demo may record your feedback and business questions it cannot answer. "
    "Saving a contact enquiry requires your confirmation. Recording does not notify "
    "staff or guarantee a reply. Draft and reviewed replies are logged locally for this demo."
)


def _error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


class FeedbackToolExecutor:
    def __init__(self, session: LeadSession, store: JsonlStore, customer_message_id: str):
        self.session, self.store = session, store
        self.customer_message_id = customer_message_id

    def execute(self, arguments: object) -> dict:
        if not isinstance(arguments, dict) or set(arguments) != {"question"}:
            return _error("invalid_arguments", "Supply only question, preserving the customer's wording.")
        return self.record_feedback(arguments["question"])

    def record_feedback(self, question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return _error("invalid_arguments", "Supply a nonempty customer feedback or question excerpt.")
        try:
            source = self.session.customer_message_text(self.customer_message_id)
        except LeadValidationError:
            return _error("invalid_customer_event", "No current customer message is bound to this tool.")
        if question not in source:
            return _error("customer_text_required", "Preserve an exact excerpt from the current customer message.")
        # Distinct excerpts in one mixed message can be logged separately. A fresh
        # model tool-call ID never changes identity. A new customer message does.
        digest = sha256(question.encode("utf-8")).hexdigest()
        event_id = f"feedback:{self.customer_message_id}:{digest}"
        try:
            result = self.store.append_once(
                session_id=self.session.session_id, event_id=event_id,
                payload={"kind": "feedback", "question": question,
                         "customer_message_id": self.customer_message_id},
            )
        except EventConflictError:
            return _error("event_conflict", "This feedback event already contains different data.")
        except StorageError:
            return _error("storage_error", "Recording could not be confirmed. Please try again.")
        return {"ok": True, "status": "recorded" if result.created else "already_recorded",
                "record_id": result.record_id}


RECORD_FEEDBACK_SCHEMA = {
    "type": "function", "function": {
        "name": "record_feedback",
        "description": "Record explicit feedback or an unanswered Chromagrid question. Supply exact wording from the current customer message, not a paraphrase. Do not log unrelated questions or known answers.",
        "strict": True,
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "Exact feedback/question excerpt, including its original wording."},
        }, "required": ["question"], "additionalProperties": False},
    },
}
