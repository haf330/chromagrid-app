"""Application-owned lead state and a guarded model-tool executor.

The chat loop registers only actual customer messages, collects fields with
source-message IDs, and passes the displayed panel's actual click to confirm_choice().
A model cannot set consent through tool arguments.
The application must serialize turns within a chat; different chats use different
LeadSession instances. This module does not classify buying interest or call an API.
"""

from dataclasses import dataclass
import json
import re
from uuid import uuid4

from .storage import EventConflictError, JsonlStore, StorageError


class LeadValidationError(ValueError):
    """Details or conversation evidence are missing or invalid."""


class ConsentError(ValueError):
    """There is no usable confirmation for the current record."""


_EMAILS = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)


def _text(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeadValidationError(f"{field} must be a nonempty string.")
    value = value.strip()
    if len(value) > maximum or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise LeadValidationError(f"{field} is too long or contains control characters.")
    return value


def _email(value: object) -> str:
    email = _text(value, "email", 254)
    if _EMAILS.fullmatch(email) is None:
        raise LeadValidationError("Supply a valid email address.")
    local, domain = email.rsplit("@", 1)
    if (len(local) > 64 or local.startswith(".") or local.endswith(".") or ".." in local
            or any(len(label) > 63 or label.startswith("-") or label.endswith("-")
                   for label in domain.split("."))):
        raise LeadValidationError("Supply a valid email address.")
    # Syntax checking is not verification that a mailbox exists or is owned.
    return email


@dataclass(frozen=True)
class LeadDetails:
    name: str
    email: str
    message: str

    @classmethod
    def validated(cls, *, name: object, email: object, message: object) -> "LeadDetails":
        return cls(_text(name, "name", 120), _email(email), _text(message, "message", 1500))


@dataclass(frozen=True)
class SaveRequest:
    request_id: str
    details: LeadDetails
    prompt: str
    session_id: str


@dataclass(frozen=True)
class _Agreement:
    request: SaveRequest
    customer_event_id: str

    @property
    def event_id(self) -> str:
        return f"lead:{self.request.request_id}:{self.customer_event_id}"


@dataclass(frozen=True)
class _ChoiceEvent:
    request: SaveRequest
    event_id: str
    choice: bool


CONFIRMATION_QUESTION = "Would you like me to record these details and your interest?"
CONFIRMATION_NOTICE = "This does not place an order or guarantee follow-up."
SAVE_LABEL = "Save my details"
DECLINE_LABEL = "Don't save"


class LeadSession:
    def __init__(self):
        self.session_id = str(uuid4())
        self._messages: dict[str, tuple[int, str]] = {}
        self._draft: dict[str, str] = {}
        self._request: SaveRequest | None = None
        self._agreement: _Agreement | None = None
        self._choice: _ChoiceEvent | None = None
        self._saved = False
        self.collection_declined = False
        self._declined_after_message = 0

    def remember_customer_message(self, text: str) -> str:
        """Called by the application with a real customer message, not model output."""
        if not isinstance(text, str) or not text.strip():
            raise LeadValidationError("A customer message must contain text.")
        message_id = str(uuid4())
        self._messages[message_id] = (len(self._messages) + 1, text)
        return message_id

    def _customer_text(self, message_id: str) -> str:
        if not isinstance(message_id, str) or message_id not in self._messages:
            raise LeadValidationError("Evidence must reference a customer message in this chat.")
        return self._messages[message_id][1]

    def customer_message_text(self, message_id: str) -> str:
        """Read actual customer evidence for application-bound feedback execution."""
        return self._customer_text(message_id)

    @property
    def latest_customer_message_id(self) -> str | None:
        return next(reversed(self._messages), None)

    def customer_evidence_ids(self, through_message_id: str) -> tuple[str, ...]:
        """Actual customer evidence through the application-bound originating turn.

        This is evidence already held by LeadSession, not a model-context retention
        policy. It excludes assistant/tool text and later customer messages.
        """
        self._customer_text(through_message_id)
        last = self._messages[through_message_id][0]
        return tuple(key for key, (sequence, _) in self._messages.items() if sequence <= last)

    def draft_snapshot(self) -> dict:
        """Return a copy of the partial draft; missing fields remain null."""
        return {field: self._draft.get(field) for field in ("name", "email", "message")}

    def confirmed_snapshot(self) -> dict | None:
        """Read only the current exact record covered by an actual agreement.

        Used to constrain generation, never to create agreement or execute a save.
        Return a fresh copy; changed/declined/unapproved drafts have no snapshot.
        """
        if self._agreement is None or self.collection_declined:
            return None
        details = self._agreement.request.details
        snapshot = {"name": details.name, "email": details.email, "message": details.message}
        return snapshot if snapshot == self.draft_snapshot() else None

    def update_details(self, *, name: str | None = None, email: str | None = None,
                       message: str | None = None, name_source: str | None = None,
                       email_source: str | None = None,
                       interest_sources: tuple[str, ...] | None = None) -> None:
        """Collect fields across turns; changes invalidate prior agreement.

        Name and email must occur in cited customer text. Notes can summarize
        cited customer turns, so their factual meaning still needs model guidance
        and customer review; Python does not prove semantic accuracy.
        Invalid attempted changes also revoke agreement, avoiding stale saves.
        """
        updates = {key: value for key, value in
                   (("name", name), ("email", email), ("message", message)) if value is not None}
        if any(value != self._draft.get(key) for key, value in updates.items()):
            self._request = None
            self._agreement = None
            self._choice = None
            self._saved = False
        draft = dict(self._draft)
        if name is not None:
            name = _text(name, "name", 120)
            # A mailbox's local part is not evidence of a customer-supplied name.
            source_text = _EMAILS.sub(" ", self._customer_text(name_source))
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                         source_text, flags=re.IGNORECASE) is None:
                raise LeadValidationError("The name was not supplied in the cited message.")
            draft["name"] = name
        if email is not None:
            email = _email(email)
            supplied = {match.group().casefold()
                        for match in _EMAILS.finditer(self._customer_text(email_source))}
            if email.casefold() not in supplied:
                raise LeadValidationError("The email was not supplied in the cited message.")
            draft["email"] = email
        if message is not None:
            message = _text(message, "message", 1500)
            if not isinstance(interest_sources, tuple) or not interest_sources:
                raise LeadValidationError("Notes require source customer messages.")
            for source in interest_sources:
                self._customer_text(source)
            draft["message"] = message
        self._draft = draft

    def _details(self) -> LeadDetails:
        return LeadDetails.validated(name=self._draft.get("name"),
                                     email=self._draft.get("email"),
                                     message=self._draft.get("message"))

    def request_confirmation(self) -> SaveRequest:
        """Create the application-owned record shown by the confirmation panel."""
        if self.collection_declined:
            raise ConsentError("The customer declined contact collection.")
        details = self._details()
        if self._request is not None and not self._saved:
            return self._request
        display = json.dumps({"name": details.name, "email": details.email,
                              "interest_notes": details.message}, ensure_ascii=False, indent=2)
        prompt = (CONFIRMATION_QUESTION + "\n" + display + "\n" + CONFIRMATION_NOTICE
                  + f" Choose {SAVE_LABEL} or {DECLINE_LABEL}.")
        self._request = SaveRequest(str(uuid4()), details, prompt, self.session_id)
        self._agreement = None
        self._choice = None
        self._saved = False
        return self._request

    @property
    def pending_confirmation(self) -> SaveRequest | None:
        if self.collection_declined or self._agreement is not None or self._saved:
            return None
        return self._request

    @property
    def confirmation_status(self) -> str:
        if self.collection_declined:
            return "declined"
        if self._saved:
            return "recorded"
        if self._agreement is not None:
            return "confirmed"
        return "awaiting_customer" if self._request is not None else "idle"

    def confirm_choice(self, *, session_id: str, request_id: str, choice: bool) -> dict:
        """Application UI callback only; never expose this method as a model tool.

        The callback captures the IDs of the displayed panel. Button clicks are
        their own customer events, not invented chat messages. Each panel accepts
        one decision. Repeated identical clicks reuse the event; opposite clicks
        or stale/cross-chat panels are rejected.
        """
        if type(choice) is not bool:
            raise LeadValidationError("Choose one of the two confirmation buttons.")
        if session_id != self.session_id:
            raise ConsentError("This confirmation belongs to a different chat.")
        previous = self._choice
        if previous is not None and previous.request.request_id == request_id:
            if previous.choice != choice:
                raise ConsentError("A choice has already been received for this request.")
            return {"ok": True, "status": "confirmed" if choice else "declined",
                    "event_id": previous.event_id, "duplicate": True}
        request = self.pending_confirmation
        if request is None or request.request_id != request_id:
            raise ConsentError("This confirmation is no longer active. Review the current details.")
        event = _ChoiceEvent(request, str(uuid4()), choice)
        self._choice = event
        if choice:
            self._agreement = _Agreement(request, event.event_id)
        else:
            self.collection_declined = True
            self._declined_after_message = len(self._messages)
            self._agreement = None
            self._request = None
        return {"ok": True, "status": "confirmed" if choice else "declined",
                "event_id": event.event_id, "duplicate": False}

    def reopen_collection(self, customer_message_id: str) -> None:
        """Allow a new offer after refusal only on an explicit customer request."""
        text = self._customer_text(customer_message_id).strip().casefold()
        if self._messages[customer_message_id][0] <= self._declined_after_message:
            raise ConsentError("A new customer request is needed after refusal.")
        if text not in {"save my details", "record my details",
                        "please save my details", "please record my details"}:
            raise ConsentError("An explicit customer request is needed to reopen collection.")
        self.collection_declined = False
        self._request = None
        self._agreement = None
        self._choice = None
        self._saved = False


def handle_confirmation_choice(session: LeadSession, *, session_id: str,
                               request_id: str, choice: bool) -> dict:
    """Validate an actual panel click without saving or calling the model."""
    try:
        return session.confirm_choice(session_id=session_id, request_id=request_id, choice=choice)
    except LeadValidationError as exc:
        return _error("invalid_choice", str(exc))
    except ConsentError as exc:
        return _error("inactive_confirmation", str(exc))


UPDATE_LEAD_DRAFT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_lead_draft",
        "description": "Update the in-chat contact draft using customer-supplied details. "
                       "Use null to retain a field. Does not request confirmation, grant consent, or save a lead. "
                       "Do not call while a confirmation is pending or collection was declined.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"],
                         "description": "Name supplied in the current customer message, or null to retain it."},
                "email": {"type": ["string", "null"],
                          "description": "Email supplied in the current customer message, or null to retain it."},
                "message": {"type": ["string", "null"],
                            "description": "Full factual interest summary from customer messages in this chat, or null to retain it. Replaces earlier notes."},
            },
            "required": ["name", "email", "message"],
            "additionalProperties": False,
        },
    },
}


class LeadDraftToolExecutor:
    """Bind draft updates to a real, current customer turn, without log access."""

    def __init__(self, session: LeadSession, customer_message_id: str):
        self.session = session
        self.customer_message_id = customer_message_id

    def execute(self, arguments: object) -> dict:
        if not isinstance(arguments, dict) or set(arguments) != {"name", "email", "message"}:
            return _error("invalid_arguments", "Supply only name, email, and message; use null to retain a field.")
        return self.update_lead_draft(**arguments)

    def update_lead_draft(self, name: str | None, email: str | None, message: str | None) -> dict:
        if self.session.collection_declined:
            return _error("collection_declined", "The customer declined. Continue helping without collection.")
        if self.session.pending_confirmation is not None:
            return _error("confirmation_pending", "Wait for the customer's current save/decline choice.")
        try:
            evidence = self.session.customer_evidence_ids(self.customer_message_id)
        except LeadValidationError:
            return _error("invalid_customer_event", "Bind this tool to the actual current customer message.")
        if self.customer_message_id != self.session.latest_customer_message_id:
            return _error("invalid_customer_event", "An older turn cannot update the current draft.")
        before = self.session.draft_snapshot()
        try:
            self.session.update_details(
                name=name, email=email, message=message,
                name_source=self.customer_message_id, email_source=self.customer_message_id,
                interest_sources=evidence if message is not None else None,
            )
        except LeadValidationError as exc:
            # Existing update_details leaves draft fields unchanged on failure,
            # but revokes approval for an attempted changed record.
            return _error("invalid_arguments", str(exc))
        draft = self.session.draft_snapshot()
        return {"ok": True, "status": "draft_updated" if draft != before else "draft_unchanged",
                "draft": draft, "missing_fields": [key for key, value in draft.items() if value is None],
                "confirmation_status": self.session.confirmation_status}


RECORD_CUSTOMER_INTEREST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_customer_interest",
        "description": "Request recording of the current customer-supported name, email and interest notes. "
                       "Shows exact-record approval buttons and waits for the real customer choice. "
                       "Python saves that snapshot only on approval; the completed result reports the save. "
                       "Does not place an order or notify staff.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer-supplied email address."},
                "name": {"type": "string", "description": "Customer-supplied name."},
                "message": {"type": "string", "description": "Interest notes shown and confirmed."},
            },
            "required": ["email", "name", "message"],
            "additionalProperties": False,
        },
    },
}


def _error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


class LeadToolExecutor:
    """Bind the business saver and confirmation request to application-owned state and storage."""

    def __init__(self, session: LeadSession, store: JsonlStore):
        self.session, self.store = session, store

    def _request_confirmation(self) -> dict:
        if self.session.confirmation_status in {"confirmed", "recorded"}:
            # Repeated model requests cannot reopen a decided panel.
            return {"ok": True, "status": self.session.confirmation_status}
        try:
            request = self.session.request_confirmation()
        except LeadValidationError as exc:
            return _error("missing_or_invalid_details", str(exc))
        except ConsentError:
            return _error("collection_declined", "The customer declined. Continue helping without collection.")
        return {"ok": True, "status": "awaiting_customer", "ui_event": {
            "type": "lead_confirmation", "request_id": request.request_id,
            "session_id": request.session_id,
            "question": CONFIRMATION_QUESTION, "notice": CONFIRMATION_NOTICE,
            "details": {"name": request.details.name, "email": request.details.email,
                        "message": request.details.message},
            "choices": [{"label": SAVE_LABEL, "value": True},
                        {"label": DECLINE_LABEL, "value": False}],
        }}

    def request_customer_interest(self, arguments: object) -> dict:
        """Model entry point: validate the draft and pause for actual consent.

        The shared chat completes this tool with the button's deterministic save
        result. This entry point never writes and cannot grant permission.
        """
        if not isinstance(arguments, dict) or set(arguments) != {"email", "name", "message"}:
            return _error("invalid_arguments", "Supply only email, name, and message.")
        try:
            details = LeadDetails.validated(**arguments)
        except LeadValidationError as exc:
            return _error("invalid_arguments", str(exc))
        if {field: getattr(details, field) for field in ("name", "email", "message")} != self.session.draft_snapshot():
            return _error("draft_mismatch", "Use exactly the customer-supported current draft before requesting approval.")
        return self._request_confirmation()

    def record_confirmed_interest(self) -> dict:
        """Application-only save: take contents from the actual approved snapshot."""
        snapshot = self.session.confirmed_snapshot()
        if snapshot is None:
            return _error("confirmation_required", "The customer has not confirmed this record.")
        return self.record_customer_interest(**snapshot)

    def record_customer_interest(self, email: str, name: str, message: str) -> dict:
        try:
            details = LeadDetails.validated(email=email, name=name, message=message)
        except LeadValidationError as exc:
            return _error("invalid_arguments", str(exc))
        agreement = self.session._agreement
        if agreement is None or self.session.collection_declined:
            return _error("confirmation_required", "The customer has not confirmed this record.")
        if details != agreement.request.details or details != self.session._details():
            mismatched = [field for field in ("name", "email", "message")
                          if getattr(details, field) != getattr(agreement.request.details, field)
                          or getattr(details, field) != self.session.draft_snapshot()[field]]
            return dict(_error("record_mismatch", "Tool details must match the customer-confirmed record."),
                        mismatched_fields=mismatched)
        try:
            result = self.store.append_once(
                session_id=self.session.session_id,
                event_id=agreement.event_id,
                payload={"kind": "customer_interest", "name": details.name,
                         "email": details.email, "message": details.message,
                         "confirmation_event_id": agreement.customer_event_id,
                         "confirmation_source": "button"},
            )
        except EventConflictError:
            return _error("event_conflict", "This event already exists with different details.")
        except StorageError:
            return _error("storage_error", "Recording could not be confirmed. Please try again.")
        self.session._saved = True
        return {"ok": True, "status": "recorded" if result.created else "already_recorded",
                "record_id": result.record_id}
