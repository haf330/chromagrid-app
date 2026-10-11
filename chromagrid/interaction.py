"""Application policy for message entry while a save choice is pending.

All future chat submission handlers must use accept_customer_message. The lower
level evidence API remains available to text-only clients and trusted setup code.
"""

from .leads import ConsentError, LeadSession


def message_entry_enabled(session: LeadSession) -> bool:
    return session.pending_confirmation is None


def accept_customer_message(session: LeadSession, text: str) -> str:
    """Reject a blocked submission before registering any new customer evidence."""
    if not message_entry_enabled(session):
        raise ConsentError("Choose Save my details or Don't save before sending another message.")
    return session.remember_customer_message(text)
