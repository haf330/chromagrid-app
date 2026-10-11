"""Fixed customer-choice panel, reusable inside the future Gradio Blocks app.

Gradio is imported only by build_confirmation_panel. State/callback tests do not
require it. Render from authoritative LeadSession state, not model-generated UI.
Increment revision_state after every chat-state change so stale panels disappear.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .interaction import message_entry_enabled
from .leads import (
    CONFIRMATION_NOTICE, CONFIRMATION_QUESTION, DECLINE_LABEL, SAVE_LABEL,
    ConsentError, LeadSession, LeadValidationError,
)


def handle_confirmation_choice(session: LeadSession, *, session_id: str,
                               request_id: str, choice: bool,
                               on_choice: Callable[[LeadSession, dict], None] | None = None) -> dict:
    """Process an actual UI event; optionally resume the app's model loop once.

    No storage write occurs here. A future model-loop continuation still needs
    to execute record_customer_interest through its guard after approval.
    """
    try:
        result = session.confirm_choice(session_id=session_id, request_id=request_id, choice=choice)
    except LeadValidationError as exc:
        return {"ok": False, "error": {"code": "invalid_choice", "message": str(exc)}}
    except ConsentError as exc:
        return {"ok": False, "error": {"code": "inactive_confirmation", "message": str(exc)}}
    if on_choice is not None and not result["duplicate"]:
        on_choice(session, result)
    return result


def choice_status_message(session: LeadSession, result: dict) -> str:
    if not result["ok"]:
        return result["error"]["message"]
    if session.confirmation_status == "recorded":
        message = "Your details and interest have been recorded."
    elif result["status"] == "declined":
        message = "I won't record this enquiry. You can keep asking about products."
    else:
        message = "Approval received. Your details have not been saved yet."
    if result.get("continuation_error"):
        message += " I couldn't complete the next reply. You can send another message."
    return message


def confirmation_choice_updates(session: LeadSession, *, session_id: str,
                                request_id: str, choice: bool,
                                on_choice: Callable[[LeadSession, dict], None] | None = None):
    """Yield accepted state for UI unlock before running an optional continuation.

    A continuation failure does not undo the choice, repeat it, or invent a save.
    Its exception text is deliberately excluded from customer-visible output.
    """
    result = handle_confirmation_choice(session, session_id=session_id,
                                        request_id=request_id, choice=choice)
    yield result
    if on_choice is not None and result["ok"] and not result["duplicate"]:
        try:
            on_choice(session, result)
        except Exception:
            result = dict(result, continuation_error=True)
        yield result


@dataclass
class ConfirmationPanel:
    status: Any
    # Include revision in outputs when the future chat loop changes session state.
    revision: Any


def build_confirmation_panel(session_state: Any, revision_state: Any, *,
                             on_choice: Callable[[LeadSession, dict], None] | None = None,
                             message_controls: tuple[Any, ...] = (),
                             concurrency_id: str = "chromagrid-chat") -> ConfirmationPanel:
    """Call inside gr.Blocks, sharing the chat's session and revision states.

    The immutable request captured in each rendered callback is crucial: reading
    only the latest request from mutable State could approve details the customer
    never saw. Buttons/events get request-specific keys to preserve that binding.
    Serialize chat and button events with the same concurrency_id in the parent app.
    """
    import gradio as gr

    status = gr.Markdown("", elem_id="confirmation-status")

    def entry_updates(session):
        return [gr.update(interactive=message_entry_enabled(session)) for _ in message_controls]

    if message_controls:
        revision_state.change(
            entry_updates, inputs=[session_state], outputs=list(message_controls),
            concurrency_id=concurrency_id, concurrency_limit=1,
            show_progress="hidden", api_visibility="private",
        )

    @gr.render(inputs=[session_state, revision_state], concurrency_id=concurrency_id,
               concurrency_limit=1, show_progress="hidden")
    def render_panel(session: LeadSession, revision: int):
        request = session.pending_confirmation
        if request is None:
            return
        prefix = request.request_id
        with gr.Group(key=(prefix, "panel")):
            gr.Markdown("### Record your interest?")
            gr.Markdown(CONFIRMATION_QUESTION)
            # Read-only plain-text fields prevent names/notes becoming HTML or Markdown.
            gr.Textbox(value=request.details.name, label="Name", interactive=False,
                       key=(prefix, "name"))
            gr.Textbox(value=request.details.email, label="Email", interactive=False,
                       key=(prefix, "email"))
            gr.Textbox(value=request.details.message, label="Interest notes", lines=2,
                       interactive=False, key=(prefix, "notes"))
            gr.Markdown(CONFIRMATION_NOTICE)
            if message_controls:
                gr.Markdown("Choose Save my details or Don't save to continue sending messages.")
            with gr.Row():
                approve = gr.Button(SAVE_LABEL, variant="primary", min_width=160,
                                    key=(prefix, "approve"))
                decline = gr.Button(DECLINE_LABEL, variant="secondary", min_width=160,
                                    key=(prefix, "decline"))

            def respond(current_session, current_revision, choice):
                updates = confirmation_choice_updates(
                    current_session, session_id=request.session_id,
                    request_id=request.request_id, choice=choice, on_choice=on_choice,
                )
                for index, result in enumerate(updates, start=1):
                    yield (current_session, current_revision + index,
                           choice_status_message(current_session, result), *entry_updates(current_session))

            def approve_choice(current_session, current_revision):
                yield from respond(current_session, current_revision, True)

            def decline_choice(current_session, current_revision):
                yield from respond(current_session, current_revision, False)

            for button, callback, action in ((approve, approve_choice, "approve"),
                                             (decline, decline_choice, "decline")):
                button.click(
                    callback,
                    inputs=[session_state, revision_state],
                    outputs=[session_state, revision_state, status, *message_controls],
                    concurrency_id=concurrency_id, concurrency_limit=1,
                    api_visibility="private", key=(prefix, action, "event"),
                )

    return ConfirmationPanel(status, revision_state)
