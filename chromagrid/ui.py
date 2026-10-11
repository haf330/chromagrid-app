"""Gradio ChatInterface with application-owned message locking and real choices.

Pinned Gradio 6.29.1 normally unconditionally unlocks its textbox after a reply.
The small subclass owns event setup to preserve the pending-confirmation policy.
No credentials or API calls are needed to build the interface.
"""

from copy import deepcopy
from pathlib import Path

from .chat import BusinessChat
from .reference import load_business_reference
from .feedback import FEEDBACK_NOTICE
from .model import LunaAdapter
from .storage import JsonlStore


def chat_factory(root, *, adapter_factory=LunaAdapter, log_directory=None):
    root = Path(root).resolve()
    reference, catalogue = load_business_reference(root)
    logs = Path(log_directory) if log_directory is not None else root / "logs"

    def create():
        return BusinessChat(adapter=adapter_factory(), business_reference=reference,
                            catalogue=catalogue, lead_store=JsonlStore(logs / "leads.jsonl"),
                            feedback_store=JsonlStore(logs / "feedback.jsonl"),
                            reflection_store=JsonlStore(logs / "reflections.jsonl"))
    return create


def diagnostic_data(bot):
    if bot is None:
        return {"routes": [], "tools": []}
    return {"routes": getattr(bot.router, "events", []), "tools": bot.events,
            "usage": getattr(bot.adapter, "usage_events", []), "reflections": bot.reflections}


def submit_message(message, history, bot, factory):
    """Testable UI boundary; never trust visible history as customer evidence."""
    current = bot if bot is not None else factory()
    result = current.submit(message)
    updated = deepcopy(history or [])
    if result.status != "blocked" and result.error_code != "invalid_message":
        updated.extend([{"role": "user", "content": message},
                        {"role": "assistant", "content": result.text}])
    return current, updated, result


def build_app(factory, *, title="Chromagrid Assistant", offline=False):
    import gradio as gr

    with gr.Blocks(title=title, analytics_enabled=False) as demo:
        gr.Markdown("# " + title)
        gr.Markdown(FEEDBACK_NOTICE)
        if offline:
            gr.Markdown("**Offline preview:** scripted responses and fictional records in temporary storage. No API calls.")
        # Lazy initialization avoids SDK/key access during notebook setup or page load.
        def cleanup(bot):
            if bot is not None and hasattr(bot.adapter, "close"):
                bot.adapter.close()
        bot_state = gr.State(None, delete_callback=cleanup)
        revision = gr.State(0)
        status = gr.Markdown("")
        with gr.Accordion("Assignment diagnostics", open=False):
            diagnostics = gr.JSON(value={"routes": [], "tools": []}, label="Routing and tool events")

        class ControlledChatInterface(gr.ChatInterface):
            def _setup_events(self):
                # Own the final textbox update instead of Gradio's unconditional unlock.
                def respond(message, history, bot, version):
                    try:
                        current, updated, result = submit_message(message, history, bot, factory)
                    except Exception:
                        # SDK initialization/key errors must not disclose bodies or secrets.
                        enabled = bot is None or bot.session.pending_confirmation is None
                        yield (gr.update(interactive=enabled), history, history, bot, version,
                               "The assistant could not start. Check the local configuration.", diagnostic_data(bot))
                        return
                    enabled = current.session.pending_confirmation is None
                    yield (gr.update(value="" if result.status != "blocked" else message, interactive=enabled),
                           updated, updated, current, version + 1,
                           result.text if result.status in ("blocked", "error") else "", diagnostic_data(current))

                self.textbox.submit(respond, inputs=[self.textbox, self.chatbot_state, bot_state, revision],
                    outputs=[self.textbox, self.chatbot, self.chatbot_state, bot_state, revision, status, diagnostics],
                    concurrency_id="chromagrid-chat", concurrency_limit=1, api_visibility="private")

        interface = ControlledChatInterface(fn=lambda message, history: "", analytics_enabled=False,
                    editable=False, save_history=False, stop_btn=False, api_visibility="private",
                    flagging_mode="never", title=None, autofocus=False)

        @gr.render(inputs=[bot_state, revision], concurrency_id="chromagrid-chat", concurrency_limit=1)
        def render_confirmation(bot, version):
            if bot is None or bot.session.pending_confirmation is None:
                return
            request = bot.session.pending_confirmation
            with gr.Group(key=(request.request_id, "confirmation")):
                gr.Markdown("### Record your interest?")
                gr.Markdown("Review these details before choosing. Recording an enquiry does not place an order or guarantee contact.")
                for label, value in (("Name", request.details.name), ("Email", request.details.email),
                                     ("Interest notes", request.details.message)):
                    gr.Textbox(value=value, label=label, interactive=False, key=(request.request_id, label))
                with gr.Row():
                    approve = gr.Button("Save my details", variant="primary", key=(request.request_id, "save"))
                    decline = gr.Button("Don't save", key=(request.request_id, "decline"))

            def choose(current, history, current_version, choice):
                updated = deepcopy(history or [])
                # IDs come from the displayed panel captured above, never latest mutable state.
                for update in current.choice_updates(session_id=request.session_id,
                                                     request_id=request.request_id, choice=choice):
                    if isinstance(update, dict):
                        text = ("Your confirmed enquiry has been recorded." if current.session.confirmation_status == "recorded" else
                                "Approval received. Saving could not be confirmed." if choice else
                                "Your choice was received.") if update["ok"] else update["error"]["message"]
                    else:
                        text = update.text
                        if not updated or updated[-1].get("content") != text:
                            updated.append({"role": "assistant", "content": text})
                    yield (current, updated, updated, current_version + 1,
                           gr.update(interactive=current.session.pending_confirmation is None),
                           text, diagnostic_data(current))

            def save(current, history, current_version):
                yield from choose(current, history, current_version, True)

            def dont_save(current, history, current_version):
                yield from choose(current, history, current_version, False)

            for button, callback in ((approve, save), (decline, dont_save)):
                button.click(callback, inputs=[bot_state, interface.chatbot_state, revision],
                    outputs=[bot_state, interface.chatbot, interface.chatbot_state, revision,
                             interface.textbox, status, diagnostics],
                    concurrency_id="chromagrid-chat", concurrency_limit=1, api_visibility="private")
    return demo
