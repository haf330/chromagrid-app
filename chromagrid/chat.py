"""Per-chat routed model/tool loop and actual customer-choice continuation.

UI clients serialize submit/choice operations. Choice updates yield accepted state
before generation so message controls can unlock. No API calls occur on import.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from threading import RLock

from .interaction import accept_customer_message
from .leads import ConsentError, LeadSession, LeadToolExecutor, handle_confirmation_choice
from .memory import MemoryError
from .model import ModelError
from .prompts import fixed_messages
from .reflection import ReplyReviewer
from .routing import IntentRouter
from .tools import BusinessToolExecutor, TOOL_SCHEMAS


@dataclass(frozen=True)
class LoopPolicy:
    max_tool_rounds: int = 8

    def __post_init__(self):
        if type(self.max_tool_rounds) is not int or self.max_tool_rounds < 1:
            raise ValueError("Tool-round limit must be a positive integer.")


@dataclass(frozen=True)
class ChatResult:
    text: str
    status: str = "complete"
    error_code: str | None = None


class BusinessChat:
    def __init__(self, *, adapter, business_reference, catalogue, lead_store,
                 feedback_store, router=None, policy=None, reflection_store=None):
        self.adapter = adapter
        self.router = router or IntentRouter(adapter)
        self.policy = policy or LoopPolicy()
        self.session = LeadSession()
        self.memory = adapter.new_memory(fixed_messages=fixed_messages(business_reference), tools=TOOL_SCHEMAS)
        self.catalogue, self.lead_store, self.feedback_store = catalogue, lead_store, feedback_store
        self._lock = RLock()
        self._route = self._message_id = self._pending = None
        self._events = []
        self._choice_results = {}
        self._reviewer = ReplyReviewer(adapter, reflection_store)
        self._turn_trace = []

    @property
    def events(self):
        return deepcopy(self._events)

    @property
    def reflections(self):
        return self._reviewer.events

    def _finish(self, text, *, error_code=None):
        self.memory.append_messages([{"role": "assistant", "content": text}], complete=True)
        self._route = self._message_id = None
        return ChatResult(text, "error" if error_code else "complete", error_code)

    def _failure(self, code):
        # Successfully executed tools are actual events, not rolled back on API failure.
        text = "I couldn't complete the reply. Please try again."
        if self.session.confirmation_status == "recorded":
            text += " Your confirmed enquiry has been recorded."
        elif self.session.confirmation_status == "confirmed":
            text += " Approval was received, but saving has not been confirmed."
        if any(event.get("recorded_feedback") for event in self._events if event.get("message_id") == self._message_id):
            text += " Feedback from this message was recorded."
        self._events.append({"type": "response_error", "code": code})
        return self._finish(text, error_code=code)

    def submit(self, text):
        with self._lock:
            if self._pending is not None or self.session.pending_confirmation is not None:
                return ChatResult("Choose Save my details or Don't save before sending another message.",
                                  "blocked", "confirmation_pending")
            if not isinstance(text, str) or not text.strip():
                return ChatResult("Please enter a message.", "error", "invalid_message")
            recent = [{"role": item["role"], "content": item["content"]}
                      for item in self.memory.transcript if item["role"] in ("user", "assistant")
                      and isinstance(item.get("content"), str)][-12:]
            self._turn_trace = []
            self._message_id = accept_customer_message(self.session, text)
            self.memory.start_turn(text)
            try:
                self._route = self.router.route(text, recent_messages=recent, session=self.session)
            except ModelError as exc:
                return self._finish(str(exc), error_code=exc.code)
            if not self._route.relevant:
                return self._finish(self._route.fixed_reply)
            if "lead_collection" in self._route.intents and self.session.collection_declined:
                try:
                    self.session.reopen_collection(self._message_id)
                except ConsentError:
                    pass  # Existing draft executor still blocks collection after refusal.
            return self._run()

    def _run(self):
        executor = BusinessToolExecutor(catalogue=self.catalogue, session=self.session,
            customer_message_id=self._message_id, lead_store=self.lead_store,
            feedback_store=self.feedback_store, allowed_tools=self._route.allowed_tools)
        for round_index in range(self.policy.max_tool_rounds + 1):
            selected = self._route.tool_schemas if round_index < self.policy.max_tool_rounds else []
            if self.session.confirmation_status in {"confirmed", "recorded"}:
                selected = [schema for schema in selected if schema["function"]["name"] != "record_customer_interest"]
                executor.allowed_tools = executor.allowed_tools - {"record_customer_interest"}
            state = json.dumps({"draft": self.session.draft_snapshot(),
                                "confirmation_status": self.session.confirmation_status,
                                "tools_exhausted": not selected}, ensure_ascii=False)
            try:
                prepared = self.memory.prepare(state_context=state, tools=selected,
                                               workflow_instructions=self._route.instructions)
                if prepared.pending_tool_call_ids:
                    raise ModelError("pending_tools", "Resolve pending tool results before generation.")
                reply = self.adapter.respond(prepared.messages, prepared.tools)
                message = reply.message
                calls = message.get("tool_calls") or []
                if not calls:
                    message["content"] = self._reviewer.review(messages=prepared.messages,
                        draft=message["content"], trace=self._turn_trace,
                        state={"draft": self.session.draft_snapshot(), "confirmation_status": self.session.confirmation_status,
                               "scope": {"has_unrelated_task": self._route.has_unrelated_task}},
                        session_id=self.session.session_id)
                    self.memory.commit(prepared, [message], complete=True)
                    self._route = self._message_id = None
                    return ChatResult(message["content"])
                if not selected:
                    raise ModelError("tool_round_limit", "Tools were requested after the loop limit.")
                self.memory.commit(prepared, [message])
            except (ModelError, MemoryError) as exc:
                return self._failure(getattr(exc, "code", "context_error"))
            if len(calls) != 1:
                # Resolve the entire rejected batch; execute none of its side effects.
                for rejected in calls:
                    self._turn_trace.append({"tool_call_id": rejected["id"],
                        "name": rejected["function"]["name"],
                        "arguments": rejected["function"]["arguments"],
                        "result": {"ok": False, "error": {"code": "one_tool_at_a_time"}}})
                self.memory.append_messages([{"role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps({"ok": False, "error": {"code": "one_tool_at_a_time",
                                                                 "message": "Request one tool at a time."}})} for call in calls])
                continue
            call = calls[0]
            arguments = None
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (TypeError, ValueError):
                result = {"ok": False, "error": {"code": "invalid_json", "message": "Supply valid JSON arguments."}}
            else:
                try:
                    result = executor.execute_tool(name, arguments)
                except Exception:
                    result = {"ok": False, "error": {"code": "execution_failed", "message": "The action could not be confirmed."}}
            self._turn_trace.append({"tool_call_id": call["id"], "name": name,
                                     "arguments": deepcopy(arguments), "result": deepcopy(result)})
            self._events.append({"type": "tool", "name": name, "ok": result.get("ok", False),
                                 "message_id": self._message_id,
                                 "error_code": result.get("error", {}).get("code"),
                                 "mismatched_fields": result.get("mismatched_fields", []),
                                 "recorded_feedback": name == "record_feedback" and result.get("ok", False)})
            if (name == "record_customer_interest" and result.get("ok") and
                    result.get("status") == "awaiting_customer"):
                # Keep this call unresolved until its real UI result is available.
                self._pending = {"call_id": call["id"], "result": deepcopy(result)}
                content = message.get("content") or ""
                text = (content + "\n\n" if content else "") + self.session.pending_confirmation.prompt
                return ChatResult(text, "awaiting_confirmation")
            self.memory.append_messages([{"role": "tool", "tool_call_id": call["id"],
                                         "content": json.dumps(result, ensure_ascii=False)}])
        return self._failure("tool_round_limit")

    def choice_updates(self, *, session_id, request_id, choice):
        """Yield the accepted click before resuming; duplicate clicks never rerun tools.

        The UI must render the first yield (including unlocked controls) before
        requesting the next. A real choice begins a new bounded response sequence.
        """
        with self._lock:
            result = handle_confirmation_choice(self.session, session_id=session_id,
                                                 request_id=request_id, choice=choice)
            if result["ok"] and choice and not result["duplicate"]:
                # Persist before yielding or calling the model. No model arguments
                # participate in this write; the approved snapshot is authoritative.
                save = LeadToolExecutor(self.session, self.lead_store).record_confirmed_interest()
                result["save_result"] = save
                if save.get("ok"):
                    result["status"] = self.session.confirmation_status
                self._events.append({"type": "approval_save", "name": "record_customer_interest",
                                     "ok": save.get("ok", False), "message_id": self._message_id,
                                     "error_code": save.get("error", {}).get("code")})
        # Gradio may resume successive generator steps on different worker threads.
        # Never hold a thread-owned RLock across a UI yield.
        yield deepcopy(result)
        with self._lock:
            if not result["ok"]:
                return
            if result["duplicate"]:
                response = self._choice_results.get(request_id)
            elif self._pending is None:
                response = ChatResult("Your choice was received, but the conversation could not resume.",
                                      "error", "missing_continuation")
            else:
                output = deepcopy(self._pending["result"])
                output.update(status=result["status"], customer_choice={"choice": choice, "event_id": result["event_id"]})
                if choice:
                    output["save_result"] = deepcopy(result["save_result"])
                for event in self._turn_trace:
                    if event["tool_call_id"] == self._pending["call_id"]:
                        event["result"] = deepcopy(output)
                self.memory.append_messages([{"role": "tool", "tool_call_id": self._pending["call_id"],
                                             "content": json.dumps(output, ensure_ascii=False)}])
                self._pending = None
                if not choice:
                    response = self._finish("I won't record this enquiry. You can keep asking about products.")
                elif not result["save_result"].get("ok"):
                    response = self._finish("Saving could not be confirmed. Your approval is retained, but I cannot report this enquiry as saved.",
                        error_code=result["save_result"].get("error", {}).get("code", "storage_error"))
                else:
                    response = self._run()
                self._choice_results[request_id] = response
        if response is not None:
            yield response
