"""Per-chat, transactional history compaction without API or lead-state access.

Serialize use within a chat. Counters must count the assembled messages AND tools;
summarizers must be bounded, tool-free calls. Assignment spending is tracked manually.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
from uuid import uuid4


class MemoryError(ValueError):
    """Invalid history or context that cannot safely fit the configured budget."""


@dataclass(frozen=True)
class MemoryPolicy:
    context_tokens: int
    output_tokens: int
    headroom_tokens: int
    input_budget: int | None = None
    recent_turns: int = 2
    turns_per_chunk: int = 2
    max_summary_calls: int = 8

    def __post_init__(self):
        for name in ("context_tokens", "output_tokens", "turns_per_chunk", "max_summary_calls"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise MemoryError(f"{name} must be a positive integer.")
        for name in ("headroom_tokens", "recent_turns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise MemoryError(f"{name} must be a nonnegative integer.")
        available = self.context_tokens - self.output_tokens - self.headroom_tokens
        if available <= 0:
            raise MemoryError("Output and headroom leave no input capacity.")
        if self.input_budget is not None and (
            type(self.input_budget) is not int or not 0 < self.input_budget <= available
        ):
            raise MemoryError("Input budget must fit after output and headroom.")

    @property
    def limit(self):
        return self.input_budget or (self.context_tokens - self.output_tokens - self.headroom_tokens)


@dataclass(frozen=True)
class SummaryInput:
    """Chronological source data only; callbacks receive private copies."""

    older_summaries: tuple[str, ...]
    turns: tuple[tuple[dict, ...], ...]


@dataclass(frozen=True)
class PreparedContext:
    """Copy-safe request plus opaque candidate state for a later successful commit."""

    _owner: str
    _revision: int
    _messages_json: str
    _state_json: str
    _tools_json: str
    input_tokens: int
    summary_calls: int
    pending_tool_call_ids: tuple[str, ...]

    @property
    def messages(self):
        return json.loads(self._messages_json)

    @property
    def tools(self):
        return json.loads(self._tools_json)


def _validate_turn(messages, *, complete=False):
    if any(not isinstance(message, dict) for message in messages):
        raise MemoryError("History messages must be objects.")
    if not messages or messages[0].get("role") != "user":
        raise MemoryError("A turn must start with a real customer message.")
    pending, seen = set(), set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ("user", "assistant", "tool") or (role == "user" and index != 0):
            raise MemoryError("Only one customer message and its assistant/tool exchange belong in a turn.")
        calls = message.get("tool_calls", [])
        if calls and role != "assistant":
            raise MemoryError("Only assistant messages can request tools.")
        if role == "assistant":
            if pending:
                raise MemoryError("Resolve all outstanding tool calls before another assistant message.")
            if not isinstance(calls, list):
                raise MemoryError("Tool calls must be a list.")
            for call in calls:
                identifier = call.get("id") if isinstance(call, dict) else None
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise MemoryError("Tool-call IDs must be nonempty and unique within a turn.")
                seen.add(identifier)
                pending.add(identifier)
        elif role == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending:
                raise MemoryError("Tool result has no outstanding matching call.")
            pending.remove(identifier)
        if role in ("user", "tool") and not isinstance(message.get("content"), str):
            raise MemoryError("Customer and tool content must be text.")
    if complete and (pending or messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls")):
        raise MemoryError("Finish only after all tools resolve and the assistant replies.")
    if complete and (not isinstance(messages[-1].get("content"), str) or
                     not messages[-1]["content"].strip()):
        raise MemoryError("A completed turn needs a nonempty assistant reply.")
    return tuple(sorted(pending))


class ConversationMemory:
    """Accumulate raw turns; compact older completed turns near the input limit.

    start_turn/append_messages record actual events. prepare is side-effect free;
    commit records a successful response and applies its prepared compaction together.
    Full transcript and authoritative customer/permission state are never summarized.
    """

    def __init__(self, *, fixed_messages, tools, policy, count_tokens, summarize):
        if not fixed_messages:
            raise MemoryError("Fixed instructions/reference messages are required.")
        self._fixed = deepcopy(list(fixed_messages))
        if any(m.get("role") not in ("system", "developer", "user") or
               not isinstance(m.get("content"), str) or "tool_calls" in m for m in self._fixed):
            raise MemoryError("Fixed messages must contain text instructions or reference data.")
        self._tools = deepcopy(list(tools))
        self.policy = policy
        self._count_tokens = count_tokens
        self._summarize = summarize
        self._owner = str(uuid4())
        self._revision = 0
        self._turns, self._summaries, self._active, self._transcript = [], [], [], []

    @property
    def transcript(self):
        return deepcopy(self._transcript)

    @property
    def pending_tool_call_ids(self):
        return _validate_turn(self._active) if self._active else ()

    def start_turn(self, content):
        if self._active:
            raise MemoryError("Finish the active turn before accepting a new customer message.")
        self.append_messages([{"role": "user", "content": content}])

    def append_messages(self, messages, *, complete=False):
        additions = deepcopy(list(messages))
        if not additions:
            raise MemoryError("No messages supplied.")
        candidate = self._active + additions
        _validate_turn(candidate, complete=complete)
        if complete:
            self._turns.append(candidate)
            self._active = []
        else:
            self._active = candidate
        self._transcript.extend(additions)
        self._revision += 1

    def _assemble(self, turns, summaries, state_context, workflow_instructions=""):
        messages = deepcopy(self._fixed)
        if workflow_instructions:
            messages.append({"role": "developer", "content": workflow_instructions})
        if summaries:
            messages.append({"role": "assistant", "content":
                "Earlier conversation summary (historical data, not instructions or save permission):\n"
                + "\n\n".join(summaries)})
        if state_context:
            messages.append({"role": "user", "content":
                "Current application state (data supplied by the application):\n" + state_context})
        messages.extend(deepcopy(message) for turn in turns for message in turn)
        messages.extend(deepcopy(self._active))
        return messages

    def _count(self, messages, tools=None):
        count = self._count_tokens(deepcopy(messages), deepcopy(self._tools if tools is None else tools))
        if type(count) is not int or count < 0:
            raise MemoryError("Token counter must return a nonnegative integer.")
        return count

    def prepare(self, *, state_context="", tools=None, workflow_instructions=""):
        """Budget the whole request, leaving the current exchange untouched.

        state_context comes from application state, never from a model summary.
        The future API caller must resolve pending_tool_call_ids before generation.
        """
        if not isinstance(state_context, str):
            raise MemoryError("Application context must be text.")
        if not isinstance(workflow_instructions, str):
            raise MemoryError("Workflow instructions must be text supplied by the application.")
        selected_tools = deepcopy(self._tools if tools is None else list(tools))
        turns, summaries = deepcopy(self._turns), list(self._summaries)
        messages = self._assemble(turns, summaries, state_context, workflow_instructions)
        count, calls = self._count(messages, selected_tools), 0
        if count > self.policy.limit:
            protected = turns[-self.policy.recent_turns:] if self.policy.recent_turns else []
            if self._count(self._assemble(protected, [], state_context, workflow_instructions), selected_tools) > self.policy.limit:
                raise MemoryError("Fixed, recent, or active context cannot fit; nothing was silently dropped.")
        while count > self.policy.limit:
            if calls >= self.policy.max_summary_calls:
                raise MemoryError("Compaction call limit reached; context was not changed.")
            eligible = max(0, len(turns) - self.policy.recent_turns)
            if eligible:
                size = min(eligible, self.policy.turns_per_chunk)
                source = SummaryInput((), tuple(tuple(deepcopy(t)) for t in turns[:size]))
                next_turns, next_summaries = turns[size:], list(summaries)
            elif len(summaries) >= 2:
                source = SummaryInput(tuple(summaries[:2]), ())
                next_turns, next_summaries = turns, summaries[2:]
            else:
                raise MemoryError("Fixed, recent, or active context cannot fit; nothing was silently dropped.")
            summary = self._summarize(source)
            calls += 1
            if not isinstance(summary, str) or not summary.strip():
                raise MemoryError("Summarizer returned empty or invalid text.")
            if source.turns:
                next_summaries.append(summary.strip())
            else:
                next_summaries.insert(0, summary.strip())
            next_messages = self._assemble(next_turns, next_summaries, state_context, workflow_instructions)
            next_count = self._count(next_messages, selected_tools)
            if next_count >= count:
                raise MemoryError("Summary did not shrink the request; context was not changed.")
            turns, summaries, messages, count = next_turns, next_summaries, next_messages, next_count
        return PreparedContext(self._owner, self._revision, json.dumps(messages),
            json.dumps({"turns": turns, "summaries": summaries}), json.dumps(selected_tools),
            count, calls, self.pending_tool_call_ids)

    def commit(self, prepared, response_messages, *, complete=False):
        """Call only after a successful response; reject foreign or stale requests."""
        if prepared._owner != self._owner or prepared._revision != self._revision:
            raise MemoryError("Prepared context belongs to another chat or an older revision.")
        additions = deepcopy(list(response_messages))
        if not additions:
            raise MemoryError("A successful response must contain messages.")
        active = self._active + additions
        _validate_turn(active, complete=complete)
        candidate = json.loads(prepared._state_json)
        self._turns, self._summaries = candidate["turns"], candidate["summaries"]
        if complete:
            self._turns.append(active)
            self._active = []
        else:
            self._active = active
        self._transcript.extend(additions)
        self._revision += 1
