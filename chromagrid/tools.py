"""Allowlisted tool execution for a single application-bound customer turn.

This dispatcher makes no model calls and does not classify intent. The chat loop
decodes model arguments and binds the current customer message.
"""

from .catalogue import (CALCULATE_QUOTE_SCHEMA, CHOICES, GET_PRODUCT_OPTIONS_SCHEMA,
                        Catalogue, error)
from .feedback import FeedbackToolExecutor, RECORD_FEEDBACK_SCHEMA
from .leads import (LeadDraftToolExecutor, LeadSession, LeadToolExecutor,
                    RECORD_CUSTOMER_INTEREST_SCHEMA,
                    UPDATE_LEAD_DRAFT_SCHEMA)
from .storage import JsonlStore


TOOL_SCHEMAS = [GET_PRODUCT_OPTIONS_SCHEMA, CALCULATE_QUOTE_SCHEMA,
                RECORD_FEEDBACK_SCHEMA, RECORD_CUSTOMER_INTEREST_SCHEMA,
                UPDATE_LEAD_DRAFT_SCHEMA]


class BusinessToolExecutor:
    def __init__(self, *, catalogue: Catalogue, session: LeadSession,
                 customer_message_id: str, lead_store: JsonlStore, feedback_store: JsonlStore,
                 allowed_tools=None):
        known = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        self.allowed_tools = frozenset(known if allowed_tools is None else allowed_tools)
        if not self.allowed_tools <= known:
            raise ValueError("Unknown workflow capability.")
        self.catalogue = catalogue
        self.leads = LeadToolExecutor(session, lead_store)
        self.draft = LeadDraftToolExecutor(session, customer_message_id)
        self.feedback = FeedbackToolExecutor(session, feedback_store, customer_message_id)

    def execute_tool(self, name: str, arguments: object) -> dict:
        if not isinstance(name, str):
            return error("unknown_tool", "This tool is not available.")
        if name not in {schema["function"]["name"] for schema in TOOL_SCHEMAS}:
            return error("unknown_tool", "This tool is not available.")
        if name not in self.allowed_tools:
            return error("tool_not_enabled", "This tool is not enabled for the current request.")
        if name == "record_customer_interest":
            return self.leads.request_customer_interest(arguments)
        if name == "update_lead_draft":
            return self.draft.execute(arguments)
        if name == "record_feedback":
            return self.feedback.execute(arguments)
        if name == "get_product_options":
            if not isinstance(arguments, dict) or set(arguments) != set(CHOICES):
                return error("invalid_arguments", "Supply all five filters, using null for unspecified preferences.")
            return self.catalogue.get_product_options(**arguments)
        if name == "calculate_quote":
            if not isinstance(arguments, dict) or set(arguments) != {"items"}:
                return error("invalid_arguments", "Supply only items; prices and delivery come from the catalogue.")
            return self.catalogue.calculate_quote(arguments["items"])
        return error("unknown_tool", "This tool is not available.")
