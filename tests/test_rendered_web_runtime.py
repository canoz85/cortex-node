"""Renderer uses the existing native/Controller/evidence path without fallback."""

import json
import time

from langchain_core.messages import AIMessage

from core.graph_runner import run_prompt
from core.protocol.enums import ExecutionStatus
from test_brain_outcomes import native_action
from test_planner_runtime_flow import StructuredModel, production_app
from tools import rendered_web_ops as ops


def test_rendered_info_call_through_controller_and_brain(monkeypatch):
    calls = []

    def worker(request, started):
        calls.append(request)
        data = ops.result_data(request.url)
        data.update(final_url=request.url, text="Ω", total_chars=2, total_text_bytes=4,
                    returned_chars=1, truncated=True, completion_reason="text_stable", settle_observed=True,
                    blocked_request_count=1, blocked_requests=[dict(method="POST",
                        url="https://example.com/metrics", resource_type="XHR",
                        same_origin=True, is_main_navigation=False)])
        data["warnings"].append("method_blocked")
        return ops.envelope(data, None, time.monotonic())

    monkeypatch.setattr(ops, "_run_worker", worker)
    replies = iter([
        native_action("rendered_web_fetch", {"url": "https://example.com/#app", "max_chars": 1}),
        native_action("brain_step_completed", {"message": "Read the available rendered evidence."}),
    ])

    class Model(StructuredModel):
        def __init__(self):
            super().__init__([{"result": "PLAN_PROPOSED", "objective": "Read rendered content", "steps": [{
                "step_id": "read", "title": "Read rendered content", "description": "Read the JavaScript page",
                "deliverable_kind": "semantic", "primary_tool": "rendered_web_fetch", "dependencies": [],
            }]}], route="info")
            self.observations = []

        def invoke(self, messages):
            if any(message.content.startswith("Active step:") for message in messages):
                self.observations.append(messages)
                return next(replies)
            return AIMessage(content="Rendered evidence summary.")

    model = Model()
    app = production_app(model, ops.get_rendered_web_tools())
    run_prompt(app, "Read the JavaScript rendered page", run_id="rendered-runtime")
    state = app.get_state({"configurable": {"thread_id": "rendered-runtime"}}).values["execution_state"]
    assert state.protocol_visible.status == ExecutionStatus.COMPLETED
    record, = state.working.tool_execution_history
    assert record.result.success and record.result.pagination is None
    assert record.result.integrity.is_truncated and record.result.integrity.captured_bytes == 2
    assert calls[0].url.endswith("#app") and len(calls) == 1
    message = next(message.content for message in model.observations[1]
                   if message.content.startswith("Execution evidence v1:"))
    evidence = json.loads(message.split("\n", 1)[1])["current_attempts"][0]["evidence"]
    assert evidence["capture_status"] == "rendered" and evidence["truncated"]
    assert evidence["completion_reason"] == "text_stable"
    assert evidence["settle_observed"] and "method_blocked" in evidence["warnings"]
    assert evidence["blocked_requests"][0]["method"] == "POST"
