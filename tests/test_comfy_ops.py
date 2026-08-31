import json

import pytest

import tools.comfy_ops as comfy_ops
from core.models import ComfyWorkflowParams, RunWorkflowRequest
from core.protocol.enums import AsyncJobStatus
from core.runtime.gpu_resources import GpuResourceHandoffError
from conftest import get_tool, parse_result
from tools.comfy_ops import (
    DEFAULT_COMFY_CHECKPOINT,
    build_comfy_workflow,
    get_comfy_tools,
    parse_comfy_checkpoint_catalog,
    validate_comfy_workflow,
)


class FakeHttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _tool(tools, name: str):
    return get_tool(tools, name)


def _known_good_params() -> ComfyWorkflowParams:
    return ComfyWorkflowParams(
        positive_prompt="a cute cat, high quality, highly detailed, 8k wallpaper",
        negative_prompt="bad quality, blurry, low resolution, distorted",
        seed=42,
        steps=20,
        cfg=7,
        width=1024,
        height=1024,
        checkpoint=DEFAULT_COMFY_CHECKPOINT,
    )


def _known_good_tool_arguments() -> dict:
    return _known_good_params().model_dump(exclude={"checkpoint"})


def _checkpoint_catalog(*names: str) -> dict:
    return {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [list(names), {"tooltip": "Checkpoint name"}],
                }
            }
        }
    }


def test_build_comfy_workflow_matches_known_good_template():
    assert build_comfy_workflow(_known_good_params()) == {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 42,
                "steps": 20,
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 1024, "width": 1024},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": "a cute cat, high quality, highly detailed, 8k wallpaper",
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": "bad quality, blurry, low resolution, distorted",
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "CortexNode", "images": ["8", 0]},
        },
    }


def test_generated_ksampler_uses_required_prompt_references_and_cfg():
    k_sampler = build_comfy_workflow(_known_good_params())["3"]

    assert k_sampler["inputs"]["positive"] == ["6", 0]
    assert k_sampler["inputs"]["negative"] == ["7", 0]
    assert k_sampler["inputs"]["cfg"] == 7


def test_generated_text_nodes_are_clip_text_encode():
    workflow = build_comfy_workflow(_known_good_params())

    assert workflow["6"]["class_type"] == "CLIPTextEncode"
    assert workflow["7"]["class_type"] == "CLIPTextEncode"


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", 0), ("width", -1), ("height", 0), ("height", -1)],
)
def test_invalid_dimensions_are_rejected(field, value):
    values = _known_good_params().model_dump()
    values[field] = value

    with pytest.raises(ValueError, match="greater than 0"):
        ComfyWorkflowParams.model_validate(values)


def test_empty_checkpoint_is_rejected():
    values = _known_good_params().model_dump()
    values["checkpoint"] = "   "

    with pytest.raises(ValueError, match="must not be blank"):
        ComfyWorkflowParams.model_validate(values)


def test_empty_positive_prompt_is_rejected():
    values = _known_good_params().model_dump()
    values["positive_prompt"] = ""

    with pytest.raises(ValueError):
        ComfyWorkflowParams.model_validate(values)


def test_invalid_graph_reference_is_rejected():
    workflow = build_comfy_workflow(_known_good_params())
    workflow["8"]["inputs"]["samples"] = ["999", 0]

    with pytest.raises(ValueError, match="missing node '999'"):
        validate_comfy_workflow(workflow)


def test_invalid_graph_reference_output_index_is_rejected():
    workflow = build_comfy_workflow(_known_good_params())
    workflow["8"]["inputs"]["samples"] = ["3", -1]

    with pytest.raises(ValueError, match="invalid output index"):
        validate_comfy_workflow(workflow)


def test_integer_graph_reference_is_rejected():
    workflow = build_comfy_workflow(_known_good_params())
    workflow["9"]["inputs"]["images"] = [8, 0]

    with pytest.raises(ValueError, match="node ID must be a string"):
        validate_comfy_workflow(workflow)


def test_integer_workflow_node_id_is_rejected():
    workflow = build_comfy_workflow(_known_good_params())
    workflow[8] = workflow.pop("8")

    with pytest.raises(ValueError, match="node IDs must be strings"):
        validate_comfy_workflow(workflow)


def test_malformed_template_is_rejected_before_http_post(monkeypatch):
    post_calls = []

    def fake_urlopen(request, **_kwargs):
        post_calls.append(request)
        return FakeHttpResponse({"prompt_id": "prompt-1"})

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(comfy_ops, "build_comfy_workflow", lambda _params: {})
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")

    result = parse_result(run_workflow.invoke(_known_good_tool_arguments()))

    assert result["success"] is False
    assert result["error_code"] == "COMFY_INVALID_PAYLOAD"
    assert post_calls == []


def test_llm_facing_schema_has_no_unrestricted_workflow_json():
    properties = RunWorkflowRequest.model_json_schema()["properties"]

    assert "workflow_json" not in properties
    assert "checkpoint" not in properties
    assert set(properties) >= {
        "positive_prompt",
        "negative_prompt",
        "seed",
        "steps",
        "cfg",
        "width",
        "height",
        "filename_prefix",
    }


def test_llm_cannot_override_application_owned_checkpoint():
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")
    arguments = _known_good_tool_arguments()
    arguments["checkpoint"] = "untrusted.safetensors"

    with pytest.raises(ValueError):
        run_workflow.invoke(arguments)


def test_parse_comfy_checkpoint_catalog_returns_exact_provider_names():
    assert parse_comfy_checkpoint_catalog(
        _checkpoint_catalog(DEFAULT_COMFY_CHECKPOINT, "nested/other.safetensors")
    ) == (
        DEFAULT_COMFY_CHECKPOINT,
        "nested/other.safetensors",
    )


def test_missing_default_checkpoint_stops_before_prompt_post(monkeypatch):
    requested_urls: list[str] = []

    def fake_urlopen(request, **_kwargs):
        requested_urls.append(request.full_url)
        return FakeHttpResponse(_checkpoint_catalog("other-model.safetensors"))

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")

    result = parse_result(run_workflow.invoke(_known_good_tool_arguments()))

    assert result["success"] is False
    assert result["error_code"] == "COMFY_CHECKPOINT_NOT_INSTALLED"
    assert result["is_async_job"] is False
    assert result["data"] == {
        "required_checkpoint": DEFAULT_COMFY_CHECKPOINT,
        "installed_checkpoints": ["other-model.safetensors"],
    }
    assert len(requested_urls) == 1
    assert requested_urls[0].endswith("/object_info/CheckpointLoaderSimple")


def test_malformed_checkpoint_catalog_stops_before_prompt_post(monkeypatch):
    requested_urls: list[str] = []

    def fake_urlopen(request, **_kwargs):
        requested_urls.append(request.full_url)
        return FakeHttpResponse({"CheckpointLoaderSimple": {}})

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")

    result = parse_result(run_workflow.invoke(_known_good_tool_arguments()))

    assert result["success"] is False
    assert result["error_code"] == "COMFY_CHECKPOINT_CATALOG_UNAVAILABLE"
    assert "checkpoint catalog" in result["message"].lower()
    assert len(requested_urls) == 1


def test_checkpoint_catalog_connection_failure_is_not_ambiguous_submission(
    monkeypatch,
):
    def fail_catalog(*_args, **_kwargs):
        raise comfy_ops.urllib.error.URLError("catalog offline")

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fail_catalog)
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")
    arguments = _known_good_tool_arguments()
    arguments.update({
        "prompt_id": "preallocated-prompt-1",
        "client_id": "run-1",
    })

    result = parse_result(run_workflow.invoke(arguments))

    assert result["success"] is False
    assert result["error_code"] == "COMFY_CHECKPOINT_CATALOG_UNAVAILABLE"
    assert result["is_async_job"] is False
    assert result["async_job_id"] is None
    assert result["data"] is None


def test_run_comfy_workflow_reports_submission_evidence(monkeypatch):
    submitted_payloads = []

    def fake_urlopen(request, **_kwargs):
        if request.full_url.endswith("/object_info/CheckpointLoaderSimple"):
            return FakeHttpResponse(_checkpoint_catalog(DEFAULT_COMFY_CHECKPOINT))
        submitted_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeHttpResponse({"prompt_id": "prompt-1"})

    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")

    result = parse_result(run_workflow.invoke(_known_good_tool_arguments()))

    assert result["success"] is True
    assert result["prompt_id"] == "prompt-1"
    assert result["is_async_job"] is True
    assert result["async_job_id"] == "prompt-1"
    assert result["async_job_status"] == AsyncJobStatus.SUBMITTED.value
    assert result["async_terminal"] is False
    assert submitted_payloads == [{
        "prompt": build_comfy_workflow(_known_good_params()),
        "client_id": "cortex_node_agent",
    }]


def test_run_comfy_workflow_handoffs_after_catalog_before_prompt_post(monkeypatch):
    events: list[str] = []

    class Coordinator:
        def prepare_for_comfy(self):
            events.append("ollama_verified_empty")

    def fake_urlopen(request, **_kwargs):
        if request.full_url.endswith("/object_info/CheckpointLoaderSimple"):
            events.append("checkpoint_catalog")
            return FakeHttpResponse(_checkpoint_catalog(DEFAULT_COMFY_CHECKPOINT))
        events.append("prompt_post")
        return FakeHttpResponse({"prompt_id": "prompt-1"})

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    run_workflow = _tool(
        get_comfy_tools(".", resource_coordinator=Coordinator()),
        "run_comfy_workflow",
    )

    result = parse_result(run_workflow.invoke(_known_good_tool_arguments()))

    assert result["success"] is True
    assert events == [
        "checkpoint_catalog",
        "ollama_verified_empty",
        "prompt_post",
    ]


def test_run_comfy_workflow_blocks_post_when_ollama_handoff_fails(monkeypatch):
    requested_urls: list[str] = []

    class Coordinator:
        def prepare_for_comfy(self):
            raise GpuResourceHandoffError("Ollama model is still loaded")

    def fake_urlopen(request, **_kwargs):
        requested_urls.append(request.full_url)
        return FakeHttpResponse(_checkpoint_catalog(DEFAULT_COMFY_CHECKPOINT))

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    run_workflow = _tool(
        get_comfy_tools(".", resource_coordinator=Coordinator()),
        "run_comfy_workflow",
    )

    result = parse_result(run_workflow.invoke(_known_good_tool_arguments()))

    assert result["success"] is False
    assert result["error_code"] == "GPU_RESOURCE_HANDOFF_FAILED"
    assert result["is_async_job"] is False
    assert result["error_details"]["submission_attempted"] is False
    assert requested_urls == [
        "http://127.0.0.1:8188/object_info/CheckpointLoaderSimple"
    ]


def test_run_comfy_workflow_preserves_preallocated_id_on_ambiguous_failure(
    monkeypatch,
):
    def fail_connection(request, **_kwargs):
        if request.full_url.endswith("/object_info/CheckpointLoaderSimple"):
            return FakeHttpResponse(_checkpoint_catalog(DEFAULT_COMFY_CHECKPOINT))
        raise comfy_ops.urllib.error.URLError("timed out")

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fail_connection)
    run_workflow = _tool(get_comfy_tools("."), "run_comfy_workflow")

    arguments = _known_good_tool_arguments()
    arguments.update({
        "prompt_id": "preallocated-prompt-1",
        "client_id": "run-1",
    })
    result = parse_result(run_workflow.invoke(arguments))

    assert result["success"] is False
    assert result["prompt_id"] == "preallocated-prompt-1"
    assert result["async_job_id"] == "preallocated-prompt-1"
    assert result["async_job_status"] == AsyncJobStatus.UNKNOWN.value
    assert result["data"] == {"submission_outcome": "ambiguous"}


def test_get_comfy_history_treats_missing_job_as_nonterminal_unknown(monkeypatch):
    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeHttpResponse({}),
    )
    history = _tool(get_comfy_tools("."), "get_comfy_history")

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is True
    assert result["async_job_status"] == AsyncJobStatus.UNKNOWN.value
    assert result["async_terminal"] is False
    assert result["data"] == {"provider_visibility": "absent"}


def test_get_comfy_history_reconciles_queue_before_reporting_absence(
    monkeypatch,
):
    def fake_urlopen(request, **_kwargs):
        if request.full_url.endswith("/queue"):
            return FakeHttpResponse({
                "queue_running": [],
                "queue_pending": [[1, "prompt-1", {}, {}, []]],
            })
        return FakeHttpResponse({})

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    history = _tool(get_comfy_tools("."), "get_comfy_history")

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is True
    assert result["async_job_status"] == AsyncJobStatus.RUNNING.value
    assert result["data"] == {"provider_visibility": "queue_pending"}


@pytest.mark.parametrize(
    (
        "provider_status",
        "provider_completed",
        "expected_status",
        "expected_success",
        "expected_terminal",
    ),
    [
        ("running", False, AsyncJobStatus.RUNNING, True, False),
        ("success", True, AsyncJobStatus.COMPLETED, True, True),
        ("error", True, AsyncJobStatus.FAILED, False, True),
        ("error", False, AsyncJobStatus.FAILED, False, True),
        ("cancelled", True, AsyncJobStatus.CANCELLED, False, True),
    ],
)
def test_get_comfy_history_normalizes_provider_lifecycle(
    monkeypatch,
    provider_status,
    provider_completed,
    expected_status,
    expected_success,
    expected_terminal,
):
    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeHttpResponse(
            {
                "prompt-1": {
                    "outputs": {"9": {"images": [{"filename": "result.png"}]}},
                    "status": {
                        "completed": provider_completed,
                        "status_str": provider_status,
                    },
                }
            }
        ),
    )
    history = _tool(get_comfy_tools("."), "get_comfy_history")

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is expected_success
    assert result["async_job_id"] == "prompt-1"
    assert result["async_job_status"] == expected_status.value
    assert result["async_terminal"] is expected_terminal


def test_terminal_comfy_history_frees_resources_before_returning_evidence(
    monkeypatch,
):
    calls: list[str] = []

    class Coordinator:
        def prepare_for_llm(self):
            calls.append("comfy_released")

    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeHttpResponse(
            {
                "prompt-1": {
                    "outputs": {"9": {"images": [{"filename": "result.png"}]}},
                    "status": {"completed": True, "status_str": "success"},
                }
            }
        ),
    )
    history = _tool(
        get_comfy_tools(".", resource_coordinator=Coordinator()),
        "get_comfy_history",
    )

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert calls == ["comfy_released"]
    assert result["async_job_status"] == AsyncJobStatus.COMPLETED.value
    assert result["async_terminal"] is True


def test_terminal_comfy_history_remains_nonterminal_until_handoff_succeeds(
    monkeypatch,
):
    class Coordinator:
        def prepare_for_llm(self):
            raise GpuResourceHandoffError("ComfyUI still owns model VRAM")

    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeHttpResponse(
            {
                "prompt-1": {
                    "outputs": {"9": {"images": [{"filename": "result.png"}]}},
                    "status": {"completed": True, "status_str": "success"},
                }
            }
        ),
    )
    history = _tool(
        get_comfy_tools(".", resource_coordinator=Coordinator()),
        "get_comfy_history",
    )

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is False
    assert result["error_code"] == "GPU_RESOURCE_HANDOFF_FAILED"
    assert result["async_job_status"] == AsyncJobStatus.UNKNOWN.value
    assert result["async_terminal"] is False
    assert result["data"] == {
        "resource_handoff": "pending",
        "provider_terminal_status": AsyncJobStatus.COMPLETED.value,
    }
