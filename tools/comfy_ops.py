import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from langchain_core.tools import tool

from core.error_codes import (
    COMFY_API_ERROR,
    COMFY_CHECKPOINT_CATALOG_UNAVAILABLE,
    COMFY_CHECKPOINT_NOT_INSTALLED,
    COMFY_CONNECTION_FAILED,
    COMFY_FILE_NOT_FOUND,
    COMFY_INVALID_PAYLOAD,
    COMFY_PROMPT_FAILED,
    GPU_RESOURCE_HANDOFF_FAILED,
)
from core.models import (
    ComfyDownloadImageRequest,
    ComfyHistoryRequest,
    ComfyHistoryResult,
    ComfyPromptResult,
    ComfyWorkflowParams,
    RunWorkflowRequest,
)
from core.protocol.enums import AsyncJobStatus
from core.runtime.gpu_resources import (
    GpuResourceCoordinator,
    GpuResourceHandoffError,
)
from tools.sandbox_paths import resolve_safe_path, resolve_workspace

# Timeout in seconds for standard HTTP API calls
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_COMFY_CHECKPOINT = (
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
)
_CHECKPOINT_CATALOG_ENDPOINT = "object_info/CheckpointLoaderSimple"

_COMFY_NODE_TYPES: dict[str, str] = {
    "3": "KSampler",
    "4": "CheckpointLoaderSimple",
    "5": "EmptyLatentImage",
    "6": "CLIPTextEncode",
    "7": "CLIPTextEncode",
    "8": "VAEDecode",
    "9": "SaveImage",
}
_COMFY_NODE_INPUTS: dict[str, set[str]] = {
    "3": {
        "cfg",
        "denoise",
        "latent_image",
        "model",
        "negative",
        "positive",
        "sampler_name",
        "scheduler",
        "seed",
        "steps",
    },
    "4": {"ckpt_name"},
    "5": {"batch_size", "height", "width"},
    "6": {"clip", "text"},
    "7": {"clip", "text"},
    "8": {"samples", "vae"},
    "9": {"filename_prefix", "images"},
}


class ComfyWorkflowValidationError(ValueError):
    """Raised when a ComfyUI workflow does not match the fixed template."""


class ComfyCheckpointCatalogError(ValueError):
    """Raised when ComfyUI returns a malformed checkpoint catalog."""


def parse_comfy_checkpoint_catalog(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Extract exact checkpoint filenames from CheckpointLoaderSimple metadata."""

    if not isinstance(payload, Mapping):
        raise ComfyCheckpointCatalogError("checkpoint catalog response must be an object")

    node_info = payload.get("CheckpointLoaderSimple")
    if not isinstance(node_info, Mapping):
        raise ComfyCheckpointCatalogError(
            "checkpoint catalog is missing CheckpointLoaderSimple metadata"
        )

    input_info = node_info.get("input")
    required_inputs = (
        input_info.get("required")
        if isinstance(input_info, Mapping)
        else None
    )
    checkpoint_spec = (
        required_inputs.get("ckpt_name")
        if isinstance(required_inputs, Mapping)
        else None
    )
    if (
        not isinstance(checkpoint_spec, (list, tuple))
        or not checkpoint_spec
        or not isinstance(checkpoint_spec[0], (list, tuple))
    ):
        raise ComfyCheckpointCatalogError(
            "checkpoint catalog has an invalid ckpt_name choice specification"
        )

    checkpoint_names: list[str] = []
    for raw_name in checkpoint_spec[0]:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ComfyCheckpointCatalogError(
                "checkpoint catalog contains an invalid checkpoint filename"
            )
        if raw_name not in checkpoint_names:
            checkpoint_names.append(raw_name)

    return tuple(checkpoint_names)


def build_comfy_workflow(
    params: ComfyWorkflowParams | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the fixed image-generation graph from typed parameters.

    The graph shape, node IDs, class types, fixed sampler settings, and all
    connections are application-owned.  Only values represented by
    ``ComfyWorkflowParams`` can enter the resulting workflow.
    """

    if not isinstance(params, ComfyWorkflowParams):
        params = ComfyWorkflowParams.model_validate(params)

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": params.cfg,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": params.seed,
                "steps": params.steps,
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": params.checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": params.height,
                "width": params.width,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": params.positive_prompt,
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": params.negative_prompt,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": params.filename_prefix,
                "images": ["8", 0],
            },
        },
    }


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComfyWorkflowValidationError(f"{path} must be an object")
    return value


def _require_input(
    inputs: Mapping[str, Any],
    node_id: str,
    input_name: str,
) -> Any:
    if input_name not in inputs:
        raise ComfyWorkflowValidationError(
            f"Node {node_id} is missing required input '{input_name}'"
        )
    return inputs[input_name]


def _require_positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ComfyWorkflowValidationError(f"{path} must be a positive integer")
    return value


def _require_positive_number(value: object, path: str) -> float | int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ComfyWorkflowValidationError(f"{path} must be positive")
    return value


def _require_non_blank_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComfyWorkflowValidationError(f"{path} must be non-empty")
    return value


def _require_exact_reference(
    value: object,
    expected_node_id: str,
    expected_output_index: int,
    path: str,
) -> None:
    if value != [expected_node_id, expected_output_index]:
        raise ComfyWorkflowValidationError(
            f"{path} must reference [{expected_node_id}, {expected_output_index}]"
        )


def _validate_graph_references(
    value: object,
    path: str,
    node_ids: set[str],
) -> None:
    """Validate every list-shaped graph reference found in input values."""

    if not isinstance(value, (list, tuple)):
        return

    if len(value) == 2:
        reference_node, output_index = value
        if not isinstance(reference_node, str) or not reference_node:
            raise ComfyWorkflowValidationError(
                f"{path} graph reference node ID must be a string"
            )

        if reference_node not in node_ids:
            raise ComfyWorkflowValidationError(
                f"{path} references missing node {reference_node!r}"
            )
        if type(output_index) is not int or output_index < 0:
            raise ComfyWorkflowValidationError(
                f"{path} has an invalid output index {output_index!r}"
            )
        return

    for index, item in enumerate(value):
        _validate_graph_references(item, f"{path}[{index}]", node_ids)


def validate_comfy_workflow(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the fixed ComfyUI template before it can be submitted.

    This validator intentionally fails closed.  It does not add missing nodes,
    repair references, or infer connectivity.
    """

    workflow = _require_mapping(workflow, "workflow")
    if any(not isinstance(node_id, str) for node_id in workflow):
        raise ComfyWorkflowValidationError("workflow node IDs must be strings")
    node_ids = set(workflow)
    required_node_ids = set(_COMFY_NODE_TYPES)

    missing = required_node_ids - node_ids
    if missing:
        raise ComfyWorkflowValidationError(
            f"workflow is missing required node IDs: {', '.join(sorted(missing))}"
        )
    unexpected = node_ids - required_node_ids
    if unexpected:
        raise ComfyWorkflowValidationError(
            f"workflow contains unexpected node IDs: {', '.join(sorted(unexpected))}"
        )

    nodes: dict[str, Mapping[str, Any]] = {}
    for raw_node_id, expected_class_type in _COMFY_NODE_TYPES.items():
        node = workflow.get(raw_node_id)
        node_mapping = _require_mapping(node, f"workflow[{raw_node_id!r}]")
        unexpected_node_fields = set(node_mapping) - {"class_type", "inputs"}
        if unexpected_node_fields:
            raise ComfyWorkflowValidationError(
                f"Node {raw_node_id} contains unexpected fields: "
                f"{', '.join(sorted(map(str, unexpected_node_fields)))}"
            )
        actual_class_type = node_mapping.get("class_type")
        if actual_class_type != expected_class_type:
            raise ComfyWorkflowValidationError(
                f"Node {raw_node_id} must have class_type '{expected_class_type}'"
            )
        nodes[raw_node_id] = _require_mapping(
            node_mapping.get("inputs"),
            f"workflow[{raw_node_id!r}].inputs",
        )

        expected_inputs = _COMFY_NODE_INPUTS[raw_node_id]
        missing_inputs = expected_inputs - set(nodes[raw_node_id])
        if missing_inputs:
            raise ComfyWorkflowValidationError(
                f"Node {raw_node_id} is missing required inputs: "
                f"{', '.join(sorted(missing_inputs))}"
            )
        unexpected_inputs = set(nodes[raw_node_id]) - expected_inputs
        if unexpected_inputs:
            raise ComfyWorkflowValidationError(
                f"Node {raw_node_id} contains unexpected inputs: "
                f"{', '.join(sorted(map(str, unexpected_inputs)))}"
            )

    for node_id, inputs in nodes.items():
        for input_name, value in inputs.items():
            _validate_graph_references(
                value,
                f"workflow[{node_id!r}].inputs[{input_name!r}]",
                required_node_ids,
            )

    k_sampler = nodes["3"]
    _require_exact_reference(k_sampler["model"], "4", 0, "Node 3 input 'model'")
    _require_exact_reference(k_sampler["positive"], "6", 0, "Node 3 input 'positive'")
    _require_exact_reference(k_sampler["negative"], "7", 0, "Node 3 input 'negative'")
    _require_exact_reference(
        k_sampler["latent_image"], "5", 0, "Node 3 input 'latent_image'"
    )
    if k_sampler["sampler_name"] != "euler":
        raise ComfyWorkflowValidationError("Node 3 input 'sampler_name' must be 'euler'")
    if k_sampler["scheduler"] != "normal":
        raise ComfyWorkflowValidationError("Node 3 input 'scheduler' must be 'normal'")
    if type(k_sampler["denoise"]) is not int or k_sampler["denoise"] != 1:
        raise ComfyWorkflowValidationError("Node 3 input 'denoise' must be 1")
    if type(k_sampler["seed"]) is not int:
        raise ComfyWorkflowValidationError("Node 3 input 'seed' must be an integer")
    _require_positive_int(k_sampler["steps"], "Node 3 input 'steps'")
    _require_positive_number(k_sampler["cfg"], "Node 3 input 'cfg'")

    checkpoint = _require_input(nodes["4"], "4", "ckpt_name")
    _require_non_blank_text(checkpoint, "Node 4 input 'ckpt_name'")

    latent = nodes["5"]
    batch_size = _require_input(latent, "5", "batch_size")
    if type(batch_size) is not int or batch_size != 1:
        raise ComfyWorkflowValidationError("Node 5 input 'batch_size' must be 1")
    _require_positive_int(_require_input(latent, "5", "width"), "Node 5 input 'width'")
    _require_positive_int(_require_input(latent, "5", "height"), "Node 5 input 'height'")

    positive = nodes["6"]
    negative = nodes["7"]
    _require_exact_reference(
        _require_input(positive, "6", "clip"), "4", 1, "Node 6 input 'clip'"
    )
    _require_exact_reference(
        _require_input(negative, "7", "clip"), "4", 1, "Node 7 input 'clip'"
    )
    _require_non_blank_text(
        _require_input(positive, "6", "text"), "Node 6 input 'text'"
    )
    negative_text = _require_input(negative, "7", "text")
    if not isinstance(negative_text, str):
        raise ComfyWorkflowValidationError("Node 7 input 'text' must be a string")

    _require_exact_reference(
        _require_input(nodes["8"], "8", "samples"),
        "3",
        0,
        "Node 8 input 'samples'",
    )
    _require_exact_reference(
        _require_input(nodes["8"], "8", "vae"),
        "4",
        2,
        "Node 8 input 'vae'",
    )
    _require_exact_reference(
        _require_input(nodes["9"], "9", "images"),
        "8",
        0,
        "Node 9 input 'images'",
    )
    _require_non_blank_text(
        _require_input(nodes["9"], "9", "filename_prefix"),
        "Node 9 input 'filename_prefix'",
    )

    return workflow


class _ComfyHttpError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _history_async_status(status: object) -> AsyncJobStatus:
    """Map ComfyUI history status evidence to the protocol async-job vocabulary."""
    if not isinstance(status, dict):
        return AsyncJobStatus.RUNNING

    status_text = str(status.get("status_str", "")).strip().lower()
    if "cancel" in status_text:
        return AsyncJobStatus.CANCELLED
    if any(token in status_text for token in ("error", "fail")):
        return AsyncJobStatus.FAILED
    if status.get("completed") is not True:
        return AsyncJobStatus.RUNNING
    return AsyncJobStatus.COMPLETED


def get_comfy_tools(
    workspace_root: str,
    comfy_base_url: str = "http://127.0.0.1:8188",
    resource_coordinator: GpuResourceCoordinator | None = None,
) -> list[Any]:
    """Factory function returning ComfyUI tools bound to a specific workspace root."""
    base_url = comfy_base_url.rstrip("/")
    workspace = resolve_workspace(workspace_root)

    def _http_request(
        endpoint: str, data: Optional[Dict[str, Any]] = None, method: str = "GET"
    ) -> Dict[str, Any]:
        """Helper to send HTTP requests to the ComfyUI API endpoint with clean error handling."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}

        encoded_data = None
        if data is not None:
            encoded_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(
            url, data=encoded_data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_HTTP_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode("utf-8", errors="ignore")
            raise _ComfyHttpError(
                http_err.code,
                f"ComfyUI HTTP {http_err.code}: {error_body or http_err.reason}",
            )
        except urllib.error.URLError as url_err:
            raise ConnectionError(f"Could not connect to ComfyUI server at {base_url}: {url_err.reason}")

    def _queue_location(queue_data: object, prompt_id: str) -> str | None:
        if not isinstance(queue_data, dict):
            return None

        for queue_name in ("queue_running", "queue_pending"):
            items = queue_data.get(queue_name)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) > 1:
                    candidate = item[1]
                elif isinstance(item, dict):
                    candidate = item.get("prompt_id") or item.get("job_id")
                else:
                    continue
                if str(candidate) == prompt_id:
                    return queue_name
        return None

    @tool("run_comfy_workflow", args_schema=RunWorkflowRequest)
    def run_comfy_workflow(
        positive_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        negative_prompt: str = "",
        filename_prefix: str = "CortexNode",
        client_id: str = "cortex_node_agent",
        prompt_id: str | None = None,
    ) -> str:
        """Queue an image-generation workflow built from the fixed template.

        The LLM-facing contract contains only validated generation parameters.
        Node IDs, class types, graph connectivity, and checkpoint selection are
        application-owned.  This tool uses the fixed checkpoint
        ``Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`` and refuses to
        submit if that exact filename is not installed in ComfyUI.
        """
        try:
            params = ComfyWorkflowParams(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                cfg=cfg,
                width=width,
                height=height,
                checkpoint=DEFAULT_COMFY_CHECKPOINT,
                filename_prefix=filename_prefix,
            )
            workflow = build_comfy_workflow(params)
            validate_comfy_workflow(workflow)
        except (TypeError, ValueError) as exc:
            return ComfyPromptResult(
                success=False,
                message=f"Invalid ComfyUI workflow parameters: {exc}",
                error_code=COMFY_INVALID_PAYLOAD,
                error_details={"exception_type": type(exc).__name__},
            ).to_tool_output()

        try:
            checkpoint_catalog = parse_comfy_checkpoint_catalog(
                _http_request(_CHECKPOINT_CATALOG_ENDPOINT)
            )
        except ConnectionError as conn_err:
            return ComfyPromptResult(
                success=False,
                message=(
                    "Could not verify the required ComfyUI checkpoint before "
                    f"submission: {conn_err}"
                ),
                error_code=COMFY_CHECKPOINT_CATALOG_UNAVAILABLE,
                error_details={
                    "exception_type": "ConnectionError",
                    "catalog_endpoint": _CHECKPOINT_CATALOG_ENDPOINT,
                },
            ).to_tool_output()
        except _ComfyHttpError as http_err:
            return ComfyPromptResult(
                success=False,
                message=(
                    "Could not read the ComfyUI checkpoint catalog before "
                    f"submission: {http_err}"
                ),
                error_code=COMFY_CHECKPOINT_CATALOG_UNAVAILABLE,
                error_details={
                    "exception_type": type(http_err).__name__,
                    "status_code": http_err.status_code,
                    "catalog_endpoint": _CHECKPOINT_CATALOG_ENDPOINT,
                },
            ).to_tool_output()
        except ComfyCheckpointCatalogError as catalog_err:
            return ComfyPromptResult(
                success=False,
                message=f"Invalid ComfyUI checkpoint catalog: {catalog_err}",
                error_code=COMFY_CHECKPOINT_CATALOG_UNAVAILABLE,
                error_details={
                    "exception_type": type(catalog_err).__name__,
                    "catalog_endpoint": _CHECKPOINT_CATALOG_ENDPOINT,
                },
            ).to_tool_output()

        if DEFAULT_COMFY_CHECKPOINT not in checkpoint_catalog:
            return ComfyPromptResult(
                success=False,
                message=(
                    "Required ComfyUI checkpoint is not installed: "
                    f"{DEFAULT_COMFY_CHECKPOINT}. Submission was not attempted."
                ),
                error_code=COMFY_CHECKPOINT_NOT_INSTALLED,
                error_details={
                    "required_checkpoint": DEFAULT_COMFY_CHECKPOINT,
                    "installed_checkpoints": list(checkpoint_catalog),
                },
                data={
                    "required_checkpoint": DEFAULT_COMFY_CHECKPOINT,
                    "installed_checkpoints": list(checkpoint_catalog),
                },
            ).to_tool_output()

        if resource_coordinator is not None:
            try:
                resource_coordinator.prepare_for_comfy()
            except GpuResourceHandoffError as handoff_err:
                return ComfyPromptResult(
                    success=False,
                    message=(
                        "ComfyUI submission was blocked because Ollama GPU "
                        f"handoff could not be verified: {handoff_err}"
                    ),
                    error_code=GPU_RESOURCE_HANDOFF_FAILED,
                    error_details={
                        "exception_type": type(handoff_err).__name__,
                        "handoff_direction": "ollama_to_comfy",
                        "submission_attempted": False,
                    },
                ).to_tool_output()

        try:
            payload: Dict[str, Any] = {"prompt": workflow}
            if client_id:
                payload["client_id"] = client_id
            if prompt_id:
                payload["prompt_id"] = prompt_id

            # Send prompt request to ComfyUI
            response_data = _http_request("prompt", data=payload, method="POST")
            prompt_id = response_data.get("prompt_id")

            if not prompt_id:
                requested_prompt_id = (
                    str(payload["prompt_id"])
                    if payload.get("prompt_id")
                    else None
                )
                return ComfyPromptResult(
                    success=False,
                    message="ComfyUI response did not contain a valid prompt_id",
                    prompt_id=requested_prompt_id,
                    error_code=COMFY_PROMPT_FAILED,
                    error_details={"response": response_data},
                    data=(
                        {"submission_outcome": "ambiguous"}
                        if requested_prompt_id
                        else None
                    ),
                    is_async_job=bool(requested_prompt_id),
                    async_job_id=requested_prompt_id,
                    async_job_status=(
                        AsyncJobStatus.UNKNOWN if requested_prompt_id else None
                    ),
                    async_terminal=False,
                    async_observed_at_utc=(
                        datetime.now(timezone.utc) if requested_prompt_id else None
                    ),
                ).to_tool_output()

            return ComfyPromptResult(
                success=True,
                message=f"Successfully queued validated ComfyUI workflow with prompt_id: {prompt_id}",
                prompt_id=prompt_id,
                node_errors=response_data.get("node_errors"),
                is_async_job=True,
                async_job_id=str(prompt_id),
                async_job_status=AsyncJobStatus.SUBMITTED,
                async_terminal=False,
                async_observed_at_utc=datetime.now(timezone.utc),
            ).to_tool_output()

        except _ComfyHttpError as http_err:
            return ComfyPromptResult(
                success=False,
                message=str(http_err),
                prompt_id=prompt_id,
                error_code=(
                    COMFY_INVALID_PAYLOAD
                    if http_err.status_code == 400
                    else COMFY_API_ERROR
                ),
                error_details={
                    "exception_type": type(http_err).__name__,
                    "status_code": http_err.status_code,
                },
                data=(
                    {"submission_outcome": "ambiguous"}
                    if prompt_id and http_err.status_code != 400
                    else None
                ),
                is_async_job=bool(prompt_id and http_err.status_code != 400),
                async_job_id=(prompt_id if http_err.status_code != 400 else None),
                async_job_status=(
                    AsyncJobStatus.UNKNOWN
                    if prompt_id and http_err.status_code != 400
                    else None
                ),
                async_terminal=False,
                async_observed_at_utc=(
                    datetime.now(timezone.utc)
                    if prompt_id and http_err.status_code != 400
                    else None
                ),
            ).to_tool_output()
        except ConnectionError as conn_err:
            return ComfyPromptResult(
                success=False,
                message=str(conn_err),
                prompt_id=prompt_id,
                error_code=COMFY_CONNECTION_FAILED,
                error_details={"exception_type": "ConnectionError"},
                data=(
                    {"submission_outcome": "ambiguous"}
                    if prompt_id
                    else None
                ),
                is_async_job=bool(prompt_id),
                async_job_id=prompt_id,
                async_job_status=(AsyncJobStatus.UNKNOWN if prompt_id else None),
                async_terminal=False,
                async_observed_at_utc=(
                    datetime.now(timezone.utc) if prompt_id else None
                ),
            ).to_tool_output()
        except Exception as exc:
            return ComfyPromptResult(
                success=False,
                message=f"Error submitting workflow to ComfyUI: {exc}",
                prompt_id=prompt_id,
                error_code=COMFY_API_ERROR,
                error_details={"exception_type": type(exc).__name__},
                data=(
                    {"submission_outcome": "ambiguous"}
                    if prompt_id
                    else None
                ),
                is_async_job=bool(prompt_id),
                async_job_id=prompt_id,
                async_job_status=(AsyncJobStatus.UNKNOWN if prompt_id else None),
                async_terminal=False,
                async_observed_at_utc=(
                    datetime.now(timezone.utc) if prompt_id else None
                ),
            ).to_tool_output()

    @tool("download_comfy_output_image", args_schema=ComfyDownloadImageRequest)
    def download_comfy_output_image(
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
        save_path: str = "generated_image.png",
    ) -> str:
        """Download an image output from the ComfyUI server and save it into the workspace directory.
        
        Capabilities: comfy_download, download_file, comfyui
        """
        try:
            # Resolve destination path safely inside the workspace boundary
            destination = resolve_safe_path(workspace, save_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

            query_params = urllib.parse.urlencode(
                {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": folder_type,
                }
            )
            url = f"{base_url}/view?{query_params}"

            # Fetch image bytes from ComfyUI server and write to workspace
            with urllib.request.urlopen(
                url, timeout=DEFAULT_HTTP_TIMEOUT
            ) as resp:
                image_data = resp.read()
                destination.write_bytes(image_data)

            return ComfyPromptResult(
                success=True,
                message=f"Saved ComfyUI image output to workspace path: {save_path}",
            ).to_tool_output()

        except Exception as exc:
            return ComfyPromptResult(
                success=False,
                message=f"Failed to download image to workspace: {exc}",
                error_code=COMFY_API_ERROR,
                error_details={"exception_type": type(exc).__name__},
            ).to_tool_output()

    @tool("get_comfy_history", args_schema=ComfyHistoryRequest)
    def get_comfy_history(prompt_id: str) -> str:
        """Fetch execution history and output details for a specific ComfyUI prompt_id."""
        try:
            safe_prompt_id = prompt_id.strip() if isinstance(prompt_id, str) else str(prompt_id)
            endpoint = f"history/{urllib.parse.quote(safe_prompt_id)}"

            history_data = _http_request(endpoint, method="GET")

            if safe_prompt_id not in history_data:
                queue_data = _http_request("queue", method="GET")
                queue_location = _queue_location(queue_data, safe_prompt_id)
                if queue_location is not None:
                    return ComfyHistoryResult(
                        success=True,
                        message=(
                            "ComfyUI job is present in "
                            f"{queue_location} for prompt_id: {safe_prompt_id}"
                        ),
                        prompt_id=safe_prompt_id,
                        completed=False,
                        data={"provider_visibility": queue_location},
                        status_details={"queue_location": queue_location},
                        is_async_job=True,
                        async_job_id=safe_prompt_id,
                        async_job_status=AsyncJobStatus.RUNNING,
                        async_terminal=False,
                        async_observed_at_utc=datetime.now(timezone.utc),
                    ).to_tool_output()

                # Close the queue-to-history transition race before declaring absence.
                history_data = _http_request(endpoint, method="GET")

            if safe_prompt_id not in history_data:
                return ComfyHistoryResult(
                    success=True,
                    message=f"ComfyUI history is not visible yet for prompt_id: {safe_prompt_id}",
                    prompt_id=safe_prompt_id,
                    completed=False,
                    data={"provider_visibility": "absent"},
                    status_details={"provider_visibility": "absent"},
                    is_async_job=True,
                    async_job_id=safe_prompt_id,
                    async_job_status=AsyncJobStatus.UNKNOWN,
                    async_terminal=False,
                    async_observed_at_utc=datetime.now(timezone.utc),
                ).to_tool_output()

            item_data = history_data[safe_prompt_id]
            outputs = item_data.get("outputs", {})
            status = item_data.get("status", {})

            extracted_filenames = []
            if isinstance(outputs, dict):
                for node_output in outputs.values():
                    if isinstance(node_output, dict) and "images" in node_output:
                        for img in node_output.get("images", []):
                            if isinstance(img, dict) and "filename" in img:
                                extracted_filenames.append(img["filename"])

            primary_filename = extracted_filenames[0] if extracted_filenames else None
            async_status = _history_async_status(status)

            if (
                resource_coordinator is not None
                and async_status
                in {
                    AsyncJobStatus.COMPLETED,
                    AsyncJobStatus.FAILED,
                    AsyncJobStatus.CANCELLED,
                }
            ):
                try:
                    resource_coordinator.prepare_for_llm()
                except GpuResourceHandoffError as handoff_err:
                    return ComfyHistoryResult(
                        success=False,
                        message=(
                            "ComfyUI reported provider-terminal status "
                            f"{async_status.value}, but GPU handoff to Ollama "
                            f"is not ready: {handoff_err}"
                        ),
                        prompt_id=safe_prompt_id,
                        completed=False,
                        outputs=outputs,
                        status_details=status,
                        error_code=GPU_RESOURCE_HANDOFF_FAILED,
                        error_details={
                            "exception_type": type(handoff_err).__name__,
                            "handoff_direction": "comfy_to_ollama",
                            "provider_terminal_status": async_status.value,
                        },
                        data={
                            "resource_handoff": "pending",
                            "provider_terminal_status": async_status.value,
                        },
                        is_async_job=True,
                        async_job_id=safe_prompt_id,
                        async_job_status=AsyncJobStatus.UNKNOWN,
                        async_terminal=False,
                        async_observed_at_utc=datetime.now(timezone.utc),
                    ).to_tool_output()

            if async_status == AsyncJobStatus.COMPLETED and extracted_filenames:
                files_str = ", ".join(extracted_filenames)
                msg = f"Retrieved execution history for prompt_id: {safe_prompt_id}. Output images: [{files_str}] (Primary: {primary_filename})"
            elif async_status == AsyncJobStatus.COMPLETED:
                msg = f"ComfyUI workflow completed for prompt_id: {safe_prompt_id}. No output images were reported."
            elif async_status == AsyncJobStatus.FAILED:
                msg = f"ComfyUI workflow failed for prompt_id: {safe_prompt_id}."
            elif async_status == AsyncJobStatus.CANCELLED:
                msg = f"ComfyUI workflow was cancelled for prompt_id: {safe_prompt_id}."
            else:
                msg = f"ComfyUI workflow is still running for prompt_id: {safe_prompt_id}."

            return ComfyHistoryResult(
                success=async_status not in {AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED},
                message=msg,
                prompt_id=safe_prompt_id,
                completed=async_status == AsyncJobStatus.COMPLETED,
                filenames=extracted_filenames,
                primary_filename=primary_filename,
                outputs=outputs,
                status_details=status,
                is_async_job=True,
                async_job_id=safe_prompt_id,
                async_job_status=async_status,
                async_terminal=async_status in {
                    AsyncJobStatus.COMPLETED,
                    AsyncJobStatus.FAILED,
                    AsyncJobStatus.CANCELLED,
                },
                async_observed_at_utc=datetime.now(timezone.utc),
            ).to_tool_output()

        except ConnectionError as conn_err:
            safe_id = prompt_id if isinstance(prompt_id, str) else ""
            return ComfyHistoryResult(
                success=False,
                message=str(conn_err),
                prompt_id=safe_id,
                error_code=COMFY_CONNECTION_FAILED,
                error_details={"exception_type": "ConnectionError"},
                is_async_job=bool(safe_id),
                async_job_id=safe_id or None,
                async_job_status=(AsyncJobStatus.UNKNOWN if safe_id else None),
                async_terminal=False,
                async_observed_at_utc=(datetime.now(timezone.utc) if safe_id else None),
            ).to_tool_output()
        except Exception as exc:
            safe_id = prompt_id if isinstance(prompt_id, str) else ""
            return ComfyHistoryResult(
                success=False,
                message=f"Error fetching ComfyUI history: {exc}",
                prompt_id=safe_id,
                error_code=COMFY_API_ERROR,
                error_details={
                    "prompt_id": safe_id,
                    "exception_type": type(exc).__name__,
                },
                is_async_job=bool(safe_id),
                async_job_id=safe_id or None,
                async_job_status=(AsyncJobStatus.UNKNOWN if safe_id else None),
                async_terminal=False,
                async_observed_at_utc=(datetime.now(timezone.utc) if safe_id else None),
            ).to_tool_output()

    return [run_comfy_workflow, download_comfy_output_image, get_comfy_history]
