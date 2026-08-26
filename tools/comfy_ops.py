import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union
from langchain_core.tools import tool

from core.error_codes import (
    COMFY_API_ERROR,
    COMFY_CONNECTION_FAILED,
    COMFY_FILE_NOT_FOUND,
    COMFY_INVALID_PAYLOAD,
    COMFY_PROMPT_FAILED,
)
from core.models import (
    ComfyDownloadImageRequest,
    ComfyHistoryRequest,
    ComfyHistoryResult,
    ComfyPromptRequest,
    ComfyPromptResult,
    RunWorkflowRequest,
)
from tools.sandbox_paths import resolve_safe_path, resolve_workspace

# Timeout in seconds for standard HTTP API calls
DEFAULT_HTTP_TIMEOUT = 30.0


def get_comfy_tools(
    workspace_root: str,
    comfy_base_url: str = "http://127.0.0.1:8188",
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
            raise RuntimeError(f"ComfyUI HTTP {http_err.code}: {error_body or http_err.reason}")
        except urllib.error.URLError as url_err:
            raise ConnectionError(f"Could not connect to ComfyUI server at {base_url}: {url_err.reason}")

    def _normalize_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all node IDs and output references are strings as required by ComfyUI API."""
        normalized = {}
        for node_id, node_data in workflow.items():
            # Convert Node ID keys to strings if they are not already
            str_node_id = str(node_id)
            if isinstance(node_data, dict):
                new_node_data = node_data.copy()
                if "inputs" in new_node_data and isinstance(new_node_data["inputs"], dict):
                    new_inputs = {}
                    for k, v in new_node_data["inputs"].items():
                        # Convert connection reference lists like [5, 0] to ["5", 0] format
                        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (int, str)):
                            new_inputs[k] = [str(v[0]), v[1]]
                        else:
                            new_inputs[k] = v

                    if new_node_data.get("class_type") == "KSampler":
                        if "positive" not in new_inputs and "6" in workflow:
                            new_inputs["positive"] = ["6", 0]
                        if "negative" not in new_inputs and "7" in workflow:
                            new_inputs["negative"] = ["7", 0]
                            
                    new_node_data["inputs"] = new_inputs
                normalized[str_node_id] = new_node_data
            else:
                normalized[str_node_id] = node_data
        return normalized

    @tool("run_comfy_workflow", args_schema=RunWorkflowRequest)
    def run_comfy_workflow(
        workflow_json: Union[str, Dict[str, Any]], client_id: str = "cortex_node_agent"
    ) -> str:
        """Queues a prompt workflow to the local ComfyUI instance for image generation.
        
        The `workflow_json` can be provided as a serialized JSON string, a raw Python dict, or a relative file path in the workspace.
        
        Example JSON payload structure:
        {
          "3": {"inputs": {"seed": 42, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
          "4": {"inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"}, "class_type": "CheckpointLoaderSimple"},
          "5": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptyLatentImage"},
          "6": {"inputs": {"text": "a cute cat, high quality", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
          "7": {"inputs": {"text": "bad quality, blurry", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
          "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
          "9": {"inputs": {"filename_prefix": "CortexNode", "images": ["8", 0]}, "class_type": "SaveImage"}
        }
        """
        try:
            parsed_prompt: Dict[str, Any] = {}
            source_info = "raw payload"

            # Handle dictionary directly provided by Brain LLM
            if isinstance(workflow_json, dict):
                parsed_prompt = workflow_json
                source_info = "raw JSON object"
            elif isinstance(workflow_json, str):
                trimmed_workflow = workflow_json.strip()
                if trimmed_workflow.startswith("{"):
                    try:
                        parsed_prompt = json.loads(trimmed_workflow)
                        source_info = "raw JSON string"
                    except json.JSONDecodeError as err:
                        return ComfyPromptResult(
                            success=False,
                            message=f"Invalid JSON string in workflow payload: {err}",
                            error_code=COMFY_INVALID_PAYLOAD,
                            error_details={"raw_prompt": trimmed_workflow[:200]},
                        ).to_tool_output()
                else:
                    # Resolve safe path inside workspace boundary if string is a file path
                    try:
                        target_file = resolve_safe_path(workspace, trimmed_workflow)
                        if not target_file.exists() or not target_file.is_file():
                            return ComfyPromptResult(
                                success=False,
                                message=f"Error: Workflow file does not exist in workspace: {trimmed_workflow}",
                                error_code=COMFY_FILE_NOT_FOUND,
                                error_details={"path": trimmed_workflow},
                            ).to_tool_output()

                        raw_content = target_file.read_text(encoding="utf-8")
                        parsed_prompt = json.loads(raw_content)
                        source_info = f"file '{trimmed_workflow}'"
                    except Exception as file_err:
                        return ComfyPromptResult(
                            success=False,
                            message=f"Error loading workflow file from workspace: {file_err}",
                            error_code=COMFY_INVALID_PAYLOAD,
                            error_details={"path": trimmed_workflow},
                        ).to_tool_output()
            else:
                return ComfyPromptResult(
                    success=False,
                    message="Invalid workflow_json format. Expected dict, JSON string, or file path string.",
                    error_code=COMFY_INVALID_PAYLOAD,
                ).to_tool_output()

            parsed_prompt = _normalize_workflow(parsed_prompt)

            payload: Dict[str, Any] = {"prompt": parsed_prompt}
            if client_id:
                payload["client_id"] = client_id

            # Send prompt request to ComfyUI
            response_data = _http_request("prompt", data=payload, method="POST")
            prompt_id = response_data.get("prompt_id")

            if not prompt_id:
                return ComfyPromptResult(
                    success=False,
                    message="ComfyUI response did not contain a valid prompt_id",
                    error_code=COMFY_PROMPT_FAILED,
                    error_details={"response": response_data},
                ).to_tool_output()

            return ComfyPromptResult(
                success=True,
                message=f"Successfully queued ComfyUI workflow from {source_info} with prompt_id: {prompt_id}",
                prompt_id=prompt_id,
                node_errors=response_data.get("node_errors"),
            ).to_tool_output()

        except ConnectionError as conn_err:
            return ComfyPromptResult(
                success=False,
                message=str(conn_err),
                error_code=COMFY_CONNECTION_FAILED,
                error_details={"exception_type": "ConnectionError"},
            ).to_tool_output()
        except Exception as exc:
            return ComfyPromptResult(
                success=False,
                message=f"Error submitting workflow to ComfyUI: {exc}",
                error_code=COMFY_API_ERROR,
                error_details={"exception_type": type(exc).__name__},
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
                return ComfyHistoryResult(
                    success=False,
                    message=f"No execution history found for prompt_id: {safe_prompt_id}",
                    prompt_id=safe_prompt_id,
                    completed=False,
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

            if extracted_filenames:
                files_str = ", ".join(extracted_filenames)
                msg = f"Retrieved execution history for prompt_id: {safe_prompt_id}. Output images: [{files_str}] (Primary: {primary_filename})"
            else:
                msg = f"Retrieved execution history for prompt_id: {safe_prompt_id}. No output images generated yet."

            return ComfyHistoryResult(
                success=True,
                message=msg,
                prompt_id=safe_prompt_id,
                completed=status.get("completed", True),
                filenames=extracted_filenames,
                primary_filename=primary_filename,
                outputs=outputs,
                status_details=status,
            ).to_tool_output()

        except ConnectionError as conn_err:
            safe_id = prompt_id if isinstance(prompt_id, str) else ""
            return ComfyHistoryResult(
                success=False,
                message=str(conn_err),
                prompt_id=safe_id,
                error_code=COMFY_CONNECTION_FAILED,
                error_details={"exception_type": "ConnectionError"},
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
            ).to_tool_output()

    return [run_comfy_workflow, download_comfy_output_image, get_comfy_history]