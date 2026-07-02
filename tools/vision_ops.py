import base64

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from core.error_codes import FILE_PATH_NOT_FOUND, VISION_ANALYSIS_FAILED, VISION_MODEL_UNAVAILABLE
from core.models import ToolResult
from tools.sandbox_paths import resolve_safe_path, resolve_workspace


def get_vision_tools(workspace_dir: str):
    workspace_root = resolve_workspace(workspace_dir)

    @tool
    def describe_image(image_path: str) -> str:
        """Describe an image in the workspace using the local Ollama llava multimodal model."""
        try:
            safe_path = resolve_safe_path(workspace_root, image_path)
            if not safe_path.exists() or not safe_path.is_file():
                return ToolResult(
                    success=False,
                    message=f"Image not found: {image_path}",
                    error_code=FILE_PATH_NOT_FOUND,
                    error_details={"path": image_path},
                ).to_tool_output()

            suffix = safe_path.suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(suffix, "image/jpeg")

            with safe_path.open("rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("ascii")

            try:
                llava_model = ChatOllama(model="llava", temperature=0)
                result = llava_model.invoke(
                    [
                        {"type": "text", "text": "Describe this image concisely."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                    ]
                )
                description = str(getattr(result, "content", "") or "").strip() or "No description generated."
                return ToolResult(
                    success=True,
                    message=description,
                    data={"path": str(safe_path), "description": description},
                ).to_tool_output()
            except Exception as exc:
                return ToolResult(
                    success=False,
                    message="Vision model (llava) is unavailable or failed to process the image.",
                    error_code=VISION_MODEL_UNAVAILABLE,
                    error_details={
                        "path": str(safe_path),
                        "exception_type": type(exc).__name__,
                    },
                ).to_tool_output()
        except ValueError as exc:
            return ToolResult(
                success=False,
                message=str(exc),
                error_code=FILE_PATH_NOT_FOUND,
                error_details={"path": image_path},
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Image analysis failed: {exc}",
                error_code=VISION_ANALYSIS_FAILED,
                error_details={
                    "path": image_path,
                    "exception_type": type(exc).__name__,
                },
            ).to_tool_output()

    return [describe_image]