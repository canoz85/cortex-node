from pathlib import Path


def resolve_workspace(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_safe_path(workspace_root: Path, target: str) -> Path:
    candidate = (workspace_root / target).resolve()
    if candidate != workspace_root and workspace_root not in candidate.parents:
        raise ValueError(f"Path '{target}' is outside sandbox workspace")
    return candidate
