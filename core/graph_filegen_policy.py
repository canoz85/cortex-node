
def last_tool_missing_required_args(tool_output: dict[str, object] | str) -> bool:
    if isinstance(tool_output, str):
        stderr = tool_output
    elif isinstance(tool_output, dict):
        data = tool_output.get("data")
        if not isinstance(data, dict):
            return False
        stderr = str(data.get("stderr", "") or "")
    else:
        return False
    return "the following arguments are required" in stderr.lower()


def last_tool_stderr(tool_output: dict[str, object] | str) -> str:
    if not isinstance(tool_output, dict):
        return ""
    data = tool_output.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("stderr", "") or "").strip()


def last_tool_has_args_nameerror(tool_output: dict[str, object] | str) -> bool:
    stderr = last_tool_stderr(tool_output)
    if not stderr:
        return False
    lowered = stderr.lower()
    return "nameerror" in lowered and "args" in lowered and "not defined" in lowered
