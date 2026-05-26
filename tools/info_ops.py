from datetime import datetime
import ast
import operator
import re

from langchain_core.tools import tool

from core.graph_constants import MAX_REASONING_STEPS
from core.models import ToolResult

_runtime: dict = {}

_RATIO_QUESTION_PATTERN = re.compile(
    r"(?P<a>-?\d+(?:\.\d+)?)\s*(?P<u1>[a-zA-Z]+)s?\s*(?:is|=|equals?)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?P<u2>[a-zA-Z]+)s?.*?(?:what\s+is|how\s+much\s+is|calculate)\s*(?P<c>-?\d+(?:\.\d+)?)\s*(?P<u3>[a-zA-Z]+)s?",
    re.IGNORECASE,
)
_ARITHMETIC_EXPRESSION_PATTERN = re.compile(r"^[\d\s+\-*/().]+$")
_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_UNIT_QUERY_PATTERN = re.compile(
    r"(?:what\s+is|how\s+much(?:\s+is)?|calculate)\s*(?P<c>-?\d+(?:\.\d+)?)\s*(?P<u3>[a-zA-Z]+)s?(?:\s*(?:in|to)\s*(?P<u4>[a-zA-Z]+)s?)?\s*\??$",
    re.IGNORECASE,
)


def _normalize_unit(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.endswith("s") and len(lowered) > 1:
        return lowered[:-1]
    return lowered


def _eval_arithmetic_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        operand = _eval_arithmetic_node(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)
    raise ValueError("Unsupported arithmetic expression")


def _get_math_relations() -> dict[str, dict[str, float]]:
    relations = _runtime.get("math_relations")
    if not isinstance(relations, dict):
        relations = {}
        _runtime["math_relations"] = relations
    return relations


def _store_math_relation(source_unit: str, target_unit: str, ratio: float) -> None:
    if ratio == 0:
        return
    relations = _get_math_relations()
    source_map = relations.setdefault(source_unit, {})
    target_map = relations.setdefault(target_unit, {})
    source_map[target_unit] = ratio
    target_map[source_unit] = 1.0 / ratio


def _lookup_ratio(source_unit: str, target_unit: str) -> float | None:
    relations = _get_math_relations()
    source_map = relations.get(source_unit)
    if not isinstance(source_map, dict):
        return None
    ratio = source_map.get(target_unit)
    if not isinstance(ratio, (int, float)):
        return None
    return float(ratio)


def _infer_single_target(source_unit: str) -> str | None:
    relations = _get_math_relations()
    source_map = relations.get(source_unit)
    if not isinstance(source_map, dict):
        return None
    targets = [unit for unit, ratio in source_map.items() if isinstance(ratio, (int, float))]
    if len(targets) == 1:
        return targets[0]
    return None


def _format_proportional_message(
    c: float,
    u3: str,
    result: float,
    target_unit: str,
    assumption_note: str,
) -> str:
    return f"Based on your stated relation (assumption), {c:g} {u3} is {result:g} {target_unit}. {assumption_note}"


def _solve_math_question(question: str) -> tuple[bool, str, dict]:
    text = (question or "").strip()
    if not text:
        return False, "Math question is empty.", {}

    ratio_match = _RATIO_QUESTION_PATTERN.search(text)
    if ratio_match:
        a = float(ratio_match.group("a"))
        b = float(ratio_match.group("b"))
        c = float(ratio_match.group("c"))
        u1 = _normalize_unit(ratio_match.group("u1"))
        u2 = _normalize_unit(ratio_match.group("u2"))
        u3 = _normalize_unit(ratio_match.group("u3"))

        if a == 0 or b == 0:
            return False, "Cannot solve proportional conversion with zero base value.", {"question": text}

        if u3 == u1:
            result = c * b / a
            target_unit = u2
        elif u3 == u2:
            result = c * a / b
            target_unit = u1
        else:
            return False, "Could not infer conversion target from units in question.", {
                "question": text,
                "known_units": [u1, u2],
                "query_unit": u3,
            }

        provided_ratio = b / a
        _store_math_relation(u1, u2, provided_ratio)

        assumption_note = "Used only the relationship provided in the question."
        message = _format_proportional_message(
            c=c,
            u3=u3,
            result=result,
            target_unit=target_unit,
            assumption_note=assumption_note,
        )

        return True, message, {
            "question": text,
            "result": result,
            "source_unit": u3,
            "target_unit": target_unit,
            "method": "proportional",
            "assumption": assumption_note,
            "provided_ratio": provided_ratio,
            "warning": None,
        }

    query_match = _UNIT_QUERY_PATTERN.search(text)
    if query_match:
        c = float(query_match.group("c"))
        u3 = _normalize_unit(query_match.group("u3"))
        explicit_target = query_match.group("u4")
        target_unit = _normalize_unit(explicit_target) if explicit_target else _infer_single_target(u3)
        if not target_unit:
            return False, "No known relation for this unit yet. Provide a relation like '1 meter is 150 cm' first.", {
                "question": text,
                "source_unit": u3,
            }
        ratio = _lookup_ratio(u3, target_unit)
        if ratio is None:
            return False, "No stored relation found for the requested unit conversion.", {
                "question": text,
                "source_unit": u3,
                "target_unit": target_unit,
            }
        result = c * ratio
        assumption_note = "Used the previously stated relationship from this session."
        message = _format_proportional_message(
            c=c,
            u3=u3,
            result=result,
            target_unit=target_unit,
            assumption_note=assumption_note,
        )
        return True, message, {
            "question": text,
            "result": result,
            "source_unit": u3,
            "target_unit": target_unit,
            "method": "proportional_from_context",
            "assumption": assumption_note,
            "provided_ratio": ratio,
            "warning": None,
        }

    expr = text
    lowered = text.lower()
    if lowered.startswith("what is"):
        expr = text[7:].strip().rstrip("?").strip()

    if _ARITHMETIC_EXPRESSION_PATTERN.fullmatch(expr):
        try:
            parsed = ast.parse(expr, mode="eval")
            result = _eval_arithmetic_node(parsed)
            return True, f"Result: {result:g}", {
                "question": text,
                "expression": expr,
                "result": result,
                "method": "arithmetic",
            }
        except Exception as exc:
            return False, f"Could not evaluate arithmetic expression: {exc}", {"question": text}

    return False, "Could not parse math question into a deterministic calculation.", {"question": text}


def _to_non_negative_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def get_info_tools(model: str, workspace_dir: str):
    _runtime["model"] = model
    _runtime["workspace_dir"] = workspace_dir
    _runtime["context_window"] = "~128k tokens"
    _runtime["max_steps"] = MAX_REASONING_STEPS

    @tool
    def agent_info() -> str:
        """Return current runtime configuration and the last known token usage."""
        usage = _runtime.get("token_usage")
        return ToolResult(
            success=True,
            message="CortexNode runtime info",
            data={
                "model": _runtime.get("model", "unknown"),
                "context_window": _runtime.get("context_window", "unknown"),
                "workspace": _runtime.get("workspace_dir", "unknown"),
                "max_steps": _runtime.get("max_steps", "unknown"),
                "token_usage": usage,
            },
        ).to_tool_output()

    @tool
    def token_usage() -> str:
        """Return token counts from the most recent brain node response."""
        usage = _runtime.get("token_usage")
        if not usage:
            return ToolResult(
                success=False,
                message="No token usage recorded yet.",
            ).to_tool_output()
        return ToolResult(
            success=True,
            message="Most recent token usage",
            data=usage,
        ).to_tool_output()

    @tool
    def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Return the current local system time using an optional strftime format."""
        try:
            now = datetime.now()
            formatted = now.strftime(format)
            return ToolResult(
                success=True,
                message="Current local system time",
                data={
                    "iso": now.isoformat(),
                    "formatted": formatted,
                    "format": format,
                },
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Error formatting current time: {exc}",
            ).to_tool_output()

    @tool
    def solve_math(question: str = "") -> str:
        """Solve deterministic arithmetic/proportional math questions without freeform LLM reasoning."""
        success, message, data = _solve_math_question(question)
        return ToolResult(
            success=success,
            message=message,
            data=data,
            display=message,
        ).to_tool_output()

    return [agent_info, token_usage, current_time, solve_math]


def update_token_usage(usage: dict) -> None:
    """Accumulate token usage across turns; only update non-zero values."""
    if not usage:
        return

    prompt_tokens = _to_non_negative_int(usage.get("prompt_tokens", 0))
    completion_tokens = _to_non_negative_int(usage.get("completion_tokens", 0))
    total_tokens = _to_non_negative_int(usage.get("total_tokens", 0))

    existing = _runtime.get("token_usage", {}) or {}

    existing_prompt = _to_non_negative_int(existing.get("prompt_tokens", 0))
    existing_completion = _to_non_negative_int(existing.get("completion_tokens", 0))
    existing_total = _to_non_negative_int(existing.get("total_tokens", 0))

    _runtime["token_usage"] = {
        "prompt_tokens": existing_prompt + prompt_tokens,
        "completion_tokens": existing_completion + completion_tokens,
        "total_tokens": existing_total + total_tokens,
    }
