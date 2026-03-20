"""
ToolRegistry — structured function-calling framework for the autonomous agent.

Provides a registry of tools that the LLM can invoke, with:
- JSON schema definitions for each tool
- Automatic validation of tool calls
- Execution routing
- Result formatting

This replaces the regex-based file_actions parsing with proper tool use.
"""
from __future__ import annotations

import inspect
import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""
    name: str
    type: str  # "string" | "integer" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """Complete definition of a tool the LLM can invoke."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    category: str = "general"  # research | experiment | writing | system

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for the LLM."""
        properties = {}
        required = []
        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolResult:
    """Result of a tool invocation."""

    def __init__(self, success: bool, data: Any = None, error: str = ""):
        self.success = success
        self.data = data
        self.error = error

    def to_message(self) -> str:
        """Format as a message string for the LLM."""
        if self.success:
            if isinstance(self.data, str):
                return self.data
            return json.dumps(self.data, indent=2, default=str)
        return f"ERROR: {self.error}"

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        preview = str(self.data)[:100] if self.data else self.error[:100]
        return f"ToolResult({status}: {preview})"


class ToolRegistry:
    """
    Central registry for all tools available to the agent.

    Tools are registered with their schemas and handlers, then exposed
    to the LLM for structured function calling.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (%s)", tool.name, tool.category)

    def register_function(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: List[ToolParameter] = None,
        category: str = "general",
    ) -> None:
        """Register a function as a tool with auto-detected parameters."""
        if parameters is None:
            parameters = self._infer_parameters(handler)

        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            category=category,
        )
        self.register(tool)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        """List all registered tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_schemas(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools (for sending to the LLM)."""
        return [t.to_schema() for t in self.list_tools(category)]

    def get_tools_description(self, category: Optional[str] = None) -> str:
        """
        Generate a human-readable description of available tools.
        Used in system prompts when the LLM doesn't support native function calling.
        """
        tools = self.list_tools(category)
        if not tools:
            return "(no tools available)"

        lines = ["## Available Tools\n"]
        lines.append("You can invoke tools by including a tool_calls JSON block in your response:\n")
        lines.append("```tool_calls")
        lines.append('[{"tool": "tool_name", "args": {"param1": "value1"}}]')
        lines.append("```\n")

        for tool in tools:
            params_desc = ", ".join(
                f"{p.name}: {p.type}" + (" (optional)" if not p.required else "")
                for p in tool.parameters
            )
            lines.append(f"### {tool.name}")
            lines.append(f"  {tool.description}")
            if params_desc:
                lines.append(f"  Parameters: {params_desc}")
            lines.append("")

        return "\n".join(lines)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(False, error=f"Unknown tool: {tool_name}")

        if not tool.handler:
            return ToolResult(False, error=f"Tool {tool_name} has no handler")

        # Validate required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return ToolResult(
                    False,
                    error=f"Missing required parameter: {param.name}",
                )

        # Apply defaults for optional params
        for param in tool.parameters:
            if param.name not in arguments and param.default is not None:
                arguments[param.name] = param.default

        try:
            result = tool.handler(**arguments)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(True, data=result)
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            return ToolResult(
                False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:500]}",
            )

    def parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool invocations from an LLM response.

        Supports two formats:
        1. ```tool_calls [...] ``` blocks
        2. Native OpenAI function calling format
        """
        calls = []

        # Format 1: ```tool_calls block
        match = re.search(r"```tool_calls\s*\n(.*?)```", response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    calls.extend(parsed)
                elif isinstance(parsed, dict):
                    calls.append(parsed)
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool_calls JSON block")

        # Format 2: individual tool call markers
        if not calls:
            for match in re.finditer(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                response,
                re.DOTALL,
            ):
                try:
                    calls.append(json.loads(match.group(1)))
                except json.JSONDecodeError:
                    pass

        # Format 3: ```json block with tool/args structure
        if not calls:
            for match in re.finditer(
                r'```json\s*\n(\{[^}]*"tool"[^}]*\})\s*```',
                response,
                re.DOTALL,
            ):
                try:
                    calls.append(json.loads(match.group(1)))
                except json.JSONDecodeError:
                    pass

        return calls

    def execute_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse and execute all tool calls found in an LLM response.

        Returns a list of {tool, args, result} dicts.
        """
        calls = self.parse_tool_calls(response)
        results = []

        for call in calls:
            tool_name = call.get("tool") or call.get("name") or call.get("function", "")
            args = call.get("args") or call.get("arguments") or call.get("parameters", {})

            result = self.execute(tool_name, args)
            results.append({
                "tool": tool_name,
                "args": args,
                "result": result,
            })
            logger.info("Tool %s → %s", tool_name, "OK" if result.success else f"FAIL: {result.error[:80]}")

        return results

    @staticmethod
    def _infer_parameters(func: Callable) -> List[ToolParameter]:
        """Infer tool parameters from a function's signature."""
        params = []
        sig = inspect.signature(func)
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            ptype = "string"
            if param.annotation != inspect.Parameter.empty:
                ptype = type_map.get(param.annotation, "string")

            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            params.append(ToolParameter(
                name=name,
                type=ptype,
                description=f"Parameter: {name}",
                required=required,
                default=default,
            ))

        return params
