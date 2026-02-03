"""Agent context management for LLM requests.

This module provides context variables and utilities for passing agent metadata
through the async call stack to HTTP transport layers without modifying function
signatures. The context is used to build structured user_input data for LLM Gateway.
"""

from contextvars import ContextVar
from typing import Any

_agent_definition: ContextVar[Any] = ContextVar("agent_definition", default=None)

_user_input: ContextVar[str] = ContextVar("user_input", default="")


def set_agent_definition(agent_def: Any) -> None:
    """Set the agent definition for the current context.

    This should be called once per graph creation and persists for the graph's
    lifetime. Also computes and caches the user_input string for LLM requests.

    Args:
        agent_def: The agent definition object from agent configuration.
    """
    _agent_definition.set(agent_def)
    if agent_def is not None:
        user_input = build_user_input_from_definition(agent_def)
        _user_input.set(user_input)
    else:
        _user_input.set("")


def get_agent_definition() -> Any:
    """Get the agent definition for the current context.

    Returns:
        The agent definition object, or None if not set.
    """
    return _agent_definition.get()


def get_user_input() -> str:
    """Get the cached user_input string for the current context.

    Returns:
        The pre-computed user_input string for LLM Gateway requests.
    """
    return _user_input.get()


def build_user_input_from_definition(
    agent_def: Any,
    tool_names: list[str] | None = None,
) -> str:
    """Build user_input string directly from agent definition for LLM Gateway.

    Extracts system/user prompts, tools, contexts, escalations, and examples
    from the agent definition and formats them matching C# ExtractUserInput format.

    Args:
        agent_def: The agent definition containing messages and resources.
        tool_names: Optional list of tool names to include. If None, extracted from resources.

    Returns:
        Formatted user_input string for LLM Gateway.
    """
    from uipath.agent.models.agent import (
        AgentContextResourceConfig,
        AgentEscalationResourceConfig,
        AgentMessageRole,
    )

    sections = []

    system_message = next(
        (msg for msg in agent_def.messages if msg.role == AgentMessageRole.SYSTEM),
        None,
    )
    system_prompt = system_message.content if system_message else ""
    sections.append(f"System Prompt:\n    {system_prompt}")

    user_message = next(
        (msg for msg in agent_def.messages if msg.role == AgentMessageRole.USER),
        None,
    )
    user_prompt = user_message.content if user_message else ""
    sections.append(f"User Prompt:\n    {user_prompt}")

    if tool_names:
        tools_lines = "\n".join(f"    {t}" for t in tool_names)
        sections.append(f"Tools:\n{tools_lines}")

    contexts = [
        r.name for r in agent_def.resources if isinstance(r, AgentContextResourceConfig)
    ]
    if contexts:
        contexts_lines = "\n".join(f"    {c}" for c in contexts)
        sections.append(f"Contexts:\n{contexts_lines}")

    escalations = [
        r.name
        for r in agent_def.resources
        if isinstance(r, AgentEscalationResourceConfig)
    ]
    if escalations:
        escalations_lines = "\n".join(f"    {e}" for e in escalations)
        sections.append(f"Escalations:\n{escalations_lines}")

    examples = []
    for feature in agent_def.features:
        feature_dict = (
            feature if isinstance(feature, dict) else getattr(feature, "__dict__", {})
        )

        feature_examples = (
            feature_dict.get("examples")
            if isinstance(feature, dict)
            else getattr(feature, "examples", None)
        )

        if feature_examples:
            for ex in feature_examples:
                if isinstance(ex, dict):
                    examples.append(
                        {
                            "input": ex.get("input", ""),
                            "output": ex.get("output", ""),
                            "instructions": ex.get("instructions", ""),
                        }
                    )
                else:
                    examples.append(
                        {
                            "input": getattr(ex, "input", ""),
                            "output": getattr(ex, "output", ""),
                            "instructions": getattr(ex, "instructions", ""),
                        }
                    )

    if examples:
        examples_parts = []
        for ex in examples:
            example_str = f"""   Input:
       {ex.get("input", "")}
   Output:
       {ex.get("output", "")}
   Instructions:
       {ex.get("instructions", "")}"""
            examples_parts.append(example_str)
        examples_section = f"Examples:\n{chr(10).join(examples_parts)}"
        sections.append(examples_section)

    return "\n\n".join(sections)
