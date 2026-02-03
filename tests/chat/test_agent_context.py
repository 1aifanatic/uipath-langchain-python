"""Tests for agent context management."""

from unittest.mock import MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentContextSettings,
    AgentEscalationResourceConfig,
    AgentMessage,
    AgentMessageRole,
)

from uipath_langchain.chat.helpers.agent_context import (
    build_user_input_from_definition,
    get_agent_definition,
    get_user_input,
    set_agent_definition,
)


def test_set_and_get_agent_definition():
    """Test setting and getting agent definition."""
    test_def = MagicMock()
    test_def.name = "test_agent"

    with patch(
        "uipath_langchain.chat.helpers.agent_context.build_user_input_from_definition"
    ) as mock_build:
        mock_build.return_value = "mocked user_input"
        set_agent_definition(test_def)
        assert get_agent_definition() == test_def
        mock_build.assert_called_once_with(test_def)


def test_get_user_input_returns_cached_value():
    """Test that get_user_input returns the cached value from set_agent_definition."""
    test_def = MagicMock()

    with patch(
        "uipath_langchain.chat.helpers.agent_context.build_user_input_from_definition"
    ) as mock_build:
        mock_build.return_value = "cached user_input"
        set_agent_definition(test_def)
        assert get_user_input() == "cached user_input"


def test_get_user_input_returns_empty_when_no_agent_definition():
    """Test that get_user_input returns empty string when no agent definition is set."""
    set_agent_definition(None)
    assert get_user_input() == ""


class TestBuildUserInputFromDefinition:
    """Tests for build_user_input_from_definition serialization logic."""

    @pytest.fixture
    def system_message(self):
        """Create a system message."""
        return AgentMessage(
            role=AgentMessageRole.SYSTEM, content="You are a helpful assistant."
        )

    @pytest.fixture
    def user_message(self):
        """Create a user message."""
        return AgentMessage(role=AgentMessageRole.USER, content="Help me with my task.")

    @pytest.fixture
    def context_resource(self):
        """Create a context resource."""
        return AgentContextResourceConfig(
            name="knowledge_base",
            description="Knowledge base context",
            folder_path="/contexts",
            index_name="kb_index",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            ),
        )

    @pytest.fixture
    def escalation_resource(self):
        """Create an escalation resource."""
        return AgentEscalationResourceConfig(
            name="human_review",
            description="Escalate to human review",
            channels=[],
        )

    @pytest.fixture
    def basic_agent_def(self, system_message, user_message):
        """Create a basic agent definition with just messages."""
        agent_def = MagicMock()
        agent_def.messages = [system_message, user_message]
        agent_def.resources = []
        agent_def.features = []
        return agent_def

    def test_basic_prompts_only(self, basic_agent_def):
        """Test output with only system and user prompts."""
        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task."
        )
        assert result == expected

    def test_with_tool_names(self, basic_agent_def):
        """Test output with tool names."""
        tool_names = ["search_tool", "calculator_tool", "email_tool"]

        result = build_user_input_from_definition(
            basic_agent_def, tool_names=tool_names
        )

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Tools:\n"
            "    search_tool\n"
            "    calculator_tool\n"
            "    email_tool"
        )
        assert result == expected

    def test_with_single_context(self, basic_agent_def, context_resource):
        """Test output with a single context resource."""
        basic_agent_def.resources = [context_resource]

        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Contexts:\n"
            "    knowledge_base"
        )
        assert result == expected

    def test_with_single_escalation(self, basic_agent_def, escalation_resource):
        """Test output with a single escalation resource."""
        basic_agent_def.resources = [escalation_resource]

        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Escalations:\n"
            "    human_review"
        )
        assert result == expected

    def test_with_multiple_contexts_and_escalations(
        self, basic_agent_def, context_resource, escalation_resource
    ):
        """Test output with multiple contexts and escalations."""
        context2 = AgentContextResourceConfig(
            name="faq_context",
            description="FAQ context",
            folder_path="/faq",
            index_name="faq_index",
            settings=AgentContextSettings(
                result_count=3,
                retrieval_mode=AgentContextRetrievalMode.STRUCTURED,
            ),
        )
        escalation2 = AgentEscalationResourceConfig(
            name="manager_escalation",
            description="Escalate to manager",
            channels=[],
        )
        basic_agent_def.resources = [
            context_resource,
            context2,
            escalation_resource,
            escalation2,
        ]

        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Contexts:\n"
            "    knowledge_base\n"
            "    faq_context\n"
            "\n"
            "Escalations:\n"
            "    human_review\n"
            "    manager_escalation"
        )
        assert result == expected

    def test_with_examples_dict_format(self, basic_agent_def):
        """Test output with examples in dict format."""
        feature = {
            "examples": [
                {
                    "input": "What is 2+2?",
                    "output": "4",
                    "instructions": "Do basic math",
                },
            ]
        }
        basic_agent_def.features = [feature]

        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Examples:\n"
            "   Input:\n"
            "       What is 2+2?\n"
            "   Output:\n"
            "       4\n"
            "   Instructions:\n"
            "       Do basic math"
        )
        assert result == expected

    def test_with_multiple_examples(self, basic_agent_def):
        """Test output with multiple examples."""
        feature = {
            "examples": [
                {
                    "input": "What is 2+2?",
                    "output": "4",
                    "instructions": "Do basic math",
                },
                {
                    "input": "Hello",
                    "output": "Hi there!",
                    "instructions": "Be friendly",
                },
            ]
        }
        basic_agent_def.features = [feature]

        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Examples:\n"
            "   Input:\n"
            "       What is 2+2?\n"
            "   Output:\n"
            "       4\n"
            "   Instructions:\n"
            "       Do basic math\n"
            "   Input:\n"
            "       Hello\n"
            "   Output:\n"
            "       Hi there!\n"
            "   Instructions:\n"
            "       Be friendly"
        )
        assert result == expected

    def test_with_examples_object_format(self, basic_agent_def):
        """Test output with examples in object format."""
        example1 = MagicMock()
        example1.input = "Search for cats"
        example1.output = "Found 10 results"
        example1.instructions = "Use the search tool"

        feature = MagicMock()
        feature.examples = [example1]

        basic_agent_def.features = [feature]

        result = build_user_input_from_definition(basic_agent_def)

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Examples:\n"
            "   Input:\n"
            "       Search for cats\n"
            "   Output:\n"
            "       Found 10 results\n"
            "   Instructions:\n"
            "       Use the search tool"
        )
        assert result == expected

    def test_full_serialization_all_sections(
        self, system_message, user_message, context_resource, escalation_resource
    ):
        """Test complete output with all sections."""
        agent_def = MagicMock()
        agent_def.messages = [system_message, user_message]
        agent_def.resources = [context_resource, escalation_resource]
        feature = {
            "examples": [
                {
                    "input": "Test input",
                    "output": "Test output",
                    "instructions": "Test instructions",
                }
            ]
        }
        agent_def.features = [feature]

        result = build_user_input_from_definition(agent_def, tool_names=["my_tool"])

        expected = (
            "System Prompt:\n"
            "    You are a helpful assistant.\n"
            "\n"
            "User Prompt:\n"
            "    Help me with my task.\n"
            "\n"
            "Tools:\n"
            "    my_tool\n"
            "\n"
            "Contexts:\n"
            "    knowledge_base\n"
            "\n"
            "Escalations:\n"
            "    human_review\n"
            "\n"
            "Examples:\n"
            "   Input:\n"
            "       Test input\n"
            "   Output:\n"
            "       Test output\n"
            "   Instructions:\n"
            "       Test instructions"
        )
        assert result == expected
