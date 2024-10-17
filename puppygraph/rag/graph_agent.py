"""PuppyGraphAgent."""

import logging
from copy import deepcopy
from typing import Iterable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, create_model
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import StructuredTool, Tool

from puppygraph.client.client import PuppyGraphClient

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Prevent "No handlers could be found" warnings


class PuppyGraphAgent:
    """PuppyGraphAgent is the agent that interacts with the PuppyGraph via natural language.

    It enables the user to interact with the graph via natural lanaguage queries.
    """

    def __init__(
        self,
        puppy_graph_client: PuppyGraphClient,
        llm: BaseChatModel,
        chat_prompt_template: ChatPromptTemplate,
        query_language: str = "cypher",
        additional_tools: Optional[List[Tool]] = None,
    ):
        """Initializes the PuppyGraphAgent.

        Args:
            puppy_graph_client: The PuppyGraph client
            llm: The language model
            chat_prompt_template: The chat prompt template
            query_language: The query language to use, either "cypher" or "gremlin" or "both"
            additional_tools: Additional tools for the agent to use
        """

        self._puppy_graph_client = puppy_graph_client

        # Set up tools
        self._cypher_query_tool = _get_cypher_query_tool(
            puppy_graph_client=puppy_graph_client
        )
        self._gremlin_query_tool = _get_gremlin_query_tool(
            puppy_graph_client=puppy_graph_client
        )
        if query_language == "cypher":
            self._tools = [self._cypher_query_tool]
        elif query_language == "gremlin":
            self._tools = [self._gremlin_query_tool]
        elif query_language == "both":
            self._tools = [self._cypher_query_tool, self._gremlin_query_tool]

        if additional_tools is not None:
            self._tools.extend(additional_tools)

        self._tool_dict = {tool.name: tool for tool in self._tools}

        # Set up llm chain
        self._chat_prompt_template = chat_prompt_template

        self._llm = llm.bind_tools(tools=self._tools)
        self._llm_chain: RunnableSequence = self._chat_prompt_template | self._llm

        self._llm_no_tool_output = llm.bind_tools(tools=self._tools, tool_choice="none")
        self._llm_no_tool_output_chain: RunnableSequence = (
            self._chat_prompt_template | self._llm_no_tool_output
        )

        # Set up other global variables
        self._message_history = []

    def query(self, user_input: str, max_iters: int = 10) -> Iterable[BaseMessage]:
        """Query the graph using the given user input.

        Args:
            user_input: The user input
            max_iters: The maximum number of iterations to run

        Yields:
            BaseMessage, can be either AIMessage or ToolMessage
        """
        # We have to copy the message history to avoid side effects
        # if query() is called multiple times
        message_history = deepcopy(self._message_history)

        new_messages = [HumanMessage(content=user_input)]

        iters = 0

        wait_for_user_input = False
        while iters < max_iters and not wait_for_user_input:
            # If we are at the last iteration, we don't want to call the tool
            # This is because we want to show the user the final message
            tool_call_allowed = True
            if iters + 1 == max_iters:
                tool_call_allowed = False

            # Predict
            ai_message = self._llm_predict(
                message_history=message_history + new_messages,
                tool_call_allowed=tool_call_allowed,
            )

            # Add AI message to new messages
            new_messages.append(ai_message)
            logger.info("AI message: %s", ai_message.content)
            yield ai_message

            if ai_message.tool_calls:
                # Execute the actual tool
                tool_messages = self._execute_tool_calls(ai_message)

                # Add tool messages to new messages
                new_messages.extend(tool_messages)
                logger.info("Tool messages: %s", tool_messages)
                yield from tool_messages

            iters += 1

            # Check if we need to wait for user input
            if not ai_message.tool_calls:
                wait_for_user_input = True

        # Update the message history
        self._message_history += new_messages

    def reset_messages(self):
        """Reset the message history."""
        self._message_history = []

    def _llm_predict(
        self,
        message_history: List[BaseMessage],
        tool_call_allowed: bool,
    ) -> AIMessage:
        """Predict the AI message using llm.

        Args:
            message_history: The message history
            tool_call_allowed: Whether tool calls are allowed

        Returns:
            The predicted AI message
        """
        input_dict = {
            "message_history": message_history,
        }

        if not tool_call_allowed:
            return self._llm_no_tool_output_chain.invoke(input=input_dict)

        return self._llm_chain.invoke(input=input_dict)

    def _execute_tool_calls(self, ai_message: AIMessage) -> List[ToolMessage]:
        """Execute the tool calls in the AI message.

        Args:
            ai_message: The AI message which might contain tool calls

        Returns:
            The tool messages
        """
        tool_messages = []
        for tool_call in ai_message.tool_calls:
            tool_messages.append(self._execute_tool_call(tool_call))
        return tool_messages

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        """Execute the given tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            The tool message
        """
        tool = self._tool_dict[tool_call["name"]]
        logger.info(
            "Calling tool: %s with args: %s", tool_call["name"], tool_call["args"]
        )
        try:
            tool_output = str(tool.invoke(input=tool_call["args"]))
        except Exception as e:
            tool_output = f"While executing tool, an error occurred : {e}"

        return ToolMessage(tool_output, tool_call_id=tool_call["id"])


def _get_cypher_query_tool(puppy_graph_client: PuppyGraphClient):
    """Get the Cypher query tool."""
    return StructuredTool.from_function(
        func=puppy_graph_client.cypher_query,
        name="query_graph_cypher",
        description="Query the graph database using Cypher.",
        args_schema=create_model(
            "", query=(str, Field(description="The Cypher query to run"))
        ),
    )


def _get_gremlin_query_tool(puppy_graph_client: PuppyGraphClient):
    """Get the Gremlin query tool."""
    return StructuredTool.from_function(
        func=puppy_graph_client.gremlin_query,
        name="query_graph_gremlin",
        description="Query the graph database using Gremlin.",
        args_schema=create_model(
            "", query=(str, Field(description="The Gremlin query to run"))
        ),
    )
