import argparse
import os
import time
from functools import partial
from typing import Iterable, List, Optional, Union

import gradio as gr
import yaml
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import Field, create_model
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from puppygraph import PuppyGraphClient, PuppyGraphHostConfig
from puppygraph.rag import PuppyGraphAgent


def _get_graph_schema_prompt(query_language: str) -> str:

    schema_prompt = """
    Nodes are the following:
    - person:
        properties: 
            - name: primaryName
            type: String
            description: The name of the person, as listed in the IMDb database.
            - name: birthYear
            type: Int
            description: The birth year of the person (if available).
            - name: deathYear
            type: Int
            description: The death year of the person (if available).

    - title:
        properties:
            - name: titleType
            type: String
            description: The type/format of the title (e.g., movie, short, tvseries, tvepisode, video, etc.).
            - name: primaryTitle
            type: String
            description: The more popular title or the title used by filmmakers on promotional materials at the point of release.
            - name: originalTitle
            type: String
            description: The original title, in the original language.
            - name: isAdult
            type: Boolean
            description: Indicates whether the title is for adults (1: adult title, 0: non-adult title).
            - name: startYear
            type: Int
            description: Represents the release year of a title. For TV Series, this is the series start year.
            - name: endYear
            type: Int
            description: For TV Series, this is the series end year. '\\N' for all other title types.
            - name: runtimeMinutes
            type: Int
            description: The primary runtime of the title, in minutes.

    Edges are the following:
    - cast_and_crew:
        from: title
        to: person
        properties:
            - name: ordering
            type: Int
            description: A unique identifier for the row, used to determine the order of people associated with this title.
            - name: category
            type: String
            description: The category of job that the person was in (e.g., actor, director).
            - name: job
            type: String
            description: The specific job title if applicable, else '\\N'.
            - name: characters
            type: String
            description: The name of the character played if applicable, else '\\N'.
"""
    if query_language == "cypher":
        additional_instructions = ""
    elif query_language == "gremlin":
        additional_instructions = """
            The relationships are the following:
            g.V().hasLabel('title').out('cast_and_crew').hasLabel('person'),
            g.V().hasLabel('person').in('cast_and_crew').hasLabel('title'),

            if filter by category, you must use outE() or inE(), because the category is stored in the EDGE properties.
        """
    else:
        raise NotImplementedError(f"Query language {query_language} is not supported.")

    return schema_prompt + additional_instructions


def _get_chat_prompt_template(
    graph_schema_prompt: str, search_tool_enabled: bool
) -> ChatPromptTemplate:

    if search_tool_enabled:
        additional_conclusion_prompt = ", please also cite the source [🌐] or [📈] indicating whether the information is from the internet or from the graph database if applicable"
    else:
        additional_conclusion_prompt = ""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant to help answer user questions about imdb."
                "You will need to use the information stored in the graph database to answer the user's questions."
                "Here is some information about the graph database schema.\n"
                f"{graph_schema_prompt}",
            ),
            (
                "system",
                "You must first output a PLAN, then you can use the PLAN to call the tools.\n"
                "Each STEP of the PLAN should be corresponding to one or more function calls (but not less), either simple or complex.\n"
                "Minimize the number of steps in the PLAN, but make sure the PLAN is workable.\n"
                "Remember, each step can be converted to a Gremlin query, since Gremlin query can handle quite complex queries,"
                "each step can be complex as well as long as it can be converted to a Gremlin query.",
            ),
            MessagesPlaceholder(variable_name="message_history"),
            (
                "system",
                "Always use the JSON format {\n"
                "'THINKING': <the thought process in PLAIN TEXT>,"
                "'PLAN': <the plan contains multiple steps in PLAIN TEXT, Your Original plan or Update plan after seeing some executed results>,"
                f"'CONCLUSION': <Keep your conclusion simple and clear if you decide to conclude {additional_conclusion_prompt} >",
            ),
        ],
        template_format="jinja2",
    )


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-2024-08-06",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _get_puppy_graph_client(ip) -> PuppyGraphClient:
    return PuppyGraphClient(PuppyGraphHostConfig(ip=ip))


def _display_ai_message_content(
    ai_message_content: str, is_last_message: bool
) -> Iterable[str]:
    if not is_last_message:
        conclusion_emoji = "📝"
    else:
        conclusion_emoji = "✅"

    try:
        text_dict = yaml.safe_load(ai_message_content)
        if "THINKING" in text_dict:
            yield f"📝 {text_dict['THINKING']}"

        if "PLAN" in text_dict:
            yield f"📝 {text_dict['PLAN']}"
        if "CONCLUSION" in text_dict:
            yield f"{conclusion_emoji} {text_dict['CONCLUSION']}"
    except Exception as _:
        text_split = ai_message_content.split("'CONCLUSION':")
        seps = "\n} "
        yield f"{conclusion_emoji} {text_split[-1].strip(seps)}"


def _display_ai_message_tool_calls(tool_calls: List[str]) -> Iterable[str]:
    for tool_call in tool_calls:
        yield f"🔨 Calling {tool_call['name']} with args: {tool_call['args']}"


def _display_tool_message(tool_message: ToolMessage) -> Iterable[str]:
    yield f"🔨 Response: {tool_message.content}"


def _display_message(
    message: Optional[Union[AIMessage, ToolMessage]], is_last_message: bool = False
) -> Iterable[str]:
    if message is None:
        return

    if isinstance(message, AIMessage):
        yield from _display_ai_message_content(message.content, is_last_message)
        yield from _display_ai_message_tool_calls(message.tool_calls)
    elif isinstance(message, ToolMessage):
        yield from _display_tool_message(message)


def _get_displayable_responses(
    pg_agent: PuppyGraphAgent, user_message: str
) -> Iterable[str]:
    response_iter = pg_agent.query(user_input=user_message, max_iters=20)
    previous_message = None
    while True:
        try:
            current_message = next(response_iter)
            for display_string in _display_message(previous_message):
                yield display_string
            previous_message = current_message
        except StopIteration:
            for display_string in _display_message(
                previous_message, is_last_message=True
            ):
                yield display_string
            break


def _gradio_respond(pg_agent: PuppyGraphAgent, verbose_mode: bool, message, _):
    all_responses = ""
    for response in _get_displayable_responses(pg_agent=pg_agent, user_message=message):
        all_responses += response + "\n"
        if verbose_mode:
            yield all_responses
        else:
            time.sleep(0.5)
            yield response


def _get_gradio_chatbot(
    pg_agent: PuppyGraphAgent, verbose_mode: bool = False
) -> gr.ChatInterface:
    clear_btn = gr.Button("Clear", variant="secondary", size="sm", min_width=60)
    chat_bot = gr.ChatInterface(
        fn=partial(_gradio_respond, pg_agent, verbose_mode), clear_btn=clear_btn
    )
    with chat_bot:
        clear_btn.click(pg_agent.reset_messages)

    return chat_bot


def main():
    """Main function for running the PuppyGraphAgent."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Configure PuppyGraphAgent settings.")
    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="The IP address for the PuppyGraph.",
    )
    parser.add_argument(
        "--query_language",
        type=str,
        default="gremlin",
        help="The query language to be used (choose from 'gremlin' or 'cypher').",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    parser.add_argument(
        "--search", action="store_true", help="Enable search tool through internet."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the arguments
    ip = args.ip
    query_language = args.query_language
    verbose_mode = args.verbose
    search_tool_enabled = args.search

    pg_agent = PuppyGraphAgent(
        puppy_graph_client=_get_puppy_graph_client(ip=ip),
        llm=_get_llm(),
        chat_prompt_template=_get_chat_prompt_template(
            graph_schema_prompt=_get_graph_schema_prompt(query_language=query_language),
            search_tool_enabled=search_tool_enabled,
        ),
        query_language=query_language,
        additional_tools=(
            [
                StructuredTool.from_function(
                    func=GoogleSerperAPIWrapper().run,
                    name="google_serper",
                    description="Query the internet.",
                    args_schema=create_model(
                        "", query=(str, Field(description="query"))
                    ),
                )
            ]
            if search_tool_enabled
            else None
        ),
    )

    _get_gradio_chatbot(pg_agent=pg_agent, verbose_mode=verbose_mode).launch()


if __name__ == "__main__":

    main()
