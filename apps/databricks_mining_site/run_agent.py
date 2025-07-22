import logging
import os
from typing import List

import yaml
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from puppygraph import PuppyGraphClient, PuppyGraphHostConfig
from puppygraph.rag import PuppyGraphAgent
from langchain_core.messages import BaseMessage
from typing import Iterable, List

logging.basicConfig(level=logging.INFO)


def _get_graph_schema_prompt() -> str:
    return """
    Nodes are the following:
    - failure_type:
        properties: 
            - name: failure_type_name
              type: String
              description: The name of the failure type, only takes values from "Electrical", "Mechanical", "Software", "Pneumatic", "Hydraulic"
    - asset:
        description: The asset node represents a physical asset in the mining site
        properties:
            - name: asset_id
              type: String
            - name: asset_name
              type: String
            - name: asset_type
              type: String
              description: the type of an asset, only takes values from "Heavy Machinery", "Drilling", "Material Handling", "Transport", "Processing", "Safety"
            - name: location
              type: String 
            - name: acquisition_date
              type: String
              description: the format is "YYYY-MM-DD"
            - name: status 
              type: String
              description: The status of the asset, only takes values from "Active", "Inactive", "Under Maintenance"
    - work_order:
        description: The work order node represents a work order raised for an asset
        properties:
            - name: work_order_id
              type: String
            - name: date
              type: String
            - name: action_taken
              type: String
            - name: technician
              type: String
            - name: component_replaced_description
              type: String
              description: Description of the component replaced, only takes values from "Alternator", "Brake Assembly", "Control Panel", "Conveyor Belt", "Cooling Fan", "Engine Oil Filter", "Exhaust Manifold", "Fuel Injector", "Hydraulic Cylinder", "Hydraulic Hose", "Hydraulic Pump", "Pressure Sensor", "Swing Motor", "Track Chain", "Transmission". 
            - name: component_replaced_material_num
              type: String
            - name: successful_fix
              type: Boolean
              description: Whether the work order successfully fixed the asset or not.
    Edges are the following:
    - can_have_failure:
        description: The potential failure mode of an asset
        from: asset
        to: failure_type
        properties:
            - name: steps_to_follow
              type: String
              description: The steps to follow to troubleshoot the failure. (remember, this property is on the EDGE, not the NODE)
            - name: reference_source
              type: String
              descritpion: The reference source of the troubleshooting steps, only takes values from "Documentum", "OEM", "Internal Manual"
            - name: recommended_actions
              type: String
    - worked_on:
        description: A work order is working on an asset
        from: work_order
        to: asset
        properties: NONE
    - related_to_failure:
        description: A work order identifies a failure type on a specific asset
        properties: NONE
        from: work_order
        to: failure_type
    The relationships are the following:
    (:asset)-[:can_have_failure]->(:failure_type),
    (:work_order)-[:worked_on]->(:asset),
    (:work_order)-[:related_to_failure]->(:failure_type)
"""


def _get_chat_prompt_template(graph_schema_prompt: str) -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant to help answer user questions about assets in a mining site."
                "You will need to use the information stored in the graph database to answer the user's questions."
                "Here is some information about the graph database schema.\n"
                f"{graph_schema_prompt}",
            ),
            (
                "system",
                "You must first output a PLAN, then you can use the PLAN to call the tools.\n"
                "Each STEP of the PLAN should be corresponding to one or more function calls (but not less), either simple or complex.\n"
                "Minimize the number of steps in the PLAN, but make sure the PLAN is workable.\n"
                "Remember, each step can be converted to a Cypher query, since Cypher query can handle quite complex queries,"
                "each step can be complex as well as long as it can be converted to a Cypher query.",
            ),
            MessagesPlaceholder(variable_name="message_history"),
            (
                "system",
                "For COUNT(), ONLY use COUNT(*) in your Cypher queries, as COUNT(something) is not supported yet.When calculating failures for a particular asset, also first find out the work orders that are related to the asset, then count the work orders that are related to the failure using related_to_failure. DO NOT USE can_have_failure for counting total number of failures, USE related_to_failure instead.",
            ),
            (
                "system",
                "Always use the format {\n"
                "'THINKING': <the thought process in PLAIN TEXT>,"
                "'PLAN': <the plan contains multiple steps in PLAIN TEXT, Your Original plan or Update plan after seeing some executed results>,"
                "'CONCLUSION': <Keep your conclusion simple and clear if you decide to conclude>}",
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


def _process_answer(answers: Iterable[BaseMessage]) -> str:
    reversed_answers = reversed(list(answers))
    for answer in reversed_answers:
        text = answer.content
        try:
            text_dict = yaml.safe_load(text)
            if "CONCLUSION" in text_dict:
                return text_dict["CONCLUSION"]
        except:
            text_split = text.split("'CONCLUSION':")
            return text_split[-1].strip("\n}")
        return text


def _run_queries(pg_agent: PuppyGraphAgent) -> List[str]:
    queries = [
        "How do I check engine oil levels?",
        "How many mechanical failures has Excavator 3000 had?",
        "How many times did we need to replace pressure sensor on Haul Truck 400T?",
        "How many mechanical work orders were unsuccessful?",
        "When was a work order raised for Load-Haul-Dump Machine to update fuel injector?",
        "What are the Asset IDs of Heavy Machinery?",
        "How many assets are active in Site A?",
        "When should I replace Pressure Sensor?",
        "How many troubleshooting steps are from the Documentum?",
        "Which asset had the most work orders and how many of them?",
        "How do I safeguard against system errors in my Hydraulic Shovels?",
        "Was the transmission tested under load in WO008?",
        "Was the troubleshooting guide for Excavator 3000 followed for WO001 order?",
        "Where is Excavator 3000 located?",
        "What are the previous failures type for Excavator 3000 from work order logs?",
        "Did we replace the Cooling fan on Crusher CR6000?",
        "What component have we replaced the most?",
        "How many Electrical failures has Crusher CR6000 had?",
        "How many Electrical failures have Heavy Machinery Had?",
    ]

    answers = []
    for i, query in enumerate(queries):
        print(f"======{i}======")
        print(f"User: {query}")

        # We are doing single user query, not a conversation
        # so need to reset history for each turn
        pg_agent.reset_messages()
        answer = _process_answer(pg_agent.query(query))
        answers.append(answer)

        print(f"System: {answer}")
        print(f"=====================")

    return answers


def main():
    """Main function to run the puppygraph agent.

    We first run a set of queries and then enter free chat mode.
    """
    pg_agent = PuppyGraphAgent(
        puppy_graph_client=_get_puppy_graph_client("127.0.0.1"),
        llm=_get_llm(),
        chat_prompt_template=_get_chat_prompt_template(
            graph_schema_prompt=_get_graph_schema_prompt()
        ),
    )

    _run_queries(pg_agent=pg_agent)

    print("\n=======Entering Free Chat Mode=======\n")
    pg_agent.reset_messages()
    while True:
        user_input = input("User: ")
        response = pg_agent.query(user_input=user_input)
        print(f"System: {_process_answer(response)}")


if __name__ == "__main__":
    main()
