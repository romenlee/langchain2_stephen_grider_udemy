from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv
import sqlite3
from pydantic.v1 import BaseModel
from typing import List
from langchain_core.tools import Tool
from langchain.globals import set_verbose, set_debug
from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen
load_dotenv()

set_verbose(True)
# set_debug(True)

def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n\n\n========= Sending Messages =========\n\n")

        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")

            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}",
                    title=message.type,
                    color="cyan"
                )

            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")

            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="purple")

            else:
                boxen_print(message.content, title=message.type)


conn = sqlite3.connect("db.sqlite")


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)


def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"


class RunQueryArgsSchema(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)


def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)


class DescribeTablesArgsSchema(BaseModel):
    tables_names: List[str]


describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


def write_report(filename, html):
    with open(filename, 'w') as f:
        f.write(html)


class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema
)

handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist. Instead, use the 'describe_tables' function"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [
    run_query_tool,
    describe_tables_tool,
    write_report_tool
]

agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
    memory=memory
)

# agent_executor.invoke(input={"input": "Summarize the top 5 most popular products. Write the results to a report file."})
# agent_executor.invoke(input={"input": "how many users are there?"})
agent_executor.invoke(input={"input": "How many orders are there? Write the result to an html report."})
agent_executor.invoke(input={"input": "Repeat the exact same process for users."})
