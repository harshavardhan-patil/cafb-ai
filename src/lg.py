from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)


# Define helper functions to pretty print the messages in the graph while we debug it and to give our tool node error handling (by adding the error to the chat history).
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant"
            " Use the provided tools to search for birthday's if the use asks so"
        ),
        ("placeholder", "{messages}"),
    ]
)

# Define Tool
@tool
def get_bday(query: str)-> str:
    """Fetch the birthday based on the query provided"""
    return "Harsh's bday is 16th nov"

tools = [get_bday]

# Assistant for LLM Node
assistant_runnable = assistant_prompt | llm.bind_tools(tools)

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_edge(START, "assistant")
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# In-memory chat history persistence
memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)



#########
# Usage #
#########
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "1",
    }
}

_printed = set()
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        events = graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        snapshot = graph.get_state(config)
        while snapshot.next:
            human_interrupt = input(
                "Press Y to continue or N to abort"
            )
            if human_interrupt.lower().strip() == "y":
                tool_result = graph.invoke(
                    None,
                    config
                )
                print(tool_result["messages"][-1].content)
            else:
                # handle user says no part
                tool_result = graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config
                )
                print(tool_result["messages"][-1].content)
              
            snapshot = graph.get_state(config)
    except Exception as e:
        print(f"Uh-oh: {e}")
        break