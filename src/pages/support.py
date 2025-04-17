import os
import streamlit as st
import openai
from openai import OpenAI
import ollama
from src.utils.constants import MODEL_OPENAI
from dotenv import load_dotenv
from src.data.rag import retrieve_relevant_context
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from src.data.jira import check_ticket_exists, get_issue_context, get_issue_kb, classify_issue, create_and_store_issue
from src.utils.helpers import prettify_category, prettify_tool
from src.utils.constants import CATEGORIES
from src.utils.constants import JIRA_FIELD_MAPPING
from langchain.tools import Tool
from src.agents.tools import close_ticket
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.agents.agent import State, Assistant
from src.agents.tools import create_tool_node_with_fallback
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from src.agents.tools import close_ticket, update_ticket_priority, add_order_number_to_ticket, add_conversation_summary_to_ticket
import uuid
import logging


# Set logging level (DEBUG, INFO, WARNING, ERROR)
logging.basicConfig(level=logging.INFO)


load_dotenv()

model = os.getenv("MODEL")
project_key = os.getenv("PROJECT_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", 
                api_key=OPENAI_API_KEY,
                timeout=5,
                max_retries=2,)

placeholder_option = "-- Select Issue Category --"
category_list = [placeholder_option] + list(CATEGORIES.keys())

# Initialize session states if they don't exist
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'issue_key' not in st.session_state:
    st.session_state.issue_key = ""
if 'issue_description' not in st.session_state:
    st.session_state.issue_description = ""
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False
if 'setup_graph' not in st.session_state:
    st.session_state.setup_graph = False
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'memory' not in st.session_state:
    st.session_state.memory = None

# Function to handle option selection
def handle_option_selection(option):
    st.session_state.selected_option = option
    st.session_state.chat_started = False  # Reset chat when option changes

# Function to start chat session
def start_chat():
    st.session_state.chat_started = True
    st.session_state.setup_graph = True
    # Clear previous chat messages
    st.session_state.messages = []

# Main selection interface
if st.session_state.selected_option is None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Existing Ticket", use_container_width=True):
            handle_option_selection("existing")
    
    with col2:
        if st.button("New Issue", use_container_width=True):
            handle_option_selection("new")

# Handle existing ticket workflow
elif st.session_state.selected_option == "existing":
    st.markdown("### Ticket Information")
    
    with st.form("ticket_form"):
        st.session_state.issue_key = st.text_input("Enter Ticket ID/Key (e.g., CAFBSS-123)")
        submit_button = st.form_submit_button("Find Ticket")
        
        if submit_button and st.session_state.issue_key:
            if not check_ticket_exists(st.session_state.issue_key):
                st.warning('Please enter a valid/existing ticket', icon="⚠️")
            else:    
                start_chat()
    
    # Button to go back to main menu
    if st.button("Back to Main Menu"):
        # Clear all relevant session state variables
        for key in list(st.session_state.keys()):
            if key in ['selected_option', 'issue_key', 'issue_description', 'chat_started', 
                    'langchain_messages', 'messages']:
                del st.session_state[key]
        
        # Initialize them with default values
        st.session_state.selected_option = None
        st.session_state.issue_key = ""
        st.session_state.issue_description = ""
        st.session_state.chat_started = False
        
        # Force refresh the page
        st.rerun()

# Handle new issue workflow
elif st.session_state.selected_option == "new":
    st.markdown("#### Please enter the following details to get started:")

    partner_name = st.text_input("Partner Name")
    
    selected_category = st.selectbox("Issue Category", category_list, key="category_selection")
    selected_subcategory = None
    if selected_category != placeholder_option:
        selected_subcategory = st.selectbox("Issue Specifics", CATEGORIES[selected_category], key="subcategory_selection")
    else:
        st.info("Please select an issue category to continue.")

    with st.form("new_issue_form"):
        st.session_state.issue_description = st.text_area("Brief Description")

        submit_button = st.form_submit_button("Create Issue")

        if submit_button:
            if selected_category == placeholder_option or not selected_subcategory:
                st.warning("Please select both an issue category and subcategory.")
            elif not st.session_state.issue_description:
                st.warning("Please enter a brief description.")
            else:
                try:
                    category, subcategory = prettify_category(selected_category), prettify_category(selected_subcategory)
                    priority = classify_issue(st.session_state.issue_description)

                    issue_dict = {
                        "project": project_key,
                        "summary": f"{category}/{subcategory}",
                        "description": st.session_state.issue_description,
                        "issue_type": "Story",
                        "priority": priority,
                        "main_category": selected_category,
                        "sub_category": selected_subcategory,
                        "partner_names": partner_name,
                    }
                    response_dict = create_and_store_issue(issue_dict)

                    if response_dict['status'] == 200 or response_dict['status'] == 201:    
                        st.success(response_dict['display'])
                        st.session_state.issue = response_dict["new_issue"]
                        st.session_state.issue_key = st.session_state.issue.key
                        start_chat()
                    else:
                        st.error(response_dict['display'])

                except Exception as e:
                    st.error(f"Failed to classify or create issue: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
    
    # Button to go back to main menu
    if st.button("Back to Main Menu"):
        # Clear all relevant session state variables
        for key in list(st.session_state.keys()):
            if key in ['selected_option', 'issue_key', 'issue_description', 'chat_started', 
                    'langchain_messages', 'messages']:
                del st.session_state[key]
        
        # Initialize them with default values
        st.session_state.selected_option = None
        st.session_state.issue_key = ""
        st.session_state.issue_description = ""
        st.session_state.chat_started = False
        
        # Force refresh the page
        st.rerun()


# Set a unique key for the button to avoid conflicts
APPROVE_BUTTON_KEY = "approve_tool_button"
DENY_BUTTON_KEY = "deny_tool_button"


if 'awaiting_tool_confirmation' not in st.session_state:
    st.session_state.awaiting_tool_confirmation = False
if 'tool_call_id' not in st.session_state:
    st.session_state.tool_call_id = None
if 'tool_description' not in st.session_state:
    st.session_state.tool_description = ""
if 'tool_approved' not in st.session_state:
    st.session_state.tool_approved = False
if 'tool_denied' not in st.session_state:
    st.session_state.tool_denied = False

# Setup LangGraph and Memory as session variables for persistence
if st.session_state.chat_started and st.session_state.setup_graph:
    llm = get_llm()  
    # If a partner confirms that the ticket has been resolved, you should invoke the close_ticket tool
    # Create appropriate system prompt based on context
    if st.session_state.selected_option == "existing":
        issue_context = get_issue_context(st.session_state.issue_key).replace("{", "{{").replace("}", "}}")
        issue_kb = get_issue_kb(st.session_state.issue_key).replace("{", "{{").replace("}", "}}")
        system_prompt = f"""
        You are a frontdesk assistant for Capital Area Food Bank (CAFB). 
        A partner is inquiring about an existing ticket with ID: {st.session_state.issue_key}.
        
        Try to provide helpful context and solutions related to this ticket.
        Use the Ticket Context and Knowledge Base to answer questions.
        The Knowledege Base provides helpful examples but does not have LIVE data (for example, Kowledge Base does not have current schedules, simply examples)
        If the answer cannot be found in the context or if no context is given, say so clearly and suggest how the user might refine their question. 
        DO NOT MAKE UP OR SIMULATE INFORMATION.

        You also have access to tools. 
        1. If the customer want to escalate the issue, or if they say its urgent, you should call the update_ticket_priority tool.
        2. If you need to update the ticket with the order number, use add_order_number_to_ticket
        3. At the end of the conversation, you should ask the partner if their issue is resolved and if they would like to close the ticket. If their issue is resolved you should call close_ticket tool.
        4. Regardless of the outcome, you should call add_conversation_summary_to_ticket tool at the end of the conversation.
        
        #############
        Ticket Context: {issue_context}
         
        #############
        Knowledge Base: {issue_kb}
        """
        
    else: 
        issue_kb = get_issue_kb(st.session_state.issue_key).replace("{", "{{").replace("}", "}}")
        system_prompt = f"""
        You are a frontdesk assistant for Capital Area Food Bank (CAFB). 
        A partner has created a new issue with the initial description: {st.session_state.issue_description}
        
        Help them by gathering more information about their issue so it can be properly addressed.
        Use the Ticket Context and Knowledge Base to answer questions.
        The Knowledege Base provides helpful examples but does not have LIVE data (for example, Kowledge Base does not have current schedules, simply examples)
        If the answer cannot be found in the context or if no context is given, say so clearly and suggest how the user might refine their question.

        You also have access to tools. 
        1. If the customer want to escalate the issue, or if they say its urgent, you should call the update_ticket_priority tool.
        2. If you need to update the ticket with the order number, use add_order_number_to_ticket
        3. At the end of the conversation, you should ask the partner if their issue is resolved and if they would like to close the ticket. If their issue is resolved you should call close_ticket tool.
        4. Regardless of the outcome, you should call add_conversation_summary_to_ticket tool at the end of the conversation.
        
        ##############
        Knowledge Base: {issue_kb}
        """
    
    assistant_prompt = ChatPromptTemplate.from_messages([
        ("system",
        system_prompt),
        ("placeholder", 
         "{messages}"),
    ])

    # Add tools
    tools = [close_ticket, update_ticket_priority, add_order_number_to_ticket, add_conversation_summary_to_ticket]
    # Assistant for LLM Node
    assistant_runnable = assistant_prompt | llm.bind_tools(tools)

    # Build Graph
    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_edge(START, "assistant")
    builder.add_node("tools", create_tool_node_with_fallback(tools))
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # In-memory chat history persistence
    st.session_state.memory = MemorySaver()
    # Compile graph
    st.session_state.graph = builder.compile(
        checkpointer=st.session_state.memory,
        interrupt_before=["tools"]
    )

    st.session_state.setup_graph = False

#################
# Chat Inteface #
#################
if st.session_state.chat_started and not st.session_state.setup_graph:  
    # Display ticket/issue info at the top of chat
    st.info(f"Discussing ticket: {st.session_state.issue_key}")

    config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "1",
        }
    }

    # Function to process user input
    def process_user_input(user_input):
        # Invoke the graph with user input
        events = st.session_state.graph.invoke(
            {"messages": ("user", user_input)}, 
            config, 
            stream_mode="values"
        )
        
        # Check if we need to handle tool calls
        snapshot = st.session_state.graph.get_state(config)
        
        if snapshot.next and "tool_calls" in events["messages"][-1].__dict__:
            # We have a tool call that needs confirmation
            st.session_state.awaiting_tool_confirmation = True
            tool_calls = events["messages"][-1].tool_calls
            
            if tool_calls and len(tool_calls) > 0:
                st.session_state.tool_call_id = tool_calls[0]["id"]
                
                # Get tool description for user to confirm
                tool_name = tool_calls[0]["name"]
                tool_args = tool_calls[0]["args"]
                
                st.session_state.tool_description = prettify_tool(tool_name)
                
                # Don't add the assistant's message yet since we're waiting for confirmation
                return None
        
        # If we're not handling a tool call, return the assistant's message
        return events["messages"][-1].content
    # Callback functions that directly update the UI without a page refresh
    def handle_tool_approval():
        """Callback function for approving tool usage without page refresh"""
        try:
            # Get the current state snapshot
            snapshot = st.session_state.graph.get_state(config)
            
            # Continue the graph execution with empty input (since the state already has what it needs)
            # But we need to make sure we pass the expected 'messages' input even if it's empty
            tool_result = st.session_state.graph.invoke(None, config)
            
            assistant_response = tool_result["messages"][-1].content
            
            # Update the chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Reset approval state
            st.session_state.awaiting_tool_confirmation = False
            st.session_state.tool_call_id = None
            st.session_state.tool_description = ""
        except Exception as e:
            st.error(f"Error during tool execution: {str(e)}")

    def handle_tool_denial():
        """Callback function for denying tool usage without page refresh"""
        try:
            # Continue the graph but with a denial message
            # We need to include both the required 'messages' input and our tool message
            tool_result = st.session_state.graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=st.session_state.tool_call_id,
                            content="API call denied by user. Continue assisting without using the tool.",
                        )
                    ]
                },
                config
            )
            
            assistant_response = tool_result["messages"][-1].content
            
            # Update chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Reset state
            st.session_state.awaiting_tool_confirmation = False
            st.session_state.tool_call_id = None
            st.session_state.tool_description = ""
        except Exception as e:
            st.error(f"Error handling tool denial: {str(e)}")


    # Initialize or get chat history from session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    if st.session_state.awaiting_tool_confirmation:
        st.info(f"The assistant wants to {st.session_state.tool_description}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve", key=APPROVE_BUTTON_KEY, on_click=handle_tool_approval):
                pass
        
        with col2:
            if st.button("Deny", key=DENY_BUTTON_KEY, on_click=handle_tool_denial):
                pass
    else:
        # Get user input if not awaiting tool confirmation
        if user_input := st.chat_input("How can I help?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "human", "content": user_input})
            
            # Process the user input
            assistant_response = process_user_input(user_input)
            
            # Add assistant response to chat history if not awaiting tool confirmation
            if assistant_response and not st.session_state.awaiting_tool_confirmation:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Force refresh to show new messages
            st.rerun()