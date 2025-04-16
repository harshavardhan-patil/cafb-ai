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
from src.utils.helpers import prettify_category
from src.utils.constants import CATEGORIES
from src.utils.constants import JIRA_FIELD_MAPPING
from langchain.tools import Tool
from src.data.jira import (
    update_ticket_priority, 
    add_order_information, 
    add_conversation_summary,
    extract_order_number,
    close_ticket
)

load_dotenv()

model = os.getenv("MODEL")
project_key = os.getenv("PROJECT_KEY")

placeholder_option = "-- Select Issue Category --"
category_list = [placeholder_option] + list(CATEGORIES.keys())

def setup_jira_tools(issue_key):
    """
    Create tools for updating JIRA tickets.
    
    Args:
        issue_key: The JIRA issue key to operate on
        
    Returns:
        A list of tools for the agent to use
    """
    escalate_tool = Tool(
        name="EscalateTicket",
        func=lambda x: update_ticket_priority(
            issue_key=issue_key,
            new_priority="High",
            reason=x
        ),
        description="Use this when the partner requests escalation or the issue is urgent. Input should be the reason for escalation."
    )
    
    add_order_tool = Tool(
        name="AddOrderInfo",
        func=lambda x: add_order_information(
            issue_key=issue_key,
            order_number=extract_order_number(x) or "Unknown",
            additional_details=x
        ),
        description="Use this when the partner provides order number or order details. Input should be the complete text with order information."
    )
    
    add_summary_tool = Tool(
        name="AddConversationSummary",
        func=lambda x: add_conversation_summary(
            issue_key=issue_key,
            conversation_summary=x
        ),
        description="Use this at the end of the conversation to add a summary to the ticket. Input should be the summary text."
    )
    
    close_ticket_tool = Tool(
        name="CloseTicket",
        func=lambda x: close_ticket(
            issue_key=issue_key,
            resolution_reason=x
        ),
        description="Use this when the partner's issue has been resolved and they confirm the ticket can be closed. Input should be the resolution reason."
    )
    
    return [escalate_tool, add_order_tool, add_summary_tool, close_ticket_tool]


# Initialize session states if they don't exist
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'issue_key' not in st.session_state:
    st.session_state.issue_key = ""
if 'issue_description' not in st.session_state:
    st.session_state.issue_description = ""
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False

# Function to handle option selection
def handle_option_selection(option):
    st.session_state.selected_option = option
    st.session_state.chat_started = False  # Reset chat when option changes

# Function to start chat session
def start_chat():
    st.session_state.chat_started = True
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

# Chat interface - only show when a chat has been started
if st.session_state.chat_started:
    llm = ChatOllama(
        model="gemma3",
    )
    
    # Set up memory
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
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

        Ticket Context: {issue_context}
         
        ##############
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

        ##############
        Knowledge Base: {issue_kb}
        """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Display ticket/issue info at the top of chat
    if st.session_state.selected_option == "existing":
        st.info(f"Discussing ticket: {st.session_state.issue_key}")
    
    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        if msg.type == 'AIMessageChunk':
            st.chat_message('ai').write(msg.content)
        else:
            st.chat_message(msg.type).write(msg.content)
    
    # If user inputs a new prompt, generate and draw a new response
    if user_input := st.chat_input("How can I help?"):
        st.chat_message("human").write(user_input)
        # New messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        st.chat_message('ai').write_stream(chain_with_history.stream({"question": user_input}, config))
         