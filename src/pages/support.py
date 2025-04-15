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
from src.data.jira import check_ticket_exists, get_issue_context, get_issue_kb

load_dotenv()

model = os.getenv("MODEL")


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
        st.session_state.selected_option = None
        st.rerun()

# todo Handle new issue workflow
elif st.session_state.selected_option == "new":
    st.markdown("### New Issue Details")
    
    with st.form("new_issue_form"):
        issue_type = st.selectbox(
            "Issue Type", 
            ["Question", "Problem/Bug", "Feature Request", "Delivery Issue", "Other"]
        )
        st.session_state.issue_description = st.text_area("Brief Description")
        submit_button = st.form_submit_button("Create Issue")
        
        if submit_button and st.session_state.issue_description:
            # Here you would create a new ticket in Jira
            # For now, we'll just start the chat
            start_chat()
    
    # Button to go back to main menu
    if st.button("Back to Main Menu"):
        st.session_state.selected_option = None
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
        Use the Ticket Context and Knowledge Base to answer questions
        If the answer cannot be found in the context or if no context is given, say so clearly and suggest how the user might refine their question. 
        DO NOT MAKE UP OR SIMULATE INFORMATION.

        Ticket Context: {issue_context}
         
        ##############
        Knowledge Base: {issue_kb}
        """
        
    else:  # todo new issue flow
        system_prompt = f"""
        You are a frontdesk assistant for Capital Area Food Bank (CAFB). 
        A partner is creating a new issue of type: {issue_type}.
        Their initial description is: {st.session_state.issue_description}
        
        Help them by gathering more information about their issue so it can be properly addressed.
        Use the information retrieved from the Jira database to answer questions.
        If the answer cannot be found in the context or if no context is given, say so clearly 
        and suggest how the user might refine their question. 
        DO NOT MAKE UP OR SIMULATE INFORMATION.
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
    else:
        st.info(f"Creating new {issue_type} issue: {st.session_state.issue_description[:50]}...")
    
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
         