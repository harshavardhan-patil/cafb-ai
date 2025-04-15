import os
import streamlit as st
import openai
import json
import psycopg2
from psycopg2 import sql
from openai import OpenAI
import ollama
import requests
from requests.auth import HTTPBasicAuth
from src.utils.constants import MODEL_OPENAI
from dotenv import load_dotenv
from pathlib import Path
from src.data.rag import retrieve_relevant_context
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from src.data.jira import check_ticket_exists, get_issue_context

# Please set the path to your .env file 
env_path = Path("/Users/pragatirao/AI_Food_Insecurity_Case/cafb-ai/.env")
load_dotenv(dotenv_path=env_path, override=True)

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

def store_ticket_postgres(jira_data):
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO jira_issues (
                issue_key, summary, status, created_at, updated_at, issue_type, project
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
        """

        cursor.execute(insert_query, (
            jira_data["key"],
            jira_data["fields"]["summary"],
            jira_data["fields"]["status"]["name"],
            jira_data["fields"]["created"],
            jira_data["fields"]["updated"],
            jira_data["fields"]["issuetype"]["name"],
            jira_data["fields"]["project"]["key"]
        ))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Jira issue '{jira_data['key']}' stored in DB.")

    except Exception as e:
        import traceback
        print("❌ Failed to insert into DB:")
        traceback.print_exc()

# Helper function to extract fields from ADF description
def extract_from_description(adf, label):
    try:
        for block in adf["content"]:
            if "text" in block["content"][0]:
                text = block["content"][0]["text"]
                if text.startswith(f"{label}:"):
                    return text.split(f"{label}:")[1].strip()
    except:
        pass
    return None

# Function to classify issue
def classify_issue(issue_description):
    # Set up a simple rule-based system to determine priority
    high_priority_keywords = ['emergency', 'urgent', 'critical', 'immediate', 'shutdown']
    low_priority_keywords = ['inquiry', 'clarification', 'educational', 'optional']
    
    description_lower = issue_description.lower()
    priority = "Medium"  # Default priority

    if any(word in description_lower for word in high_priority_keywords):
        priority = "High"
    elif any(word in description_lower for word in low_priority_keywords):
        priority = "Low"

    return priority

# Function to create a Jira issue
def create_jira_issue(summary, description, category, subcategory, priority, order_number):
    """Creates a Jira issue in ADF format with classification info"""

    # Load environment variables
    jira_domain = os.getenv("JIRA_DOMAIN")
    project_key = os.getenv("PROJECT_KEY")
    jira_user = os.getenv("JIRA_EMAIL")
    jira_token = os.getenv("JIRA_API_TOKEN")
    
    url = f"{jira_domain}/rest/api/3/issue"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    description_adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": f"Issue Description: {description}"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": f"Category: {category}"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": f"Subcategory: {subcategory}"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": f"Priority: {priority}"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": f"Order Number: {order_number}"}]}
        ]
    }

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": description_adf,
            "issuetype": {"name": "Story"},
            "priority": {"name": priority}
        }
    }

    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(jira_user, jira_token),
        json=payload,
    )

    if response.status_code == 201:
        issue_key = response.json().get("key")
        issue_url = f"{jira_domain}/browse/{issue_key}"

        full_issue_response = requests.get(
            f"{jira_domain}/rest/api/3/issue/{issue_key}",
            headers=headers,
            auth=HTTPBasicAuth(jira_user, jira_token)
        )
        full_issue_data = full_issue_response.json()
        store_ticket_postgres(full_issue_data)

        return f"✅ Jira issue created successfully: [{issue_key}]({issue_url})"
    else:
        return f"❌ Failed to create issue: {response.status_code} - {response.text}"

# Function to start chat session
def start_chat():
    st.session_state.chat_started = True
    # Clear previous chat messages
    st.session_state.messages = []

categories = {
    'order-modifications': ['item-additions', 'item-removals', 'quantity-adjustments', 'late-modification-requests'],
    'order-cancellations': ['standard-cancellations', 'urgent-cancellations', 'rescheduled-orders'],
    'delivery-issues': ['late-deliveries', 'missed-deliveries', 'incomplete-deliveries', 'damaged-goods', 'delivery-confirmation-issues'],
    'pickup-scheduling-&-rescheduling': ['new-pickup-requests', 'rescheduling-pickup', 'missed-pickups', 'pickup-policy-clarifications'],
    'product-availability-&-substitutions': ['stock-availability-inquiries', 'out-of-stock-notifications', 'product-substitution-requests', 'special-item-requests'],
    'grant-&-billing-issues': ['grant-fund-usage', 'incorrect-grant-deduction', 'billing-discrepancies', 'payment-&-credit-issues'],
    'training-&-account-access': ['training-signups', 'missed-training-sessions', 'login-issues', 'new-user-account-requests'],
    'emergency-situations': ['weather-related-disruptions', 'personal/organization-emergencies', 'food-safety-concerns', 'unexpected-facility-closures'],
    'special-requests': ['educational-materials', 'large-event-orders', 'holiday-&-seasonal-adjustments'],
    'technical-support': ['website-&-ordering-system-errors', 'email-&-communication-issues', 'data-entry-mistakes', 'general-it-assistance'],
    'other': ['miscellaneous']
}

placeholder_option = "-- Select Issue Category --"
category_list = [placeholder_option] + list(categories.keys())

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

# Handle new issue workflow
elif st.session_state.selected_option == "new":
    st.markdown("### New Issue Details")
    st.markdown("### Please enter the following details:")

    customer_name = st.text_input("Customer Name")
    order_number = st.text_input("Order Number")

    # ✅ Category selection outside the form
    selected_category = st.selectbox("Issue Category", category_list, key="category_selection")

    selected_subcategory = None
    if selected_category != placeholder_option:
        selected_subcategory = st.selectbox("Issue Subcategory", categories[selected_category], key="subcategory_selection")
    else:
        st.info("Please select an issue category to continue.")

    with st.form("new_issue_form"):
        st.session_state.issue_description = st.text_area("Brief Description")

        submit_button = st.form_submit_button("Create Issue")

        if submit_button:
            if selected_category == placeholder_option or not selected_subcategory:
                st.warning("🚨 Please select both an issue category and subcategory.")
            elif not st.session_state.issue_description:
                st.warning("🚨 Please enter a brief description.")
            else:
                try:
                    category, subcategory = selected_category, selected_subcategory
                    priority = classify_issue(st.session_state.issue_description)

                    response_msg = create_jira_issue(
                        summary=f"{category}: {subcategory}",
                        description=st.session_state.issue_description,
                        category=category,
                        subcategory=subcategory,
                        priority=priority,
                        order_number=order_number
                        )

                    st.success(response_msg)
                    start_chat()

                except Exception as e:
                    st.error(f"❌ Failed to classify or create issue: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())

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
        system_prompt = f"""
        You are a frontdesk assistant for Capital Area Food Bank (CAFB). 
        A partner is inquiring about an existing ticket with ID: {st.session_state.issue_key}.
        
        Try to provide helpful context and solutions related to this ticket.
        Use the information retrieved from the Jira database to answer questions.
        If the answer cannot be found in the context or if no context is given, say so clearly and suggest how the user might refine their question. 
        DO NOT MAKE UP OR SIMULATE INFORMATION.

        Ticket Context: {issue_context}
        """
    else:
        system_prompt = f"""
        You are a frontdesk assistant for Capital Area Food Bank (CAFB). 
        A partner is creating a new issue of type: {selected_category}.
        Their initial description is: {st.session_state.issue_description}

        This issue has been classified as:
        - Category: {selected_category}
        - Subcategory: {selected_subcategory}
        - Priority: {priority}

        Help them by gathering more information or helping resolve it.
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
        st.info(f"Creating new {selected_category} issue: {st.session_state.issue_description[:50]}...")
    
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
         
