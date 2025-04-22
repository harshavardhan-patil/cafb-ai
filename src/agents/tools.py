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
from src.data.jira import connect_to_jira
from src.data.db import connect_to_db
from datetime import datetime
import os
import re



@tool
def close_ticket(issue_key: str, resolution_reason: str, summary: str = None):
    """
    Use this when the partner's issue has been resolved and they confirm the ticket can be closed. Input should be the resolution reason.
    """
    jira = connect_to_jira()
    
    try:
        issue = jira.issue(issue_key)
        
        # Add summary comment if provided
        if summary:
            jira.add_comment(issue, f"Final Summary:\n\n{summary}")
        
        # Add resolution comment
        jira.add_comment(issue, f"Resolution: {resolution_reason}")
        
        # Close the ticket - transition the issue to the "Done" status
        # Note: The exact transition ID/name may vary based on your JIRA workflow
        transitions = jira.transitions(issue)
        close_transition = None
        
        # Find the right transition ID for closing the issue
        for t in transitions:
            if t['name'].lower() in ['close', 'done', 'resolve', 'closed', 'complete']:
                close_transition = t['id']
                break
        
        if close_transition:
            jira.transition_issue(
                issue,
                close_transition
            )
            
            # Update the database record
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE jira_issues
                SET status = 'Closed', resolved_at = NOW(), updated_at = NOW()
                WHERE issue_key = %s
                """,
                (issue_key,)
            )
            conn.commit()
            conn.close()
            
            return  "Ticket {issue_key} has been closed successfully"
        else:
            return  "Could not find a transition to close the ticket"
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to close ticket: {str(e)}"
        }
    
@tool
def update_ticket_priority(issue_key: str, new_priority: str, reason: str):
    """
    Use this to update the priority of a JIRA ticket. For escalations use priority High
    
    Args:
        issue_key: The key of the JIRA issue to update
        new_priority: The new priority level (e.g., "High", "Medium", "Low")
        reason: Reason for escalation
        
    Returns:
        Dict with status and display message
    """
    jira = connect_to_jira()
    
    try:
        issue = jira.issue(issue_key)
        
        # Directly update the priority field
        issue.update(fields={'priority': {'name': new_priority}})
        
        # Add comment explaining the escalation
        comment_body = f"Priority escalated to {new_priority}.\nReason: {reason}"
        jira.add_comment(issue, comment_body)
        
        # Update the database record
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE jira_issues
            SET priority = %s, updated_at = NOW()
            WHERE issue_key = %s
            """,
            (new_priority, issue_key)
        )
        conn.commit()
        conn.close()
        
        return f"✓ Ticket {issue_key} priority updated to {new_priority}"
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to update ticket priority: {str(e)}"
        }


@tool
def add_order_number_to_ticket(issue_key: str, order_number: str, additional_details: str=None):
    """
    Use this to add order number to a JIRA ticket.
    
    Args:
        issue_key: The key of the JIRA issue to update
        order_number: The order number to add
        additional_details: Any additional context about the order
        
    Returns:
        Dict with status and display message
    """
    jira = connect_to_jira()
    
    try:
        issue = jira.issue(issue_key)
        
        # Create comment with order information
        comment_body = f"Order Number: {order_number}"
        if additional_details:
            comment_body += f"\nAdditional Details: {additional_details}"
        
        jira.add_comment(issue, comment_body)
        
        # Add the order number as a label
        current_labels = issue.fields.labels
        if current_labels is None:
            current_labels = []
        
        order_label = f"order-{order_number}"
        if order_label not in current_labels:
            current_labels.append(order_label)
            issue.update(fields={"labels": current_labels})
        
        return f"Order number added to ticket {issue_key}"
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to add order information: {str(e)}"
        }

@tool
def add_conversation_summary_to_ticket(issue_key, conversation_summary):
    """
    Use this to add a summary of the conversation to a JIRA ticket.
    
    Args:
        issue_key: The key of the JIRA issue to update
        conversation_summary: Summary of the conversation
        
    Returns:
        Dict with status and display message
    """
    jira = connect_to_jira()
    
    try:
        issue = jira.issue(issue_key)
        
        # Format the conversation summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment_body = f"Conversation Summary ({timestamp}):\n\n{conversation_summary}"
        
        jira.add_comment(issue, comment_body)
        
        return f"✓ Conversation summary added to ticket {issue_key}"
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to add conversation summary: {str(e)}"
        }
    



# Helper functions to pretty print the messages in the graph while we debug it and to give our tool node error handling (by adding the error to the chat history).
# todo
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