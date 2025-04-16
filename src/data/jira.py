import os
import json
import re
from datetime import datetime
from jira import JIRA
import psycopg2
import requests
from requests.auth import HTTPBasicAuth
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from src.data.db import connect_to_db

load_dotenv()

# Jira connection parameters
jira_url = os.getenv("JIRA_URL")
jira_email = os.getenv("JIRA_EMAIL")
jira_api_token = os.getenv("JIRA_API_TOKEN")
project_key = os.getenv("PROJECT_KEY")


def connect_to_jira():
    """Establish connection to Jira."""
    return JIRA(
        server=jira_url,
        basic_auth=(jira_email, jira_api_token)
    )

def parse_time_string(time_str):
    """Parse time string in the format HH:MM:SS into minutes."""
    if not time_str:
        return None
    
    try:
        parts = time_str.split(':')
        if len(parts) == 2:  # HH:MM
            hours, minutes = map(int, parts)
            return hours * 60 + minutes
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 60 + minutes + (seconds // 60)
        else:
            return None
    except (ValueError, TypeError):
        return None

def check_ticket_exists(issue_key):
    """Check if a ticket with the given key exists in the database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT COUNT(*) 
        FROM jira_issues
        WHERE issue_key = %s
        """,
        (issue_key,)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] > 0

def get_issue_context(issue_key):
    conn = connect_to_db()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute("""
    SELECT 
    issue_key, summary, description, region, status, created_at, updated_at, resolved_at, assignee, reporter, reporter_email, issue_type, priority, labels, project, components, affects_versions, fix_versions, resolution, votes, remaining_estimate, time_spent, original_estimate, rank, main_category, sub_category, partner_names, relevant_departments, request_category, request_type, request_language, resolution_action, source, time_to_first_response, time_to_resolution, status_category_changed, date_of_first_response
    FROM jira_issues
    WHERE issue_key = %s
    """,
    (str(issue_key),)
    )

    issue = dict(cursor.fetchone())
    cursor.execute("""
    SELECT body FROM jira_comments        
    WHERE issue_key = %s
    """,
    (str(issue_key),)
    )

    comments = cursor.fetchall()
    conn.close()

    # todo need bigger context window for comments OR cleaned comments
    #issue['Comments'] = comments
    return json.dumps(dict(issue), default=str, indent=2)


def get_issue_info(issue_key):
    """Get basic information about a Jira issue."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT summary, status
        FROM jira_issues
        WHERE issue_key = %s
        """,
        (issue_key,)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "summary": result[0],
            "status": result[1]
        }
    else:
        return {
            "summary": "Unknown issue",
            "status": "Unknown"
        }

def get_comment_info(comment_id):
    """Get basic information about a Jira comment."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT issue_key, author
        FROM jira_comments
        WHERE id = %s
        """,
        (comment_id,)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "issue_key": result[0],
            "author": result[1]
        }
    else:
        return {
            "issue_key": "Unknown",
            "author": "Unknown"
        }


def get_issue_kb(issue_key):
    """Get knowledge base for handling issue"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT main_category, sub_category
        FROM jira_issues
        WHERE issue_key = %s
        """,
        (issue_key,)
    )
    
    result = cursor.fetchone()
    
    cursor.execute(
        """
        SELECT kb
        FROM knowledge_base
        WHERE 
        main_category = %s AND sub_category = %s
        """,
        (result[0], result[1])
    )

    result = cursor.fetchone()
    conn.close
    return result[0]

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
def create_and_store_issue(issue_dict):
    """
    Create a new issue in Jira and store it in the database.
    
    Args:
        issue_dict: Dictionary containing all the fields for the new issue
        
    Returns:
        The key of the created issue
    """
    # Connect to Jira and database
    jira = connect_to_jira()
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Prepare fields dictionary for Jira issue creation
    jira_fields = {
        'project': {'key': issue_dict['project']},
        'summary': issue_dict['summary'],
        'description': issue_dict['description'],
        'issuetype': {'name': issue_dict['issue_type']},
    }

    # todo: no way to have a unauthenticated reporter
    if issue_dict.get('reporter_name') and issue_dict.get('reporter_email'):
        # For Jira Cloud, use accountId if available
        # For Jira Server, use name/emailAddress
        try:
            # First try to find the user by email
            users = jira.search_users(query=issue_dict['reporter_email'])
            if users:
                # If found by email, use the first match
                if hasattr(users[0], 'accountId'):
                    jira_fields['reporter'] = {'accountId': users[0].accountId}
                else:
                    jira_fields['reporter'] = {'name': users[0].name}
            else:
                # If not found by email, try by name
                users = jira.search_users(query=issue_dict['reporter_name'])
                if users:
                    if hasattr(users[0], 'accountId'):
                        jira_fields['reporter'] = {'accountId': users[0].accountId}
                    else:
                        jira_fields['reporter'] = {'name': users[0].name}
                else:
                    # If user not found, use the current authenticated user
                    myself = jira.myself()
                    if hasattr(myself, 'accountId'):
                        jira_fields['reporter'] = {'accountId': myself['accountId']}
                    else:
                        jira_fields['reporter'] = {'name': myself['name']}
        except Exception as e:
            # If any error occurs while setting reporter, use the current user
            try:
                myself = jira.myself()
                if hasattr(myself, 'accountId'):
                    jira_fields['reporter'] = {'accountId': myself['accountId']}
                else:
                    jira_fields['reporter'] = {'name': myself['name']}
            except:
                # If we can't even get current user, let JIRA handle it
                pass
    else:
        # If no reporter info is provided, use the current authenticated user
        try:
            myself = jira.myself()
            if hasattr(myself, 'accountId'):
                jira_fields['reporter'] = {'accountId': myself['accountId']}
            else:
                jira_fields['reporter'] = {'name': myself['name']}
        except Exception as e:
            pass
            # Let JIRA handle the reporter field in this case
    
    # Add optional fields if provided
    if issue_dict.get('assignee'):
        jira_fields['assignee'] = {'name': issue_dict['assignee']}
    
    if issue_dict.get('priority'):
        jira_fields['priority'] = {'name': issue_dict['priority']}
    
    if issue_dict.get('labels'):
        jira_fields['labels'] = issue_dict['labels']
    
    if issue_dict.get('components'):
        jira_fields['components'] = [{'name': c} for c in issue_dict['components']]
    
    # Add custom fields
    if issue_dict.get('region'):
        jira_fields['customfield_10103'] = issue_dict['region']
    
    if issue_dict.get('main_category'):
        jira_fields['customfield_10101'] = [issue_dict['main_category']]
    
    if issue_dict.get('sub_category'):
        jira_fields['customfield_10096'] = [issue_dict['sub_category']]
    
    if issue_dict.get('request_category'):
        jira_fields['customfield_10098'] = issue_dict['request_category']
    
    if issue_dict.get('partner_names'):
        jira_fields['customfield_10108'] = issue_dict['partner_names']
    
    if issue_dict.get('request_language'):
        jira_fields['customfield_10109'] = issue_dict['request_language']
    
    if issue_dict.get('source'):
        jira_fields['customfield_10111'] = issue_dict['source']
    
    if issue_dict.get('relevant_departments'):
        jira_fields['customfield_10104'] = issue_dict['relevant_departments']
    
    # Create the issue in Jira
    try:
        new_issue = jira.create_issue(fields=jira_fields)
        print(f"Created new issue {new_issue.key}")
        
        # Update the issue_dict with the new key and any system-generated fields
        issue_dict['issue_key'] = new_issue.key
        issue_dict['created_at'] = new_issue.fields.created
        issue_dict['updated_at'] = new_issue.fields.updated
        issue_dict['status'] = new_issue.fields.status.name
        issue_dict['reporter'] = new_issue.fields.reporter.displayName if new_issue.fields.reporter else None
        issue_dict['reporter_email'] = new_issue.fields.reporter.emailAddress if new_issue.fields.reporter else None
        issue_dict['raw_data'] = json.dumps(new_issue.raw)
        
        # Prepare the issue for database insertion
        db_record = (
            issue_dict['issue_key'],
            issue_dict['summary'],
            issue_dict['description'],
            issue_dict['status'],
            issue_dict['created_at'],
            issue_dict['updated_at'],
            issue_dict.get('resolved_at'),
            issue_dict.get('assignee'),
            issue_dict.get('reporter'),
            issue_dict.get('reporter_email'),
            issue_dict['issue_type'],
            issue_dict.get('priority'),
            issue_dict.get('labels', []),
            issue_dict['project'],
            issue_dict.get('components', []),
            issue_dict.get('affects_versions', []),
            issue_dict.get('fix_versions', []),
            issue_dict.get('resolution'),
            issue_dict.get('votes', 0),
            issue_dict.get('remaining_estimate'),
            issue_dict.get('time_spent'),
            issue_dict.get('original_estimate'),
            issue_dict.get('rank'),
            issue_dict.get('main_category'),
            issue_dict.get('sub_category'),
            issue_dict.get('partner_names'),
            issue_dict.get('relevant_departments'),
            issue_dict.get('request_category'),
            issue_dict.get('request_type'),
            issue_dict.get('request_language'),
            issue_dict.get('resolution_action'),
            issue_dict.get('source'),
            issue_dict.get('time_to_first_response'),
            issue_dict.get('time_to_resolution'),
            issue_dict.get('status_category_changed'),
            issue_dict.get('date_of_first_response'),
            issue_dict['raw_data']
        )
        
        # Insert issue into database
        cursor.execute(
            """
            INSERT INTO jira_issues 
            (issue_key, summary, description, status, created_at, updated_at, 
             resolved_at, assignee, reporter, reporter_email, issue_type, priority, 
             labels, project, components, affects_versions, fix_versions, resolution, 
             votes, remaining_estimate, time_spent, original_estimate, rank, 
             main_category, sub_category, partner_names, relevant_departments, 
             request_category, request_type, request_language, resolution_action, 
             source, time_to_first_response, time_to_resolution, status_category_changed, 
             date_of_first_response, raw_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            db_record
        )
        
        conn.commit()
        print(f"Stored issue {new_issue.key} in database")
        
        return {"status": 200, 
                "display": f"Please save this ticket number for future reference: [{new_issue.key}]({new_issue.permalink()})",
                "new_issue": new_issue}
        
    except Exception as e:
        conn.rollback()
        print(f"Error creating issue: {str(e)}")
        return {"status": 500, 
                "display": f"❌ Failed to create issue: {str(e)}"}
    finally:
        conn.close()


def get_field_value(issue, field_name):
    """
    Get a field value from a JIRA issue, handling nested fields and custom field name
    To get a field using the readable name:
    value = get_field_value(issue, JIRA_FIELD_MAPPING["region"])
    Args:
        issue: The JIRA issue object
        field_name: The field name, possibly with dots for nested fields
        
    Returns:
        The field value or None if not found
    """
    if '.' not in field_name:
        # Direct field access
        return getattr(issue.fields, field_name, None)
    else:
        # Nested field access
        parts = field_name.split('.')
        value = issue.fields
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
            if value is None:
                return None
        return value


def update_ticket_priority(issue_key, new_priority, reason):
    """
    Update the priority of a JIRA ticket.
    
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
        
        # Update priority
        jira.transition_issue(
            issue,
            transition="Update Priority", 
            fields={'priority': {'name': new_priority}}
        )
        
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
        
        return {
            "status": 200,
            "display": f"✓ Ticket {issue_key} priority updated to {new_priority}"
        }
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to update ticket priority: {str(e)}"
        }

def add_order_information(issue_key, order_number, additional_details=None):
    """
    Add order information to a JIRA ticket.
    
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
        
        return {
            "status": 200,
            "display": f"✓ Order information added to ticket {issue_key}"
        }
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to add order information: {str(e)}"
        }

def add_conversation_summary(issue_key, conversation_summary):
    """
    Add a summary of the conversation to a JIRA ticket.
    
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
        
        return {
            "status": 200,
            "display": f"✓ Conversation summary added to ticket {issue_key}"
        }
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to add conversation summary: {str(e)}"
        }
    
def close_ticket(issue_key, resolution_reason, summary=None):
    """
    Close a JIRA ticket with resolution information.
    
    Args:
        issue_key: The key of the JIRA issue to close
        resolution_reason: The reason for closing the ticket
        summary: Optional conversation summary to add before closing
        
    Returns:
        Dict with status and display message
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
                close_transition,
                fields={'resolution': {'name': 'Done'}}
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
            
            return {
                "status": 200,
                "display": f"✓ Ticket {issue_key} has been closed successfully"
            }
        else:
            return {
                "status": 400,
                "display": f"❌ Could not find a transition to close the ticket"
            }
        
    except Exception as e:
        return {
            "status": 500,
            "display": f"❌ Failed to close ticket: {str(e)}"
        }

def extract_order_number(text):
    """
    Extract order number from text using regex patterns.
    
    Args:
        text: The text to extract from
        
    Returns:
        The extracted order number or None
    """
    # Common order number patterns
    patterns = [
        r'order\s*(?:#|number|num|no)?[:\s]*(\w{2,}-\d{4,})',  # Example: Order #: AB-12345
        r'order\s*(?:#|number|num|no)?[:\s]*(\d{5,})',         # Example: Order number: 12345678
        r'order\s*id[:\s]*(\w{2,}-\d{4,}|\d{5,})',             # Example: Order ID: 12345678
        r'#(\w{2,}-\d{4,}|\d{5,})',                           # Example: #AB-12345
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

##################
# One Time Setup #
##################
def extract_and_load_issues(project_key, max_results=100):
    """Extract issues from Jira and load them into the database."""
    jira = connect_to_jira()
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # JQL query to get all issues from a project
    jql = f"project = {project_key} ORDER BY created DESC"
    
    # Get issues
    issues = jira.search_issues(jql, maxResults=max_results)
    
    # Prepare data for batch insert
    issue_data = []
    for issue in issues:
        # Extract basic issue data
        issue_dict = {
            "issue_key": issue.key,
            "summary": issue.fields.summary,
            "description": issue.fields.description or "",
            "status": issue.fields.status.name,
            "created_at": issue.fields.created,
            "updated_at": issue.fields.updated,
            "resolved_at": getattr(issue.fields, 'resolutiondate', None),
            "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
            "reporter": issue.fields.reporter.displayName if issue.fields.reporter else None,
            "reporter_email": issue.fields.reporter.emailAddress if issue.fields.reporter else None,
            "issue_type": issue.fields.issuetype.name,
            "priority": issue.fields.priority.name if issue.fields.priority else None,
            "labels": issue.fields.labels,
            "project": issue.fields.project.key,
            "components": [c.name for c in issue.fields.components] if hasattr(issue.fields, 'components') else [],
            "affects_versions": [v.name for v in issue.fields.versions] if hasattr(issue.fields, 'versions') else [],
            "fix_versions": [v.name for v in issue.fields.fixVersions] if hasattr(issue.fields, 'fixVersions') else [],
            "resolution": issue.fields.resolution.name if issue.fields.resolution else None,
            "votes": issue.fields.votes.votes if hasattr(issue.fields.votes, 'votes') else 0,
            "remaining_estimate": parse_time_string(getattr(issue.fields, 'timeestimate', None)),
            "time_spent": parse_time_string(getattr(issue.fields, 'timespent', None)),
            "original_estimate": parse_time_string(getattr(issue.fields, 'timeoriginalestimate', None)),
            "rank": getattr(issue.fields, 'customfield_10019', None),  
            "raw_data": json.dumps(issue.raw)
        }
        
        # Extract custom fields
        try:
            issue_dict["region"] = getattr(issue.fields, 'customfield_10103', None)
            issue_dict["main_category"] = getattr(issue.fields, 'customfield_10101', None)[0]
            issue_dict["sub_category"] = getattr(issue.fields, 'customfield_10096', None)[0]
            issue_dict["request_category"] = getattr(issue.fields, 'customfield_10098', None)
            issue_dict["partner_names"] = getattr(issue.fields, 'customfield_10108', None)
            issue_dict["request_language"] = getattr(issue.fields, 'customfield_10109', None)
            issue_dict["source"] = getattr(issue.fields, 'customfield_10111', None)
            issue_dict["time_to_first_response"] = getattr(issue.fields, 'customfield_10094', None)
            issue_dict["relevant_departments"] = getattr(issue.fields, 'customfield_10104', None)
            issue_dict["resolution_action"] = getattr(issue.fields, 'customfield_10106', None)
            issue_dict["time_to_resolution"] = getattr(issue.fields, 'customfield_10107', None)
            issue_dict["status_category_changed"] = getattr(issue.fields, 'customfield_10100', None)
            issue_dict["date_of_first_response"] = getattr(issue.fields, 'customfield_10095', None)
            issue_dict["request_type"] = getattr(issue.fields, 'customfield_10099', None)
        except AttributeError:
            # Handle missing fields
            pass
        
        # Flatten the data for database insertion
        db_record = (
            issue_dict["issue_key"],
            issue_dict["summary"],
            issue_dict["description"],
            issue_dict["status"],
            issue_dict["created_at"],
            issue_dict["updated_at"],
            issue_dict["resolved_at"],
            issue_dict["assignee"],
            issue_dict["reporter"],
            issue_dict["reporter_email"],
            issue_dict["issue_type"],
            issue_dict["priority"],
            issue_dict["labels"],
            issue_dict["project"],
            issue_dict["components"],
            issue_dict["affects_versions"],
            issue_dict["fix_versions"],
            issue_dict["resolution"],
            issue_dict["votes"],
            issue_dict["remaining_estimate"],
            issue_dict["time_spent"],
            issue_dict["original_estimate"],
            issue_dict["rank"],
            issue_dict.get("main_category"),
            issue_dict.get("sub_category"),
            issue_dict.get("partner_names"),
            issue_dict.get("relevant_departments"),
            issue_dict.get("request_category"),
            issue_dict.get("request_type"),
            issue_dict.get("request_language"),
            issue_dict.get("resolution_action"),
            issue_dict.get("source"),
            issue_dict.get("time_to_first_response"),
            issue_dict.get("time_to_resolution"),
            issue_dict.get("status_category_changed"),
            issue_dict.get("date_of_first_response"),
            issue_dict["raw_data"]
        )
        
        issue_data.append(db_record)
    
    # Batch insert issues
    if issue_data:
        execute_values(
            cursor,
            """
            INSERT INTO jira_issues 
            (issue_key, summary, description, status, created_at, updated_at, 
             resolved_at, assignee, reporter, reporter_email, issue_type, priority, 
             labels, project, components, affects_versions, fix_versions, resolution, 
             votes, remaining_estimate, time_spent, original_estimate, rank, 
             main_category, sub_category, partner_names, relevant_departments, 
             request_category, request_type, request_language, resolution_action, 
             source, time_to_first_response, time_to_resolution, status_category_changed, 
             date_of_first_response, raw_data)
            VALUES %s
            ON CONFLICT (issue_key) DO UPDATE SET
                summary = EXCLUDED.summary,
                description = EXCLUDED.description,
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at,
                resolved_at = EXCLUDED.resolved_at,
                assignee = EXCLUDED.assignee,
                priority = EXCLUDED.priority,
                labels = EXCLUDED.labels,
                resolution = EXCLUDED.resolution,
                votes = EXCLUDED.votes,
                remaining_estimate = EXCLUDED.remaining_estimate,
                time_spent = EXCLUDED.time_spent,
                rank = EXCLUDED.rank,
                main_category = EXCLUDED.main_category,
                sub_category = EXCLUDED.sub_category,
                request_category = EXCLUDED.request_category,
                raw_data = EXCLUDED.raw_data
            """,
            issue_data
        )
    
    # Extract and insert comments and attachments for each issue
    for issue in issues:
        extract_and_load_comments(issue.key, jira, cursor)
    
    conn.commit()
    conn.close()
    
    print(f"Extracted {len(issues)} issues from Jira project {project_key}")

def extract_and_load_comments(issue_key, jira, cursor):
    """Extract comments for a Jira issue and load them into the database."""
    issue = jira.issue(issue_key)
    comments = issue.fields.comment.comments
    
    comment_data = []
    for comment in comments:
        # Extract creation date text from the PDF format (e.g., "09/03/2024 11:49")
        comment_date = None
        if hasattr(comment, 'body') and comment.body:
            date_match = re.search(r'(\d{2}/\d{2}/\d{4}\s\d{2}:\d{2})', comment.body)
            if date_match:
                comment_date = date_match.group(1)
        
        comment_dict = {
            "issue_key": issue_key,
            "author": 'Unknown',#comment.author.displayName if hasattr(comment.author, 'displayName') else None,
            "body": comment.body,
            "created_at": comment.created,
            "updated_at": comment.updated,
            "comment_date": comment_date,  # Add the extracted date
            "raw_data": json.dumps(comment.raw)
        }
        
        comment_data.append(
            (
                comment_dict["issue_key"],
                comment_dict["author"],
                comment_dict["body"],
                comment_dict["created_at"],
                comment_dict["updated_at"],
                comment_dict["comment_date"],
                comment_dict["raw_data"]
            )
        )
    
    # Delete existing comments for this issue and insert new ones
    cursor.execute("DELETE FROM jira_comments WHERE issue_key = %s", (issue_key,))
    
    if comment_data:
        execute_values(
            cursor,
            """
            INSERT INTO jira_comments 
            (issue_key, author, body, created_at, updated_at, comment_date, raw_data)
            VALUES %s
            """,
            comment_data
        )
