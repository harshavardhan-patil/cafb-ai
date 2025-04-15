import os
import json
import re
from datetime import datetime
from jira import JIRA
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from src.data.db import connect_to_db

load_dotenv()

# Jira connection parameters
jira_url = os.getenv("JIRA_URL")
jira_email = os.getenv("JIRA_EMAIL")
jira_api_token = os.getenv("JIRA_API_TOKEN")


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
