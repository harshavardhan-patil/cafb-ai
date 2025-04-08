from src.data.db import setup_db
from src.data.jira import extract_and_load_issues
from src.data.rag import process_jira_issues, process_jira_comments

if __name__ == "__main__":
    setup_db()
    extract_and_load_issues("CAFBSS", 0) #0 for all
    process_jira_issues()
    process_jira_comments()
