import os
import psycopg2
from src.utils.constants import MODEL_OPENAI
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

# Get database connection parameters from environment variables
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
dbname = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
model = os.getenv("MODEL")

def connect_to_db():
    """Establish connection to PostgreSQL database."""
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    register_vector(conn)
    return conn

def setup_db():
    """ One-time setup of Postgres """
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{dbname}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {dbname}")

    # Connect to the newly created database
    conn.close()
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    cursor = conn.cursor()

    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jira_issues (
        id SERIAL PRIMARY KEY,
        issue_key TEXT UNIQUE NOT NULL,
        summary TEXT NOT NULL,
        description TEXT,
        region TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        resolved_at TIMESTAMP,
        assignee TEXT,
        reporter TEXT,
        reporter_email TEXT,
        issue_type TEXT NOT NULL,
        priority TEXT,
        labels TEXT[],
        project TEXT NOT NULL,
        components TEXT[],
        affects_versions TEXT[],
        fix_versions TEXT[],
        resolution TEXT,
        votes INTEGER,
        remaining_estimate INTERVAL,  -- in minutes
        time_spent INTERVAL,          -- in minutes
        original_estimate INTERVAL,   -- in minutes
        rank TEXT,
        main_category TEXT,
        sub_category TEXT,
        partner_names TEXT,
        relevant_departments TEXT,
        request_category TEXT,
        request_type TEXT,
        request_language TEXT,
        resolution_action TEXT,
        source TEXT,
        time_to_first_response TEXT,
        time_to_resolution TEXT,
        status_category_changed TEXT,
        date_of_first_response TEXT,
        raw_data JSONB
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jira_comments (
        id SERIAL PRIMARY KEY,
        issue_key TEXT REFERENCES jira_issues(issue_key),
        author TEXT,
        body TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        comment_date TEXT,      -- Additional field to capture date in text format
        raw_data JSONB
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jira_attachments (
        id SERIAL PRIMARY KEY,
        issue_key TEXT REFERENCES jira_issues(issue_key),
        filename TEXT NOT NULL,
        content_type TEXT,
        size INTEGER,
        created_at TIMESTAMP,
        author TEXT,
        raw_data JSONB
    )
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        main_category TEXT,
        sub_category TEXT,
        kb TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )
    """)

    # for nomic-embed
    emb_size = 768
    # for openai
    if model == MODEL_OPENAI:
        emb_size = 1536
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS issue_embeddings (
        id SERIAL PRIMARY KEY,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding vector({emb_size}),
        chunk_number INTEGER,
        UNIQUE(source_type, source_id, chunk_number)
    )
    """)
    # todo source id should be issue key
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS comment_embeddings (
        id SERIAL PRIMARY KEY,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding vector({emb_size}),
        chunk_number INTEGER,
        UNIQUE(source_type, source_id, chunk_number)
    )
    """)

    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jira_issues_issue_key ON jira_issues(issue_key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jira_issues_project ON jira_issues(project)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jira_issues_main_category ON jira_issues(main_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jira_issues_status ON jira_issues(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jira_comments_issue_key ON jira_comments(issue_key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jira_attachments_issue_key ON jira_attachments(issue_key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_source ON issue_embeddings(source_type, source_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_source ON comment_embeddings(source_type, source_id)")
    # ivfflat -> more accurate; hnsw -> faster less accurate
    # todo: update index after every insertion
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_embedding ON issue_embeddings USING ivfflat (embedding vector_cosine_ops)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_embedding ON comment_embeddings USING ivfflat (embedding vector_cosine_ops)")

    conn.commit()
    conn.close()

    print("Database setup completed successfully.")
