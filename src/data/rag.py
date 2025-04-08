import os
import re
import psycopg2
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from src.utils.constants import MODEL_OPENAI
from src.data.db import connect_to_db
from src.data.jira import get_issue_info, get_comment_info

load_dotenv()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

model = os.getenv("MODEL")


def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove Jira/Markdown formatting
    text = re.sub(r'\{[^}]*\}', '', text)  # Remove Jira macros
    text = re.sub(r'\[.*?\|.*?\]', '', text)  # Remove Jira links
    text = re.sub(r'!.*!', '', text)  # Remove image references
    
    # Convert to plaintext
    text = re.sub(r'[*_~^]+', '', text)  # Remove formatting chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into smaller chunks for embedding."""
    if not text or len(text) < 100:
        return [text] if text else []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_text(text)

def generate_embedding(text):
    """Generate embedding vector for a text"""
    # if model == MODEL_OPENAI:
    #     response = openai.Embedding.create(
    #         input=text,
    #         model="text-embedding-ada-002"
    #     )["data"][0]["embedding"]
    # else:
    response = ollama.embed(
        model='nomic-embed-text',
        input=text
    ).embeddings[0]

    return response

# todo fix similarity search not returning any content
def retrieve_relevant_context(query, limit=5, issue_key=None):
    """
    Retrieve relevant context from the vector database based on query similarity.
    
    Args:
        query (str): The query to search for
        limit (int): Max number of contexts to retrieve
        issue_key (str, optional): If provided, limit search to this specific issue
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    
    # Perform vector similarity search
    # todo not returning ticket context
    if issue_key:
        # If issue_key is provided, only search within that issue's context
        cursor.execute(
            """
            SELECT 
                e.source_type, 
                e.source_id, 
                e.content,
                (e.embedding <=> %s::vector) as distance
            FROM issue_embeddings e
            WHERE
                (e.source_type = 'issue' AND e.source_id = %s) OR
                (e.source_type = 'comment' AND e.source_id IN (
                    SELECT CAST(c.id AS TEXT) 
                    FROM jira_comments c 
                    WHERE c.issue_key = %s
                ))
            ORDER BY distance ASC
            LIMIT %s
            """,
            (query_embedding, issue_key, issue_key, limit)
        )
    else:
        # General search across all content - doesnt work great
        cursor.execute(
            """
            SELECT 
                e.source_type, 
                e.source_id, 
                e.content,
                (e.embedding <=> %s::vector) as distance
            FROM issue_embeddings e
            ORDER BY distance ASC
            LIMIT %s
            """,
            (query_embedding, limit)
        )
    
    results = cursor.fetchall()
    conn.close()
    
    # Format results as context
    context = []
    for source_type, source_id, content, distance in results:
        # Get additional information based on source type
        if source_type == 'issue':
            issue_info = get_issue_info(source_id)
            context_str = f"JIRA ISSUE {source_id} - {issue_info['summary']} (Status: {issue_info['status']}):\n{content}"
        elif source_type == 'comment':
            comment_info = get_comment_info(source_id)
            context_str = f"COMMENT on {comment_info['issue_key']} by {comment_info['author']}:\n{content}"
        else:
            context_str = f"{source_type.upper()} {source_id}:\n{content}"
        
        context.append(context_str)
    
    return "\n\n Ticket Context: ".join(context)

def process_jira_issues():
    """Process Jira issues, generate chunks and embeddings, and store them."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Get all issues that don't have embeddings yet
    cursor.execute("""
    SELECT i.issue_key, i.summary, i.description
    FROM jira_issues i
    LEFT JOIN issue_embeddings e ON e.source_id = i.issue_key AND e.source_type = 'issue'
    WHERE e.id IS NULL
    """)
    
    issues = cursor.fetchall()
    
    for issue_key, summary, description in issues:
        # First, clean and prepare the text content
        clean_description = clean_text(description)
        
        # Split the content into chunks
        content_text = f"Issue: {summary}\n\nDescription: {clean_description}"
        chunks = chunk_text(content_text)
        
        # Process each chunk and prepend the issue key to EACH chunk
        for i, chunk in enumerate(chunks):
            # Add issue key to each individual chunk
            chunk_with_key = f"Issue Key: {issue_key} | {chunk}"
            
            # Generate embedding
            embedding = generate_embedding(chunk_with_key)
            
            if embedding:
                cursor.execute(
                    """
                    INSERT INTO issue_embeddings 
                    (source_type, source_id, content, embedding, chunk_number)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_type, source_id, chunk_number) 
                    DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                    """,
                    ('issue', issue_key, chunk_with_key, embedding, i)
                )
        
        conn.commit()
    print(f"Embedded {len(issues)} issues")

def process_jira_comments():
    """Process Jira comments, generate chunks and embeddings, and store them."""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Get all comments that don't have embeddings yet
    cursor.execute("""
    SELECT c.id, c.issue_key, c.author, c.body
    FROM jira_comments c
    LEFT JOIN comment_embeddings e ON e.source_id = CAST(c.id AS TEXT) AND e.source_type = 'comment'
    WHERE e.id IS NULL AND LENGTH(c.body) > 50
    """)
    
    comments = cursor.fetchall()
    
    for comment_id, issue_key, author, body in comments:
        # Clean text
        clean_body = clean_text(body)
        
        # Format comment content
        content_text = f"Comment by {author}:\n{clean_body}"
        
        # Split into chunks
        chunks = chunk_text(content_text)
        
        # Process each chunk and prepend the issue key to EACH chunk
        for i, chunk in enumerate(chunks):
            # Add issue key to each individual chunk
            chunk_with_key = f"Issue Key: {issue_key} | {chunk}"
            
            # Generate embedding
            embedding = generate_embedding(chunk_with_key)
            
            if embedding:
                cursor.execute(
                    """
                    INSERT INTO comment_embeddings 
                    (source_type, source_id, content, embedding, chunk_number)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_type, source_id, chunk_number) 
                    DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                    """,
                    ('comment', str(comment_id), chunk_with_key, embedding, i)
                )
        
        conn.commit()
    print(f"Embedded {len(comments)} comments")

