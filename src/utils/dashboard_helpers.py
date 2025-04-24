import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
import threading
import time
from dotenv import load_dotenv
import os
from src.data.db import connect_to_db
from loguru import logger

# Cache for dataframes
cache = {
    "issues_df": None,
    "last_refresh": 0
}

# Refresh interval in seconds
REFRESH_INTERVAL = 10

def initialize_data_refresh():
    """Start a background thread to refresh data periodically"""
    def refresh_loop():
        while True:
            with st.session_state.get("db_lock", threading.Lock()):
                refresh_all_dataframes()
            time.sleep(REFRESH_INTERVAL)
    
    # Initialize the lock if it doesn't exist
    if "db_lock" not in st.session_state:
        st.session_state.db_lock = threading.Lock()
    
    # Start background thread
    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()

def get_issues_dataframe(force_refresh=False):
    """
    Get a dataframe of jira issues from the database
    
    Args:
        force_refresh: Whether to force a refresh regardless of cache
        
    Returns:
        DataFrame: Pandas dataframe with jira issues
    """
    current_time = time.time()
    
    # Return cached dataframe if it exists and is fresh enough
    if not force_refresh and cache["issues_df"] is not None and (current_time - cache["last_refresh"]) < REFRESH_INTERVAL:
        return cache["issues_df"]
    
    # Otherwise, query the database
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                issue_key, 
                summary, 
                description, 
                status, 
                region as "Region", 
                created_at, 
                resolved_at,
                priority as "Priority", 
                main_category,
                sub_category,
                partner_names,
                assignee as "Assignee",
                source as "Source",
                votes as "Satisfaction rating",
                project
            FROM jira_issues
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        
        # Update cache
        cache["issues_df"] = df
        cache["last_refresh"] = current_time
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching issues data: {str(e)}")
        # Return empty dataframe or cached one if available
        return cache.get("issues_df", pd.DataFrame())

def get_comments_dataframe():
    """
    Get a dataframe of jira comments from the database
    
    Returns:
        DataFrame: Pandas dataframe with jira comments
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                issue_key, 
                author, 
                body, 
                created_at, 
                updated_at
            FROM jira_comments
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching comments data: {str(e)}")
        return pd.DataFrame()

def get_attachments_dataframe():
    """
    Get a dataframe of jira attachments from the database
    
    Returns:
        DataFrame: Pandas dataframe with jira attachments
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                issue_key, 
                filename, 
                content_type, 
                size, 
                created_at, 
                author
            FROM jira_attachments
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching attachments data: {str(e)}")
        return pd.DataFrame()

def get_knowledge_base_dataframe():
    """
    Get a dataframe of knowledge base from the database
    
    Returns:
        DataFrame: Pandas dataframe with knowledge base
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                id,
                main_category, 
                sub_category, 
                kb, 
                created_at, 
                updated_at
            FROM knowledge_base
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching knowledge base data: {str(e)}")
        return pd.DataFrame()

def get_issues_by_category():
    """
    Get count of issues grouped by main_category and sub_category
    
    Returns:
        DataFrame: Pandas dataframe with issue counts by category
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                main_category, 
                sub_category, 
                COUNT(*) as issue_count
            FROM jira_issues
            GROUP BY main_category, sub_category
            ORDER BY main_category, sub_category
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching category data: {str(e)}")
        return pd.DataFrame()

def get_monthly_issues():
    """
    Get count of issues created each month
    
    Returns:
        DataFrame: Pandas dataframe with monthly issue counts
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                DATE_TRUNC('month', created_at) as month,
                COUNT(*) as issue_count
            FROM jira_issues
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY month
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        if not df.empty:
            df['month'] = pd.to_datetime(df['month'])
        return df
    
    except Exception as e:
        logger.error(f"Error fetching monthly data: {str(e)}")
        return pd.DataFrame()

def get_monthly_resolved_issues():
    """
    Get count of issues resolved each month
    
    Returns:
        DataFrame: Pandas dataframe with monthly resolved issue counts
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                DATE_TRUNC('month', resolved_at) as month,
                COUNT(*) as resolved_count
            FROM jira_issues
            WHERE resolved_at IS NOT NULL
            GROUP BY DATE_TRUNC('month', resolved_at)
            ORDER BY month
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        if not df.empty:
            df['month'] = pd.to_datetime(df['month'])
        return df
    
    except Exception as e:
        logger.error(f"Error fetching monthly resolved data: {str(e)}")
        return pd.DataFrame()

def get_yearly_issues():
    """
    Get count of issues created each year
    
    Returns:
        DataFrame: Pandas dataframe with yearly issue counts
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                DATE_TRUNC('year', created_at) as year,
                COUNT(*) as issue_count
            FROM jira_issues
            GROUP BY DATE_TRUNC('year', created_at)
            ORDER BY year
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        if not df.empty:
            df['year'] = pd.to_datetime(df['year'])
        return df
    
    except Exception as e:
        logger.error(f"Error fetching yearly data: {str(e)}")
        return pd.DataFrame()

def get_daily_issues():
    """
    Get count of issues created each day
    
    Returns:
        DataFrame: Pandas dataframe with daily issue counts
    """
    try:
        conn = connect_to_db()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT 
                DATE_TRUNC('day', created_at) as day,
                COUNT(*) as issue_count
            FROM jira_issues
            GROUP BY DATE_TRUNC('day', created_at)
            ORDER BY day
            """
            cursor.execute(query)
            results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        if not df.empty:
            df['day'] = pd.to_datetime(df['day'])
        return df
    
    except Exception as e:
        logger.error(f"Error fetching daily data: {str(e)}")
        return pd.DataFrame()

def refresh_all_dataframes():
    """Force refresh all cached dataframes"""
    get_issues_dataframe(force_refresh=True)
    # Add other dataframes that need refreshing here

def preprocess_dataframe(df):
    """
    Apply preprocessing similar to the original preprocess function
    
    Args:
        df: DataFrame to preprocess
        
    Returns:
        DataFrame: Preprocessed dataframe
    """
    # Copy to avoid modifying the original
    processed_df = df.copy()
    
    # Ensure datetime columns are in the right format
    for col in ['created_at', 'resolved_at', 'updated_at']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col])

    # Handle missing values for important columns
    if 'Region' in processed_df.columns:
        processed_df['Region'] = processed_df['Region'].fillna('Unknown').astype(str)
    
    if 'Source' in processed_df.columns:
        processed_df['Source'] = processed_df['Source'].fillna('Unknown').astype(str)
        
    if 'Priority' in processed_df.columns:
        processed_df['Priority'] = processed_df['Priority'].fillna('Unknown')
        
    if 'Satisfaction rating' in processed_df.columns:
        processed_df['Satisfaction rating'] = processed_df['Satisfaction rating'].fillna(0)
    
    return processed_df

def get_processed_dataframe(filtered=True):
    """
    Get processed dataframe with optional filtering
    
    Args:
        filtered: Whether to apply filters
        
    Returns:
        DataFrame: Processed dataframe
    """
    df = get_issues_dataframe()
    processed_df = preprocess_dataframe(df)
    
    if filtered and 'filter_params' in st.session_state:
        # Apply filters if they exist in session state
        filters = st.session_state.filter_params
        
        if 'priority' in filters and filters['priority']:
            processed_df = processed_df[processed_df['Priority'].isin(filters['priority'])]
            
        if 'assignee' in filters and filters['assignee']:
            processed_df = processed_df[processed_df['Assignee'].isin(filters['assignee'])]
            
        if 'region' in filters and filters['region']:
            processed_df = processed_df[processed_df['Region'].isin(filters['region'])]
            
        if 'source' in filters and filters['source']:
            processed_df = processed_df[processed_df['Source'].isin(filters['source'])]
    
    return processed_df

def apply_filters(df, priority_filter, assignee_filter, region_filter, source_filter):
    """
    Apply filters to dataframe
    
    Args:
        df: DataFrame to filter
        priority_filter: List of priority values
        assignee_filter: List of assignee values
        region_filter: List of region values
        source_filter: List of source values
        
    Returns:
        DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    # Store filter parameters in session state for reuse
    st.session_state.filter_params = {
        'priority': priority_filter,
        'assignee': assignee_filter,
        'region': region_filter,
        'source': source_filter
    }
    
    # Apply filters
    if priority_filter:
        filtered_df = filtered_df[filtered_df['Priority'].isin(priority_filter)]
    
    if assignee_filter:
        filtered_df = filtered_df[filtered_df['Assignee'].isin(assignee_filter)]
    
    if region_filter:
        filtered_df = filtered_df[filtered_df['Region'].isin(region_filter)]
    
    if source_filter:
        filtered_df = filtered_df[filtered_df['Source'].isin(source_filter)]
    
    return filtered_df

def get_satisfaction_rating(df):
    """
    Calculate average satisfaction rating
    
    Args:
        df: DataFrame with satisfaction ratings
        
    Returns:
        float: Average satisfaction rating
    """
    if 'Satisfaction rating' in df.columns:
        return int(df['Satisfaction rating'].mean())
    return 0

def get_closed_complaints_count(df):
    """
    Get count of closed complaints
    
    Args:
        df: DataFrame with status column
        
    Returns:
        int: Count of closed complaints
    """
    return len(df[df['status'] == 'Closed'])

def get_completed_complaints_count(df):
    """
    Get count of completed complaints
    
    Args:
        df: DataFrame with status column
        
    Returns:
        int: Count of completed complaints
    """
    return len(df[df['status'] == 'Completed'])

def get_cancelled_complaints_count(df):
    """
    Get count of cancelled complaints
    
    Args:
        df: DataFrame with status column
        
    Returns:
        int: Count of cancelled complaints
    """
    return len(df[df['status'] == 'Canceled'])