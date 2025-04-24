import streamlit as st
import pandas as pd
import numpy as np
from src.utils.dashboard_helpers import (
    initialize_data_refresh, 
    get_processed_dataframe,
    apply_filters
)

st.title('🛃 Complaints Detailed Information')

# Get data from database
df = get_processed_dataframe(filtered=False)

# Create filter UI with multiselect
st.markdown("""
    <style>
    .custom-multiselect {
        width: 400px;
    }
    </style>
""", unsafe_allow_html=True)

dash_1 = st.container()
with st.container():
    st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        priority_options = sorted(df['Priority'].unique().tolist()) if 'Priority' in df.columns else []
        priority_filter = st.multiselect('Select Priority', options=priority_options, default=priority_options)
    
    with col2:
        assignee_options = sorted(df['Assignee'].unique().tolist()) if 'Assignee' in df.columns else []
        assignee_filter = st.multiselect('Select Assignee', options=assignee_options, default=assignee_options)
    
    with col3:
        region_options = sorted(df['Region'].unique().tolist()) if 'Region' in df.columns else []
        region_filter = st.multiselect('Select Region', options=region_options, default=region_options)
    
    with col4:
        source_options = sorted(df['Source'].unique().tolist()) if 'Source' in df.columns else []
        source_filter = st.multiselect('Select Source', options=source_options, default=source_options)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
filtered_df = apply_filters(df, priority_filter, assignee_filter, region_filter, source_filter)

# Display detailed information
if not filtered_df.empty:
    # Add search functionality
    search_term = st.text_input("Search complaints (by issue key, summary, or description):", "")
    
    if search_term:
        # Apply search filter to the dataframe
        mask = (
            filtered_df['issue_key'].str.contains(search_term, case=False, na=False) |
            filtered_df['summary'].str.contains(search_term, case=False, na=False) |
            filtered_df['description'].str.contains(search_term, case=False, na=False)
        )
        search_filtered_df = filtered_df[mask]
    else:
        search_filtered_df = filtered_df
    
    # Display the table with key information
    display_columns = ['issue_key', 'summary', 'status', 'Priority', 'Region', 'Source', 'Assignee', 'created_at']
    display_cols = [col for col in display_columns if col in search_filtered_df.columns]
    
    # Show record count
    st.write(f"Displaying {len(search_filtered_df)} records")
    
    # Display the dataframe
    st.dataframe(search_filtered_df[display_cols])
    
    # Allow user to select a complaint for detailed view
    selected_issue = st.selectbox(
        "Select an issue to view details:",
        options=search_filtered_df['issue_key'].tolist(),
        index=0 if len(search_filtered_df) > 0 else None
    )
    
    if selected_issue:
        # Get the selected issue
        issue_detail = search_filtered_df[search_filtered_df['issue_key'] == selected_issue].iloc[0]
        
        st.write("### Issue Details")
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Issue Key:** {issue_detail['issue_key']}")
            st.write(f"**Summary:** {issue_detail['summary']}")
            st.write(f"**Status:** {issue_detail['status']}")
            st.write(f"**Priority:** {issue_detail['Priority']}")
            
        with col2:
            st.write(f"**Region:** {issue_detail['Region']}")
            st.write(f"**Source:** {issue_detail['Source']}")
            st.write(f"**Assignee:** {issue_detail['Assignee']}")
            st.write(f"**Created At:** {issue_detail['created_at']}")
        
        st.write("**Description:**")
        st.text_area("", value=issue_detail['description'] if 'description' in issue_detail and issue_detail['description'] else "No description available.", height=200, disabled=True)
        
else:
    st.warning("No complaints match the selected filters. Please adjust your filter criteria.")