import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from src.utils.dashboard_helpers import (
    initialize_data_refresh, 
    get_processed_dataframe,
    apply_filters,
    get_closed_complaints_count,
    get_completed_complaints_count,
    get_cancelled_complaints_count,
    get_satisfaction_rating,
    get_yearly_issues,
    get_monthly_issues,
    get_daily_issues,
    get_issues_by_category
)
import plotly.express as px


# Get data from database
df = get_processed_dataframe(filtered=False)

st.title("📊 Trends Monitoring System")

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

# Metrics dashboard
dash_2 = st.container()
with dash_2:
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            closed_count = get_closed_complaints_count(filtered_df)
            st.metric("Closed Complaints", f"{closed_count:,}")
    except Exception as e:
        st.error("Unable to display closed complaints. Please check your filter selection.")
    
    try:
        with col2:
            completed_count = get_completed_complaints_count(filtered_df)
            st.metric("Completed Complaints", f"{completed_count:,}")
    except Exception as e:
        st.error("Unable to display completed complaints. Please check your filter selection.")
    
    try:
        with col3:
            cancelled_count = get_cancelled_complaints_count(filtered_df)
            st.metric("Cancelled Complaints", f"{cancelled_count:,}")
    except Exception as e:
        st.error("Unable to display cancelled complaints. Please check your filter selection.")
    
    try:
        with col4:
            satisfaction_rating = get_satisfaction_rating(filtered_df)
            st.metric("Average Satisfaction Rating", f"{satisfaction_rating}")
    except Exception as e:
        st.error("Unable to display satisfaction rating. Please check your filter selection.")
    
    style_metric_cards(border_left_color="#4E8226")

# Function to create yearly trends chart
def yearly_trends(df):
    yearly_data = get_yearly_issues()
    
    if yearly_data.empty:
        yearly_counts = pd.DataFrame({
            'year': pd.date_range(start='2020-01-01', periods=5, freq='Y'),
            'issue_count': [0, 0, 0, 0, 0]
        })
    else:
        yearly_counts = yearly_data
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_counts['year'],
        y=yearly_counts['issue_count'],
        marker_color='#4E8226'
    ))
    
    fig.update_layout(
        title="Yearly Complaint Trends",
        xaxis_title="Year",
        yaxis_title="Number of Complaints",
        template="plotly_white"
    )
    
    return fig

# Function to create monthly trends chart
def monthly_trends(df):
    monthly_data = get_monthly_issues()
    
    if monthly_data.empty:
        # Create sample data for demonstration
        monthly_counts = pd.DataFrame({
            'month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'issue_count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
    else:
        monthly_counts = monthly_data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_counts['month'],
        y=monthly_counts['issue_count'],
        mode='lines+markers',
        marker_color='#4E8226',
        line=dict(color='#4E8226', width=2)
    ))
    
    fig.update_layout(
        title="Monthly Complaint Trends",
        xaxis_title="Month",
        yaxis_title="Number of Complaints",
        template="plotly_white"
    )
    
    return fig

# Function to create daily trends chart
def daily_trends(df):
    daily_data = get_daily_issues()
    
    if daily_data.empty:
        # Create sample data for demonstration
        daily_counts = pd.DataFrame({
            'day': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'issue_count': [0] * 30
        })
    else:
        daily_counts = daily_data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_counts['day'],
        y=daily_counts['issue_count'],
        mode='lines',
        marker_color='#4E8226',
        line=dict(color='#4E8226', width=2)
    ))
    
    fig.update_layout(
        title="Daily Complaint Trends",
        xaxis_title="Day",
        yaxis_title="Number of Complaints",
        template="plotly_white"
    )
    
    return fig

# Function to create category bubble chart
def category_bubble(df):
    category_data = get_issues_by_category()
    
    if category_data.empty:
        return go.Figure().update_layout(
            title="No category data available",
            template="plotly_white"
        )
    
    # Group by main category to get size for bubbles
    main_category_counts = category_data.groupby('main_category')['issue_count'].sum().reset_index()
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly if 'px' in globals() else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (idx, row) in enumerate(main_category_counts.iterrows()):
        color = colors[i % len(colors)]
        size = row['issue_count'] * 2  # Scale size for visibility
        
        # Get all subcategories for this main category
        subcategories = category_data[category_data['main_category'] == row['main_category']]
        
        for _, subcat in subcategories.iterrows():
            subsize = subcat['issue_count']
            
            fig.add_trace(go.Scatter(
                x=[row['main_category']],
                y=[subcat['sub_category']],
                mode='markers',
                marker=dict(
                    size=subsize * 2,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=f"{row['main_category']} - {subcat['sub_category']}",
                text=f"Count: {subcat['issue_count']}",
                hoverinfo='text'
            ))
    
    fig.update_layout(
        title="Issue Categories",
        xaxis_title="Main Category",
        yaxis_title="Subcategory",
        template="plotly_white",
        height=600
    )
    
    return fig

# Create tabs for different time periods
dash_3 = st.container()
tab1, tab2, tab3 = st.tabs(["Yearly Trends", "Monthly Trends", "Daily Trends"])

with tab1:
    st.plotly_chart(yearly_trends(filtered_df))

with tab2:
    st.plotly_chart(monthly_trends(filtered_df))

with tab3:
    st.plotly_chart(daily_trends(filtered_df))

# Category bubble chart
dash_4 = st.container()
with dash_4:
    st.plotly_chart(category_bubble(filtered_df))