import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(layout="wide", page_title="Complaints Dashboard")

@st.cache_data
def load_data():
    file_path = 'data/raw/cafb_data_case2.xlsx'
    df = pd.read_excel(file_path)
    
    # Convert relevant columns to datetime
    df['Created'] = pd.to_datetime(df['Created'])
    df['Resolved'] = pd.to_datetime(df['Resolved'])
    
    # Extract the timestamp from the first comment in 'Comment.1'
    df['First Comment Time'] = df['Comment.1'].str.split(';').str[0]
    df['First Comment Time'] = pd.to_datetime(df['First Comment Time'], errors='coerce')
    
    # Calculate Response Time using the first comment time
    df['Response Time'] = df['First Comment Time'] - df['Created']
    df['Response Time (Hours)'] = df['Response Time'].dt.total_seconds() / 3600
    
    # Calculate Resolution Time
    df['Resolution Time'] = df['Resolved'] - df['Created']
    df['Resolution Time (Hours)'] = df['Resolution Time'].dt.total_seconds() / 3600
    df['Resolution Time (Days)'] = df['Resolution Time'].dt.total_seconds() / (3600 * 24)
    
    # Extract day, month, and year from 'Created' and 'Resolved' dates
    df['Created Day'] = df['Created'].dt.day
    df['Created Month'] = df['Created'].dt.month
    df['Created Year'] = df['Created'].dt.year
    
    df["Created Daily"] = df['Created'].dt.day.astype(str) + "-" + df['Created'].dt.month.astype(str) + "-" + df['Created'].dt.year.astype(str)
    
    df['Resolved Day'] = df['Resolved'].dt.day
    df['Resolved Month'] = df['Resolved'].dt.month
    df['Resolved Year'] = df['Resolved'].dt.year
    
    # Create month-year for easier grouping
    df['Created Month-Year'] = df['Created'].dt.strftime('%b-%Y')
    df['Month-Year Sort'] = df['Created'].dt.year * 100 + df['Created'].dt.month
    
    return df

def main():
    st.title("Complaints Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Time period filters in sidebar
    st.sidebar.header("Time Period Filters")
    
    # Get available years and current year
    available_years = sorted(df['Created Year'].unique())
    current_year = datetime.now().year
    if current_year not in available_years and available_years:
        current_year = max(available_years)
    
    # Time period selection using radio buttons
    time_period = st.sidebar.radio(
        "Select Time Period View:",
        ["Year", "Month", "YTD"]
    )
    
    # Filter data based on selected time period
    if time_period == "Year":
        selected_year = st.sidebar.selectbox("Select Year:", available_years, index=available_years.index(current_year) if current_year in available_years else 0)
        filtered_df = df[df['Created Year'] == selected_year]
        period_title = f"Year: {selected_year}"
        
    elif time_period == "Month":
        selected_year = st.sidebar.selectbox("Select Year:", available_years, index=available_years.index(current_year) if current_year in available_years else 0)
        month_options = [i for i in range(1, 13) if i in df[df['Created Year'] == selected_year]['Created Month'].unique()]
        if not month_options:
            st.error(f"No data available for {selected_year}")
            return
        
        selected_month = st.sidebar.selectbox(
            "Select Month:", 
            month_options,
            index=min(datetime.now().month - 1 if datetime.now().year == selected_year else 0, len(month_options) - 1)
        )
        filtered_df = df[(df['Created Year'] == selected_year) & (df['Created Month'] == selected_month)]
        month_name = datetime(selected_year, selected_month, 1).strftime('%B')
        period_title = f"{month_name} {selected_year}"
        
    else:  # YTD
        selected_year = st.sidebar.selectbox("Select Year:", available_years, index=available_years.index(current_year) if current_year in available_years else 0)
        current_month = datetime.now().month if datetime.now().year == selected_year else 12
        filtered_df = df[(df['Created Year'] == selected_year) & (df['Created Month'] <= current_month)]
        period_title = f"YTD {selected_year} (through {datetime(selected_year, current_month, 1).strftime('%B')})"
    
    # Display filtered information
    st.subheader(f"Complaints Analysis - {period_title}")
    
    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning(f"No data available for the selected time period: {period_title}")
        return
    
    # Create two columns for the top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Complaints", filtered_df.shape[0])
    
    with col2:
        resolved_count = filtered_df['Resolved'].notna().sum()
        st.metric("Resolved Complaints", resolved_count)
    
    with col3:
        avg_response_time = filtered_df['Response Time (Hours)'].mean()
        st.metric("Avg. Response Time (Hours)", f"{avg_response_time:.2f}")
    
    with col4:
        avg_resolution_time = filtered_df['Resolution Time (Hours)'].mean()
        st.metric("Avg. Resolution Time (Hours)", f"{avg_resolution_time:.2f}")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Complaint Trends", "Response Time", "Resolution Time", "Category Analysis"])
    
    with tab1:
        st.subheader("Complaint Volume Trends")
        
        if time_period == "Year":
            # Monthly trend for the selected year
            monthly_counts = filtered_df.groupby('Created Month').size().reset_index(name='count')
            monthly_counts['Month Name'] = monthly_counts['Created Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
            
            fig = px.bar(
                monthly_counts, 
                x='Month Name', 
                y='count',
                title=f"Monthly Complaint Volume for {selected_year}",
                labels={'count': 'Number of Complaints', 'Month Name': 'Month'},
                text='count',
                category_orders={"Month Name": [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif time_period == "Month":
            # Daily trend for the selected month
            daily_counts = filtered_df.groupby('Created Day').size().reset_index(name='count')
            
            # Ensure all days of the month are represented
            days_in_month = (datetime(selected_year, selected_month % 12 + 1, 1) - timedelta(days=1)).day if selected_month < 12 else 31
            all_days = pd.DataFrame({'Created Day': range(1, days_in_month + 1)})
            daily_counts = all_days.merge(daily_counts, on='Created Day', how='left').fillna(0)
            
            fig = px.bar(
                daily_counts, 
                x='Created Day', 
                y='count',
                title=f"Daily Complaint Volume for {datetime(selected_year, selected_month, 1).strftime('%B %Y')}",
                labels={'count': 'Number of Complaints', 'Created Day': 'Day of Month'},
                text='count'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # YTD
            # Monthly trend for YTD
            monthly_counts = filtered_df.groupby(['Created Month', 'Month-Year Sort']).size().reset_index(name='count')
            monthly_counts['Month Name'] = monthly_counts['Created Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
            monthly_counts = monthly_counts.sort_values('Month-Year Sort')
            
            fig = px.bar(
                monthly_counts, 
                x='Month Name', 
                y='count',
                title=f"YTD Complaint Volume for {selected_year}",
                labels={'count': 'Number of Complaints', 'Month Name': 'Month'},
                text='count',
                category_orders={"Month Name": [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Response Time Analysis")
        
        # Filter out rows with invalid response times
        valid_response_df = filtered_df[filtered_df['Response Time (Hours)'] > 0]
        
        if valid_response_df.empty:
            st.warning("No valid response time data available for the selected period.")
        else:
            if time_period == "Year":
                # Monthly average response times
                monthly_response = valid_response_df.groupby('Created Month')['Response Time (Hours)'].mean().reset_index()
                monthly_response['Month Name'] = monthly_response['Created Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
                
                fig = px.line(
                    monthly_response,
                    x='Month Name',
                    y='Response Time (Hours)',
                    title=f"Monthly Average Response Time for {selected_year}",
                    markers=True,
                    category_orders={"Month Name": [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif time_period == "Month":
                # Daily average response times
                daily_response = valid_response_df.groupby('Created Day')['Response Time (Hours)'].mean().reset_index()
                
                fig = px.line(
                    daily_response,
                    x='Created Day',
                    y='Response Time (Hours)',
                    title=f"Daily Average Response Time for {datetime(selected_year, selected_month, 1).strftime('%B %Y')}",
                    markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # YTD
                # Monthly average response times for YTD
                monthly_response = valid_response_df.groupby(['Created Month', 'Month-Year Sort'])['Response Time (Hours)'].mean().reset_index()
                monthly_response['Month Name'] = monthly_response['Created Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
                monthly_response = monthly_response.sort_values('Month-Year Sort')
                
                fig = px.line(
                    monthly_response,
                    x='Month Name',
                    y='Response Time (Hours)',
                    title=f"YTD Monthly Average Response Time for {selected_year}",
                    markers=True,
                    category_orders={"Month Name": [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Response time distribution
            fig = px.histogram(
                valid_response_df,
                x='Response Time (Hours)',
                nbins=20,
                title="Response Time Distribution",
                labels={'Response Time (Hours)': 'Response Time (Hours)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Resolution Time Analysis")
        
        # Filter out rows with invalid resolution times
        valid_resolution_df = filtered_df[filtered_df['Resolution Time (Hours)'] > 0]
        
        if valid_resolution_df.empty:
            st.warning("No valid resolution time data available for the selected period.")
        else:
            if time_period == "Year":
                # Monthly average resolution times
                monthly_resolution = valid_resolution_df.groupby('Created Month')['Resolution Time (Days)'].mean().reset_index()
                monthly_resolution['Month Name'] = monthly_resolution['Created Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
                
                fig = px.line(
                    monthly_resolution,
                    x='Month Name',
                    y='Resolution Time (Days)',
                    title=f"Monthly Average Resolution Time (Days) for {selected_year}",
                    markers=True,
                    category_orders={"Month Name": [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif time_period == "Month":
                # Daily average resolution times
                daily_resolution = valid_resolution_df.groupby('Created Day')['Resolution Time (Days)'].mean().reset_index()
                
                fig = px.line(
                    daily_resolution,
                    x='Created Day',
                    y='Resolution Time (Days)',
                    title=f"Daily Average Resolution Time (Days) for {datetime(selected_year, selected_month, 1).strftime('%B %Y')}",
                    markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # YTD
                # Monthly average resolution times for YTD
                monthly_resolution = valid_resolution_df.groupby(['Created Month', 'Month-Year Sort'])['Resolution Time (Days)'].mean().reset_index()
                monthly_resolution['Month Name'] = monthly_resolution['Created Month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
                monthly_resolution = monthly_resolution.sort_values('Month-Year Sort')
                
                fig = px.line(
                    monthly_resolution,
                    x='Month Name',
                    y='Resolution Time (Days)',
                    title=f"YTD Monthly Average Resolution Time (Days) for {selected_year}",
                    markers=True,
                    category_orders={"Month Name": [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Resolution time distribution
            fig = px.histogram(
                valid_resolution_df,
                x='Resolution Time (Days)',
                nbins=20,
                title="Resolution Time Distribution (Days)",
                labels={'Resolution Time (Days)': 'Resolution Time (Days)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Complaint Categories Analysis")
        
        # Check if Category column exists
        if 'Category' in filtered_df.columns:
            # Category distribution
            category_counts = filtered_df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            fig = px.pie(
                category_counts, 
                values='Count', 
                names='Category', 
                title="Complaint Categories Distribution",
                hole=0.4
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Check if there's a Status column
            if 'Status' in filtered_df.columns:
                # Category by status
                category_status = filtered_df.groupby(['Category', 'Status']).size().reset_index(name='Count')
                
                fig = px.bar(
                    category_status,
                    x='Category',
                    y='Count',
                    color='Status',
                    title="Complaints by Category and Status",
                    barmode='group'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            # If no Category column, check for Type or another similar field
            potential_category_cols = [col for col in filtered_df.columns if col in ['Type', 'Issue', 'Problem']]
            
            if potential_category_cols:
                category_col = potential_category_cols[0]
                category_counts = filtered_df[category_col].value_counts().reset_index()
                category_counts.columns = [category_col, 'Count']
                
                fig = px.pie(
                    category_counts, 
                    values='Count', 
                    names=category_col, 
                    title=f"Complaint {category_col} Distribution",
                    hole=0.4
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No category or type column found in the data for analysis.")

if __name__ == "__main__":
    main()