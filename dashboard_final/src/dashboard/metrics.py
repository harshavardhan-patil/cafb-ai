import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output,callback

def get_monthly_complaints(df: pd.DataFrame) -> go.Figure:
    # Group by 'Created' date and calculate the average response time for each day
    daily_response_time = df.groupby(df['Created Month'])['Issue id'].count()

    # Create a plotly figure
    fig = go.Figure()

    # Add a line trace
    fig.add_trace(
        go.Scatter(
            x=daily_response_time.index,
            y=daily_response_time.values,
            mode='lines+markers',
            name='Monthly Complaints'
        )
    )

    fig.update_layout(
        title='Monthly Created Complaints',
        xaxis=dict(
            title='Month',
            tickangle=45
        ),
        yaxis=dict(
            title='Number of Complaints Created'
        ),
        width=1000,
        height=600,
        template='plotly_white'  # Clean white background with grid
    )

    return fig

def monthly_resolved_complaints(df: pd.DataFrame) -> go.Figure:
    monthly_complaints = df.groupby(df['Resolved Month'])['Issue id'].count()
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add a line trace
    fig.add_trace(
        go.Scatter(
            x=monthly_complaints.index,
            y=monthly_complaints.values,
            mode='lines+markers',
            name='Monthly Resolved Complaints'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Monthly Resolved Complaints',
        xaxis=dict(
            title='Month',
            tickangle=45
        ),
        yaxis=dict(
            title='Number of Complaints Resolved'
        ),
        width=1000,
        height=600,
        template='plotly_white'  # Clean white background with grid
    )

    return fig


def yearly_trends(df: pd.DataFrame) -> go.Figure:
    # df['Resolved Year'] = df['Resolved Year'].astype(int)
    # df['Created Year'] = df['Created Year'].astype(int)
    # st.write(df["Resolved Year"].drop_duplicates())
    # st.write(df["Created Year"].drop_duplicates())
    
    # yearly_resolved_complaints = df.groupby(df['Resolved Year'].astype(int))['Issue id'].count()
    # yearly_created_complaints = df.groupby(df['Created Year'].astype(int))['Issue id'].count()
    
    # # Create a plotly figure
    # fig = go.Figure()
    
    # # Add a line trace
    # trace1 = go.Scatter(
    #         x=yearly_resolved_complaints.index,
    #         y=yearly_resolved_complaints.values,
    #         mode='lines+markers',
    #         name='Yearly Resolved Complaints'
    #     )
    # trace2 = go.Scatter(
    #         x=yearly_created_complaints.index,
    #         y=yearly_created_complaints.values,
    #         mode='lines+markers',
    #         name='Yearly Created Complaints'
    #     )
    
    # # Create layout
    # layout = go.Layout(title='Yearly Trend of Complaints',
    #                    xaxis=dict(title='Year'),
    #                    yaxis=dict(title='No. of Complaints'))
    
    # # Create figure
    # fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    # fig.update_xaxes(categoryorder='array', categoryarray=['2021', '2022', '2023','2024','2025'])

    df['Resolved Year'] = df['Resolved Year'].astype(int)
    df['Created Year'] = df['Created Year'].astype(int)
    # st.write(df["Resolved Year"].drop_duplicates())
    # st.write(df["Created Year"].drop_duplicates())

    yearly_resolved_complaints = df.groupby(df['Resolved Year'])['Issue id'].count()
    yearly_created_complaints = df.groupby(df['Created Year'])['Issue id'].count()

    # st.write(yearly_resolved_complaints)
    # st.write(yearly_created_complaints)

    # Create a plotly figure
    fig = go.Figure()

    # Add a line trace
    trace1 = go.Scatter(
            x=yearly_resolved_complaints.index,
            y=yearly_resolved_complaints.values,
            mode='lines+markers',
            name='Yearly Resolved Complaints'
        )
    trace2 = go.Scatter(
            x=yearly_created_complaints.index,
            y=yearly_created_complaints.values,
            mode='lines+markers',
            name='Yearly Created Complaints'
        )

    # Create layout
    layout = go.Layout(title='Yearly Trend of Complaints',
                       xaxis=dict(title='Year'),
                       yaxis=dict(title='No. of Complaints'))

    # Create figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    fig.update_xaxes(categoryorder='array', categoryarray=['2021', '2022', '2023','2024','2025'])
    
    # Show figure
    return fig

# def monthly_trends(df: pd.DataFrame) -> go.Figure:
    # app = Dash()
    # app.layout = html.Div([
    #     dcc.Dropdown(df['Year'], 'Year', id='demo-dropdown'),
    #     html.Div(id='dd-output-container')
    # ])
    
    
    # @callback(
    #     Output('dd-output-container', 'children'),
    #     Input('demo-dropdown', 'value')
    # )
    # def update_output(value):
    #     return f'You have selected {value}'
    
    
    # if __name__ == '__main__':
    #     app.run(debug=True)

    # selected_year = "2024"
    
    
    # df["Resolved Month Name"] = df["Resolved Month"].map({1:"January",2:"Feburary",3:"March",4:"April",5:"May",6:"June",
    #                                                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"})
    
    # df["Created Month Name"] = df["Created Month"].map({1:"January",2:"Feburary",3:"March",4:"April",5:"May",6:"June",
    #                                                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"})
    
    # df_month = df[["Resolved Year", "Resolved Month", "Resolved Month Name","Resolved Day", "Created Year", "Created Month", "Created Month Name", "Created Day","Issue id"]] \
    #             .sort_values(by=["Created Year", "Created Month"])
    
    # # Group by 'Created' date and calculate the average response time for each day
    # monthly_resolved_complaints = df_month[df_month["Resolved Year"].astype(str)==selected_year].groupby(df_month['Resolved Month Name'])['Issue id'].count()
    # monthly_created_complaints = df_month[df_month["Created Year"].astype(str)==selected_year].groupby(df_month['Created Month Name'])['Issue id'].count()
    
    # # print(yearly_complaints.index)
    
    # # print(monthly_resolved_complaints)
    
    # # Create a plotly figure
    # fig = go.Figure()
    
    # # Add a line trace
    # trace1 = go.Scatter(
    #         x=monthly_resolved_complaints.index,
    #         y=monthly_resolved_complaints.values,
    #         mode='lines+markers',
    #         name='Monthly Resolved Complaints'
    #     )
    # trace2 = go.Scatter(
    #         x=monthly_created_complaints.index,
    #         y=monthly_created_complaints.values,
    #         mode='lines+markers',
    #         name='Monthly Created Complaints'
    #     )
    
    # # Create layout
    # layout = go.Layout(title='Monthly Trend of Complaints in Selected Year',
    #                    xaxis=dict(title='Month'),
    #                    yaxis=dict(title='No. of Complaints'))
    
    # # Create figure
    # fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    
    # fig.update_xaxes(categoryorder='array', categoryarray=["January","Feburary","March","April","May","June","July","August","September","October","November","December"])
    
    
    # # Show figure
    # fig.show()


def test_monthly(df):
    with st.container():
        st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # Convert to strings for consistent comparison
            created_years = sorted([str(year) for year in df['Created Year'].unique()])
            created_ticket_filter = st.selectbox(
                'Select Year for Created Ticket', 
                options=created_years, 
                key="created_ticket"
            )
        with col2:
            # Convert to strings for consistent comparison
            resolved_years = sorted([str(year) for year in df['Resolved Year'].unique()])
            resolved_ticket_filter = st.selectbox(
                'Select Year for Ticket Resolved', 
                options=resolved_years
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Debug information
    # st.write(f"Selected Created Year: {created_ticket_filter}")
    # st.write(f"Selected Resolved Year: {resolved_ticket_filter}")
    
    # Display data types to help with debugging
    # st.write("Data types in dataframe:")
    # st.write(f"Created Year column type: {df['Created Year'].dtype}")
    # st.write(f"Resolved Year column type: {df['Resolved Year'].dtype}")
    # st.write(f"Created Year filter type: {type(created_ticket_filter)}")
    # st.write(f"Resolved Year filter type: {type(resolved_ticket_filter)}")
    
    # Correctly spelled month names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 
        5: "May", 6: "June", 7: "July", 8: "August", 
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    # Create mapping for sorting
    month_order = {month: i for i, month in enumerate(month_names.values())}
    
    df["Resolved Month Name"] = df["Resolved Month"].map(month_names)
    df["Created Month Name"] = df["Created Month"].map(month_names)
    
    df_month = df[["Resolved Year", "Resolved Month", "Resolved Month Name", "Resolved Day", 
                   "Created Year", "Created Month", "Created Month Name", "Created Day", "Issue id"]] \
                .sort_values(by=["Created Year", "Created Month"])
    
    # Convert both the dataframe column and filter value to string for reliable comparison
    resolved_df = df_month[df_month["Resolved Year"].astype(str) == resolved_ticket_filter]
    # st.write(f"Filtered Resolved Tickets (found {len(resolved_df)} rows):")
    # st.write(resolved_df)
    
    created_df = df_month[df_month["Created Year"].astype(str) == created_ticket_filter]
    # st.write(f"Filtered Created Tickets (found {len(created_df)} rows):")
    # st.write(created_df)
    
    # If filters still don't work, let's add a more direct debugging test
    st.subheader("Testing filtering directly:")
    # Check first 5 values in each column
    # st.write("First 5 Created Year values:", df_month["Created Year"].head().tolist())
    # st.write("First 5 Resolved Year values:", df_month["Resolved Year"].head().tolist())
    
    # Try manual filtering for debugging
    test_created_year = df_month["Created Year"].iloc[0]  # Get first value
    test_created_df = df_month[df_month["Created Year"] == test_created_year]
    # st.write(f"Test filter with exact first value ({test_created_year}): {len(test_created_df)} rows found")
    
    # Try string conversion for debugging
    test_created_df_str = df_month[df_month["Created Year"].astype(str) == str(test_created_year)]
    # st.write(f"Test filter with string conversion ({str(test_created_year)}): {len(test_created_df_str)} rows found")




def monthly_trends(df: pd.DataFrame) -> go.Figure:


#     with st.container():
#         st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             created_ticket_filter = st.selectbox('Select Year for Created Ticket', options=list(df['Created Year'].unique()), key="created_ticket")
#         with col2:
#             resolved_ticket_filter = st.selectbox('Select Year for Ticket Resolved', options=list(df['Resolved Year'].unique()))
#         # with col3:
#         #     region_filter = st.multiselect('Select Region', options=list(df['Region'].unique()), default=list(df['Region'].unique()))
#         # with col4:
#         #     source_filter = st.multiselect('Select Source', options=list(df['Source'].unique()), default=list(df['Source'].unique()))
#         st.markdown('</div>', unsafe_allow_html=True)

#     # filtered_df = df[df['Created Year'].isin(created_ticket_filter) & df['Resolved Year'].isin(resolved_ticket_filter)]


#     # filtered_df = df[(df['Created Year']==created_ticket_filter) & df['Resolved Year']==resolved_ticket_filter]
#     # df = filtered_df
#     # st.write(df)
# #     dash_2 = st.container()
# #     with dash_2:
# #         created_ticket_filter = st.multiselect('Select Year for Created Ticket', options=list(df['Created Year'].unique()), key="created_ticket")
        
    
# #     st.markdown("""
# #     <style>
# #     .custom-multiselect {
# #         width: 400px;  /* Change this to your desired width */
# #     }
# #     </style>
# # """, unsafe_allow_html=True)

    
    
# #     st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
# #     created_ticket_filter = st.multiselect('Select Year for Created Ticket', options=list(df['Created Year'].unique()))
# #     st.markdown('</div>', unsafe_allow_html=True)

# #     st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
# #     resolved_ticket_filter = st.multiselect('Select Year for Ticket Resolved', options=list(df['Resolved Year'].unique()))
# #     st.markdown('</div>', unsafe_allow_html=True)

#     # selected_year = created_ticket_filter
#     # resolved_year = resolved_ticket_filter
#     st.write(df)
#     selected_year = f"{created_ticket_filter.astype(str)}"
#     st.write(selected_year)
#     # st.write(type(selected_year))
#     resolved_year = f"{resolved_ticket_filter.astype(str)}"
#     st.write(resolved_year)
#     # st.write(type(resolved_year))
# #     st.write(selected_year)
# #     st.write(resolved_year)
    
#     # Correctly spelled month names
#     month_names = {
#         1: "January", 2: "February", 3: "March", 4: "April", 
#         5: "May", 6: "June", 7: "July", 8: "August", 
#         9: "September", 10: "October", 11: "November", 12: "December"
#     }
    
#     # Create a mapping for proper sorting
#     month_order = {month: i for i, month in enumerate(month_names.values())}
    
#     df["Resolved Month Name"] = df["Resolved Month"].map(month_names)
#     df["Created Month Name"] = df["Created Month"].map(month_names)
    
#     df_month = df[["Resolved Year", "Resolved Month", "Resolved Month Name","Resolved Day", 
#                    "Created Year", "Created Month", "Created Month Name", "Created Day","Issue id"]] \
#                 .sort_values(by=["Created Year", "Created Month"])
    
#     # Filter by selected year
#     # resolved_df = df_month[df_month["Resolved Year"].astype(str)==resolved_year]
#     resolved_df = df_month[df_month["Resolved Year"]==resolved_year]
#     st.write(resolved_df)
#     created_df = df_month[df_month["Created Year"]==selected_year]
#     st.write(created_df)

    with st.container():
        st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # Convert to strings for consistent comparison
            created_years = sorted([str(year) for year in df['Created Year'].unique()])
            created_ticket_filter = st.selectbox(
                'Select Year for Created Ticket', 
                options=created_years, 
                key="created_ticket"
            )
        with col2:
            # Convert to strings for consistent comparison
            resolved_years = sorted([str(year) for year in df['Resolved Year'].unique()])
            resolved_ticket_filter = st.selectbox(
                'Select Year for Ticket Resolved', 
                options=resolved_years
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Debug information
    # st.write(f"Selected Created Year: {created_ticket_filter}")
    # st.write(f"Selected Resolved Year: {resolved_ticket_filter}")
    
    # Display data types to help with debugging
    # st.write("Data types in dataframe:")
    # st.write(f"Created Year column type: {df['Created Year'].dtype}")
    # st.write(f"Resolved Year column type: {df['Resolved Year'].dtype}")
    # st.write(f"Created Year filter type: {type(created_ticket_filter)}")
    # st.write(f"Resolved Year filter type: {type(resolved_ticket_filter)}")
    
    # Correctly spelled month names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 
        5: "May", 6: "June", 7: "July", 8: "August", 
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    # Create mapping for sorting
    month_order = {month: i for i, month in enumerate(month_names.values())}
    
    df["Resolved Month Name"] = df["Resolved Month"].map(month_names)
    df["Created Month Name"] = df["Created Month"].map(month_names)
    
    df_month = df[["Resolved Year", "Resolved Month", "Resolved Month Name", "Resolved Day", 
                   "Created Year", "Created Month", "Created Month Name", "Created Day", "Issue id"]] \
                .sort_values(by=["Created Year", "Created Month"])
    
    # Convert both the dataframe column and filter value to string for reliable comparison
    resolved_df = df_month[df_month["Resolved Year"].astype(str) == resolved_ticket_filter]
    # st.write(f"Filtered Resolved Tickets (found {len(resolved_df)} rows):")
    # st.write(resolved_df)
    
    created_df = df_month[df_month["Created Year"].astype(str) == created_ticket_filter]
    # st.write(f"Filtered Created Tickets (found {len(created_df)} rows):")
    # st.write(created_df)
    
    # Group by month name
    monthly_resolved_complaints = resolved_df.groupby('Resolved Month Name')['Issue id'].count()
    monthly_created_complaints = created_df.groupby('Created Month Name')['Issue id'].count()
    
    # Convert to DataFrame for easier sorting
    resolved_counts = pd.DataFrame({
        'Month': monthly_resolved_complaints.index,
        'Count': monthly_resolved_complaints.values
    })
    created_counts = pd.DataFrame({
        'Month': monthly_created_complaints.index,
        'Count': monthly_created_complaints.values
    })
    
    created_resolved_df = pd.merge(resolved_counts, created_counts, left_on="Month", right_on="Month", how="outer")
    created_resolved_df.fillna(0, inplace=True)
    created_resolved_df['Count_x'] = created_resolved_df['Count_x'].astype(int)
    created_resolved_df['Count_y'] = created_resolved_df['Count_y'].astype(int)
    # print(created_resolved_df)
    
    resolved_counts = created_resolved_df[["Month","Count_x"]].rename(columns={"Count_x":"Count"})
    created_counts = created_resolved_df[["Month", "Count_y"]].rename(columns={"Count_y":"Count"})
    
    # resolved_counts.insert({"Month":"January","Count":0})
    # # Sort by the correct month order
    resolved_counts['MonthOrder'] = resolved_counts['Month'].map(month_order)
    created_counts['MonthOrder'] = created_counts['Month'].map(month_order)
    resolved_counts = resolved_counts.sort_values('MonthOrder')
    # # print(resolved_counts.sort_values('MonthOrder'))
    # # resolved_counts.append({"January":0, "February":0, "March":0})
    # print(resolved_counts.sort_values('MonthOrder'))
    # # print(created_counts.sort_values('MonthOrder'))
    created_counts = created_counts.sort_values('MonthOrder')
    # print(created_counts)
    
    # Create a plotly figure
    fig = go.Figure()
    
    # # Add line traces with sorted data
    fig.add_trace(go.Scatter(
        x=resolved_counts['Month'],
        y=resolved_counts['Count'],
        mode='lines+markers',
        name='Monthly Resolved Complaints'
    ))
    
    fig.add_trace(go.Scatter(
        x=created_counts['Month'],
        y=created_counts['Count'],
        mode='lines+markers',
        name='Monthly Created Complaints'
    ))
    
    # Create layout
    fig.update_layout(
        title='Monthly Trend of Complaints in Selected Year',
        xaxis=dict(title='Month'),
        yaxis=dict(title='No. of Complaints')
    )
    
    # Show figure
    return fig


def daily_trends(df: pd.DataFrame) -> go.Figure:

    # Extract date components
    df['Created Day'] = df['Created'].dt.day
    df['Created Month'] = df['Created'].dt.month
    df['Created Year'] = df['Created'].dt.year
    
    # Create proper date strings in sortable YYYY-MM-DD format
    df["Created Daily"] = pd.to_datetime(df['Created'].dt.date)
    
    df['Resolved Day'] = df['Resolved'].dt.day
    df['Resolved Month'] = df['Resolved'].dt.month
    df['Resolved Year'] = df['Resolved'].dt.year
    
    # Create proper date strings in sortable YYYY-MM-DD format
    df["Resolved Daily"] = pd.to_datetime(df['Resolved'].dt.date)
    
    # Create month name columns using a mapping
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 
        5: "May", 6: "June", 7: "July", 8: "August", 
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    df["Resolved Month Name"] = df["Resolved Month"].map(month_names)
    df["Created Month Name"] = df["Created Month"].map(month_names)
    
    # Keep relevant columns in a new dataframe
    df_month = df[["Resolved Year", "Resolved Month", "Resolved Month Name", "Resolved Day",
                   "Created Year", "Created Month", "Created Month Name", "Created Day", 
                   "Issue id", "Created Daily", "Resolved Daily"]]
    
    # Initial filtering (you can add your filter conditions here)
    resolved_df = df_month
    created_df = df_month
    
    # Group by actual datetime objects for proper sorting
    daily_created_complaints = created_df.groupby('Created Daily')['Issue id'].count()
    daily_resolved_complaints = resolved_df.groupby('Resolved Daily')['Issue id'].count()
    
    # Convert to DataFrames
    resolved_counts = pd.DataFrame({
        'Day': daily_resolved_complaints.index,
        'Count': daily_resolved_complaints.values
    })
    created_counts = pd.DataFrame({
        'Day': daily_created_complaints.index,
        'Count': daily_created_complaints.values
    })
    
    # Merge the dataframes
    created_resolved_df = pd.merge(created_counts, resolved_counts, 
                                   left_on="Day", right_on="Day", 
                                   how="outer", suffixes=('_created', '_resolved'))
    created_resolved_df.fillna(0, inplace=True)
    created_resolved_df['Count_created'] = created_resolved_df['Count_created'].astype(int)
    created_resolved_df['Count_resolved'] = created_resolved_df['Count_resolved'].astype(int)
    
    # Sort the merged dataframe by date
    created_resolved_df = created_resolved_df.sort_values('Day')
    
    # Create individual dataframes for plotting
    resolved_counts = created_resolved_df[["Day", "Count_resolved"]].rename(columns={"Count_resolved": "Count"})
    created_counts = created_resolved_df[["Day", "Count_created"]].rename(columns={"Count_created": "Count"})
    
    # Format dates for display (optional)
    created_resolved_df['Day_formatted'] = created_resolved_df['Day'].dt.strftime('%d-%b-%Y')
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add line traces
    fig.add_trace(go.Scatter(
        x=resolved_counts['Day'],
        y=resolved_counts['Count'],
        mode='lines+markers',
        name='Daily Resolved Complaints'
    ))
    
    fig.add_trace(go.Scatter(
        x=created_counts['Day'],
        y=created_counts['Count'],
        mode='lines+markers',
        name='Daily Created Complaints'
    ))
    
    # Update layout
    fig.update_layout(
        title='Daily Trend of Complaints',
        xaxis=dict(
            title='Date',
            tickformat='%d-%b-%Y',  # Format as DD-MMM-YYYY
            tickangle=-45
        ),
        yaxis=dict(title='No. of Complaints'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
# Show figure
# st.plotly_chart(fig)

# # Optional: Display the data in a table
# with st.expander("Show Data Table"):
#     display_df = created_resolved_df[['Day_formatted', 'Count_created', 'Count_resolved']].rename(
#         columns={
#             'Day_formatted': 'Date',
#             'Count_created': 'Created Complaints',
#             'Count_resolved': 'Resolved Complaints'
#         }
#     )
#     st.dataframe(display_df)

def category_bubble(df: pd.DataFrame) -> go.Figure:
    ## Improvement - This can be shown in different tab on the basis of number of complaints and resolution time
    
    df['Created Year'] = df['Created Year'].astype(str)
    grouped = df.groupby(['Created Year', 'Custom field (Request Type)']).agg(count=('Issue id', 'count')).reset_index()
    
    fig = px.scatter(
        grouped,
        x='Created Year',
        y='Custom field (Request Type)',
        size='count',
        color='count',
        hover_name='Custom field (Request Type)',
        size_max=60,
        color_continuous_scale='RdYlGn_r',  # Red for longer resolution times, green for shorter
        hover_data={
            'count': True
        },
        title='Analysis of Precised Issues'
    )
    
    # Update layout for better readability
    fig.update_layout(
        height=800,
        width=1200,
        xaxis_title='Year',
        yaxis_title='Issue Type',
        coloraxis_colorbar_title='Complaints',
        legend_title='Issue Count',
        font=dict(size=12),
        template='plotly_white'
    )
    
    return fig
        
