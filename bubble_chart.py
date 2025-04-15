import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the Excel file
df = pd.read_excel('data/raw/cafb_data_case2.xlsx', sheet_name='PSUP_Jira_Data_Email_Scrambled')

# Data preprocessing
# Convert date columns to datetime
date_columns = ['Created', 'Updated', 'Resolved', 'Status Category Changed']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate resolution time in days
df['Resolution_Time_Days'] = (df['Resolved'] - df['Created']).dt.total_seconds() / (60*60*24)

# Clean and preprocess the data
# Focus on Issue Type and Request Type as main categories
df = df[df['Resolution_Time_Days'] >= 0]  # Remove any negative resolution times (data errors)

# Group by Issue Type and Request Type to count issues and calculate metrics
grouped = df.groupby(['Issue Type', 'Custom field (Request Type)']).agg(
    count=('Issue id', 'count'),
    avg_resolution_time=('Resolution_Time_Days', 'mean'),
    priority_high=('Priority', lambda x: (x == 'High').sum()),
    priority_medium=('Priority', lambda x: (x == 'Medium').sum()),
    priority_low=('Priority', lambda x: (x == 'Low').sum()),
).reset_index()

# Calculate priority ratio (high vs low+medium) for color intensity
grouped['priority_ratio'] = grouped['priority_high'] / (grouped['priority_medium'] + grouped['priority_low'] + 1)

# Create a bubble chart
fig = px.scatter(
    grouped,
    x='Issue Type',
    y='Custom field (Request Type)',
    size='count',
    color='avg_resolution_time',
    hover_name='Custom field (Request Type)',
    size_max=60,
    color_continuous_scale='RdYlGn_r',  # Red for longer resolution times, green for shorter
    hover_data={
        'count': True,
        'avg_resolution_time': ':.2f',
        'priority_high': True,
        'priority_medium': True,
        'priority_low': True
    },
    title='Service Request Analysis: Issue Type vs Request Type'
)

# Update layout for better readability
fig.update_layout(
    height=800,
    width=1200,
    xaxis_title='Issue Type',
    yaxis_title='Request Category',
    coloraxis_colorbar_title='Avg. Resolution<br>Time (Days)',
    legend_title='Issue Count',
    font=dict(size=12),
    template='plotly_white'
)

# Create a second chart - Resolution time by request type
request_type_analysis = df.groupby('Custom field (Request Type)').agg(
    count=('Issue id', 'count'),
    avg_resolution_time=('Resolution_Time_Days', 'mean'),
    median_resolution_time=('Resolution_Time_Days', 'median'),
    max_resolution_time=('Resolution_Time_Days', 'max')
).reset_index().sort_values('count', ascending=False).head(10)

fig2 = px.bar(
    request_type_analysis,
    x='Custom field (Request Type)', 
    y='avg_resolution_time',
    color='count',
    text='count',
    title='Top 10 Request Categories by Volume with Resolution Time',
    labels={'Custom field (Request Type)': 'Request Category', 
            'avg_resolution_time': 'Average Resolution Time (Days)',
            'count': 'Number of Requests'},
    color_continuous_scale='Viridis'
)

fig2.update_layout(
    height=600,
    width=1100,
    xaxis_title='Request Category',
    yaxis_title='Average Resolution Time (Days)',
    coloraxis_colorbar_title='Request<br>Count',
    template='plotly_white'
)

# Create a third chart - Priority distribution by Issue Type
# priority_dist = df.groupby(['Issue Type', 'Priority']).size().unstack().fillna(0)
# priority_dist = priority_dist.div(priority_dist.sum(axis=1), axis=0) * 100

# fig3 = px.bar(
#     priority_dist.reset_index(),
#     x='Issue Type',
#     y=['High', 'Medium', 'Low'],
#     title='Priority Distribution by Issue Type',
#     labels={'value': 'Percentage', 'variable': 'Priority Level'},
#     barmode='stack'
# )

# fig3.update_layout(
#     height=600,
#     width=1100,
#     xaxis_title='Issue Type',
#     yaxis_title='Percentage',
#     legend_title='Priority',
#     template='plotly_white'
# )

# # Create a fourth chart - Monthly trends
# df['month_year'] = df['Created'].dt.strftime('%Y-%m')
# monthly_trends = df.groupby('month_year').agg(
#     count=('Issue id', 'count'),
#     avg_resolution_time=('Resolution_Time_Days', 'mean')
# ).reset_index()

# fig4 = make_subplots(specs=[[{"secondary_y": True}]])

# fig4.add_trace(
#     go.Bar(
#         x=monthly_trends['month_year'],
#         y=monthly_trends['count'],
#         name='Request Volume',
#         marker_color='lightblue'
#     ),
#     secondary_y=False
# )

# fig4.add_trace(
#     go.Scatter(
#         x=monthly_trends['month_year'],
#         y=monthly_trends['avg_resolution_time'],
#         name='Avg Resolution Time',
#         marker_color='red',
#         mode='lines+markers'
#     ),
#     secondary_y=True
# )

# fig4.update_layout(
#     title_text='Monthly Request Volume and Resolution Time Trends',
#     height=600,
#     width=1100,
#     template='plotly_white'
# )

# fig4.update_xaxes(title_text='Month')
# fig4.update_yaxes(title_text='Number of Requests', secondary_y=False)
# fig4.update_yaxes(title_text='Avg Resolution Time (Days)', secondary_y=True)

# Save or display the figures
# fig.show()
# fig2.show()
# fig3.show()
# fig4.show()

# Generate insights and recommendations based on the analysis
def generate_insights():
    insights = []
    
    # Find request types with longest resolution times
    slow_resolution = df.groupby('Custom field (Request Type)').filter(lambda x: len(x) >= 10).groupby('Custom field (Request Type)')['Resolution_Time_Days'].mean().sort_values(ascending=False).head(3)
    insights.append(f"Request types with longest resolution times: {', '.join(slow_resolution.index.tolist())}")
    
    # Find request types with highest volume
    high_volume = df['Custom field (Request Type)'].value_counts().head(3)
    insights.append(f"Highest volume request types: {', '.join(high_volume.index.tolist())}")
    
    # Find issues with high priority that take longest to resolve
    high_priority_slow = df[df['Priority'] == 'High'].groupby('Custom field (Request Type)')['Resolution_Time_Days'].mean().sort_values(ascending=False).head(3)
    insights.append(f"High priority request types with longest resolution times: {', '.join(high_priority_slow.index.tolist())}")
    
    # Check if there's a trend in resolution time over months
    # if len(monthly_trends) > 3:
    #     recent_trend = monthly_trends.iloc[-3:]['avg_resolution_time'].tolist()
    #     if recent_trend[-1] > recent_trend[0]:
    #         insights.append(f"Resolution times are increasing over the last 3 months: {recent_trend}")
    #     else:
    #         insights.append(f"Resolution times are decreasing over the last 3 months: {recent_trend}")
    
    return insights

# Print out insights
insights = generate_insights()
for insight in insights:
    print(insight)

# Return HTML or save figures to files
# You can adjust the output format based on your needs
pio.write_html(fig, 'issue_type_vs_request_type_bubble_chart.html')
pio.write_html(fig2, 'top_request_categories.html')
# pio.write_html(fig3, 'priority_distribution.html')
# pio.write_html(fig4, 'monthly_trends.html')

print("Analysis complete. Visualizations and insights generated.")