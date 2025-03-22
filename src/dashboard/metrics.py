import pandas as pd
import numpy as np
import plotly.graph_objects as go


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