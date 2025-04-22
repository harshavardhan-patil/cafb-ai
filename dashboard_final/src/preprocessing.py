import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
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

    # Extract day, month, and year from 'Created' and 'Resolved' dates
    df['Created Day'] = df['Created'].dt.day
    df['Created Month'] = df['Created'].dt.month
    df['Created Year'] = df['Created'].dt.year

    df['Resolved Day'] = df['Resolved'].dt.day
    df['Resolved Month'] = df['Resolved'].dt.month
    df['Resolved Year'] = df['Resolved'].dt.year

    return df