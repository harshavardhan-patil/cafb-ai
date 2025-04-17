import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards 
from src.preprocessing import preprocess
from src.dashboard.metrics import get_monthly_complaints
from src.dashboard.metrics import monthly_resolved_complaints
from src.dashboard.metrics import yearly_trends
from src.dashboard.metrics import monthly_trends
# from src.dashboard.metrics import daily_trends
from src.dashboard.metrics import category_bubble
from streamlit_dynamic_filters import DynamicFilters


print(f"{RAW_DATA_DIR}")
stats_path = RAW_DATA_DIR / "cafb_data_case2.xlsx"
df = pd.read_excel(str(stats_path))
logger.info("Read File")

df = preprocess(df)
####



st.title('🛃 Complaints Detailed Information')
### Single Selection Filters 
# add_selectbox = st.sidebar.selectbox(
#     "Priority",
#     (df['Priority'].drop_duplicates().values[0])
# )

# add_selectbox = st.sidebar.selectbox(
#     "Assignee",
#     tuple(df['Assignee'].drop_duplicates().values)
# )

# add_selectbox = st.sidebar.selectbox(
#     "Regions",
#     tuple(df['Custom field (Region)'].drop_duplicates().values)
# )

# add_selectbox = st.sidebar.selectbox(
#     "Source",
#     tuple(df['Custom field (Source)'].drop_duplicates().values)
# )

# modified_df = df.query(
#     "Priority == @Priority & Assignee == @Assignee &  Regions == @Regions & Source == @Source"
# )

class IndexValueError(Exception):
    pass

df['Custom field (Region)'] = df['Custom field (Region)'].astype(str)
df['Custom field (Source)'] = df['Custom field (Source)'].astype(str)
df.rename(columns={'Custom field (Region)':'Region','Custom field (Source)':'Source'}, inplace=True)


## Attempt on Dynamic Filtering
# # dynamic_filters = DynamicFilters(df, filters=['Priority', 'Assignee', 'Custom field (Region)', 'Custom field (Source)'],)
# dynamic_filters = DynamicFilters(df, filters=['Priority', 'Assignee', 'Custom field (Region)', 'Custom field (Source)'],)
# # filters_name=['Priority', 'Assignee', 'Region', 'Source']
# with st.sidebar:
#     dynamic_filters.display_filters()
# dynamic_filters.display_df()

st.markdown("""
    <style>
    .custom-multiselect {
        width: 400px;  /* Change this to your desired width */
    }
    </style>
""", unsafe_allow_html=True)

dash_1 = st.container()
with st.container():
    st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        priority_filter = st.multiselect('Select Priority', options=list(df['Priority'].unique()), default=list(df['Priority'].unique()))
    with col2:
        assignee_filter = st.multiselect('Select Assignee', options=list(df['Assignee'].unique()), default=list(df['Assignee'].unique()))
    with col3:
        region_filter = st.multiselect('Select Region', options=list(df['Region'].unique()), default=list(df['Region'].unique()))
    with col4:
        source_filter = st.multiselect('Select Source', options=list(df['Source'].unique()), default=list(df['Source'].unique()))
    st.markdown('</div>', unsafe_allow_html=True)


filtered_df = df[df['Priority'].isin(priority_filter) & df['Assignee'].isin(assignee_filter) & df['Region'].isin(region_filter) & df['Source'].isin(source_filter)]

st.write(filtered_df)

# print(filtered_df)

df = filtered_df
