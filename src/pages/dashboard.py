import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards 
from src.data.preprocessing import preprocess
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

# Monthly Complaints
st.markdown("""
    <style>
        .top-bar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 200px;
            background-color: black;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            z-index: 1000;
        }
        .main {
            margin-top: 60px; /* Adjust content to avoid overlap with the top bar */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-bar">', unsafe_allow_html=True)Dashboard Header
st.markdown('</div>', unsafe_allow_html=True)



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

# st.write(filtered_df)

# print(filtered_df)

df = filtered_df


### Dashboard Content 
    # Metrics row
dash_2 = st.container()
with dash_2:
    col1, col2, col3, col4 = st.columns(4)
    try:
        with col1:
            st.metric("Closed Complaints", f"{df[df['Status']=='Closed'].groupby('Status')['Issue id'].count().values[0]:,}")
    except IndexError:
        st.error("We do not have closed complaints based upon the filter selection. Please change your selction criteria to monitor the complaints")

    try:        
        with col2:
            st.metric("Completed Complaints", f"{df[df['Status']=='Completed'].groupby('Status')['Issue id'].count().values[0]:,}")
    except IndexError:
        st.error("We do not have completed complaints based upon the filter selection. Please change your selction criteria to monitor the complaints")

    try:
        with col3:
            st.metric("Cancelled Complaints", f"{df[df['Status']=='Canceled'].groupby('Status')['Issue id'].count().values[0]:,}")
    except IndexError:
        st.error("We do not have cancelled complaints based upon the filter selection. Please change your selction criteria to monitor the complaints")
    
    try:   
        with col4:
            st.metric("Average Satisfaction Rating", f"{int(df['Satisfaction rating'].mean())}")
    except IndexError:
        st.error("We do not have average satisfaction rating based upon the filter selection. Please change your selction criteria to monitor the complaints")
    
    style_metric_cards(border_left_color="#4E8226")



dash_3 = st.container()
tab1, tab2, tab3 = st.tabs(["Yearly Trends", "Monthly Trends", "Daily Trends"])

with tab1:
    st.plotly_chart(yearly_trends(df))

with tab2:
    st.plotly_chart(monthly_trends(df))
    
with tab3:
    st.plotly_chart(daily_trends(df))

dash_4 = st.container()
with dash_4:
    st.plotly_chart(category_bubble(filtered_df))

### Cancelling this as of now
# # Charts row
# dash_3 = st.container()
# with dash_3:
#     chart_col1, chart_col2 = st.columns(2)
    
#     with chart_col1:
#         # st.subheader("Sales Trend")
#         # monthly_sales = df.groupby(pd.Grouper(key='date', freq='M')).agg({'sales': 'sum'}).reset_index()
#         # fig, ax = plt.subplots(figsize=(10, 6))
#         # ax.plot(monthly_sales['date'], monthly_sales['sales'], marker='o', linewidth=2)
#         # ax.set_ylabel('Sales ($)')
#         # ax.grid(True, alpha=0.3)
#         # st.pyplot(fig)
#         st.plotly_chart(get_monthly_complaints(df))
    
#     with chart_col2:
#         # st.subheader("Sales by Category")
#         # category_sales = df.groupby('category').agg({'sales': 'sum'}).reset_index()
#         # fig, ax = plt.subplots(figsize=(10, 6))
#         # sns.barplot(x='category', y='sales', data=category_sales, ax=ax)
#         # ax.set_ylabel('Sales ($)')
#         # ax.set_xlabel('')
#         # ax.grid(True, axis='y', alpha=0.3)
#         # st.pyplot(fig)
#         st.plotly_chart(monthly_resolved_complaints(df))