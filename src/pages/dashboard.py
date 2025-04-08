import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go
from src.data.preprocessing import preprocess
from src.dashboard.metrics import get_monthly_complaints

with open('src/pages/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

stats_path = RAW_DATA_DIR / "cafb_data_case2.xlsx"
df = pd.read_excel(str(stats_path))
logger.info("Read File")

df = preprocess(df)
####


st.markdown('### KPI')
col1, col2, col3 = st.columns(3)
col1.metric("Closed", int(df[df['Status']=="Closed"].groupby("Status")["Issue id"].count().values[0]))

# Monthly Complaints
st.plotly_chart(get_monthly_complaints(df))
