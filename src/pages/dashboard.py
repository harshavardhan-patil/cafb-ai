import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go
from src.data.preprocessing import preprocess
from src.dashboard.metrics import get_monthly_complaints


stats_path = RAW_DATA_DIR / "cafb_data_case2.xlsx"
df = pd.read_excel(str(stats_path))
logger.info("Read File")

df = preprocess(df)
####

# Monthly Complaints
st.plotly_chart(get_monthly_complaints(df))
