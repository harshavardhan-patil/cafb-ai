import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go

st.set_page_config(layout="wide")

pg = st.navigation([st.Page("src/pages/Overview.py"), st.Page("src/pages/Support.py"), st.Page("src/pages/Trends.py"),  st.Page("src/pages/Topic Modelling.py"), st.Page("src/pages/Complaints Information.py")])
pg.run()
