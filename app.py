import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go

#converting static image and setting as website background
st.title("CAFB AI Support System")

pg = st.navigation([st.Page("src/pages/support.py"), st.Page("src/pages/dashboard.py")])
pg.run()
