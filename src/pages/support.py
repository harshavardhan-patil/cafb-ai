import streamlit as st
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR
import base64
from loguru import logger
import plotly.graph_objects as go

#converting static image and setting as website background
st.title("How can we help you?")

prompt = st.chat_input("Start typing...")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")


