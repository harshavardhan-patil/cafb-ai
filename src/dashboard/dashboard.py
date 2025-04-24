import streamlit as st
from src.config import DASHBOARD_PAGES_DIR

st.set_page_config(layout="wide")

pg = st.navigation([st.Page(DASHBOARD_PAGES_DIR / "Overview.py"), st.Page(DASHBOARD_PAGES_DIR / "Trends.py"),  st.Page(DASHBOARD_PAGES_DIR / "Topic Modelling.py"), st.Page(DASHBOARD_PAGES_DIR / "Complaints Information.py")])
pg.run()
