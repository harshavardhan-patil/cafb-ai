
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime
from src.config import RAW_DATA_DIR
from loguru import logger

print(f"{RAW_DATA_DIR}")
stats_path = RAW_DATA_DIR / "jira_issues.csv"
df = pd.read_csv(str(stats_path))
logger.info("Read File")

# Load data
# df = pd.read_csv("jira_issues.csv")

# st.set_page_config(layout="wide")
st.title("📊 JIRA Issues Dashboard")

# ---- 1. Document Heatmap ----
st.header("📅 Document Heatmap (Issue Creation Timeline)")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['created_date'] = df['created_at'].dt.date

heat_data = df.groupby('created_date').size().reset_index(name='count')
fig1 = px.density_heatmap(heat_data, x="created_date", y="count", nbinsx=20, color_continuous_scale="Viridis")
st.plotly_chart(fig1, use_container_width=True)

# ---- 2. Word Cloud ----
st.header("☁️ Word Clouds")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Summary Word Cloud")
    summary_text = " ".join(df['summary'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(summary_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.subheader("Description Word Cloud")
    desc_text = " ".join(df['description'].dropna().astype(str))
    wc2 = WordCloud(width=800, height=400, background_color='white').generate(desc_text)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wc2, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)

# ---- 3. Topic Modeling ----
st.header("📌 Topic Modeling & Distribution")

text_data = df['summary'].fillna('') + " " + df['description'].fillna('')
vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(text_data)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(doc_term_matrix)

topic_words = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
    topic_words.append(", ".join(top_words))

topic_distribution = lda.transform(doc_term_matrix)
topic_counts = np.argmax(topic_distribution, axis=1)
topic_freq = pd.Series(topic_counts).value_counts().sort_index()
topic_labels = [f"Topic {i+1}: {tw}" for i, tw in enumerate(topic_words)]

fig3 = px.pie(values=topic_freq.values, names=topic_labels, title="Topic Distribution", hole=0.4)
st.plotly_chart(fig3, use_container_width=True)
