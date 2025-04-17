
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import seaborn as sns
import plotly.graph_objects as go

# # Load data
# df = pd.read_csv("jira_issues.csv")

from src.config import RAW_DATA_DIR
from loguru import logger

print(f"{RAW_DATA_DIR}")
stats_path = RAW_DATA_DIR / "jira_issues.csv"
df = pd.read_csv(str(stats_path))
logger.info("Read File")
df["text"] = df["summary"].fillna('') + " " + df["description"].fillna('')

# # Vectorize and apply LDA
# vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words="english")
# dt_matrix = vectorizer.fit_transform(df["text"])
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# lda.fit(dt_matrix)
# topic_dist = lda.transform(dt_matrix)
# topic_keywords = [
#     ", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]])
#     for topic in lda.components_
# ]
# topic_names = ["Pickup-scheduling-&-rescheduling", "Technical-support", "Order-cancellations", "Order-modifications", "Delivery-issues",
#                "Grant-&-billing-issues", "Product-availability-&-substitutions", "Other", "Training-&-account-access", "Emergency Situations"]
# df["Dominant Topic"] = topic_dist.argmax(axis=1)

# # Streamlit UI
# st.title("🧠 AI-Powered Topic Dashboard")

# tab1, tab2, tab3 = st.tabs(["📊 Topic-Document Heatmap", "☁️ Word Clouds", "📈 Topic Distribution"])

# with tab1:
#     st.subheader("Topic Distribution Across Documents")
#     heatmap_df = pd.DataFrame(topic_dist, columns=topic_names)
#     heatmap_df["Document"] = df["summary"].fillna("").apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
#     heatmap_pivot = pd.melt(heatmap_df, id_vars=["Document"], var_name="Topic", value_name="Score")

#     fig = px.density_heatmap(
#         heatmap_pivot,
#         x="Topic",
#         y="Document",
#         z="Score",
#         color_continuous_scale="Blues"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with tab2:
#     st.subheader("Topic-wise Word Clouds")
#     topic_selector = st.selectbox("Select Topic", topic_names)
#     selected_topic_idx = topic_names.index(topic_selector)
#     topic_docs = df[topic_dist[:, selected_topic_idx] > 0.2]["text"]

#     wc_text = " ".join(topic_docs)
#     wc = WordCloud(width=800, height=400, background_color="white").generate(wc_text)

#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     ax2.imshow(wc, interpolation="bilinear")
#     ax2.axis("off")
#     st.pyplot(fig2)

# with tab3:
#     st.subheader("Topic Distribution per Document")
#     doc_selector = st.selectbox("Select Document:", df["summary"].fillna("Unknown Summary"))
#     selected_doc_idx = df[df["summary"] == doc_selector].index[0]
#     doc_topics = topic_dist[selected_doc_idx]

#     fig3 = go.Figure(go.Bar(
#         x=doc_topics * 100,
#         y=topic_names,
#         orientation="h",
#         text=[f"{v*100:.1f}%" for v in doc_topics],
#         textposition="auto",
#         marker=dict(color=px.colors.qualitative.Set2)
#     ))
#     fig3.update_layout(xaxis_title="Topic Relevance (%)", yaxis_title="Topics")
#     st.plotly_chart(fig3, use_container_width=True)



# df = pd.read_csv("jira_issues.csv")

# st.set_page_config(layout="wide")
# st.title("📊 JIRA Issues Dashboard")

# # ---- 1. Document Heatmap ----
# st.header("📅 Document Heatmap (Issue Creation Timeline)")
# df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
# df['created_date'] = df['created_at'].dt.date

# heat_data = df.groupby('created_date').size().reset_index(name='count')
# fig1 = px.density_heatmap(heat_data, x="created_date", y="count", nbinsx=20, color_continuous_scale="Viridis")
# st.plotly_chart(fig1, use_container_width=True)

# # ---- 2. Word Cloud ----
# st.header("☁️ Word Clouds")

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Summary Word Cloud")
#     summary_text = " ".join(df['summary'].dropna().astype(str))
#     wc = WordCloud(width=800, height=400, background_color='white').generate(summary_text)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis("off")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Description Word Cloud")
#     desc_text = " ".join(df['description'].dropna().astype(str))
#     wc2 = WordCloud(width=800, height=400, background_color='white').generate(desc_text)
#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     ax2.imshow(wc2, interpolation='bilinear')
#     ax2.axis("off")
#     st.pyplot(fig2)

# # ---- 3. Topic Modeling ----
# st.header("📌 Topic Modeling & Distribution")

# text_data = df['summary'].fillna('') + " " + df['description'].fillna('')
# vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
# doc_term_matrix = vectorizer.fit_transform(text_data)

# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# lda.fit(doc_term_matrix)

# topic_words = []
# for topic_idx, topic in enumerate(lda.components_):
#     top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
#     topic_words.append(", ".join(top_words))

# topic_distribution = lda.transform(doc_term_matrix)
# topic_counts = np.argmax(topic_distribution, axis=1)
# topic_freq = pd.Series(topic_counts).value_counts().sort_index()
# topic_labels = ['pickup-scheduling-&-rescheduling', 'technical-support', 'order-cancellations', 'order-modifications', 'delivery-issues', 'grant-&-billing-issues', 'product-availability-&-substitutions', 'other', 'training-&-account-access', 'emergency-situations']

# fig3 = px.pie(values=topic_freq.values, names=topic_labels, title="Topic Distribution", hole=0.4)
# st.plotly_chart(fig3, use_container_width=True)



import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import streamlit.components.v1 as components

# # Load CSV
# df = pd.read_csv("jira_issues_export.csv")
data = df[['main_category', 'summary', 'description']].dropna()

# Topic Distribution
def plot_topic_distribution(data):
    topic_counts = data['main_category'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.bar(topic_counts, x='Topic', y='Count', title='Topic Distribution')
    return fig

# Topic-Document Heatmap
def plot_topic_document_heatmap(data):
    pivot = data.groupby(['main_category', 'summary']).size().reset_index(name='Counts')
    fig = px.density_heatmap(pivot, x='main_category', y='summary', z='Counts',
                             title='Topic-Document Heatmap', nbinsx=20, nbinsy=20)
    return fig

# Word Cloud Generator
def generate_wordcloud(text, title='Word Cloud'):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" />'

# Streamlit UI
def main():
    st.title("📊 Jira Issues Topic Dashboard")

    st.subheader("🔹 Topic Distribution")
    st.plotly_chart(plot_topic_distribution(data), use_container_width=True)

    st.subheader("🔸 Topic-Document Heatmap")
    st.plotly_chart(plot_topic_document_heatmap(data), use_container_width=True)

    st.subheader("☁️ Word Cloud")
    combined_text = " ".join(data['summary'].astype(str) + " " + data['description'].astype(str))
    wordcloud_html = generate_wordcloud(combined_text)
    components.html(wordcloud_html, height=450)

if __name__ == "__main__":
    main()





