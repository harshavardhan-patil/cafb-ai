import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import streamlit.components.v1 as components
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from io import BytesIO
import base64
from src.utils.dashboard_helpers import initialize_data_refresh, get_issues_dataframe

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Preprocess function for text data
def preprocess_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords and lemmatize
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    return ''

# Function to generate a word cloud
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


st.title("🧠 AI-Powered Topic Dashboard")

# Get data from the database
df = get_issues_dataframe()

# Combine summary and description for topic analysis
df['text'] = df['summary'].fillna('') + " " + df['description'].fillna('')

# Preprocess the text
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove rows with empty processed text
df = df[df['processed_text'].str.strip() != '']

# Topic modeling parameters
num_topics = st.sidebar.slider("Number of Topics", 2, 15, 5)
num_words = st.sidebar.slider("Top Words per Topic", 5, 20, 10)

# Topic modeling with LDA
try:
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    dtm = vectorizer.fit_transform(df['processed_text'])
    
    # Create and fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=10,
        learning_method='online'
    )
    lda_output = lda_model.fit_transform(dtm)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get keywords for each topic
    topic_keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-num_words-1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        top_weights = [topic[i] for i in top_features_ind]
        topic_keywords.append((top_features, top_weights))
    
    # Assign dominant topic to each document
    df_topic_distribution = pd.DataFrame(lda_output)
    df['dominant_topic'] = df_topic_distribution.idxmax(axis=1)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Topic Distribution", "Topic Words", "Word Clouds"])
    
    # Tab 1: Topic Distribution
    with tab1:
        st.subheader("Topic Distribution Across Documents")
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        
        fig = px.pie(
            values=topic_counts.values,
            names=[f"Topic {i+1}" for i in topic_counts.index],
            title="Topic Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show documents per topic
        st.subheader("Documents by Topic")
        selected_topic = st.selectbox(
            "Select a topic to view documents:",
            options=[f"Topic {i+1}" for i in range(num_topics)]
        )
        
        topic_idx = int(selected_topic.split()[1]) - 1
        topic_docs = df[df['dominant_topic'] == topic_idx]
        
        if not topic_docs.empty:
            st.write(f"{len(topic_docs)} documents in {selected_topic}")
            st.dataframe(topic_docs[['issue_key', 'summary']])
        else:
            st.write(f"No documents found for {selected_topic}")
        
    # Tab 2: Topic Words
    with tab2:
        st.subheader("Top Words by Topic")
        
        for i, (keywords, weights) in enumerate(topic_keywords):
            # Create a bar chart for each topic
            fig = px.bar(
                x=weights,
                y=keywords,
                orientation='h',
                title=f"Topic {i+1} - Top {num_words} Words",
                labels={'x': 'Weight', 'y': 'Word'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Word Clouds
    with tab3:
        st.subheader("Word Clouds by Topic")
        
        selected_topic_wc = st.selectbox(
            "Select a topic for word cloud:",
            options=[f"Topic {i+1}" for i in range(num_topics)],
            key="wordcloud_select"
        )
        
        topic_idx_wc = int(selected_topic_wc.split()[1]) - 1
        
        # Create word frequency dict for the topic
        topic_words, topic_weights = topic_keywords[topic_idx_wc]
        word_freq = {word: weight for word, weight in zip(topic_words, topic_weights)}
        
        # Generate word cloud
        wc = WordCloud(width=800, height=400, 
                        background_color='white', 
                        max_words=100).generate_from_frequencies(word_freq)
        
        # Display the generated image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Show example documents for this topic
        topic_examples = df[df['dominant_topic'] == topic_idx_wc].head(5)
        if not topic_examples.empty:
            st.subheader(f"Example Documents for {selected_topic_wc}")
            for _, row in topic_examples.iterrows():
                st.write(f"**Issue Key:** {row['issue_key']}")
                st.write(f"**Summary:** {row['summary']}")
                st.write("---")
        
except Exception as e:
    st.error(f"Error in topic modeling: {str(e)}")
    import traceback
    st.write(traceback.format_exc())

