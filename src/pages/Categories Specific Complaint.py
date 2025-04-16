# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import base64
# from io import BytesIO
# import re
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.feature_extraction.text import CountVectorizer
# from loguru import logger
# from src.config import RAW_DATA_DIR


# # Set page configuration
# # st.set_page_config(
# #     page_title="Interactive Topic Modeling Visualization",
# #     page_icon="📊",
# #     layout="wide"
# # )

# # Demo data generation function
# @st.cache_data
# def generate_demo_data(num_docs=100, num_topics=5):
#     # Create sample vocabulary
#     vocab_by_topic = {
#         'Technology': ['technology', 'digital', 'computer', 'software', 'data', 'cloud', 'ai', 
#                       'innovation', 'algorithm', 'automation', 'interface', 'programming', 'application'],
#         'Finance': ['finance', 'market', 'investment', 'bank', 'stock', 'trading', 'fund', 
#                    'economic', 'portfolio', 'asset', 'financial', 'growth', 'capital'],
#         'Healthcare': ['health', 'medical', 'patient', 'hospital', 'doctor', 'treatment', 'care', 
#                       'disease', 'clinical', 'therapy', 'diagnosis', 'research', 'medicine'],
#         'Entertainment': ['movie', 'music', 'game', 'entertainment', 'film', 'streaming', 'media', 
#                          'series', 'television', 'show', 'audience', 'performance', 'video'],
#         'Politics': ['political', 'government', 'policy', 'election', 'party', 'voter', 'campaign', 
#                     'democracy', 'legislation', 'reform', 'congress', 'leader', 'public']
#     }
    
#     topic_names = list(vocab_by_topic.keys())
    
#     # Generate documents
#     documents = []
#     doc_topic_dist = []
    
#     for i in range(num_docs):
#         # Random primary topic for this document
#         primary_topic_idx = np.random.randint(0, num_topics)
#         primary_topic = topic_names[primary_topic_idx]
        
#         # Create topic distribution for this document
#         topic_dist = np.random.dirichlet(np.ones(num_topics) * 0.5)
#         # Ensure primary topic has higher probability
#         topic_dist = topic_dist / sum(topic_dist)
#         topic_dist[primary_topic_idx] = np.random.uniform(0.4, 0.7)
#         # Renormalize
#         topic_dist = topic_dist / sum(topic_dist)
#         doc_topic_dist.append(topic_dist)
        
#         # Create document text
#         doc_text = []
        
#         # Add words from primary topic
#         num_primary_words = np.random.randint(20, 40)
#         primary_words = np.random.choice(vocab_by_topic[primary_topic], 
#                                          size=num_primary_words, 
#                                          replace=True)
#         doc_text.extend(primary_words)
        
#         # Add some words from other topics
#         for j, topic in enumerate(topic_names):
#             if j != primary_topic_idx:
#                 num_words = int(topic_dist[j] * 30)
#                 if num_words > 0:
#                     other_words = np.random.choice(vocab_by_topic[topic], 
#                                                   size=num_words, 
#                                                   replace=True)
#                     doc_text.extend(other_words)
                    
#         # Shuffle words and join
#         np.random.shuffle(doc_text)
#         document = ' '.join(doc_text)
#         documents.append({
#             'id': i+1,
#             'title': f"Document {i+1}: About {primary_topic}",
#             'text': document,
#             'primary_topic': primary_topic
#         })
    
#     # Convert to dataframe
#     doc_df = pd.DataFrame(documents)
    
#     # Create topic-term matrix
#     topic_term_matrix = []
#     all_terms = set()
    
#     for topic, terms in vocab_by_topic.items():
#         all_terms.update(terms)
        
#     all_terms = sorted(list(all_terms))
    
#     for topic, topic_terms in vocab_by_topic.items():
#         # Count frequency of each term in this topic's vocabulary
#         term_counts = {}
#         for term in all_terms:
#             term_counts[term] = topic_terms.count(term)
            
#         # Add random weights
#         for term in topic_terms:
#             term_counts[term] = np.random.randint(1, 10)
            
#         topic_term_matrix.append({
#             'topic': topic,
#             **term_counts
#         })
    
#     topic_term_df = pd.DataFrame(topic_term_matrix)
    
#     # Create document-topic matrix
#     doc_topic_matrix = pd.DataFrame(doc_topic_dist, 
#                                    columns=topic_names)
#     doc_topic_matrix['doc_id'] = range(1, num_docs + 1)
    
#     return doc_df, topic_term_df, doc_topic_matrix, topic_names

# # Function to create a wordcloud
# def generate_wordcloud(topic_idx, topic_term_df, topic_names):
#     topic_name = topic_names[topic_idx]
    
#     # Extract term frequencies for this topic
#     topic_row = topic_term_df[topic_term_df['topic'] == topic_name].iloc[0].drop('topic')
#     word_freq = {word: freq for word, freq in topic_row.items() if freq > 0}
    
#     # Generate wordcloud
#     wc = WordCloud(width=400, height=300, 
#                   background_color='white', 
#                   colormap='viridis', 
#                   max_words=50).generate_from_frequencies(word_freq)
    
#     # Convert to image
#     img = wc.to_image()
    
#     # Save to BytesIO object
#     buf = BytesIO()
#     img.save(buf, format='PNG')
    
#     # Encode to base64
#     data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
#     return f"data:image/png;base64,{data}"

# # Function to create bubble chart of document-topic distribution
# def create_document_topic_bubble_chart(doc_topic_matrix, topic_names, selected_topics=None):
#     # Filter by selected topics if provided
#     display_topics = selected_topics if selected_topics else topic_names
    
#     # Prepare data for visualization
#     plot_data = []
    
#     # Melt the dataframe to get it in the right format for plotly
#     doc_topic_long = pd.melt(doc_topic_matrix, 
#                             id_vars=['doc_id'], 
#                             value_vars=display_topics,
#                             var_name='topic', 
#                             value_name='weight')
    
#     # Create color scale
#     colors = px.colors.qualitative.Bold
#     color_map = {topic: colors[i % len(colors)] for i, topic in enumerate(topic_names)}
    
#     # Create the bubble chart
#     fig = px.scatter(doc_topic_long, 
#                     x='doc_id', 
#                     y='topic', 
#                     size='weight', 
#                     color='topic',
#                     color_discrete_map=color_map,
#                     hover_data=['weight'],
#                     size_max=30,
#                     title="Document-Topic Distribution")
    
#     fig.update_layout(
#         xaxis_title="Document ID",
#         yaxis_title="Topic",
#         showlegend=True,
#         height=500
#     )
    
#     return fig

# # Function to create topic similarity network
# def create_topic_similarity_network(topic_term_df, topic_names):
#     # Calculate topic similarity based on term co-occurrence
#     topic_term_matrix = topic_term_df.drop('topic', axis=1).values
    
#     # Cosine similarity between topics
#     from sklearn.metrics.pairwise import cosine_similarity
#     similarity_matrix = cosine_similarity(topic_term_matrix)
    
#     # Create nodes
#     nodes = []
#     for i, topic in enumerate(topic_names):
#         nodes.append({
#             'id': i,
#             'label': topic,
#             'size': 20
#         })
    
#     # Create edges where similarity is above threshold
#     edges = []
#     threshold = 0.2  # Minimum similarity to show an edge
    
#     for i in range(len(topic_names)):
#         for j in range(i+1, len(topic_names)):
#             similarity = similarity_matrix[i, j]
#             if similarity > threshold:
#                 edges.append({
#                     'source': i,
#                     'target': j,
#                     'weight': similarity
#                 })
    
#     # Create network graph
#     edge_x = []
#     edge_y = []
#     edge_weights = []
    
#     # Position nodes in a circle
#     import math
#     radius = 1
#     node_positions = {}
    
#     for i, node in enumerate(nodes):
#         angle = 2 * math.pi * i / len(nodes)
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         node_positions[node['id']] = (x, y)
    
#     # Create edge traces
#     for edge in edges:
#         x0, y0 = node_positions[edge['source']]
#         x1, y1 = node_positions[edge['target']]
        
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])
#         edge_weights.append(edge['weight'])
    
#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=1, color='#888'),
#         hoverinfo='none',
#         mode='lines')
    
#     # Create node traces
#     node_x = []
#     node_y = []
#     node_text = []
#     node_colors = []
    
#     for node in nodes:
#         x, y = node_positions[node['id']]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node['label'])
#         node_colors.append(px.colors.qualitative.Bold[node['id'] % len(px.colors.qualitative.Bold)])
    
#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers+text',
#         hoverinfo='text',
#         text=node_text,
#         textposition="top center",
#         marker=dict(
#             showscale=False,
#             color=node_colors,
#             size=30,
#             line_width=2))
    
#     # Create figure
#     fig = go.Figure(data=[edge_trace, node_trace],
#                   layout=go.Layout(
#                       title='Topic Similarity Network',
#                       showlegend=False,
#                       hovermode='closest',
#                       margin=dict(b=20, l=5, r=5, t=40),
#                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                   )
    
#     return fig

# # Function to create topic bar chart
# def create_top_terms_chart(topic_term_df, selected_topic, n_terms=10):
#     # Get the selected topic's term distribution
#     topic_row = topic_term_df[topic_term_df['topic'] == selected_topic].iloc[0].drop('topic')
    
#     # Sort terms by frequency
#     sorted_terms = topic_row.sort_values(ascending=False).head(n_terms)
    
#     # Create bar chart
#     fig = px.bar(
#         x=sorted_terms.values,
#         y=sorted_terms.index,
#         orientation='h',
#         title=f"Top {n_terms} terms in topic: {selected_topic}",
#         labels={'x': 'Weight', 'y': 'Term'}
#     )
    
#     fig.update_layout(height=400)
    
#     return fig

# # Function to preprocess text for LDA
# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove punctuation and numbers
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\d+', '', text)
    
#     return text

# # Function to run LDA on custom data
# def run_lda(docs, n_topics=5, n_top_words=10):
#     # Preprocess
#     preprocessed_docs = [preprocess_text(doc) for doc in docs]
    
#     # Create document-term matrix
#     vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
#     dtm = vectorizer.fit_transform(preprocessed_docs)
    
#     # Get feature names
#     feature_names = vectorizer.get_feature_names_out()
    
#     # Run LDA
#     lda = LatentDirichletAllocation(
#         n_components=n_topics,
#         random_state=0,
#         learning_method='online'
#     )
    
#     lda.fit(dtm)
    
#     # Extract topic-term matrix
#     topic_term_matrix = []
#     for topic_idx, topic in enumerate(lda.components_):
#         top_features_ind = topic.argsort()[:-n_top_words-1:-1]
#         top_features = [feature_names[i] for i in top_features_ind]
#         weights = topic[top_features_ind]
        
#         # Create row for this topic
#         topic_dict = {'topic': f'Topic {topic_idx+1}'}
#         for term, weight in zip(top_features, weights):
#             topic_dict[term] = weight
            
#         topic_term_matrix.append(topic_dict)
    
#     # Extract document-topic matrix
#     doc_topic_dist = lda.transform(dtm)
#     doc_topic_df = pd.DataFrame(doc_topic_dist, 
#                                columns=[f'Topic {i+1}' for i in range(n_topics)])
#     doc_topic_df['doc_id'] = range(1, len(docs) + 1)
    
#     # Create document dataframe
#     doc_df = pd.DataFrame({
#         'id': range(1, len(docs) + 1),
#         'text': docs,
#         'primary_topic': [f'Topic {i+1}' for i in doc_topic_dist.argmax(axis=1)]
#     })
    
#     # Convert topic_term_matrix to dataframe
#     topic_term_df = pd.DataFrame(topic_term_matrix)
    
#     topic_names = [f'Topic {i+1}' for i in range(n_topics)]
#     print("LDA Run")
#     st.write(doc_df)
#     st.write(topic_term_df)
#     st.write(doc_topic_df)
#     st.write(topic_names)
    
#     return doc_df, topic_term_df, doc_topic_df, topic_names

# # --- Main application ---
# def main():
#     st.title("📊 Interactive Topic Modeling Visualization")
    
#     # Sidebar for controls
#     # st.sidebar.title("Controls")
    
#     # # Choose between demo and custom data
#     # data_source = st.sidebar.radio(
#     #     "Data Source",
#     #     ("Demo Data", "Upload Your Own Data")
#     # )
    
#     # if data_source == "Demo Data":
#     #     num_topics = st.sidebar.slider("Number of Topics", 3, 8, 5)
#     #     num_docs = st.sidebar.slider("Number of Documents", 50, 200, 100)
        
#     #     # Generate demo data
#     #     doc_df, topic_term_df, doc_topic_matrix, topic_names = generate_demo_data(
#     #         num_docs=num_docs, 
#     #         num_topics=num_topics
#     #     )
        
#     # else:
#     #     # File upload
#     #     uploaded_file = st.sidebar.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])
        
#     #     if uploaded_file is not None:
#     #         # Determine file type
#     #         file_type = uploaded_file.name.split(".")[-1].lower()
            
#     #         if file_type == "csv":
#     #             # Read CSV
#     #             df = pd.read_csv(uploaded_file)
                
#     #             # Let user select text column
#     #             text_col = st.sidebar.selectbox(
#     #                 "Select text column",
#     #                 df.columns.tolist()
#     #             )
                
#     #             docs = df[text_col].tolist()
                
#     #         elif file_type == "txt":
#     #             # Read TXT (one document per line)
#     #             content = uploaded_file.read().decode("utf-8")
#     #             docs = content.split("\n")
#     #             docs = [doc for doc in docs if doc.strip()]  # Remove empty lines
                
#     #         # LDA parameters
#     #         n_topics = st.sidebar.slider("Number of Topics", 2, 15, 5)
#     #         n_top_words = st.sidebar.slider("Number of Top Words per Topic", 5, 30, 10)
            
#             # Run LDA
#     print(f"{RAW_DATA_DIR}")
#     stats_path = RAW_DATA_DIR / "jira_issues.csv"
#     df = pd.read_csv(str(stats_path))
#     logger.info("Read File")
#     docs = df["description"]
#     doc_df, topic_term_df, doc_topic_matrix, topic_names = run_lda(docs, n_topics=20, n_top_words=10)
#     st.write(doc_df)
#     st.write(topic_term_df)
#     st.write(doc_topic_matrix)
#     st.write(topic_names)
#         # else:
#         #     st.info("Please upload a file to continue.")
#         #     return
    
#     # # Topic selection for filtering
#     # st.sidebar.subheader("Filter Topics")
#     # selected_topics = st.sidebar.multiselect(
#     #     "Select topics to display",
#     #     topic_names,
#     #     default=topic_names
#     # )
    
#     # if not selected_topics:
#     #     selected_topics = topic_names  # Show all if none selected
    
#     # # Select topic for detailed view
#     # st.sidebar.subheader("Detailed Topic View")
#     # selected_topic_detail = st.sidebar.selectbox(
#     #     "Select a topic to view details",
#     #     topic_names
#     # )
    
#     # # Layout
#     # tab1, tab2, tab3, tab4 = st.tabs([
#     #     "Topic-Document Distribution", 
#     #     "Topic Term Analysis", 
#     #     "Topic Similarity Network",
#     #     "Document Explorer"
#     # ])
    
#     # with tab1:
#     #     st.subheader("Topic Distribution Across Documents")
        
#     #     # Display bubble chart
#     #     bubble_chart = create_document_topic_bubble_chart(
#     #         doc_topic_matrix, 
#     #         topic_names, 
#     #         selected_topics
#     #     )
#     #     st.plotly_chart(bubble_chart, use_container_width=True)
        
#     #     # Topic distribution summary
#     #     st.subheader("Topic Distribution Summary")
#     #     topic_summary = doc_topic_matrix[topic_names].mean().sort_values(ascending=False)
        
#     #     summary_fig = px.bar(
#     #         x=topic_summary.index,
#     #         y=topic_summary.values,
#     #         title="Average Topic Weight Across All Documents",
#     #         labels={'x': 'Topic', 'y': 'Average Weight'},
#     #         color=topic_summary.index,
#     #         color_discrete_sequence=px.colors.qualitative.Bold
#     #     )
        
#     #     st.plotly_chart(summary_fig, use_container_width=True)
    
#     # with tab2:
#     #     col1, col2 = st.columns(2)
        
#     #     with col1:
#     #         st.subheader(f"Top Terms in {selected_topic_detail}")
            
#     #         # Display top terms chart
#     #         top_terms_chart = create_top_terms_chart(
#     #             topic_term_df, 
#     #             selected_topic_detail
#     #         )
#     #         st.plotly_chart(top_terms_chart, use_container_width=True)
        
#     #     with col2:
#     #         st.subheader(f"Word Cloud: {selected_topic_detail}")
            
#     #         # Get the topic index
#     #         topic_idx = topic_names.index(selected_topic_detail)
            
#     #         # Display wordcloud
#     #         wc_data = generate_wordcloud(topic_idx, topic_term_df, topic_names)
#     #         st.image(wc_data)
    
#     # with tab3:
#     #     st.subheader("Topic Similarity Network")
        
#     #     # Display network graph
#     #     network_fig = create_topic_similarity_network(topic_term_df, topic_names)
#     #     st.plotly_chart(network_fig, use_container_width=True)
        
#     #     st.info("""
#     #     This network shows relationships between topics based on shared terms.
#     #     Topics that are closer together or have stronger connections share more common terms.
#     #     """)
    
#     # with tab4:
#     #     st.subheader("Document Explorer")
        
#     #     # Select document
#     #     doc_id = st.selectbox(
#     #         "Select a document",
#     #         doc_df['id'].tolist()
#     #     )
        
#     #     # Display document details
#     #     selected_doc = doc_df[doc_df['id'] == doc_id].iloc[0]
        
#     #     st.markdown(f"### Document {doc_id}")
        
#     #     # Display document text
#     #     st.markdown("#### Document Text")
#     #     st.write(selected_doc['text'][:500] + "..." if len(selected_doc['text']) > 500 else selected_doc['text'])
        
#     #     # Document topic distribution
#     #     st.markdown("#### Topic Distribution")
#     #     doc_topics = doc_topic_matrix[doc_topic_matrix['doc_id'] == doc_id][topic_names].iloc[0]
        
#     #     doc_topic_fig = px.bar(
#     #         x=doc_topics.index,
#     #         y=doc_topics.values,
#     #         title=f"Topic Distribution for Document {doc_id}",
#     #         labels={'x': 'Topic', 'y': 'Weight'},
#     #         color=doc_topics.index,
#     #         color_discrete_sequence=px.colors.qualitative.Bold
#     #     )
        
#     #     st.plotly_chart(doc_topic_fig, use_container_width=True)

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.config import RAW_DATA_DIR
from loguru import logger
import streamlit as st

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

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

def load_and_process_data(df):
    # """Load data from file and preprocess"""
    # # Detect file type and load accordingly
    # if file_path.endswith('.csv'):
    #     stats_path = RAW_DATA_DIR / "jira_issues.csv"
    #     df = pd.read_csv(str(file_path))
    # elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
    #     df = pd.read_excel(file_path)
    # elif file_path.endswith('.txt'):
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         text = file.read()
    #     # Split by paragraphs or sentences for txt files
    #     texts = re.split(r'\n\n|\. ', text)
    #     df = pd.DataFrame({'text': texts})
    # else:
    #     raise ValueError("Unsupported file format. Please use CSV, Excel, or TXT files.")
    
    # Identify the text column (if it's not already called 'text')
    st.write("Enetered Load")
    text_col = 'text'
    if text_col not in df.columns:
        # Try to find a column that might contain text
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                break
        else:
            raise ValueError("Could not identify a text column in the data.")
    
    # Preprocess the text
    df['processed_text'] = df[text_col].apply(preprocess_text)
    # Remove empty entries
    df = df[df['processed_text'].str.strip() != '']
    
    return df

def perform_topic_modeling(df, num_topics=5, num_words=10):
    """Perform LDA topic modeling on the text data"""
    # Create document-term matrix
    # vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=1000)
    vectorizer = CountVectorizer()
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
        topic_keywords.append(top_features)
    
    # Get dominant topic for each document
    df_topic_distribution = pd.DataFrame(lda_output)
    dominant_topic = df_topic_distribution.idxmax(axis=1)
    df['dominant_topic'] = dominant_topic
    
    return df, lda_model, vectorizer, topic_keywords, lda_output

def create_topic_visualizations(df, lda_model, vectorizer, topic_keywords, lda_output):
    """Create visualizations for topic model results"""
    # Prepare data for visualizations
    num_topics = len(topic_keywords)
    
    # 1. Topic-Word Distribution
    fig_word_dist = make_subplots(rows=1, cols=1, subplot_titles=["Top Words in Each Topic"])
    
    for i, keywords in enumerate(topic_keywords):
        topic_probs = lda_model.components_[i]
        top_indices = topic_probs.argsort()[:-len(keywords)-1:-1]
        top_probs = [topic_probs[idx] for idx in top_indices]
        
        # Normalize for better visualization
        top_probs = np.array(top_probs) / sum(top_probs)
        
        fig_word_dist.add_trace(
            go.Bar(
                x=keywords,
                y=top_probs,
                name=f'Topic {i+1}',
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            )
        )
    
    fig_word_dist.update_layout(
        title="Top Words in Each Topic",
        xaxis_title="Words",
        yaxis_title="Relative Importance",
        barmode='group',
        height=600
    )
    
    # 2. Document-Topic Distribution
    topic_counts = df['dominant_topic'].value_counts().sort_index()
    
    fig_doc_dist = go.Figure(data=[
        go.Bar(
            x=[f'Topic {i+1}' for i in range(num_topics)],
            y=topic_counts.values,
            marker_color=px.colors.qualitative.Plotly[:num_topics]
        )
    ])
    
    fig_doc_dist.update_layout(
        title="Number of Documents per Topic",
        xaxis_title="Topic",
        yaxis_title="Number of Documents",
        height=500
    )
    
    # 3. Topic Keywords Word Cloud (text representation)
    fig_keywords = go.Figure()
    
    for i, keywords in enumerate(topic_keywords):
        topic_text = f"<b>Topic {i+1}:</b> " + ", ".join(keywords)
        
        fig_keywords.add_annotation(
            x=0.5,
            y=1 - (i / num_topics),
            xref="paper",
            yref="paper",
            text=topic_text,
            showarrow=False,
            font=dict(size=14),
            align="left"
        )
    
    fig_keywords.update_layout(
        title="Keywords for Each Topic",
        height=300 + (num_topics * 40),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # 4. Topic Similarity Heatmap
    topic_similarities = np.zeros((num_topics, num_topics))
    
    for i in range(num_topics):
        for j in range(num_topics):
            # Calculate cosine similarity between topic vectors
            v1 = lda_model.components_[i]
            v2 = lda_model.components_[j]
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            topic_similarities[i, j] = similarity
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=topic_similarities,
        x=[f'Topic {i+1}' for i in range(num_topics)],
        y=[f'Topic {i+1}' for i in range(num_topics)],
        colorscale='Viridis',
        showscale=True
    ))
    
    fig_heatmap.update_layout(
        title="Topic Similarity Matrix",
        height=500
    )
    
    # 5. Interactive 3D visualization for documents (if we have enough topics)
    if num_topics >= 3:
        # Select 3 topics for visualization
        topics_to_viz = [0, 1, 2]
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=lda_output[:, topics_to_viz[0]],
            y=lda_output[:, topics_to_viz[1]],
            z=lda_output[:, topics_to_viz[2]],
            mode='markers',
            marker=dict(
                size=5,
                color=df['dominant_topic'],
                colorscale=px.colors.qualitative.Plotly,
                opacity=0.8
            ),
            text=[f"Doc {i}" for i in range(len(df))]
        )])
        
        fig_3d.update_layout(
            title="Document Distribution in Topic Space (3D)",
            scene=dict(
                xaxis_title=f'Topic {topics_to_viz[0]+1}',
                yaxis_title=f'Topic {topics_to_viz[1]+1}',
                zaxis_title=f'Topic {topics_to_viz[2]+1}'
            ),
            height=700
        )
    else:
        fig_3d = None
    
    return fig_word_dist, fig_doc_dist, fig_keywords, fig_heatmap, fig_3d

def main(file_path, num_topics=5):
    """Main function to run the topic modeling pipeline"""
    print(f"Loading and processing data from: {file_path}")
    new_df = load_and_process_data(df)
    st.write("New DF in Maint")
    st.write(new_df)
    print(f"Found {len(new_df)} documents after preprocessing")
    
    # print(f"Performing topic modeling with {num_topics} topics...")
    new_df, lda_model, vectorizer, topic_keywords, lda_output = perform_topic_modeling(new_df, num_topics=num_topics)
    
    print("Creating visualizations...")
    fig_word_dist, fig_doc_dist, fig_keywords, fig_heatmap, fig_3d = create_topic_visualizations(
        new_df, lda_model, vectorizer, topic_keywords, lda_output
    )
    
    # Show the visualizations
    fig_word_dist.show()
    fig_doc_dist.show()
    fig_keywords.show()
    fig_heatmap.show()
    if fig_3d is not None:
        fig_3d.show()
    
    # print("Topic modeling completed successfully!")
    return new_df, topic_keywords, lda_model

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "'/Users/kritishahi/Documents/UMCP Documents/Smith AI Competition/CAFB AI Competition/cafb-ai-feature-edahp/data/raw/jira_issues.csv" 

    print(f"{RAW_DATA_DIR}")
    stats_path = RAW_DATA_DIR / "jira_issues.csv"
    df = pd.read_csv(str(stats_path))
    logger.info("Read File")
    num_topics = 5  # Adjust number of topics as needed

    st.write(df.head())
    # Run the topic modeling
    new_df, topic_keywords, lda_model = main(df, num_topics)
    
    # Print topics and keywords
    print("\nTopic Keywords:")
    for i, keywords in enumerate(topic_keywords):
        print(f"Topic {i+1}: {', '.join(keywords)}")