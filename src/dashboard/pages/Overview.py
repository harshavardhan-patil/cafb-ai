
import streamlit as st
import base64
from PIL import Image
import requests
from io import BytesIO


# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .title-container {
        text-align: right;
        margin-bottom: 1rem;
    }
    .title {
        color: green;
        font-size: 3.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #7f8c8d;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #3498db;
    }
    .stats-container {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 1rem;
    }
    .highlight {
        color: #e74c3c;
        font-weight: bold;
    }
    .cta-button {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #7f8c8d;
    }
    /* Custom styling for the support icons */
    .support-icons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 2rem;
    }
    .support-icon {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 15px;
        width: 70px;
        height: 70px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .img{
        width : 400, 
        height : 300, 
        bg_color : "f1f1f1";
        text_color: "555555"
        text : "Image"
    
    }
</style>
""", unsafe_allow_html=True)

# Hero section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<div class='title-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Customer Support Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content section
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(""" 
        <div class="feature-card">
            <h2>Our Support Team is Here For You</h2>
            <p>At our company, we believe in providing exceptional customer service 24 hours a day, 7 days a week. 
            Our dedicated team of support specialists is ready to assist you with any questions or issues you may encounter.</p> 
            <h3>What We Offer:</h3>
            <ul>
                <li>24/7 availability through multiple channels</li>
                <li>Expert technical assistance</li>
                <li>Quick response times</li>
                <li>Personalized solutions to your problems</li>
                <li>Follow-up to ensure your satisfaction</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("static/overview_img.png")


 