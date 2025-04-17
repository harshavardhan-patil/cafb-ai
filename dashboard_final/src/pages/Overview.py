import streamlit as st

# st.set_page_config(page_title="Mapping Demo", page_icon="🌍")

# text_overview = st.container()
    



# image_overview = st.container()
# with image_overview:
#     st.image("reports/figures/overview_img.png")


import streamlit as st
import base64
from PIL import Image
import requests
from io import BytesIO

# Page configuration
# st.set_page_config(
#     page_title="24/7 Customer Support",
#     page_icon="📞",
#     layout="wide"
# )

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

# Function to create a placeholder image if unable to use uploaded image
# def get_placeholder_image_url(width, height, bg_color="f1f1f1", text_color="555555", text="Image"):
#     return f"https://via.placeholder.com/{width}x{height}/{bg_color}/{text_color}?text={text}"

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
    # Use a placeholder image for the customer support representative
    # st.image(get_placeholder_image_url(500, 400, text="Customer+Support"), 
    #          caption="Our dedicated support team is ready to help", 
    #          use_column_width=True)

    st.image("reports/figures/overview_img.png")


    
# Support channels section
# st.markdown("<h2 style='text-align: center; margin-top: 3rem;'>Contact Us Through Multiple Channels</h2>", unsafe_allow_html=True)

# # Support icons
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.markdown("""
#     <div class="feature-card">
#         <div style="text-align: center;">
#             <span class="feature-icon">📞</span>
#             <h3>Phone Support</h3>
#             <p>Call us directly for immediate assistance with your questions.</p>
#             <p><strong>+1 (800) 123-4567</strong></p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="feature-card">
#         <div style="text-align: center;">
#             <span class="feature-icon">💬</span>
#             <h3>Live Chat</h3>
#             <p>Connect with a support agent instantly through our chat service.</p>
#             <p><strong>Available 24/7</strong></p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="feature-card">
#         <div style="text-align: center;">
#             <span class="feature-icon">✉️</span>
#             <h3>Email Support</h3>
#             <p>Send us your questions and receive a detailed response.</p>
#             <p><strong>support@company.com</strong></p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col4:
#     st.markdown("""
#     <div class="feature-card">
#         <div style="text-align: center;">
#             <span class="feature-icon">🌐</span>
#             <h3>Knowledge Base</h3>
#             <p>Find answers to common questions in our extensive documentation.</p>
#             <p><strong>Browse Articles</strong></p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # Stats section
# st.markdown("""
# <div class="stats-container">
#     <div style="display: flex; justify-content: space-around; text-align: center;">
#         <div>
#             <div class="stat-value">98.7%</div>
#             <div class="stat-label">Customer Satisfaction</div>
#         </div>
#         <div>
#             <div class="stat-value">&lt; 2 min</div>
#             <div class="stat-label">Average Response Time</div>
#         </div>
#         <div>
#             <div class="stat-value">24/7</div>
#             <div class="stat-label">Support Availability</div>
#         </div>
#         <div>
#             <div class="stat-value">10k+</div>
#             <div class="stat-label">Issues Resolved Monthly</div>
#         </div>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # Call to action
# st.markdown("""
# <div style="text-align: center; margin-top: 2rem; margin-bottom: 3rem;">
#     <h2>Need Help? We're Just a Click Away</h2>
#     <p style="margin-bottom: 2rem;">Our support team is standing by to assist you with any questions or issues.</p>
#     <button class="cta-button">Contact Support Now</button>
# </div>
# """, unsafe_allow_html=True)

# # Testimonials
# st.markdown("<h2 style='text-align: center;'>What Our Customers Say</h2>", unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown("""
#     <div class="feature-card">
#         <p>"The support team was incredibly helpful and resolved my issue within minutes. I'm very impressed with the level of service!"</p>
#         <p><strong>- Sarah J., Customer since 2022</strong></p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="feature-card">
#         <p>"I contacted support at 3 AM and was amazed to get an immediate response. The representative was knowledgeable and solved my problem efficiently."</p>
#         <p><strong>- Michael T., Customer since 2021</strong></p>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="feature-card">
#         <p>"The best customer support I've experienced. They went above and beyond to make sure my issue was resolved completely."</p>
#         <p><strong>- Elena R., Customer since 2023</strong></p>
#     </div>
#     """, unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <div class="footer">
#     <p>© 2025 Company Name. All rights reserved.</p>
#     <p>Privacy Policy | Terms of Service | Accessibility</p>
# </div>
# """, unsafe_allow_html=True)