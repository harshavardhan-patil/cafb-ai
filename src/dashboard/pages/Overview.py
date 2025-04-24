import streamlit as st
from src.utils.dashboard_helpers import initialize_data_refresh, get_processed_dataframe

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
    .img {
        width : 400, 
        height : 300, 
        bg_color : "f1f1f1";
        text_color: "555555"
        text : "Image"
    }
</style>
""", unsafe_allow_html=True)

# Get data from the database (unfiltered for overview)
df = get_processed_dataframe(filtered=False)

# Hero section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<div class='title-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Customer Support Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content section
col1, col2 = st.columns([3, 2])

# Key metrics
total_issues = len(df)
open_issues = len(df[df['status'] != 'Closed'])
avg_resolution_time = "N/A"  # This would require calculation based on created_at and resolved_at

with col1:
    st.markdown(f""" 
        <div class="feature-card">
            <h2>Our Support System Overview</h2>
            <p>The Capital Area Food Bank Support System provides a centralized platform for tracking 
            and managing partner inquiries and issues.</p>
            
            <h3>Current Statistics:</h3>
            <ul>
                <li><strong>Total Issues Tracked:</strong> {total_issues}</li>
                <li><strong>Currently Open Issues:</strong> {open_issues}</li>
                <li><strong>Average Resolution Time:</strong> {avg_resolution_time}</li>
            </ul>
            
            <h3>System Features:</h3>
            <ul>
                <li>Centralized issue tracking</li>
                <li>AI-powered response assistance</li>
                <li>Automated ticket categorization</li>
                <li>Real-time dashboard analytics</li>
                <li>Knowledge base integration</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    try:
        st.image("reports/figures/overview_img.png")
    except:
        st.info("Overview image not found. Please place an image file at 'reports/figures/overview_img.png'")

# Bottom section - Recent activity
st.markdown("<h2>Recent Activity</h2>", unsafe_allow_html=True)

# Get the 5 most recent issues
recent_issues = df.sort_values('created_at', ascending=False).head(5)

# Display in a table
if not recent_issues.empty:
    # Select only the columns we want to display
    display_cols = ['issue_key', 'summary', 'status', 'Priority', 'created_at']
    cols_to_display = [col for col in display_cols if col in recent_issues.columns]
    
    st.table(recent_issues[cols_to_display])
else:
    st.info("No recent activity found in the database.")