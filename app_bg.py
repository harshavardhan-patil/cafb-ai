import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Dashboard with Colored Top Bar",
        page_icon="📊",
        layout="wide",
    )

def add_colored_top_bar(color="#3366ff", height=70, text="Dashboard", text_color="white", font_size=30):
    """
    Add a colored bar at the top of the dashboard
    
    Parameters:
    color (str): CSS color code for the bar
    height (int): Height of the bar in pixels
    text (str): Text to display in the bar
    text_color (str): CSS color code for the text
    font_size (int): Font size of the text
    """
    # Custom CSS to inject
    css = f"""
    <style>
        div.top-bar {{
            background-color: {color};
            padding: 15px 25px;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: {height}px;
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: {text_color};
            font-size: {font_size}px;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }}
        div.spacer {{
            margin-top: {height+30}px;
        }}
        .stApp {{
            margin-top: {height}px;
        }}
    </style>
    """
    
    # Current time for demonstration
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # HTML content for the bar
    bar_html = f"""
    <div class="top-bar">
        <div>{text}</div>
        <div style="font-size: {font_size-10}px;">{current_time}</div>
    </div>
    <div class="spacer"></div>
    """
    
    # Inject CSS and HTML
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(bar_html, unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for the dashboard"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    sales = np.random.normal(loc=1000, scale=200, size=len(dates))
    sales = np.where(sales < 0, 0, sales)  # Ensure no negative sales
    
    # Add weekly pattern
    weekday_effect = np.array([0.8, 1.0, 1.1, 1.0, 1.2, 1.5, 0.7])
    for i in range(len(sales)):
        sales[i] *= weekday_effect[dates[i].weekday()]
    
    # Add monthly trend
    for m in range(12):
        month_mask = (dates.month == m+1)
        monthly_factor = 0.9 + 0.3 * np.sin(m / 2)
        sales[month_mask] *= monthly_factor
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'transactions': np.random.normal(loc=sales/20, scale=sales/100),
        'customers': np.random.normal(loc=sales/30, scale=sales/150),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], size=len(dates))
    })
    
    df['transactions'] = df['transactions'].astype(int)
    df['customers'] = df['customers'].astype(int)
    
    return df

def display_dashboard(df):
    """Display the main dashboard content"""
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${df['sales'].sum():,.2f}", "+8.2% vs prev")
        
    with col2:
        st.metric("Avg Daily Sales", f"${df['sales'].mean():,.2f}", "-2.1% vs prev")
        
    with col3:
        st.metric("Total Transactions", f"{df['transactions'].sum():,}", "+5.4% vs prev")
        
    with col4:
        st.metric("Unique Customers", f"{df['customers'].sum():,}", "+3.7% vs prev")
    
    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Sales Trend")
        monthly_sales = df.groupby(pd.Grouper(key='date', freq='M')).agg({'sales': 'sum'}).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(monthly_sales['date'], monthly_sales['sales'], marker='o', linewidth=2)
        ax.set_ylabel('Sales ($)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with chart_col2:
        st.subheader("Sales by Category")
        category_sales = df.groupby('category').agg({'sales': 'sum'}).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='category', y='sales', data=category_sales, ax=ax)
        ax.set_ylabel('Sales ($)')
        ax.set_xlabel('')
        ax.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig)
    
    # Data table
    st.subheader("Recent Sales Data")
    st.dataframe(df.tail(10)[['date', 'sales', 'transactions', 'customers', 'category']])

def main():
    # Page configuration
    set_page_config()
    
    # Add colored top bar
    add_colored_top_bar(
        color="#1E88E5",
        height=70, 
        text="Sales Performance Dashboard",
        text_color="white",
        font_size=28
    )
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Color picker for top bar
        bar_color = st.color_picker("Top Bar Color", "#1E88E5")
        
        # Title text input
        bar_text = st.text_input("Dashboard Title", "Sales Performance Dashboard")
        
        # Apply button
        if st.button("Apply Changes"):
            add_colored_top_bar(
                color=bar_color,
                height=70,
                text=bar_text,
                text_color="white",
                font_size=28
            )
    
    # Generate sample data for the dashboard
    df = generate_sample_data()
    
    # Display the dashboard content
    display_dashboard(df)

if __name__ == "__main__":
    main()