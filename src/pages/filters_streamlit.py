# import streamlit as st
# from streamlit_dynamic_filters import DynamicFilters
# import pandas as pd

# data = {
#     'Region': ['North America', 'North America', 'North America', 'Europe', 'Europe', 'Asia', 'Asia'],
#     'Country': ['USA', 'USA', 'Canada', 'Germany', 'France', 'Japan', 'China'],
#     'City': ['New York', 'Los Angeles', 'Toronto', 'Berlin', 'Paris', 'Tokyo', 'Beijing']
#     }

# df = pd.DataFrame(data)

# dynamic_filters = DynamicFilters(df, filters=['Region', 'Country', 'City'])

# with st.sidebar:
#     dynamic_filters.display_filters()

# # dynamic_filters.display_df()


import streamlit as st
import pandas as pd
import plotly

# Load your data (replace with your actual data source)
data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Location': ['NY', 'CA', 'TX', 'NY', 'CA'],
    'Age': [25, 30, 35, 40, 45],
})

gender_filter = st.multiselect('Select Gender', options=list(data['Gender'].unique()), default=list(data['Gender'].unique()))
location_filter = st.multiselect('Select Location', options=list(data['Location'].unique()), default=list(data['Location'].unique()))

filtered_data = data[data['Gender'].isin(gender_filter) & data['Location'].isin(location_filter)]

st.write(filtered_data)

import plotly.express as px
fig = px.bar(filtered_data, x='Location', y='Age', color='Gender')
st.plotly_chart(fig)