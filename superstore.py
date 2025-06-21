import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import json

# Load model and encodings
try:
    model = joblib.load('ETR.pkl')
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

with open('encodings.json', 'r') as f:
    encodings = json.load(f)

# Load dataset
df = pd.read_csv("Superstore.csv", encoding='latin1')
# Streamlit UI
st.set_page_config(page_title='Superstore Sales Prediction App', layout='wide')
st.title('Sales Analysis and Prediction Web App')

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ['Project Overview', 'Visualizations', 'Sales Prediction'])

if page == 'Project Overview':
    st.header("Project Overview")
    st.write("""
    With growing demands and cut-throat competition in the market, a survey was taken and analysis was done to understand regions and cities which are costing the organization. 
    This app explores a sales dataset, with comprehensive visuals to aid decision making and a sales predicting model.
    
    Using a trained regression model, you can:
    - View saved visualizations
    - Try out the model with your own inputs
    """)
    st.write('Here is a preview of the data:')
    st.dataframe(df.head())

elif page == 'Visualizations':
    st.header('Exploration with Visuals')
    selected_img = st.selectbox("Choose a chart to display", [
        'Sales distribution by region',
        'Sales trend over the years',
        'relationship between sales, state and subcategries',
        'Total loss by state',
        'Total discount by state'
    ])
    
    image_map = {
        'Sales distribution by region': 'Sales distribution by region.png',
        'Sales by customers segment': 'Sales by customers segment.png',
        'Sales trend over the years': 'sales trend over the years.png',
        'relationship between sales, state and subcategries': 'heatmat.png',
        'Total loss by state': 'Total loss by state.png',
        'Total discount by state': 'Total discount by state.png'
    }
    
    if selected_img in image_map:
        image = Image.open(image_map[selected_img])
        st.image(image, caption=selected_img)

elif page == 'Sales Prediction':
    st.header('Predicting Sales with ExtraTreeRegressor (91% Accuracy)')

    # Numeric inputs
    discount = st.slider('Discount', 0.0, 5.0, step=0.5)
    quantity = st.slider('Quantity', 1, 10, step=1)
    profit = st.slider('Profit', 5.0, 20.0, step=1.0)
    profit_margin = st.slider('Profit Margin', 0.0, 0.5, step=0.5)
    loss = st.slider('Loss', 0.0, 7000.0, step=500.0)
    order_year = st.slider('Order Year', 2011, 2014, step=1)

    # Categorical inputs
    shipmode = st.selectbox("Select Ship Mode", list(encodings['Ship Mode'].keys()))
    segment = st.selectbox("Select Segment", list(encodings['Segment'].keys()))
    city = st.selectbox("Select City", list(encodings['City'].keys()))
    state = st.selectbox("Select State", list(encodings['State'].keys()))
    region = st.selectbox("Select Region", list(encodings['Region'].keys()))
    category = st.selectbox("Select Category", list(encodings['Category'].keys()))
    sub_category = st.selectbox("Select Sub-Category", list(encodings['Sub-Category'].keys()))

    # Encode input
    encoded_input = {
        'Quantity': quantity,
        'Discount': discount,
        'Profit': profit,
        'order year': order_year,
        'loss': loss,
        'profit margin': profit_margin,
        'city_encoded': encodings['City'].get(city, -1),
        'Ship Mode_encoded': encodings['Ship Mode'].get(shipmode, -1),
        'Segment_encoded': encodings['Segment'].get(segment, -1),
        'State_encoded': encodings['State'].get(state, -1),
        'Region_encoded': encodings['Region'].get(region, -1),
        'Category_encoded': encodings['Category'].get(category, -1),
        'Sub-Category_encoded': encodings['Sub-Category'].get(sub_category, -1)
    }

    # Order of columns
    column_order = [
        'Quantity', 'Discount', 'Profit', 'order year',
        'loss', 'profit margin', 'city_encoded', 'Ship Mode_encoded',
        'Segment_encoded', 'State_encoded', 'Region_encoded',
        'Category_encoded', 'Sub-Category_encoded'
    ]

    # Create DataFrame
    input_data = pd.DataFrame([[encoded_input[col] for col in column_order]], columns=column_order)

    # Prediction
    if st.button('Predict'):
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Sales: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
