import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
import pickle


import streamlit as st

# Load the data and model
#df = pickle.load(open('carsellingdata.pkl', 'rb'))  # Absolute path
#model= pickle.load(open('pipe_rf_car.pkl','rb'))
df = pd.read_pickle('carsellingdata.pkl')
model = pd.read_pickle('pipe_rf_car.pkl')
# Set up the Streamlit application
st.title('car price prediction')
st.header("choose your car's model and its specifications")

# Select car specifications using Streamlit widgets
year = st.selectbox('Year', df['year'].unique())
km = st.slider("km:",min_value=0, max_value=807000, value=10000, step=5000)
fuel = st.selectbox('Fuel', df['fuel'].unique())
seller_type = st.selectbox('Seller type', df['seller_type'].unique())
transmission = st.selectbox('Transmission', df['transmission'].unique())
owner = st.selectbox('Owner type', df['owner'].unique())
company = st.selectbox('Choose car company', df['company'].unique().astype(str))

# Perform car price prediction upon button click
if st.button("Guess my car's price"):
   # Create a test DataFrame with selected car specifications
   d= {'year':year,'km_driven':km,'fuel':fuel,'seller_type':seller_type,'owner':owner,'company':company,'transmission':transmission}
   test=pd.DataFrame(data=d,index=[0])
   
   # Perform car price prediction and display the result
   predicted_price = model.predict(test)
   st.header("You can expect anywhere between")
   st.subheader('Minimum price')
   st.success(predicted_price - predicted_price*10/100)
   st.subheader('Maximum price')
   st.success(predicted_price + predicted_price*10/100)
