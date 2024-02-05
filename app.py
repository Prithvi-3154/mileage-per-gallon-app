#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from prediction import predict


# In[4]:


st.title("Predict Mileage Per Gallon")
st.markdown("Model to predict MPG of a Car")

st.header("Car Features")
col1,col2,col3,col4 = st.columns(4)
with col1:
    cylinders=st.slider("Cylinders",2,8,1)
    displacement=st.slider("Displacement",50,500,10)
with col2:
    horsepower=st.slider("Horsepower", 50,500,10)
    weight=st.slider("Weight", 1500,6000,250)
with col3:
    acceleration=st.slider("acceleration", 8,25,1)
    modelyear=st.slider("Model_Year", 70,85,1)
with col4:
    origin=st.slider("Origin",1,3,1)


# In[5]:


if st.button("Prediction of MPG of Car"):
    result=predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,origin]]))
    st.text(result[0])


# In[ ]:




