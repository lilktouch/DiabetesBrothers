#!/usr/bin/env python
# coding: utf-8

# In[62]:


import streamlit as st
from PIL import Image
from datetime import datetime
import requests  # pip install requests
import base64
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[21]:


st.set_page_config(page_title="PREDIDCTION",layout="wide")


# In[39]:


st.header("Diabetes: A disease that kills thousands of innocent lives.")


# In[51]:


st.write("Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose. Hyperglycaemia, also called raised blood glucose or raised blood sugar, is a common effect of uncontrolled diabetes and over time leads to serious damage to many of the body's systems, especially the nerves and blood vessels.")
st.write(' ')
with st.container():
    col1, image_col, col2 = st.columns(3)
    with col1:
        st.write(' ')
    with image_col:
        st.image('https://amritahospitals.org/wp-content/uploads/2022/07/Diabetes-1-1536x1024.jpg')
    with col2:
        st.write(' ')
        
st.header("Types of diabetes are:")

tab1, tab2, tab3 = st.tabs(["Type 1", "Type 2", "Gestational"])

with tab1:
   st.header("Type 1")
   st.markdown("Type 1 diabetes is thought to be caused by an autoimmune reaction (the body attacks itself by mistake). This reaction stops your body from making insulin. Approximately 5-10% of the people who have diabetes have type 1. Symptoms of type 1 diabetes often develop quickly. It’s usually diagnosed in children, teens, and young adults")

with tab2:
   st.header("Type 2")
   st.markdown("With type 2 diabetes, your body doesn’t use insulin well and can’t keep blood sugar at normal levels. About 90-95% of people with diabetes have type 2. It develops over many years and is usually diagnosed in adults (but more and more in children, teens, and young adults). You may not notice any symptoms, so it’s important to get your blood sugar tested if you’re at risk. Type 2 diabetes can be prevented or delayed with healthy lifestyle changes, such as:" 
"Losing weight." "Eating healthy food." "Being active.")

with tab3:
   st.header("Gestational")
   st.markdown("Gestational diabetes develops in pregnant women who have never had diabetes. If you have gestational diabetes, your baby could be at higher risk for health problems. Gestational diabetes usually goes away after your baby is born. However, it increases your risk for type 2 diabetes later in life. Your baby is more likely to have obesity as a child or teen and develop type 2 diabetes later in life.")

    
               


# In[68]:


st.sidebar.success("Direct to desired route")

PAGE_DICT = {
    "Home": "home",
    "About": "about",
    "Research": 'research'
}


def home():
    st.write("This is the home page.")


def about():
    st.write("This is the about page.")
    
def research():
    st.write("This is the research page.")


# In[69]:


def main():
    st.title("Streamlit Navigation Bar Example")

    page = st.sidebar.selectbox("Select a page", list(PAGE_DICT.keys()))

    if page == "Home":
        home()
    elif page == "About":
        about()
    elif page == "Research":
        research()


if __name__ == "_main_":
    run()


# In[70]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix

import warnings
warnings.filterwarnings('ignore')

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[71]:


data = pd.read_csv('C:/Users/naman/Downloads/diabetes.csv')  
data.head(10)


# In[72]:


import missingno as msno
msno.bar(data)
plt.show()


# In[75]:


fig = sns.pairplot(data,hue='Outcome')
fig


# In[79]:


st.pyplot()



st.set_option('deprecation.showPyplotGlobalUse', False)


# In[ ]:





# In[ ]:




