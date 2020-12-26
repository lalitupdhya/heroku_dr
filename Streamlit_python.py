#!/usr/bin/env python
# coding: utf-8

# ## Front End of our Application to detect the Disease

# ### Importing the necessary libraries

# In[23]:


import streamlit as st # using streamlit to create the UI of our application
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# In[24]:


model = load_model('diab_retin.h5') # Loading the trained model


# In[25]:


# About Page of our application
about = 'Diabetic retinopathy is caused by damage to the blood vessels in the tissue at the back of the eye (retina).        Poorly controlled blood sugar is a risk factor.Early symptoms include floaters, blurriness, dark areas of vision and        difficulty perceiving colours. Blindness can occur. Mild cases may be treated with careful diabetes management.        Advanced cases may require laser treatment or surgery. Diagnosis of this disease can be done with the fundus image of an eye.'


# In[26]:


# Defining sidebar for Navigation
sd = st.sidebar.radio('NAVIGATION',['Home','Prediction'])


# In[27]:


if sd=='Prediction': #If in the sidebar prediction is selected run the included script
    st.title('Diabetic Retinopathy Detection.')
    uploaded_image=st.file_uploader('Upload the fundus image of the eye.')# file uploader to upload and process the image
    if uploaded_image:
        # converting the uploaded image to array and resizing it to 224x224 which is the input of our trained model.
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
        image=cv2.resize(opencv_image,(224,224))
        st.write('The fundus image.')
        disp=Image.fromarray(image)
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(RGB_img) #Displaying the uploaded image in the application

        image=np.array(image).reshape(-1,224,224,3)

        CATEGORIES = ['No_Diabetic Retinopathy','Mild_Diabetic Retinopathy','Moderate_Diabetic Retinopathy',
                      'Severe_Diabetic Retinopathy','Proliferate_Diabetic Retinopathy']
        prediction=CATEGORIES[np.argmax(model.predict(image))] # predicting the reesults
        st.write('The above image has',prediction) # displaying the predicted results
else: # Home page of the appliction
    st.header('About Diabetic Retinopathy')
    st.write(about)
    st.write('In this application we can detect 5 levels of the disease ranging from NO_DR to Proliferate_DR. In between                we have Mild, Moderate and Severe.')
    st.subheader('Fundus Image of an eye')
    st.image('FUNDUS2.jpg')


# In[ ]:




