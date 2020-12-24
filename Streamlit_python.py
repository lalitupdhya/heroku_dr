#!/usr/bin/env python
# coding: utf-8

# In[52]:


import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# In[53]:


model = load_model('diab_retin.h5')


# In[54]:


st.title('Diabetic Retinopathy Detection.')
uploaded_image=st.file_uploader('Upload the fundus image of the eye.')
if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
    image=cv2.resize(opencv_image,(224,224))
    st.write('The fundus image.')
    disp=Image.fromarray(image)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(RGB_img)
    
    image=np.array(image).reshape(-1,224,224,3)
    
    CATEGORIES = ['No_Diabetic Retinopathy','Mild_Diabetic Retinopathy','Moderate_Diabetic Retinopathy',
                  'Severe_Diabetic Retinopathy','Proliferate_Diabetic Retinopathy']
    prediction=CATEGORIES[np.argmax(model.predict(image))]
    st.write('The above image has',prediction)


# In[ ]:




