import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


model = tf.keras.models.load_model(r'C:\Users\Siddhartha Devan V\jupyter ml\COVID\vgg16_cov_1')


st.title('COVID-19 Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    image_array = cv2.resize(image_array, (224, 224))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    st.image(image_array, caption='Uploaded Image', use_column_width=True)

    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)

    pred_dict = {0: 'Covid Negative', 1: 'Covid Positive', 2: 'Other Non-Covid Diseases'}
    predicted_class = pred_dict[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(predicted_class)

    st.subheader("Prediction Probabilities:")
    for i, class_label in pred_dict.items():
        st.write(f"{class_label}: {prediction[0][i]:.4f}")


