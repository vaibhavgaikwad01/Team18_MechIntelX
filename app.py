# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image

# # Load your trained model
# model = load_model("model.h5")

# # Streamlit UI
# st.set_page_config(page_title="Waste Classifier", layout="centered")
# st.title("♻️ Waste Classification App")
# st.write("Upload an image of waste and classify it as **Organic (O)** or **Recyclable (R)**.")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_container_width=True)

#     # Preprocess image
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     # Predict
#     prediction = model.predict(img_array)
#     class_name = "Recyclable (R)" if prediction[0][0] > 0.5 else "Organic (O)"
    
#     st.success(f"Predicted Class: **{class_name}**")

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Class labels (adjust if different)
class_labels = ["Wet (Organic)", "Dry (Recyclable)"]

# Preprocess image for model
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict class
def predict(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0][0]
    class_index = 1 if prediction >= 0.5 else 0
    confidence = prediction if class_index == 1 else 1 - prediction
    return class_labels[class_index], confidence

# UI
st.set_page_config(page_title="♻️ Waste Classifier", layout="centered")
st.title("♻️ Waste Classification App")

# Option Selection
option = st.radio("Choose Input Method:", ("📤 Upload Image", "📷 Capture Using Camera"))

image = None

if option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif option == "📷 Capture Using Camera":
    camera_image = st.camera_input("Take a photo of the waste")
    if camera_image:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)

# Prediction Button
if image:
    if st.button("🔍 Classify Waste"):
        label, confidence = predict(image)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2%}**")



