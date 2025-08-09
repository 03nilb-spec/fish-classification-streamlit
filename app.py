import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fish_Mobile_net_V2_1.h5")  # change to your model file
    return model

model = load_model()

# --- Class Labels ---
class_labels = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]  

# --- Streamlit UI ---
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish to identify its species.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_size = (224, 224)  # change to your training size
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence_scores = predictions[0]

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.write("Confidence Scores:")
    for label, score in zip(class_labels, confidence_scores):
        st.write(f"{label}: {score:.2%}")