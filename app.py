import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./generate_faces.h5')
    return model

generator = load_model()

NOISE_DIM = 100

# Function to generate faces
def generate_faces(num_faces):
    random_latent_vectors = tf.random.normal([num_faces, NOISE_DIM])
    generated_faces = generator(random_latent_vectors, training=False)
    generated_faces = (generated_faces * 127.5 + 127.5).numpy().astype(np.uint8)
    return generated_faces

# Streamlit UI
st.title("Face Generator")
num_faces = st.selectbox("Select number of faces to generate:", [1, 5, 10, 20])
if st.button("Generate"):
    faces = generate_faces(num_faces)
    st.write(f"Generated {num_faces} face(s)")
    for i in range(num_faces):
        st.image(faces[i], width=128, caption=f"Face {i+1}")
