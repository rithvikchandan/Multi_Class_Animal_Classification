import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set page config
st.set_page_config(page_title="Animal Classifier", layout="centered")

# Load the model
@st.cache_resource
def load_animal_model():
    return load_model("MCAR.keras")

model = load_animal_model()

# Actual 90 animal class names from dataset
class_names = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
    'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog',
    'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
    'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
    'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
    'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse',
    'octopus', 'okapi', 'opossum', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot',
    'pelican', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer',
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'skunk', 'sloth', 'snail',
    'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle',
    'walrus', 'wasp', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

# App UI
st.title("🐾 Multi-Class Animal Classifier")
st.markdown("Upload an image of an animal, and I’ll tell you what it is using a deep learning model trained on 90 animals!")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[0][predicted_idx]) * 100

    st.success(f"🧠 Prediction: **{predicted_class.capitalize()}** with **{confidence:.2f}%** confidence.")
