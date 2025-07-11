import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
from PIL import Image
import torch

# Load models (cache to avoid reloading)
@st.cache_resource
def load_models():
    blip_model = BlipForQuestionAnswering.from_pretrained("sharawy53/diploma")
    processor = BlipProcessor.from_pretrained("sharawy53/diploma")
    ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    return blip_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model

blip_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()

# Medical terms dictionary and translation functions remain unchanged
medical_terms = { ... }  # Copy from original code

# Main function (simplified)
def vqa_multilingual(image, question):
    # Copy logic from original function
    ...

# Streamlit UI
st.title("نموذج ثنائي اللغة (عربي - إنجليزي) لتحليل صور الأشعة")
uploaded_image = st.file_uploader("رفع صورة الأشعة", type=["jpg", "png"])
question = st.text_input("أدخل سؤالك (بالعربية أو الإنجليزية)")

if uploaded_image and question:
    image = Image.open(uploaded_image).convert("RGB")
    result = vqa_multilingual(image, question)
    # Display results using st.write or st.markdown
    ...
