# medical_vqa_chatbot.py
import streamlit as st
from PIL import Image
import requests
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    MarianTokenizer,
    MarianMTModel
)

# Cache resource-intensive model loading
@st.cache_resource
def load_models():
    """Load all required models with progress indicators"""
    models = {}
    
    with st.spinner("🔄 Loading Medical VQA Model..."):
        models['vqa_processor'] = AutoProcessor.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
        models['vqa_model'] = AutoModelForImageTextToText.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
    
    with st.spinner("🔄 Loading Translation Models..."):
        # English to Arabic
        models['en_ar_tokenizer'] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        models['en_ar_model'] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        
        # Arabic to English
        models['ar_en_tokenizer'] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        models['ar_en_model'] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    
    return models

def detect_language(text):
    """Detect if text is Arabic or English"""
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F')
    english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
    return 'ar' if arabic_chars > english_chars else 'en'

def translate_text(text, source_lang, target_lang, models):
    """Translate text between English and Arabic"""
    if source_lang == target_lang or not text.strip():
        return text

    if source_lang == 'en' and target_lang == 'ar':
        tokenizer = models['en_ar_tokenizer']
        model = models['en_ar_model']
    elif source_lang == 'ar' and target_lang == 'en':
        tokenizer = models['ar_en_tokenizer']
        model = models['ar_en_model']
    else:
        return text

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def process_medical_vqa(image, question, models):
    """Process medical VQA with translation support"""
    # Detect input language
    input_lang = detect_language(question)
    
    # Translate to English if needed
    if input_lang == 'ar':
        english_question = translate_text(question, 'ar', 'en', models)
    else:
        english_question = question
    
    # Prepare prompt for VQA model
    prompt = f"Question: {english_question}\nAnswer:"
    
    # Process inputs
    processor = models['vqa_processor']
    vqa_model = models['vqa_model']
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(vqa_model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = vqa_model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode response
    english_response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "Answer:" in english_response:
        english_response = english_response.split("Answer:")[-1].strip()
    
    # Translate response back to user's language
    if input_lang == 'ar':
        final_response = translate_text(english_response, 'en', 'ar', models)
    else:
        final_response = english_response
    
    return final_response, input_lang

def load_image(image_input):
    """Load image from different sources"""
    if isinstance(image_input, str):
        if image_input.startswith('http'):
            return Image.open(requests.get(image_input, stream=True).raw)
        else:
            return Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        return image_input
    return image_input

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Medical VQA Chatbot",
        page_icon="🩺",
        layout="wide"
    )
    
    # Title and description
    st.title("🩺 Medical Visual Question Answering")
    st.markdown("""
    **Upload a medical image and ask questions in English or Arabic**
    - Supported images: X-rays, CT scans, MRIs, etc.
    - Example questions: 
        - "What abnormalities are visible in this X-ray?"
        - "ما هو التشخيص المحتمل لهذه الصورة؟"
    """)
    
    # Load models (cached)
    models = load_models()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Image input section
        st.subheader("1. Medical Image")
        img_source = st.radio("Image source:", ["Upload", "URL"])
        
        if img_source == "Upload":
            image_file = st.file_uploader("Upload medical image", type=["jpg", "jpeg", "png"])
            if image_file:
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                image = None
        else:
            image_url = st.text_input("Image URL:", placeholder="https://example.com/image.jpg")
            if image_url:
                try:
                    image = Image.open(requests.get(image_url, stream=True).raw)
                    st.image(image, caption="Image from URL", use_column_width=True)
                except:
                    st.error("❌ Failed to load image from URL")
                    image = None
            else:
                image = None
    
    with col2:
        # Chat interface
        st.subheader("2. Ask a Question")
        question = st.text_area("Question (English or Arabic):", 
                               placeholder="Type your medical question here...",
                               height=100)
        
        if st.button("Get Diagnosis", type="primary") and image and question:
            with st.spinner("🔍 Analyzing image and question..."):
                try:
                    # Process VQA
                    response, lang = process_medical_vqa(image, question, models)
                    
                    # Display results
                    st.success("✅ Analysis Complete")
                    st.subheader("Answer:")
                    st.write(response)
                    
                    # Language indicator
                    lang_name = "Arabic" if lang == "ar" else "English"
                    st.caption(f"Detected question language: {lang_name}")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
        elif not image:
            st.warning("⚠️ Please provide a medical image")
        elif not question:
            st.warning("⚠️ Please enter a question")

    # Add footer
    st.markdown("---")
    st.caption("Medical VQA Chatbot | Built with [LLaVA Medical](https://huggingface.co/Mohamed264/llava-medical-VQA-lora-merged3) and Streamlit")

if __name__ == "__main__":
    main()
