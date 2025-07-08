# medical_vqa_chatbot.py
import streamlit as st
from PIL import Image
import requests
import torch
import logging
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    MarianTokenizer,
    MarianMTModel,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache resource-intensive model loading
@st.cache_resource
def load_models():
    """Load all required models with progress indicators"""
    models = {}
    
    with st.spinner("🔄 Loading Medical VQA Model..."):
        try:
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4-bit instead of 8-bit
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load processor and model
            models['vqa_processor'] = AutoProcessor.from_pretrained(
                "Mohamed264/llava-medical-VQA-lora-merged3",
                use_fast=True
            )
            
            models['vqa_model'] = AutoModelForVision2Seq.from_pretrained(
                "Mohamed264/llava-medical-VQA-lora-merged3",
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
        except Exception as e:
            logger.error(f"Error loading VQA model: {str(e)}")
            st.error(f"❌ Failed to load medical model: {str(e)}")
            return None
    
    with st.spinner("🔄 Loading Translation Models..."):
        try:
            # English to Arabic
            models['en_ar_tokenizer'] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            models['en_ar_model'] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
            
            # Arabic to English
            models['ar_en_tokenizer'] = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
            models['ar_en_model'] = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        except Exception as e:
            logger.error(f"Error loading translation models: {str(e)}")
            st.error(f"❌ Failed to load translation models: {str(e)}")
            return None
    
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
    if not models:
        return "Error: Models not loaded", "en"
    
    # Detect input language
    input_lang = detect_language(question)
    
    # Translate to English if needed
    if input_lang == 'ar':
        english_question = translate_text(question, 'ar', 'en', models)
    else:
        english_question = question
    
    # Prepare prompt for VQA model
    prompt = f"Question: {english_question}\nAnswer:"
    
    try:
        # Process inputs
        processor = models['vqa_processor']
        vqa_model = models['vqa_model']
        
        # Process image and text together
        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt",
            padding=True
        ).to(vqa_model.device)
        
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
    
    except Exception as e:
        logger.error(f"VQA processing error: {str(e)}")
        return f"Error processing request: {str(e)}", "en"

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
    
    # Display warning about initial load time
    st.info("⚠️ First-time loading may take 2-5 minutes as we download AI models. Please be patient.")
    
    # Load models (cached)
    models = load_models()
    
    if not models:
        st.error("Critical error: Failed to load AI models. Please check the logs.")
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Image input section
        st.subheader("1. Medical Image")
        img_source = st.radio("Image source:", ["Upload", "URL"])
        
        image = None
        if img_source == "Upload":
            image_file = st.file_uploader("Upload medical image", type=["jpg", "jpeg", "png"])
            if image_file:
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            image_url = st.text_input("Image URL:", placeholder="https://example.com/image.jpg")
            if image_url:
                try:
                    image = Image.open(requests.get(image_url, stream=True).raw)
                    st.image(image, caption="Image from URL", use_column_width=True)
                except:
                    st.error("❌ Failed to load image from URL")
    
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
