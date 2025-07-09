import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import langdetect

# --- Model Loading with Caching and Error Handling ---
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_translation_models():
    try:
        ar_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        en_ar_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        return {
            'ar_en': {'model': ar_en_model, 'tokenizer': ar_en_tokenizer},
            'en_ar': {'model': en_ar_model, 'tokenizer': en_ar_tokenizer}
        }
    except Exception as e:
        st.error(f"Error loading translation models: {str(e)}")
        return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_vqa_model():
    try:
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, processor
    except Exception as e:
        st.error(f"Error loading VQA model: {str(e)}")
        return None, None

# Load models (cached to avoid reloading on each run)
with st.spinner("Loading models (this may take a few minutes on first run)..."):
    translation_models = load_translation_models()
    vqa_model, vqa_processor = load_vqa_model()

if translation_models and vqa_model and vqa_processor:
    st.success("Models loaded successfully!")
else:
    st.error("Failed to load models. Check the error messages above.")

# --- Helper Functions ---
def detect_language(text):
    """Detect if the text is in Arabic or English using langdetect."""
    try:
        lang = langdetect.detect(text)
        return 'ar' if lang.startswith('ar') else 'en'
    except:
        # Fallback to simple character-based detection
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F')
        english_chars = sum(1 for char in text.lower() if 'a' <= char <= 'z')
        return 'ar' if arabic_chars > english_chars else 'en'

def translate_text(text, source_lang, target_lang, translation_models):
    """Translate text between English and Arabic."""
    if source_lang == target_lang:
        return text
    model_key = f"{source_lang}_{target_lang}"
    if model_key not in translation_models:
        return text
    model = translation_models[model_key]['model']
    tokenizer = translation_models[model_key]['tokenizer']
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def process_medical_vqa(image, question, model, processor, translation_models):
    """Process the image and question through the VQA model with translation."""
    # Detect input language
    input_lang = detect_language(question)
    
    # Translate to English if needed (LLaVA expects English)
    if input_lang == 'ar':
        english_question = translate_text(question, 'ar', 'en', translation_models)
    else:
        english_question = question
    
    # Prepare inputs for LLaVA
    prompt = f"<|begin_of_text|>USER: <image>\nQuestion: {english_question}\nAnswer: ASSISTANT:"
    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode the response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    english_response = response.split("ASSISTANT:")[-1].strip()
    
    # Translate back to input language if needed
    final_response = translate_text(english_response, 'en', 'ar', translation_models) if input_lang == 'ar' else english_response
    
    return final_response, input_lang

# --- Streamlit Interface ---
st.title("Medical Visual Question Answering (VQA)")
st.markdown("Upload a medical image and ask a question in **English** or **Arabic**. This app uses a fine-tuned LLaVA model to provide answers.")

# Input section
with st.form(key='vqa_form'):
    image = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])
    question = st.text_input("Question", placeholder="e.g., 'What abnormality is visible?' or 'ما هو الشذوذ المرئي؟'")
    submit_button = st.form_submit_button(label="Get Answer")

# Output section
if submit_button:
    if image is not None and question:
        with st.spinner("Processing your request..."):
            # Open the uploaded image
            image_pil = Image.open(image).convert("RGB")
            # Process the VQA request
            response, detected_lang = process_medical_vqa(
                image_pil, 
                question, 
                vqa_model, 
                vqa_processor, 
                translation_models
            )
            # Display results
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            st.subheader("Answer")
            st.write(response)
            st.subheader("Detected Language")
            st.write("Arabic" if detected_lang == 'ar' else "English")
    else:
        st.error("Please upload an image and enter a question.")
