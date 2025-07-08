import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, MarianTokenizer, MarianMTModel
from PIL import Image
import time

st.set_page_config(
    page_title="Medical VQA Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    try:
        processor = AutoProcessor.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
        vqa_model = AutoModelForImageTextToText.from_pretrained(
            "Mohamed264/llava-medical-VQA-lora-merged3",
            torch_dtype=torch.float32,
            device_map=None
        )

        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

        return processor, vqa_model, en_ar_tokenizer, en_ar_model, ar_en_tokenizer, ar_en_model

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None, None

processor, vqa_model, en_ar_tokenizer, en_ar_model, ar_en_tokenizer, ar_en_model = load_models()

if not all([processor, vqa_model, en_ar_tokenizer, en_ar_model, ar_en_tokenizer, ar_en_model]):
    st.stop()

# Language detection

def detect_language(text):
    ar_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    en_chars = sum(1 for c in text.lower() if 'a' <= c <= 'z')
    return 'ar' if ar_chars > en_chars else 'en'

# Translation

def translate_text(text, source, target):
    try:
        if source == 'ar' and target == 'en':
            inputs = ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = ar_en_model.generate(**inputs, max_length=512)
            return ar_en_tokenizer.decode(translated[0], skip_special_tokens=True).strip()
        elif source == 'en' and target == 'ar':
            inputs = en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = en_ar_model.generate(**inputs, max_length=512)
            return en_ar_tokenizer.decode(translated[0], skip_special_tokens=True).strip()
        return text
    except:
        return text

# Main VQA logic

def process_vqa(image, question):
    lang = detect_language(question)
    question_en = translate_text(question, 'ar', 'en') if lang == 'ar' else question
    prompt = f"USER: <image>\n{question_en}\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vqa_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    answer_en = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    answer = translate_text(answer_en, 'en', 'ar') if lang == 'ar' else answer_en
    return answer

# Streamlit UI

st.title("🏥 Medical VQA Chatbot")
st.markdown("Upload medical images and ask questions in English or Arabic | ارفع الصور الطبية واسأل بالإنجليزية أو العربية")

uploaded_file = st.file_uploader("Choose a medical image | اختر صورة طبية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
user_question = st.text_area("Enter your question in English or Arabic:", height=100)

if st.button("🚀 Send | إرسال"):
    if not uploaded_file or not user_question.strip():
        st.warning("⚠️ Please upload an image and enter a question! | يرجى رفع صورة وكتابة سؤال!")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Processing your question..."):
            response = process_vqa(image, user_question)
        st.success("✅ Response:")
        st.markdown(response)
