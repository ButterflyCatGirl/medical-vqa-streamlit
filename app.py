import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
from PIL import Image
import torch

# Load models (cached to avoid reloading)
@st.cache_resource
def load_models():
    # Load BLIP model
    blip_model = BlipForQuestionAnswering.from_pretrained("sharawy53/diploma")
    processor = BlipProcessor.from_pretrained("sharawy53/diploma")
    
    # Load translation models
    ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    
    return blip_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model

blip_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()

# Medical terms dictionary (unchanged)
medical_terms = {
    "chest x-ray": "أشعة سينية للصدر",
    "x-ray": "أشعة سينية",
    "ct scan": "تصوير مقطعي محوسب",
    "mri": "تصوير بالرنين المغناطيسي",
    "ultrasound": "تصوير بالموجات فوق الصوتية",
    "normal": "طبيعي",
    "abnormal": "غير طبيعي",
    "brain": "الدماغ",
    "fracture": "كسر",
    "no abnormality detected": "لا توجد شذوذات",
    "left lung": "الرئة اليسرى",
    "right lung": "الرئة اليمنى"
}

# Translation functions (unchanged)
def translate_ar_to_en(text):
    inputs = ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = ar_en_model.generate(**inputs)
    return ar_en_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text):
    inputs = en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = en_ar_model.generate(**inputs)
    return en_ar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_answer_medical(answer_en):
    key = answer_en.lower().strip()
    return medical_terms.get(key, translate_en_to_ar(answer_en))

# Main VQA function (unchanged)
def vqa_multilingual(image, question):
    if not image or not question.strip():
        return "يرجى رفع صورة وكتابة سؤال.", "", "", ""
    
    # Detect Arabic input
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
    if is_arabic:
        question_ar = question.strip()
        question_en = translate_ar_to_en(question_ar)
    else:
        question_en = question.strip()
        question_ar = translate_en_to_ar(question_en)
    
    # Process with BLIP
    inputs = processor(image, question_en, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    answer_en = processor.decode(output[0], skip_special_tokens=True).strip()
    answer_ar = translate_answer_medical(answer_en)
    
    return question_ar, question_en, answer_ar, answer_en

# Streamlit UI
st.set_page_config(page_title="VQA طبية ثنائية اللغة", layout="wide")
st.title("نموذج ثنائي اللغة (عربي - إنجليزي) لتحليل صور الأشعة")
st.markdown("ارفع صورة طبية واسأل بالعربية أو الإنجليزية، وستحصل على الإجابة باللغتين.")

# Image and question inputs
col1, col2 = st.columns(2)
with col1:
    uploaded_image = st.file_uploader("رفع صورة الأشعة", type=["jpg", "png"])
with col2:
    question = st.text_input("أدخل سؤالك (بالعربية أو الإنجليزية)")

# Process and display results
if uploaded_image and question:
    image = Image.open(uploaded_image).convert("RGB")
    question_ar, question_en, answer_ar, answer_en = vqa_multilingual(image, question)
    
    st.markdown("---")
    st.subheader("🔍 النتائج")
    st.markdown(f"**السؤال بالعربية:** {question_ar}")
    st.markdown(f"**السؤال بالإنجليزية:** {question_en}")
    st.markdown(f"**الإجابة بالعربية:** {answer_ar}")
    st.markdown(f"**الإجابة بالإنجليزية:** {answer_en}")
