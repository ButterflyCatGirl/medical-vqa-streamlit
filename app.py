# app.py
import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
from PIL import Image
import torch

# Set page config
st.set_page_config(
    page_title="نموذج ثنائي اللغة لصور الأشعة الطبية",
    page_icon="🩻",
    layout="centered"
)

# Load models with caching
@st.cache_resource
def load_models():
    # Load BLIP VQA model
    blip_model = BlipForQuestionAnswering.from_pretrained("sharawy53/diploma")
    processor = BlipProcessor.from_pretrained("sharawy53/diploma")
    
    # Load translation models
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    
    return {
        "blip": (blip_model, processor),
        "ar_en": (ar_en_model, ar_en_tokenizer),
        "en_ar": (en_ar_model, en_ar_tokenizer)
    }

models = load_models()

# Medical terms dictionary
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

# Translation functions
def translate_ar_to_en(text):
    model, tokenizer = models["ar_en"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text):
    model, tokenizer = models["en_ar"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_answer_medical(answer_en):
    key = answer_en.lower().strip()
    return medical_terms.get(key, translate_en_to_ar(answer_en))

# Main VQA function
def vqa_multilingual(image, question):
    if not image or not question.strip():
        return None
    
    # Detect language
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
    question_ar = question.strip() if is_arabic else translate_en_to_ar(question)
    question_en = translate_ar_to_en(question) if is_arabic else question.strip()

    # Process image and question
    blip_model, processor = models["blip"]
    inputs = processor(image, question_en, return_tensors="pt")
    
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    
    answer_en = processor.decode(output[0], skip_special_tokens=True).strip()
    answer_ar = translate_answer_medical(answer_en)
    
    return {
        "arabic_question": question_ar,
        "english_question": question_en,
        "arabic_answer": answer_ar,
        "english_answer": answer_en
    }

# UI Components
st.title("نموذج ثنائي اللغة (عربي - إنجليزي) خاص بصور الأشعة الطبية")
st.markdown("""
ارفع صورة طبية (أشعة مقطعية، أشعة سينية، رنين مغناطيسي) واسأل بالعربية أو الإنجليزية، وستحصل على الإجابة باللغتين.
""")

col1, col2 = st.columns(2)
with col1:
    uploaded_image = st.file_uploader("🔍 ارفع صورة الأشعة", type=["jpg", "jpeg", "png"])
with col2:
    question_input = st.text_input("💬 أدخل سؤالك (بالعربية أو الإنجليزية)", "")

if uploaded_image and question_input:
    image = Image.open(uploaded_image)
    with st.spinner("جارٍ معالجة السؤال والصورة..."):
        results = vqa_multilingual(image, question_input)
    
    if results:
        st.success("تم تحليل الصورة والسؤال بنجاح!")
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        
        st.subheader("🟠 السؤال بالعربية")
        st.info(results["arabic_question"])
        
        st.subheader("🟢 السؤال بالإنجليزية")
        st.info(results["english_question"])
        
        st.subheader("🟠 الإجابة بالعربية")
        st.success(results["arabic_answer"])
        
        st.subheader("🟢 الإجابة بالإنجليزية")
        st.success(results["english_answer"])
    else:
        st.error("حدث خطأ أثناء المعالجة. يرجى المحاولة مرة أخرى.")
