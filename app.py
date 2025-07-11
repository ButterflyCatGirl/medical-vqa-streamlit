# app.py
import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import sys

# Set page config
st.set_page_config(
    page_title="نموذج ثنائي اللغة لصور الأشعة الطبية",
    page_icon="🩻",
    layout="centered"
)

# Check for required libraries
try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    st.error("""
    **Missing required dependencies!**  
    Please add `sentencepiece` to your requirements.txt file:
    ```
    sentencepiece==0.2.0
    ```
    """)
    st.stop()

# Load models with caching and error handling
@st.cache_resource(show_spinner="جارٍ تحميل النماذج... (قد يستغرق بضع دقائق)")
def load_models():
    try:
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
    except Exception as e:
        st.error(f"فشل تحميل النماذج: {str(e)}")
        st.error("""
        **الاحتمالات:**
        1. مساحة التخزين غير كافية (النماذج تحتاج ~1.5GB)
        2. مشكلة في اتصال الإنترنت
        3. مشكلة في خادم Hugging Face
        """)
        st.stop()

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
    "right lung": "الرئة اليمنى",
    "heart": "القلب",
    "lungs": "الرئتين",
    "spine": "العمود الفقري"
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
    
    try:
        if is_arabic:
            question_ar = question.strip()
            question_en = translate_ar_to_en(question_ar)
        else:
            question_en = question.strip()
            question_ar = translate_en_to_ar(question_en)

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
    except Exception as e:
        st.error(f"خطأ في المعالجة: {str(e)}")
        return None

# UI Components
st.title("🧠 نموذج ثنائي اللغة (عربي/إنجليزي) لتحليل صور الأشعة")
st.markdown("""
ارفع صورة طبية (أشعة سينية، مقطعية، رنين مغناطيسي) واسأل بالعربية أو الإنجليزية
""")

with st.form("vqa_form"):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("📷 ارفع صورة الأشعة", type=["jpg", "jpeg", "png"])
    with col2:
        question_input = st.text_input("❓ أدخل سؤالك (عربي/إنجليزي)", "")
    
    submitted = st.form_submit_button("🚀 حلل الصورة")

if submitted:
    if not uploaded_image:
        st.warning("⚠️ يرجى رفع صورة أولاً")
        st.stop()
        
    if not question_input.strip():
        st.warning("⚠️ يرجى إدخال سؤال")
        st.stop()
        
    image = Image.open(uploaded_image)
    with st.spinner("جارٍ تحليل الصورة والسؤال... (قد يستغرق 10-20 ثانية)"):
        results = vqa_multilingual(image, question_input)
    
    if results:
        st.success("✅ تم التحليل بنجاح!")
        st.image(image, caption="الصورة المرفوعة", width=300)
        
        st.subheader("🗨️ النتائج")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🟠 السؤال بالعربية**")
            st.info(results["arabic_question"])
            
            st.markdown("**🟢 الإجابة بالإنجليزية**")
            st.success(results["english_answer"])
            
        with col2:
            st.markdown("**🟢 السؤال بالإنجليزية**")
            st.info(results["english_question"])
            
            st.markdown("**🟠 الإجابة بالعربية**")
            st.success(results["arabic_answer"])
    else:
        st.error("❌ فشل في التحليل. يرجى المحاولة مرة أخرى")
