import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering, MarianMTModel, MarianTokenizer
from PIL import Image
import torch

# Cache model and processor loading for performance
@st.cache_resource
def load_blip_model():
    return BlipForQuestionAnswering.from_pretrained("sharawy53/diploma")

@st.cache_resource
def load_blip_processor():
    return BlipProcessor.from_pretrained("sharawy53/diploma")

@st.cache_resource
def load_ar_en_tokenizer():
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

@st.cache_resource
def load_ar_en_model():
    return MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

@st.cache_resource
def load_en_ar_tokenizer():
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

@st.cache_resource
def load_en_ar_model():
    return MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

# Load models and processors
blip_model = load_blip_model()
processor = load_blip_processor()
ar_en_tokenizer = load_ar_en_tokenizer()
ar_en_model = load_ar_en_model()
en_ar_tokenizer = load_en_ar_tokenizer()
en_ar_model = load_en_ar_model()

# Translation functions
def translate_ar_to_en(text):
    inputs = ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = ar_en_model.generate(**inputs)
    return ar_en_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text):
    inputs = en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = en_ar_model.generate(**inputs)
    return en_ar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Manual medical terms dictionary
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

def translate_answer_medical(answer_en):
    key = answer_en.lower().strip()
    if key in medical_terms:
        return medical_terms[key]
    else:
        return translate_en_to_ar(answer_en)

# Main VQA function
def vqa_multilingual(image, question):
    if not image or not question.strip():
        return "يرجى رفع صورة وكتابة سؤال.", "", "", ""

    is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
    if is_arabic:
        question_ar = question.strip()
        question_en = translate_ar_to_en(question_ar)
    else:
        question_en = question.strip()
        question_ar = translate_en_to_ar(question_en)

    inputs = processor(image, question_en, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    answer_en = processor.decode(output[0], skip_special_tokens=True).strip()
    answer_ar = translate_answer_medical(answer_en)

    return (
        f"السؤال بالعربية:\n{question_ar}",
        f"السؤال بالإنجليزية:\n{question_en}",
        f"الإجابة بالعربية:\n{answer_ar}",
        f"الإجابة بالإنجليزية:\n{answer_en}"
    )

# Streamlit interface
st.title("نموذج ثنائي اللغة (عربي - إنجليزي) لصور الأشعة")
st.write("ارفع صورة طبية واسأل بالعربية أو الإنجليزية، وستحصل على الإجابة باللغتين.")

uploaded_image = st.file_uploader("🔍 ارفع صورة الأشعة", type=["jpg", "png", "jpeg"])
question = st.text_input("💬 أدخل سؤالك (بالعربية أو الإنجليزية)")

if st.button("Submit"):
    if uploaded_image is not None and question.strip():
        with st.spinner("جارٍ المعالجة..."):
            image = Image.open(uploaded_image)
            result = vqa_multilingual(image, question)
        st.write("🟠 السؤال بالعربية:")
        st.write(result[0])
        st.write("🟢 السؤال بالإنجليزية:")
        st.write(result[1])
        st.write("🟠 الإجابة بالعربية:")
        st.write(result[2])
        st.write("🟢 الإجابة بالإنجليزية:")
        st.write(result[3])
    else:
        st.write("يرجى رفع صورة وكتابة سؤال.")
