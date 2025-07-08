import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText, MarianMTModel, MarianTokenizer
from PIL import Image
import torch

st.set_page_config(page_title="Bilingual Medical VQA", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=False)

llava_model = AutoModelForImageTextToText.from_pretrained(
    "Mohamed264/llava-medical-VQA-lora-merged3",
    device_map=None,
    torch_dtype=torch.float32,
    quantization_config=bnb_config
)
        processor = AutoProcessor.from_pretrained(
            "Mohamed264/llava-medical-VQA-lora-merged3"
        )

        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

        return (
            llava_model,
            processor,
            ar_en_tokenizer,
            ar_en_model,
            en_ar_tokenizer,
            en_ar_model,
        )
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None, None

# Instantiate models
llava_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()

if not all([llava_model, processor, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model]):
    st.stop()

# Translation helpers
def translate_ar_to_en(text):
    inputs = ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = ar_en_model.generate(**inputs)
    return ar_en_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text):
    inputs = en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = en_ar_model.generate(**inputs)
    return en_ar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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

def translate_answer_medical(answer_en):
    key = answer_en.lower().strip()
    return medical_terms.get(key, translate_en_to_ar(answer_en))

# Main VQA function
def vqa_multilingual(image, question):
    if not image or not question.strip():
        return "", "", "", "يرجى رفع صورة وكتابة سؤال."

    is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
    question_ar = question.strip() if is_arabic else translate_en_to_ar(question)
    question_en = translate_ar_to_en(question) if is_arabic else question.strip()

    inputs = processor(image, question_en, return_tensors="pt")
    with torch.no_grad():
        output = llava_model.generate(**inputs)
    answer_en = processor.decode(output[0], skip_special_tokens=True).strip()
    answer_ar = translate_answer_medical(answer_en)

    return question_ar, question_en, answer_ar, answer_en

# Streamlit UI
st.title("🧠 نموذج الأسئلة البصرية الطبية (VQA) ثنائي اللغة")
st.markdown("ارفع صورة طبية واسأل بالعربية أو الإنجليزية، وستحصل على الإجابة باللغتين.")

uploaded_image = st.file_uploader("🔍 ارفع صورة الأشعة", type=["jpg", "jpeg", "png"])
user_question = st.text_input("💬 أدخل سؤالك (بالعربية أو الإنجليزية):")

if st.button("🔎 تحليل الصورة والإجابة"):
    if uploaded_image and user_question:
        image = Image.open(uploaded_image).convert("RGB")
        question_ar, question_en, answer_ar, answer_en = vqa_multilingual(
            image, user_question
        )

        st.subheader("📌 السؤال بالعربية")
        st.success(question_ar)

        st.subheader("📌 السؤال بالإنجليزية")
        st.success(question_en)

        st.subheader("✅ الإجابة بالعربية")
        st.info(answer_ar)

        st.subheader("✅ الإجابة بالإنجليزية")
        st.info(answer_en)
    else:
        st.error("يرجى رفع صورة وكتابة سؤال.")
