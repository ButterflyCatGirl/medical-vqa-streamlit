import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image

# Set page title and layout
st.set_page_config(page_title="BLIP VQA Model", layout="wide")
st.title("🤖 BLIP VQA Model - `sharawy53/diploma`")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model and processor with caching"""
    model_id = "sharawy53/diploma"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

def predict(image, question):
    """Run model prediction"""
    try:
        inputs = processor(image, question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load model with progress indicator
with st.spinner("Loading AI model (this may take a minute)..."):
    processor, model, device = load_model()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    # Image upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Preview uploaded image
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    # Question input section
    st.subheader("Ask a Question")
    question = st.text_input(
        "Enter your question about the image:",
        placeholder="What is in this image?",
        label_visibility="collapsed"
    )
    
    # Display prediction
    if uploaded_file and question:
        with st.spinner("Thinking..."):
            answer = predict(image, question)
        st.subheader("Answer")
        st.info(answer)
    elif question:
        st.warning("⚠️ Please upload an image to ask a question")
