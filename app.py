

# ===== STREAMLIT APPLICATION CODE =====
streamlit_app_code = '''
import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import time
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
from transformers import pipeline
import torch

# Configure page
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #2196F3;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}

class MedicalAIService:
    def __init__(self, hf_api_key=None, gemini_api_key=None):
        self.hf_api_key = hf_api_key
        self.gemini_api_key = gemini_api_key or "AIzaSyB5euCKgEEVRtkriAGxVmv2fUl5F2pODk0"
        self.medical_models = {
            "Clinical BERT": "emilyalsentzer/Bio_ClinicalBERT",
            "Medical T5": "google/flan-t5-small",
            "Bio GPT": "microsoft/biogpt",
            "General Medical": "microsoft/DialoGPT-medium"
        }
    
    def generate_medical_response(self, message, model_name="Medical T5", language="en"):
        """Generate medical response using Hugging Face or Gemini"""
        try:
            if self.hf_api_key and model_name in self.medical_models:
                return self._call_huggingface_model(message, self.medical_models[model_name])
            else:
                return self._call_gemini_model(message, language)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response. Please try again."
    
    def _call_huggingface_model(self, message, model_id):
        """Call Hugging Face model"""
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        prompt = f"answer medical question: {message}" if "t5" in model_id.lower() else f"Medical question: {message}\\n\\nAnswer:"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "").strip()
                return f"{text}\\n\\n⚠️ This is AI-generated medical information. Please consult with qualified healthcare professionals for proper medical advice."
        
        raise Exception(f"API call failed with status {response.status_code}")
    
    def _call_gemini_model(self, message, language="en"):
        """Call Gemini model as fallback"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        
        system_prompt = "You are a medical AI assistant. Provide helpful, accurate medical information while reminding users to consult healthcare professionals for serious concerns."
        if language == "ar":
            system_prompt += " Respond in Arabic."
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt} User message: {message}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        
        raise Exception(f"Gemini API call failed with status {response.status_code}")
    
    def analyze_image(self, image_bytes, message=""):
        """Analyze medical image using Gemini Vision"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            
            image_base64 = base64.b64encode(image_bytes).decode()
            
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": f"You are a medical AI assistant analyzing an image. Provide observations about what you see, but remind the user to consult healthcare professionals for diagnosis. {f'User question: {message}' if message else ''}"
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }]
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "I couldn't analyze this image. Please try again."
                
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def detect_language(self, text):
        """Simple language detection"""
        arabic_chars = sum(1 for char in text if '\\u0600' <= char <= '\\u06FF')
        return "ar" if arabic_chars > len(text) * 0.3 else "en"
    
    def calculate_confidence(self, response):
        """Calculate response confidence score"""
        # Simple heuristic based on response length and medical keywords
        medical_keywords = ["symptoms", "treatment", "diagnosis", "medication", "doctor", "patient", "medical", "health"]
        keyword_count = sum(1 for word in medical_keywords if word.lower() in response.lower())
        
        length_score = min(len(response) / 200, 1.0)
        keyword_score = min(keyword_count / 5, 1.0)
        
        return (length_score + keyword_score) / 2

def main():
    # Header
    st.markdown('<div class="main-header"><h1>🏥 Medical AI Assistant</h1><p>Advanced Medical Consultation with Multi-language Support</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys Section
        with st.expander("🔑 API Keys", expanded=True):
            hf_key = st.text_input("Hugging Face API Key", type="password", value=st.session_state.api_keys.get("hf", ""))
            if hf_key:
                st.session_state.api_keys["hf"] = hf_key
            
            st.info("💡 Get your free API key from https://huggingface.co/settings/tokens")
        
        # Model Selection
        st.subheader("🤖 Model Settings")
        selected_model = st.selectbox(
            "Choose AI Model",
            ["Medical T5", "Clinical BERT", "Bio GPT", "General Medical", "Gemini (Fallback)"]
        )
        
        # Language Selection
        language = st.selectbox("🌍 Language", ["English", "Arabic", "Auto-detect"])
        
        # Medical Specialization
        specialization = st.selectbox(
            "🩺 Medical Specialization",
            ["General Medicine", "Cardiology", "Neurology", "Dermatology", "Pediatrics", "Psychiatry", "Orthopedics"]
        )
        
        # Statistics
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            avg_conf = 0.75 if st.session_state.messages else 0
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
    
    # Initialize AI Service
    ai_service = MedicalAIService(
        hf_api_key=st.session_state.api_keys.get("hf"),
        gemini_api_key="AIzaSyB5euCKgEEVRtkriAGxVmv2fUl5F2pODk0"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🖼️ Image Analysis", "📊 Analytics", "🔧 Fine-tuning"])
    
    with tab1:
        # Chat Interface
        st.subheader("Medical Consultation Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    confidence = message.get("confidence", 0.5)
                    conf_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
                    st.markdown(f'<div class="chat-message assistant-message"><strong>AI Assistant:</strong> {message["content"]}<br><small class="{conf_class}">Confidence: {confidence:.2f}</small></div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_area("Ask a medical question...", height=100, key="chat_input")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Send Message", type="primary"):
                if user_input.strip():
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
                    
                    # Generate AI response
                    with st.spinner("AI is thinking..."):
                        lang_code = "ar" if language == "Arabic" else "en" if language == "English" else ai_service.detect_language(user_input)
                        response = ai_service.generate_medical_response(user_input, selected_model, lang_code)
                        confidence = ai_service.calculate_confidence(response)
                    
                    # Add AI response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "confidence": confidence,
                        "model": selected_model,
                        "specialization": specialization,
                        "timestamp": datetime.now()
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    with tab2:
        # Image Analysis
        st.subheader("🖼️ Medical Image Analysis")
        
        uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])
        image_question = st.text_input("Ask a question about the image (optional)")
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    image_bytes = uploaded_file.getvalue()
                    analysis = ai_service.analyze_image(image_bytes, image_question)
                
                st.subheader("Analysis Results:")
                st.write(analysis)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"[Image uploaded] {image_question if image_question else 'Please analyze this medical image'}",
                    "timestamp": datetime.now()
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": analysis,
                    "confidence": 0.8,
                    "model": "Gemini Vision",
                    "timestamp": datetime.now()
                })
    
    with tab3:
        # Analytics Dashboard
        st.subheader("📊 Conversation Analytics")
        
        if st.session_state.messages:
            # Confidence trend
            assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
            if assistant_messages:
                confidences = [msg.get("confidence", 0.5) for msg in assistant_messages]
                timestamps = [msg.get("timestamp", datetime.now()) for msg in assistant_messages]
                
                df = pd.DataFrame({
                    "Time": timestamps,
                    "Confidence": confidences
                })
                
                fig = px.line(df, x="Time", y="Confidence", title="AI Confidence Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Model usage
                models_used = [msg.get("model", "Unknown") for msg in assistant_messages]
                model_counts = pd.Series(models_used).value_counts()
                
                fig2 = px.pie(values=model_counts.values, names=model_counts.index, title="Model Usage Distribution")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Conversation export
                if st.button("📥 Export Conversation"):
                    conversation_data = {
                        "conversation_id": st.session_state.conversation_id,
                        "messages": st.session_state.messages,
                        "summary": {
                            "total_messages": len(st.session_state.messages),
                            "avg_confidence": sum(confidences) / len(confidences),
                            "models_used": list(model_counts.index)
                        }
                    }
                    
                    st.download_button(
                        label="Download Conversation JSON",
                        data=json.dumps(conversation_data, indent=2, default=str),
                        file_name=f"medical_chat_{st.session_state.conversation_id}.json",
                        mime="application/json"
                    )
        else:
            st.info("Start a conversation to see analytics!")
    
    with tab4:
        # Fine-tuning Interface
        st.subheader("🔧 Model Fine-tuning")
        
        st.info("🚀 Create custom medical AI models with your own data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            base_model = st.selectbox("Base Model", ["microsoft/DialoGPT-medium", "microsoft/biogpt", "google/flan-t5-small"])
            epochs = st.slider("Training Epochs", 1, 10, 3)
            batch_size = st.slider("Batch Size", 1, 8, 2)
            learning_rate = st.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4])
        
        with col2:
            st.subheader("Dataset")
            dataset_source = st.selectbox("Data Source", ["Upload CSV", "Hugging Face Dataset", "Custom URL"])
            
            if dataset_source == "Upload CSV":
                uploaded_dataset = st.file_uploader("Upload training data (CSV)", type=["csv"])
                if uploaded_dataset:
                    df = pd.read_csv(uploaded_dataset)
                    st.write("Data Preview:")
                    st.dataframe(df.head())
        
        if st.button("🚀 Generate Kaggle Notebook", type="primary"):
            # Generate Kaggle training code
            kaggle_code = f'''
# Medical AI Model Fine-tuning on Kaggle
# Generated by Medical AI Platform

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import json
import pandas as pd
from datasets import Dataset

# Configuration
MODEL_NAME = "{base_model}"
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
LEARNING_RATE = {learning_rate}

# Load and prepare data
def load_medical_data():
    # Load your dataset here
    # Example: df = pd.read_csv("/kaggle/input/medical-conversations/data.csv")
    conversations = []
    # Process your data
    return Dataset.from_pandas(pd.DataFrame(conversations))

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-medical-model",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,
    logging_steps=100,
)

# Load dataset
dataset = load_medical_data()

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train the model
print("Starting fine-tuning...")
trainer.train()

# Save the model
trainer.save_model("./fine-tuned-medical-model")
tokenizer.save_pretrained("./fine-tuned-medical-model")

print("Fine-tuning completed!")
'''
            
            st.download_button(
                label="📄 Download Kaggle Notebook",
                data=kaggle_code,
                file_name="medical_ai_fine_tuning.py",
                mime="text/python"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>🏥 Medical AI Assistant v1.0 | Built with Streamlit & Hugging Face</p>
            <p>⚠️ This AI assistant is for informational purposes only. Always consult qualified healthcare professionals for medical advice.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
'''

# Write the Streamlit app to a file
with open("medical_ai_chatbot.py", "w", encoding="utf-8") as f:
    f.write(streamlit_app_code)

print("📝 Streamlit app created: medical_ai_chatbot.py")

# ===== LAUNCH STREAMLIT APP =====
def launch_app():
    """Launch the Streamlit app with ngrok tunnel"""
    import subprocess
    import threading
    from pyngrok import ngrok
    
    print("🚀 Starting Streamlit app...")
    
    # Start Streamlit in background
    def run_streamlit():
        subprocess.run([sys.executable, "-m", "streamlit", "run", "medical_ai_chatbot.py", "--server.port", "8501"])
    
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    import time
    time.sleep(10)
    
    # Create ngrok tunnel
    try:
        public_url = ngrok.connect(8501)
        print(f"\\n🌐 Your Medical AI Chatbot is running at: {public_url}")
        print("\\n📱 Click the link above to access your app!")
        print("\\n⚠️  Keep this cell running to maintain the connection")
        
        # Keep the tunnel alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down...")
            ngrok.disconnect(public_url)
    except Exception as e:
        print(f"❌ Error creating tunnel: {e}")
        print("📱 Try accessing directly at: http://localhost:8501")

# Launch the app
launch_app()
