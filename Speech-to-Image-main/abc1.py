import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr

# App Title
st.title(" Speech / Text to Image Generator")
st.markdown("Image Generator ")
st.caption("Created by **Mohith Shuka")

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

# Load the Stable Diffusion model
@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    ).to(device)

    return pipe

def generate_image(prompt):
    pipe = load_pipeline()
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

# 1️⃣ Microphone speech recognition (LOCAL USE)
def recognize_by_microphone():
    try:
        with sr.Microphone() as source:
            st.info("🎤 Say something...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)
        st.success(f"🗣 Recognized Speech: {text}")
        return text
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        return None

# 2️⃣ Audio Upload Recognition (Cloud Safe)
def recognize_by_file():
    uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if uploaded_audio:
        try:
            with sr.AudioFile(uploaded_audio) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                st.success(f"🗣 Recognized Speech: {text}")
                return text
        except Exception as e:
            st.error(f"Error processing audio: {e}")
    return None


# ----------------------------------------------
# User UI Section
# ----------------------------------------------
prompt_text = st.text_input("✍ Enter text prompt:")

# Speech Recognition Options
st.subheader("🎤 Speech Input Options")
col1, col2 = st.columns(2)

with col1:
    if st.button("🎙 Speak Now (Mic)"):
        text = recognize_by_microphone()
        if text:
            prompt_text = text

with col2:
    if st.button("📤 Upload Audio File"):
        text = recognize_by_file()
        if text:
            prompt_text = text


# Generate Image Button
if st.button("🚀 Generate Image"):
    if prompt_text:
        st.write(f"📌 Prompt Used: **{prompt_text}**")
        with st.spinner("🎨 Creating image... please wait..."):
            img = generate_image(prompt_text)
            st.image(img, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a text prompt or use speech input first.")
