import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr

# Streamlit App Setup
st.set_page_config(page_title="Fast Speech/Text to Image Generator", layout="centered")
st.title("🎨 Fast Speech/Text to Image Generator (CPU Optimized)")
st.markdown("### ⚡ Powered by Stable Diffusion Turbo (Optimized for CPU)")
st.markdown("🧠 No GPU or API keys required — generates images in ~45–60 seconds.")

# -------------------- SPEECH RECOGNITION --------------------
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            st.info("🧠 Recognizing speech...")
            text = recognizer.recognize_google(audio)
            st.success(f"✅ Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"⚠️ Speech service error: {e}")
        return None

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/sd-turbo"  # Super-fast model
    device = "cpu"  # Force CPU use
    torch_dtype = torch.float32  # Safer on CPU

    st.info("🧩 Loading Stable Diffusion Turbo... (First time may take 1–2 mins)")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # Helps memory + speed
    st.success("✅ Model loaded! Ready to generate images.")
    return pipe

# -------------------- IMAGE GENERATION --------------------
def generate_image(prompt):
    pipe = load_pipeline()
    st.info(f"🎨 Generating image for: '{prompt}'")
    image = pipe(prompt, guidance_scale=1.0, num_inference_steps=4, height=512, width=512).images[0]
    return image

# -------------------- UI INPUTS --------------------
prompt_text = st.text_input("💬 Enter your text prompt:")

# Speech Input
if st.button("🎙️ Use Speech"):
    recognized_text = recognize_speech()
    if recognized_text:
        prompt_text = f"{recognized_text}, detailed, 4k, high quality"
        st.text_input("Recognized Prompt", value=prompt_text)
        with st.spinner("🎨 Generating image from speech..."):
            image = generate_image(prompt_text)
            st.image(image, caption="🖼️ Generated Image", use_column_width=True)
            image.save("generated_image.png")
            st.success("✅ Done!")

# Text Input
if st.button("⚡ Generate Image"):
    if prompt_text:
        with st.spinner("🎨 Generating image... please wait (~45–60 sec)"):
            image = generate_image(prompt_text)
            st.image(image, caption="🖼️ Generated Image", use_column_width=True)
            image.save("generated_image.png")
            st.success("✅ Done!")
    else:
        st.warning("⚠️ Please enter a prompt or use speech input.")
