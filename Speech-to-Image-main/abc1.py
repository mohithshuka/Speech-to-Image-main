import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr

# Streamlit UI
st.title("Speech/Text to Image Generator")
st.markdown("Image Generator")
st.caption("Created by Mohith Shuka")

# Speech recognition using audio upload
def recognize_speech():
    st.info("Upload an audio file to convert speech to text")
    uploaded_audio = st.file_uploader("Upload audio file (wav/mp3)", type=["wav", "mp3"])

    if uploaded_audio is not None:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(uploaded_audio) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                st.success(f"Recognized Text: {text}")
                return text
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")
    return None

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


prompt_text = st.text_input("Enter prompt for image generation:")


# Speech Button
if st.button("Recognize Speech"):
    text = recognize_speech()
    if text:
        prompt_text = text + ", high quality"
        st.write(f"Prompt: {prompt_text}")
        with st.spinner("Generating image..."):
            img = generate_image(prompt_text)
            st.image(img, caption="Generated Image", use_column_width=True)


# Generate Image Button
if st.button("Generate Image"):
    if prompt_text:
        with st.spinner("Generating image..."):
            img = generate_image(prompt_text)
            st.image(img, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a text prompt or upload audio first.")
