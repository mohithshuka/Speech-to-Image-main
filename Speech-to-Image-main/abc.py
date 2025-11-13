import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr

# Streamlit app title and info
st.title("Speech/Text to Image Generator")
st.markdown("### Powered by Stable Diffusion & Speech Recognition")
st.markdown("⚠️ Please be patient — image generation may take some time (especially on CPU).No API keys are used")

# Function: recognize speech using microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(" Listening for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            st.info(" Recognizing speech...")
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error(" Could not understand audio.")
        except sr.RequestError as e:
            st.error(f" Could not request results; {e}")
        return None

# Function: load the Stable Diffusion pipeline
@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"

    # Automatically detect if GPU (CUDA) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    )

    pipe = pipe.to(device)
    st.write(f"Using device: {device.upper()}")
    return pipe, device

# Function: generate image from text
def generate_image(prompt):
    pipe, device = load_pipeline()

    # Use autocast only if GPU is available
    if device == "cuda":
        with torch.autocast("cuda"):
            image = pipe(prompt, guidance_scale=8.5).images[0]
    else:
        image = pipe(prompt, guidance_scale=8.5).images[0]

    return image

# Text input for prompt
prompt_text = st.text_input(" Enter a prompt for the image generation:")

# Button: Speech recognition
if st.button("Recognize Speech"):
    recognized_text = recognize_speech()
    if recognized_text:
        prompt_text = f"{recognized_text}, 4k, high resolution"
        st.text_input("Recognized Prompt", value=prompt_text)
        with st.spinner(" Generating image..."):
            image = generate_image(prompt_text)
            st.image(image, caption=" Generated Image", use_column_width=True)
            image.save("generated_image.png")
            st.success(" Image generated successfully!")

# Button: Text prompt generation
if st.button(" Generate Image"):
    if prompt_text:
        with st.spinner(" Generating image..."):
            image = generate_image(prompt_text)
            st.image(image, caption=" Generated Image", use_column_width=True)
            image.save("generated_image.png")
            st.success(" Image generated successfully!")
    else:
        st.warning("Please enter a text prompt or use speech recognition first.")
