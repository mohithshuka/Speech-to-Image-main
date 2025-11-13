

import gradio as gr
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the model once when the script starts
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# Function to recognize speech
def recognize_speech(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None

# Function to generate image based on text using Stable Diffusion
def generate_image(text):
    outputs = pipe(text)
    image = outputs.images[0]  # First image from the batch
    return image

# Function to recognize speech and generate image for Gradio interface
def recognize_and_generate(audio_file_path):
    text = recognize_speech(audio_file_path)
    if text:
        image = generate_image(text)
        return text, image, audio_file_path
    else:
        return "No speech recognized", None, audio_file_path

# Gradio interface
def gradio_interface():
    iface = gr.Interface(
        fn=recognize_and_generate,
        inputs=gr.Audio(type="filepath", label="Record your voice"),
        outputs=[
            gr.Textbox(label="Recognized Text"),
            gr.Image(label="Generated Image"),
            gr.Audio(label="Recorded Audio")
        ],
        title="Acoustic Artistry - Voice to Image Generator",
        description="Record your voice, get recognized text, and generate an image."
    )
    return iface

if __name__ == "__main__":
    gradio_interface().launch()


# import streamlit as st


# st.title("Streamlit Test App")
# st.write("✅ Streamlit is installed and running correctly!")

# st.title("Hello, Streamlit!")
# st.write("This is my first Streamlit app 🚀")

# number = st.slider("Pick a number", 0, 100, 25)
# st.write("You selected:", number)
