import gradio as gr
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import torch

# Load model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# Speech recognition
def recognize_speech(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text
    except:
        return None

# Image generation
def generate_image(text):
    try:
        outputs = pipe(text, height=512, width=512)
        return outputs.images[0]
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Combined function
def text_or_voice(text_input, audio_file_path):
    if text_input and text_input.strip() != "":
        prompt = text_input
    elif audio_file_path:
        recognized_text = recognize_speech(audio_file_path)
        if recognized_text:
            prompt = recognized_text
        else:
            return "Could not recognize speech", None
    else:
        return "Please provide text or voice input", None

    image = generate_image(prompt)
    return prompt, image

# Gradio interface
iface = gr.Interface(
    fn=text_or_voice,
    inputs=[
        gr.Textbox(label="Enter text (optional)"),
        gr.Audio(type="filepath", label="Or record your voice")
    ],
    outputs=[
        gr.Textbox(label="Prompt Used"),
        gr.Image(label="Generated Image")
    ],
    title="🎨 Voice or Text to Image Generator",
    description="You can either type a prompt or record your voice to generate an image."
)

if __name__ == "__main__":
    iface.launch()
