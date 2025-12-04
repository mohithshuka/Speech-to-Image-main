🎨 Speech/Text to Image Generator

An AI-powered web application that converts typed text or spoken speech into realistic, high-quality images using Stable Diffusion.
Built using Streamlit, PyTorch, and Speech Recognition — and requires no API keys.

🚀 Features

🎤 Speech to Text conversion using microphone input

📝 Text-based prompt option for image generation

🧠 Generates AI images using Stable Diffusion v1.4

⚡ Automatically detects and runs on GPU (CUDA) if available

💾 Saves generated images locally (generated_image.png)

🖥️ Easy-to-use interface powered by Streamlit

🛠️ Tech Stack
Component	Technology Used
Frontend UI	Streamlit
Backend	Python
Image Generation	Stable Diffusion (Diffusers)
Deep Learning	PyTorch
Voice Recognition	SpeechRecognition
Image Processing	PIL (Python Imaging Library)
📦 Installation
1️⃣ Download
download the zip and open in vs code
2️⃣ Install Dependencies
pip install -r requirements.txt


OR manually install:

pip install streamlit torch torchvision torchaudio diffusers transformers accelerate safetensors pillow SpeechRecognition


🔹 GPU users should install the correct CUDA version for PyTorch from pytorch.org

▶️ How to Run
streamlit run app.py


Then open the URL shown in the terminal, usually:

http://localhost:8501

Want to see the deploye code just click on this
https://speech-to-image-main-yhtz7uwten7ru7twx9e8wb.streamlit.app

🧪 How It Works

Enter a prompt or click Recognize Speech to speak

The model processes the input prompt

Stable Diffusion generates a realistic image

The output image is displayed & saved automatically

📌 Example Prompts
Prompt	Result
“A cute dog astronaut walking on Mars”	🐶🚀 Dog in space suit
“A futuristic cyberpunk city at night”	🌆 Neon sci-fi city
📂 Project Structure
├── app.py                 # Main Streamlit application
├── generated_image.png    # Output image (auto created)
├── README.md              # Project documentation
└── requirements.txt       # Required Python libraries

📈 Results & Analysis

Smooth text input and speech recognition performance

High-quality image generation using Stable Diffusion

Faster on GPU, slower but functional on CPU

💡 Future Enhancements

Download button for generated images

Support for multiple images at a time

Add prompt history and gallery view

Upgrade to SDXL for ultra-high resolution

🤝 Contributions

Contributions are welcome!
Feel free to fork, create a branch, and submit a pull request.

📜 License

This project is released under the MIT License.
