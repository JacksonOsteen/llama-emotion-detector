# Emotion Detection Assistant

An interactive web app that uses a LLaMA 7B language model to classify emotions from user-input text.

## Features
### v1
- Detects emotions such as joy, sadness, anger, fear, love, gratitude, and neutrality
- Displays a confidence percentage for the predicted emotion
- Built with Hugging Face Transformers and Gradio for an easy-to-use web interface
### v2
- better detection
- excludes the percentage correct estimate
### v3
- utilizes a different GUI based around paragraph by paragraph emotion classification
- mood detection is done on a paragraph by paragraph basis
- main emotion detection feature that tallys all of the paraghraphs and detects the main emotion from the entire text

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
2. install required pip installs
   ```bash
   pip install gradio transformers torch huggingface_hub difflib3 python-dotenv
3. Generate HuggyFace Access token and replace the following with your own token
   ```bash
   login(token="HF KEY")
