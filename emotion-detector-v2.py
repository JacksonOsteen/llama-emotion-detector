import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import difflib
from huggingface_hub import login

login(token="ADD A KEY")

model_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def detect_emotion(text):
    prompt = f"""You are an assistant that classifies text into these emotions:
joy, sadness, anger, fear, love, gratitude, or neutrality.

Respond with only one word, the emotion name.
Do NOT add letters, numbers, or punctuation before or after the word.
Do NOT output any options, lists, or explanations.

Examples:
"I got a new puppy and I'm so happy!" => joy
"I lost my job today." => sadness
"Why did you lie to me?!" => anger
Text:
{text}

Emotion:
""".strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=15,do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("raw model output:\n", response)

    if "Emotion:" in response:
        emotion_part = response.split("Emotion:")[-1].strip()
    else:
        emotion_part = response[len(prompt):].strip()

    words = emotion_part.split()

    valid_emotions = {"joy", "sadness", "anger", "fear", "love", "gratitude", "neutrality"}

    emotion = ""
    for word in words:
        w = word.lower().strip('.,:;!?"\'')
        if w in valid_emotions:
            emotion = w
            break

    if not emotion:
        for word in words:
            w = word.lower().strip('.,:;!?"\'')
            matches = difflib.get_close_matches(w, valid_emotions, n=1, cutoff=0.6)
            if matches:
                emotion = matches[0]
                break

    return emotion

iface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs="text",
    title="Emotion Detection Assistant",
    description="Paste text below and get the dominant emotion detected by the model."
)
if __name__ == "__main__":
    iface.launch()

