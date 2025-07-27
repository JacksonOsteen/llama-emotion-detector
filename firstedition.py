import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import difflib
from huggingface_hub import login

login("***REMOVED***")

model_id = "NousResearch/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

import torch.nn.functional as F

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True
    )

    generated_ids = outputs.sequences[0]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("raw model output:\n", response)

    # Get last token score
    token_scores = outputs.scores
    if token_scores:
        last_token_logits = token_scores[-1][0]
        probs = F.softmax(last_token_logits, dim=-1)
        confidence = probs.max().item()
    else:
        confidence = 0.0

    # Extract emotion
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
        emotion = ""

    return emotion, round(confidence * 100, 2)



iface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs=[
        gr.Textbox(label="Detected Emotion"),
        gr.Textbox(label="Confidence (%)")
    ],
    title="Emotion Detection Assistant",
    description="Paste text below and get the detected emotion with a confidence score."
)
if __name__ == "__main__":
    iface.launch()
