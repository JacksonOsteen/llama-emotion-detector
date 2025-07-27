import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import difflib
from huggingface_hub import login

login(token="REMOVED_TOKEN")

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

from collections import Counter

def analyze_text(full_text):
    paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
    paragraph_emotions = []
    all_emotions = []

    for p in paragraphs:
        emotion = detect_emotion(p)
        paragraph_emotions.append((p, emotion))
        all_emotions.append(emotion)

    dominant_emotion = Counter(all_emotions).most_common(1)[0][0] if all_emotions else "neutrality"
    return dominant_emotion, paragraph_emotions

with gr.Blocks(title="Email Emotion Classifier") as app:
    gr.Markdown("## 🧠 Email Emotion Detector\nEnter your full message on the right. See emotional breakdown on the left.")

    with gr.Row():
        with gr.Column(scale=1):
            paragraph_boxes = gr.Textbox(label="Paragraph Breakdown", lines=20, interactive=False)

        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Full Message Input",
                lines=20,
                placeholder="Paste your full email or message here...",
            )
            submit_btn = gr.Button("Analyze")

            dominant_output = gr.Textbox(label="Dominant Emotion", interactive=False)

    def on_submit(text):
        dominant, breakdown = analyze_text(text)
        formatted = "\n\n".join([f"Paragraph {i+1} ({emotion}):\n{p}" for i, (p, emotion) in enumerate(breakdown)])
        return dominant, formatted

    submit_btn.click(fn=on_submit, inputs=[input_box], outputs=[dominant_output, paragraph_boxes])

if __name__ == "__main__":
    app.launch()