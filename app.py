import gradio as gr
from transformers import pipeline
from audiocraft.models import MusicGen
import torch
from audiocraft.data.audio import audio_write

# Load models
emotion_classifier = pipeline("text-classification", 
                              model="bhadresh-savani/distilbert-base-uncased-emotion", 
                              return_all_scores=False)

musicgen = MusicGen.get_pretrained('facebook/musicgen-small')
musicgen.set_generation_params(duration=10)

def generate_music(scene_text):
    emotion = emotion_classifier(scene_text)[0]['label']
    prompt = f"A {emotion} soundtrack for a film scene"
    
    wav = musicgen.generate([prompt])
    filepath = "emotion_music.wav"
    audio_write("emotion_music", wav[0].cpu(), musicgen.sample_rate, format="wav")

    return f"Detected Emotion: **{emotion}**", filepath

iface = gr.Interface(
    fn=generate_music,
    inputs=gr.Textbox(label="Enter scene description"),
    outputs=[gr.Markdown(), gr.Audio(label="Generated Music")],
    title="🎬 Scene-to-Music Generator",
    description="Describe a film scene and get AI-generated background music that fits the emotion."
)

iface.launch()
