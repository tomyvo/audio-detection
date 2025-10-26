# 🧠 Audio Personality Analyzer  
**AI model for detecting Big Five personality traits from voice recordings**

This project uses a neural network to **analyze personality traits from human speech**.  
The model estimates the **Big Five personality dimensions** from an audio sample and interprets them in natural language.

You can find the WebApp here: https://www.huggingface.co/spaces/tomyvo/soner-audio

---

## 🚀 Features
- 🎧 Voice-based personality analysis through a clean Gradio UI  
- 🔍 Prediction of the **Big Five** personality traits:
  - **Openness** – openness to experience  
  - **Conscientiousness** – self-discipline and organization  
  - **Extraversion** – social energy and expressiveness  
  - **Agreeableness** – empathy and cooperation  
  - **Neuroticism** – emotional stability  
- 🗣️ Natural language interpretations and summarized results  
- 💬 Supports multiple audio formats (`.wav`, `.mp3`, etc.)  
- ⚡ GPU-accelerated inference using PyTorch  

---

## 📊 Example Output

Openness          : 0.78 → appears imaginative and open to new experiences.
Conscientiousness : 0.66 → organized and responsible.
Extraversion      : 0.35 → calm and reserved.
Agreeableness     : 0.81 → kind, cooperative, and empathetic.
Neuroticism       : 0.42 → emotionally balanced and stable.

📋 Summary:
This person appears creative, structured, socially balanced, friendly, and emotionally stable.

---

## 🧩 Model Overview
### 1. Neural Network
The model is a **Convolutional Neural Network (CNN)** trained on Mel-spectrograms extracted from speech samples.  
It loads from a pretrained checkpoint file (`best_audio_model.pth`) via:

```python
model = AudioCNN()
checkpoint = torch.load("best_audio_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()


