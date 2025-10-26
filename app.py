import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
N_MELS = 64
MAX_DURATION = 15.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)

class ChaLearnWavDataset(Dataset):
    def __init__(self, annotations_csv, audio_dir, sample_rate=SAMPLE_RATE,
                 n_mels=N_MELS, max_len=MAX_LEN, trait_list=None):
        self.df = pd.read_csv(annotations_csv)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len
        self.traits = trait_list or ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filename):
        path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.size(1) > self.max_len:
            wav = wav[:, :self.max_len]
        elif wav.size(1) < self.max_len:
            pad = self.max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["video_id"])
        # Entferne .mp4, falls vorhanden, bevor .wav angehängt wird
        if filename.endswith(".mp4"):
            filename = filename.replace(".mp4", "")
        filename += ".wav"
        wav = self._load_audio(filename)
        mel = self.melspec(wav)
        mel_db = self.db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        label = torch.tensor([float(row[t]) for t in self.traits], dtype=torch.float32)
        return mel_db.squeeze(0), label
    
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_mels=64, num_traits=5):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Nur 1 FC-Schicht
        self.fc = nn.Linear(64, num_traits)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, n_mels, time)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === 1️⃣ Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioCNN()
checkpoint = torch.load("best_audio_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# === 2️⃣ Audio vorbereiten ===
def preprocess_audio(filepath, sample_rate=16000, n_mels=64, max_duration=15.0):
    wav, sr = torchaudio.load(filepath)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # Mono
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    max_len = int(sample_rate * max_duration)
    if wav.size(1) > max_len:
        wav = wav[:, :max_len]
    elif wav.size(1) < max_len:
        pad = max_len - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad))

    # Mel-Spektrogramm
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB()(melspec)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.unsqueeze(0)  # Batch dimension

# === 3️⃣ Prediction ===
def predict_audio(filepath, model, sample_rate=16000, max_duration=15.0):
    max_len = int(sample_rate * max_duration)
    
    # 1️⃣ Audio laden
    wav, sr = torchaudio.load(filepath)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    # 2️⃣ MelSpectrogram
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=64
    )
    db = torchaudio.transforms.AmplitudeToDB()

    # 3️⃣ Sliding Window oder Padding
    if wav.size(1) <= max_len:
        # kurz → padding
        pad = max_len - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad))
        mel = melspec(wav)
        mel_db = db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_db = mel_db.unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = model(mel_db)
    else:
        # lang → Sliding Window
        window_size = max_len
        hop_size = max_len // 2  # 50% overlap
        preds = []
        for start in range(0, wav.size(1) - window_size + 1, hop_size):
            segment = wav[:, start:start + window_size]
            mel = melspec(segment)
            mel_db = db(mel)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            mel_db = mel_db.unsqueeze(0).to(device)
            with torch.inference_mode():
                pred_seg = model(mel_db)
                preds.append(pred_seg)
        pred = torch.mean(torch.stack(preds), dim=0)

    return pred.squeeze().cpu()  # Tensor mit Trait-Werten

import random

traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

meanings = {
    "Openness": {
        "very low": [
            "wirkt sehr bodenständig und bevorzugt vertraute Routinen.",
            "zeigt starke Vorliebe für Bewährtes und bleibt gerne in bekannten Mustern.",
            "ist traditionsbewusst und probiert selten neue Dinge aus."
        ],
        "low": [
            "wirkt bodenständig, traditionsbewusst.",
            "zeigt nur geringe Neugier und hält an Gewohntem fest.",
            "bevorzugt pragmatische und bekannte Lösungen."
        ],
        "mid": [
            "zeigt ein ausgewogenes Maß an Neugier und Kreativität.",
            "ist offen für Neues, aber auch vorsichtig bei unbekannten Erfahrungen.",
            "ist neugierig, probiert Neues aus, ohne Risiken zu übertreiben."
        ],
        "high": [
            "wirkt offen, fantasievoll und interessiert an neuen Erfahrungen.",
            "zeigt viel Kreativität und Neugier für Unbekanntes.",
            "ist aufgeschlossen, experimentierfreudig und phantasievoll."
        ],
        "very high": [
            "wirkt extrem kreativ, neugierig und aufgeschlossen für alles Neue.",
            "ist ausgesprochen innovativ, originell und immer neugierig.",
            "zeigt eine starke Vorliebe für ungewöhnliche Ideen und Experimente."
        ]
    },
    "Conscientiousness": {
        "very low": [
            "handelt sehr spontan, wenig organisiert.",
            "zeigt kaum Selbstdisziplin und plant selten voraus.",
            "ist impulsiv und strukturiert seine Aufgaben kaum."
        ],
        "low": [
            "neigt dazu, spontan zu handeln und weniger strukturiert zu sein.",
            "ist nicht sehr organisiert, übernimmt Aufgaben eher locker.",
            "zeigt nur geringe Selbstdisziplin und wenig Planung."
        ],
        "mid": [
            "hat eine solide Selbstdisziplin, ohne übermäßig perfektionistisch zu sein.",
            "zeigt ein ausgewogenes Maß an Organisation und Zielstrebigkeit.",
            "ist strukturiert, ohne zu streng mit sich selbst zu sein."
        ],
        "high": [
            "wirkt sehr organisiert, verantwortungsbewusst und zielstrebig.",
            "zeigt starke Selbstdisziplin und eine klare Struktur in Aufgaben.",
            "ist verlässlich, plant sorgfältig und hält sich an Regeln."
        ],
        "very high": [
            "wirkt extrem strukturiert, perfektionistisch und sehr zuverlässig.",
            "zeigt höchste Selbstdisziplin, Genauigkeit und Organisation.",
            "ist außerordentlich verlässlich, detailorientiert und methodisch."
        ]
    },
    "Extraversion": {
        "very low": [
            "wirkt sehr ruhig, zurückhaltend und introvertiert.",
            "zeigt starke Zurückhaltung in sozialen Situationen.",
            "ist eher still, beobachtend und wenig kontaktfreudig."
        ],
        "low": [
            "wirkt ruhig, zurückhaltend und eher introvertiert.",
            "zeigt zurückhaltendes Sozialverhalten, spricht wenig in Gruppen.",
            "ist zurückhaltend, tritt nicht stark in den Vordergrund."
        ],
        "mid": [
            "zeigt ein ausgeglichenes Sozialverhalten zwischen Ruhe und Aktivität.",
            "ist weder sehr extrovertiert noch stark introvertiert.",
            "ist sozial ausgeglichen und situationsabhängig kontaktfreudig."
        ],
        "high": [
            "wirkt gesellig, energiegeladen und kontaktfreudig.",
            "zeigt aktives und lebendiges Sozialverhalten.",
            "ist kommunikativ, freundlich und aufgeschlossen."
        ],
        "very high": [
            "wirkt extrem extrovertiert, dominant und sehr kontaktfreudig.",
            "zeigt starke Energie, Begeisterung und soziale Präsenz.",
            "ist sehr aktiv, charismatisch und zieht leicht Aufmerksamkeit auf sich."
        ]
    },
    "Agreeableness": {
        "very low": [
            "kann sehr direkt und wettbewerbsorientiert auftreten.",
            "zeigt wenig Rücksicht auf andere, eher egozentrisch.",
            "ist kritisch, durchsetzungsstark und weniger kooperativ."
        ],
        "low": [
            "kann direkt und wettbewerbsorientiert auftreten.",
            "zeigt nur mäßige Kooperationsbereitschaft.",
            "ist ehrlich, manchmal ungeduldig oder direkt im Umgang."
        ],
        "mid": [
            "zeigt ein gesundes Gleichgewicht zwischen Kooperation und Eigeninteresse.",
            "ist fair, kann sich durchsetzen, ohne rücksichtslos zu sein.",
            "kooperativ, aber auch in eigenen Interessen bedacht."
        ],
        "high": [
            "wirkt freundlich, hilfsbereit und empathisch.",
            "zeigt gute Zusammenarbeit und Mitgefühl.",
            "ist aufmerksam, verständnisvoll und hilfsbereit."
        ],
        "very high": [
            "wirkt extrem freundlich, sehr empathisch und kompromissbereit.",
            "zeigt außergewöhnliche Rücksichtnahme, Geduld und Freundlichkeit.",
            "ist stark altruistisch, hilfsbereit und sehr kooperativ."
        ]
    },
    "Neuroticism": {
        "very low": [
            "wirkt extrem emotional stabil, ruhig und gelassen.",
            "zeigt kaum Stressreaktionen und bleibt ausgeglichen.",
            "ist entspannt, ruhig und wenig anfällig für Ängste."
        ],
        "low": [
            "wirkt emotional stabil, ruhig und ausgeglichen.",
            "zeigt nur geringe emotionale Schwankungen.",
            "ist insgesamt gelassen und stressresistent."
        ],
        "mid": [
            "zeigt gelegentlich emotionale Schwankungen, bleibt aber kontrolliert.",
            "ist moderat sensibel, kann Emotionen gut regulieren.",
            "zeigt normale Reaktionen auf Stress und emotionale Situationen."
        ],
        "high": [
            "wirkt sensibel, selbstkritisch und emotional reaktiver.",
            "zeigt häufig emotionale Reaktionen und Unsicherheiten.",
            "ist anfälliger für Stress, neigt zu Sorgen und Selbstkritik."
        ],
        "very high": [
            "wirkt extrem emotional, leicht gestresst und sehr sensibel.",
            "zeigt starke emotionale Reaktionen, kann leicht überfordert sein.",
            "ist hochsensibel, schnell gestresst und sehr selbstkritisch."
        ]
    }
}

summary_phrases = {
    "Openness": {
        "very low": ["sehr bodenständig", "traditionsbewusst", "wenig neugierig"],
        "low": ["bodenständig", "praktisch orientiert", "leicht neugierig"],
        "mid": ["aufgeschlossen", "ausgewogen neugierig", "interessiert an Neuem"],
        "high": ["kreativ und neugierig", "fantasievoll", "experimentierfreudig"],
        "very high": ["extrem kreativ und neugierig", "innovativ", "stark experimentierfreudig"]
    },
    "Conscientiousness": {
        "very low": ["sehr spontan", "wenig organisiert", "impulsiv"],
        "low": ["spontan", "locker", "wenig diszipliniert"],
        "mid": ["verlässlich", "ausgewogen organisiert", "diszipliniert"],
        "high": ["zielstrebig und organisiert", "verantwortungsbewusst", "strukturiert"],
        "very high": ["extrem organisiert und zuverlässig", "perfektionistisch", "außerordentlich diszipliniert"]
    },
    "Extraversion": {
        "very low": ["sehr introvertiert", "ruhig", "zurückhaltend"],
        "low": ["introvertiert", "ruhig", "etwas zurückhaltend"],
        "mid": ["ausgeglichen sozial", "situationsabhängig kontaktfreudig", "neutral extrovertiert"],
        "high": ["extrovertiert und kontaktfreudig", "gesellig", "energiegeladen"],
        "very high": ["extrem extrovertiert", "dominant", "charismatisch"]
    },
    "Agreeableness": {
        "very low": ["sehr direkt", "wenig kooperativ", "kritisch"],
        "low": ["direkt", "wenig kompromissbereit", "ehrlich"],
        "mid": ["kooperativ", "fair", "ausgewogen"],
        "high": ["freundlich und hilfsbereit", "mitfühlend", "aufmerksam"],
        "very high": ["extrem freundlich und empathisch", "altruistisch", "sehr kooperativ"]
    },
    "Neuroticism": {
        "very low": ["sehr emotional stabil", "gelassen", "ruhig"],
        "low": ["emotional stabil", "wenig stressanfällig", "ausgeglichen"],
        "mid": ["sensibel, aber kontrolliert", "moderate emotionale Reaktionen", "normal sensibel"],
        "high": ["emotional und sensibel", "selbstkritisch", "reaktiver"],
        "very high": ["extrem emotional und sensibel", "hochsensibel", "leicht gestresst"]
    }
}

# Funktionen zum zufälligen Auswählen
def get_random_meaning(trait, level):
    return random.choice(meanings[trait][level])

def get_random_summary(trait, level):
    return random.choice(summary_phrases[trait][level])

def interpret_value(trait, value):
    if value < 0.2:
        level = "very low"
    elif value < 0.4:
        level = "low"
    elif value < 0.6:
        level = "mid"
    elif value < 0.8:
        level = "high"
    else:
        level = "very high"

    # Zufällige Variante auswählen
    meaning = get_random_meaning(trait, level)
    short = get_random_summary(trait, level)
    return meaning, short

def interpret_prediction(pred):
    print("🧠 Persönlichkeitsprofil (Big Five Analyse)\n")

    summary = []
    for trait, val in zip(traits, pred):
        meaning, short = interpret_value(trait, val.item())
        summary.append(short)
        print(f"{trait:17s}: {val:.3f} → {meaning}")

    # Automatische Zusammenfassung
    print("\n📋 Zusammenfassung:")
    print(f"Die Person wirkt {', '.join(summary[:-1])} und {summary[-1]}.")

import gradio as gr

def gradio_predict(file):
    pred = predict_audio(file, model)  # file ist jetzt der Pfad
    summary_texts = []
    for trait, val in zip(traits, pred):
        meaning, short = interpret_value(trait, val.item())
        summary_texts.append(f"{trait}: {val:.2f} → {meaning}")
    summary_text = "\n".join(summary_texts)
    
    # Optionale Zusammenfassung
    summary_text += "\n\n📋 Zusammenfassung: Die Person wirkt " + \
        ", ".join([get_random_summary(trait, "mid") for trait in traits[:-1]]) + \
        f" und {get_random_summary(traits[-1], 'mid')}."
    
    return summary_text

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(label="Charakteranalyse", lines=20, max_lines=30),
    title="🎙️KI-Persönlichkeitsanalyse aus Audio",
    description="Lade eine Audioaufnahme hoch, die ungefähr 15s dauert und das Modell sagt die Big Five Persönlichkeitsmerkmale voraus."
)

iface.launch()