# 🌐 Real-Time AI Translator

An academic project for a **real-time translator** that captures voice, transcribes, translates, and synthesizes speech in the target language — all using open-source **Artificial Intelligence** models.

## 🎯 Features

| Stage | Description | Model / Technology |
|-------|-------------|---------------------|
| 🎤 **Speech-to-Text (STT)** | Captures audio from the microphone and converts it to text | OpenAI Whisper (Hugging Face) |
| 🌍 **Translation** | Translates text from one language to another | MarianMT / Helsinki-NLP |
| 🔊 **Text-to-Speech (TTS)** | Synthesizes translated text into speech | Microsoft SpeechT5 |
| 🖥️ **Graphical Interface** | Unified screen to operate the entire pipeline | CustomTkinter |

## 📂 Project Structure

```
real-time-translator/
│
├── modules/                  # AI Modules
│   ├── __init__.py           # Package exports
│   ├── stt_module.py         # Speech-to-Text (Person 1)
│   ├── tradutor_module.py    # Translation (Person 2)
│   └── tts_module.py         # Text-to-Speech (Person 3)
│
├── interface.py              # Graphical interface (Person 4)
├── requirements.txt          # Project dependencies
├── .gitignore                # Files ignored by Git
└── README.md                 # This file
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/real-time-translator.git
cd real-time-translator
```

### 2. Create and activate the virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **NVIDIA GPU (optional):** If you have an NVIDIA graphics card, install PyTorch with CUDA **before** running the command above. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and follow the instructions for your card.

### 4. Run the application

```bash
python interface.py
```

## 👥 Team

| Person | Responsibility | File |
|--------|---------------|------|
| Person 1 | Speech-to-Text (Whisper) | `modules/stt_module.py` |
| Person 2 | Translation (MarianMT) | `modules/tradutor_module.py` |
| Person 3 | Text-to-Speech (SpeechT5) | `modules/tts_module.py` |
| Person 4 | Graphical Interface | `interface.py` |
| Person 5 | MLOps / Configuration | Environment, Git, CI/CD |

## 🛠️ Technologies

- **Python 3.10+**
- **PyTorch** — Deep Learning Framework
- **Hugging Face Transformers** — Pre-trained NLP/NLU models
- **SoundDevice** — Audio capture and playback
- **CustomTkinter** — Modern graphical interface
