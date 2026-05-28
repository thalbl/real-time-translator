# Real-Time AI Voice Translator

A comprehensive end-to-end Machine Learning pipeline designed to capture live audio, transcribe it, translate it into a target language, and synthesize it back into spoken audio. Developed for Google Colab environments, this system implements a dual-architecture approach, allowing users to seamlessly switch between **Local Deep Learning Models** running entirely on the GPU and **Cloud-Based APIs**.

Beyond the core application, the project features an interactive GUI, rigorous performance benchmarking, and applied machine learning experiments (supervised classification and unsupervised clustering) using audio features extracted from the FLEURS dataset.

---

## Project Objectives

1. **End-to-End Voice Translation:** Create a cohesive pipeline combining Speech-to-Text (STT), Machine Translation, and Text-to-Speech (TTS).
2. **Architecture Comparison:** Benchmark processing time, accuracy, and privacy trade-offs of running large local models versus relying on external cloud APIs.
3. **ML Evaluation:** Quantify model performance using industry-standard metrics (WER for transcription, BLEU for translation).
4. **Feature Engineering & ML:** Extract audio features (MFCCs) to train standalone supervised and unsupervised models for language detection.

---

## System Architecture & Modules

The system is built as a modular backend pipeline where audio data flows sequentially through different transformation stages.

### 1. Speech-to-Text (STT) Module

Ingests raw audio data and transcribes it into text.

* **Browser Audio Integration:** Uses a custom JavaScript injector to leverage the browser's `MediaRecorder` API, capturing audio in Google Colab, encoding it as WebM, and converting it to 16kHz mono WAV via `ffmpeg`.
* **Local DL (Whisper):** Loads OpenAI's Whisper (`tiny`, `base`, `small`) into VRAM to extract features and generate token IDs. Evaluated using **Word Error Rate (WER)** via the `jiwer` library against the FLEURS PT-BR dataset.
* **Cloud API Fallback:** Uses the `speech_recognition` library to route the audio array to Google's free Speech-to-Text API for a lightweight alternative.

### 2. Translation Module

Handles the semantic transformation of the transcribed text from the source language to the target language.

* **Local DL (M2M100):** Utilizes Meta's M2M100 (`1.2B` & `418M`), a multilingual encoder-decoder model that translates directly between languages without pivoting through English. Evaluated using **BLEU scores** against human references, including a specific "Slang & Idioms Challenge".
* **Cloud API Fallback:** Integrates `deep-translator` to hook into Google Translate for rapid translations with zero VRAM overhead.

### 3. Text-to-Speech (TTS) Module

Converts the translated text back into an audible waveform.

* **Local DL (SpeechT5):** Uses Microsoft's SpeechT5 architecture. Text passes through a processor, applies a default speaker embedding, and a `HiFi-GAN` vocoder converts the generated spectrograms back into raw audio played directly in the notebook.
* **Cloud API Fallback:** Utilizes `gTTS` to send text to Google's TTS engine, saving and playing the output as a temporary MP3.

### 4. Interface & Benchmarks Module

Acts as the frontend and analytical engine.

* **Interactive UI:** Built with `ipywidgets`, providing a clean GUI within the notebook to orchestrate the flow of data (Record -> STT -> Translate -> TTS) using dedicated callback functions.
* **Benchmarking Engine:** Systematically times module execution side-by-side (`time.perf_counter()`). Outputs `matplotlib` bar charts comparing inference speed and overall pipeline latency to visually quantify the speed/privacy trade-offs.

### 5. MLOps & Machine Learning Pipeline Module

Manages infrastructure and standalone ML experiments.

* **Infrastructure:** Handles GPU detection, dependency installation, dataset loading (Google FLEURS), and rigorous memory management (`gc.collect()`, `torch.cuda.empty_cache()`) to prevent VRAM overflow.
* **Supervised Learning (Language Detection):** Extracts 26 audio features (mean and standard deviation of 13 MFCCs) using `librosa`. Trains **Logistic Regression** and **Random Forest** classifiers from scratch to predict the spoken language (PT, EN, NO) directly from raw audio.
* **Unsupervised Learning (Audio Clustering):** Applies **K-Means clustering** to the extracted MFCC features without providing language labels. Uses PCA for 2D visualization and Adjusted Rand Index (ARI) to evaluate if the audio naturally groups into distinct language clusters.

---

## Setup and Installation

This project is highly optimized for Google Colab environments.

1. **Hardware Preparation:** * Open Google Colab.
* Navigate to `Runtime` -> `Change runtime type` and select **T4 GPU**.


2. **Install Dependencies:**
```bash
pip install torch transformers sentencepiece scipy numpy
pip install SpeechRecognition deep-translator gTTS pydub
pip install datasets==2.20.0 transformers==4.44.0 jiwer librosa

```


3. **Execution Order:**
* Run the **MLOps / Setup** cells first to establish dependencies, clear memory, and load the heavy Deep Learning models into VRAM.
* Execute the **STT, Translation, and TTS** module definitions.
* Launch the **Interface** cells to interact with the GUI.
* Run the **Benchmark and ML Evaluation** cells at the end to generate analytical charts.



---

## Ethical Reflection & Technical Limitations

* **Data Privacy:** Voice is biometric data. The Cloud API version poses a privacy risk by transmitting user data to external servers. The Local Deep Learning pipeline solves this by ensuring all data processing remains strictly on the host machine.
* **Algorithmic Bias:** Models like Whisper and M2M100 are heavily skewed toward high-resource languages (like English). Translations and transcriptions for under-resourced languages (e.g., Norwegian) may suffer from lower accuracy or introduce cultural biases.
* **Hardware Bottlenecks:** The local pipeline requires substantial VRAM (~10GB+ depending on Whisper and M2M100 model sizes). Reaching true "real-time" latency on edge devices would require advanced techniques like model quantization or distillation.
