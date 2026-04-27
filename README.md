# 🖼️ Multilingual Image Captioning with Text-to-Speech

> An AI-powered system that generates descriptive captions for images and delivers them as spoken audio in multiple languages — built with HuggingFace Transformers and Google Text-to-Speech.

---

## 🌟 Overview

This project combines **computer vision**, **neural machine translation**, and **text-to-speech synthesis** into a single pipeline. A user uploads an image through a browser interface, selects a language, and instantly receives both a written caption and an audio file.

**Supported Languages:** English · Bengali · French · Chinese · Hindi

---

## ✨ Features

- 🖼️ Automatic image caption generation using a pre-trained vision-language model
- 🌐 Real-time translation into 5 languages
- 🔊 Audio output via Google Text-to-Speech (gTTS)
- 💻 Lightweight browser-based interface (no framework required)
- 📄 Detailed project report included

---

## 🗂️ Project Structure

```
├── main.py                  # Core pipeline: captioning + translation + TTS
├── reference_captions.py    # Caption utilities and language configuration
├── front.html               # Web interface for image upload & language selection
├── requirements.txt         # Python dependencies
└── Multilingual Captioning with Text-to-Speech.pdf  # Project report
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ForhadAhmedKhan/Multilingual-Image-Captioning-with-T2S.git
cd Multilingual-Image-Captioning-with-T2S
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

Then open `front.html` in your browser:
1. Upload any image
2. Select your target language
3. Receive a **text caption** + **audio playback**

---

## 💡 Example Output

**Input:** A photo of a boy riding a bicycle

| Language | Caption |
|---|---|
| English | "A boy is riding a bicycle on the street." |
| Bengali | "একটি ছেলে রাস্তায় সাইকেল চালাচ্ছে।" |
| French | "Un garçon fait du vélo dans la rue." |
| Chinese | "一个男孩正在街上骑自行车。" |
| Hindi | "एक लड़का सड़क पर साइकिल चला रहा है।" |

Each caption is also delivered as a `.mp3` audio file.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![gTTS](https://img.shields.io/badge/gTTS-4285F4?style=flat&logo=google&logoColor=white)

- **Caption model:** HuggingFace `nlpconnect/vit-gpt2-image-captioning`
- **Translation:** HuggingFace MarianMT / Helsinki-NLP
- **TTS:** Google Text-to-Speech (gTTS)

---

## 🔮 Future Improvements

- [ ] Deploy as a FastAPI or Streamlit web app
- [ ] Add more supported languages
- [ ] Integrate larger multilingual models for better translation quality
- [ ] Cloud storage support for uploaded images

---

## 📄 License

MIT License
