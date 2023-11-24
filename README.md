# YODA

This repository hosts a Python-based chatbot that interacts with YouTube videos. The chatbot downloads a specified YouTube video, transcribes its audio content to text using speech recognition, and then uses a pre-trained question-answering model to answer user queries about the video.

## Introduction

The chatbot combines several powerful libraries, including PyTube for downloading YouTube videos, SpeechRecognition for transcribing audio, and Hugging Face's Transformers library for question answering using a BERT-based model. This project aims to provide a simple yet effective way to interact with video content through natural language queries.

## Features

- **YouTube Video Download:** Easily download YouTube videos using the PyTube library.
- **Speech-to-Text Transcription:** Utilize the SpeechRecognition library to transcribe audio from the video to text.
- **Question Answering:** Leverage a pre-trained BERT-based question-answering model from Hugging Face's Transformers library.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- [Python](https://www.python.org/) (>=3.6)

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd yoda
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the chatbot
```bash
python Yotube_Chat.py
```
## Components
### 1. Download Video
The download_video function uses the PyTube library to download the audio stream of a specified YouTube video.

### 2. Transcribe Audio
The transcribe_audio function utilizes the SpeechRecognition library to transcribe the downloaded audio to text.

### 3. Question Answering
The answer_questions function uses a pre-trained BERT-based question-answering model to answer user queries based on the transcribed text.

## Customization
Feel free to customize the code or integrate different models based on your preferences. You can explore other speech recognition libraries, download methods, or even experiment with different question-answering models.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
