import pytube
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from moviepy.editor import VideoFileClip
import speech_recognition as sr

# Download video and extract audio
def download_video(video_url):
    try:
        yt = pytube.YouTube(video_url)
        video = yt.streams.filter(only_audio=True).first()
        if not video:
            print("No audio stream found for the specified video URL.")
            return None

        video.download()

        # Get the audio file path after downloading the video
        audio_file = video.default_filename.replace(".mp4", ".mp3")

        # Convert the video file to audio using moviepy
        video_path = video.default_filename
        audio_path = audio_file
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)

        # Close the video file clip
        video_clip.close()

        return audio_file

    except pytube.exceptions.VideoUnavailable:
        print("Video is unavailable or access is restricted.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None


# Transcribe audio to text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
            return transcript
        except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    return None

# Load question answering model
def load_qa_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

# Answer questions based on transcript
def answer_questions(transcript, question):
    tokenizer, model = load_qa_model()
    prompt_text = f"Question: {question}"
    encoded_text = tokenizer(transcript, prompt_text, return_tensors="pt")

    # Convert input_ids and attention_mask to tensors on the GPU (if available)
    if torch.cuda.is_available():
        encoded_text['input_ids'] = encoded_text['input_ids'].cuda()
        encoded_text['attention_mask'] = encoded_text['attention_mask'].cuda()

    outputs = model(**encoded_text)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Move tensors back to CPU (if necessary)
    if torch.cuda.is_available():
        start_logits = start_logits.cpu()
        end_logits = end_logits.cpu()

    answer_start = torch.argmax(start_logits, dim=-1)
    answer_end = torch.argmax(end_logits, dim=-1)

    answer = transcript[answer_start:answer_end+1]
    return answer

# Chatbot interface
def chatbot():
    while True:
        video_url = input("Enter YouTube video URL: ")
        audio_file = download_video(video_url)

        if audio_file is not None:
            # Proceed with transcription and question-answering if audio is available
            transcript = transcribe_audio(audio_file)

            if transcript is not None:
                question = input("Ask a question about the video: ")
                answer = answer_questions(transcript, question)
                print(answer)
            else:
                print("Transcription failed. Please try a different video.")
        else:
            # Handle the case where no audio stream was found
            print("No audio stream found for the specified video URL. Please try a different video.")

if __name__ == "__main__":
    chatbot()
