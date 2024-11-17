# make a streamlit app to convert image or audio input to text and display the output
from PIL import Image
import streamlit as st
import os
import pytesseract
from pydub import AudioSegment
import speech_recognition as sr

def extract_text_from_image(image_path):
    if not os.path.isfile(image_path):
        return f"File not found: {image_path}"
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        # remove groups of whitespaces and newline characters
        text = " ".join(text.split())
        return text if text else "No text found in the image."
    except pytesseract.pytesseract.TesseractError as e:
        return f"Tesseract OCR error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
    
# def extract_text_from_audio(audio_path):
#     if not os.path.isfile(audio_path):
#         return f"File not found: {audio_path}"
#     try:
#         audio = AudioSegment.from_file(audio_path)
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(audio_path) as source:
#             audio_data = recognizer.record(source)
#             text = recognizer.recognize_google(audio_data)
#             return text if text else "No text found in the audio."
#     except sr.RequestError as e:
#         return f"Could not request results from Google Speech Recognition service; {e}"
#     except sr.UnknownValueError as e:
#         return f"Google Speech Recognition could not understand the audio; {e}"
#     except Exception as e:
#         return f"An error occurred: {e}"
    
print(extract_text_from_image("multimodality/image_to_text/c.png"))