import speech_recognition as sr
from PIL import Image
import pytesseract
import streamlit as st

def audio_to_text():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        print("Recording stopped.")
        try:
            text = recognizer.recognize_google(audio)
        except Exception:
            return f"An error occurred. Please try again."
    return text

def image_to_text(image):
    try:
        image = Image.open(image)
        text = pytesseract.image_to_string(image)
        text = " ".join(text.split())
        return text if text else "No text found in the image."
    except pytesseract.pytesseract.TesseractError as e:
        return f"Tesseract OCR error: {e}"

image_tab, audio_tab = st.tabs(["Image to Text", "Audio to Text"])

with image_tab:
    st.write("Upload an image and convert it to text.")
    image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if image is not None:
        try:
            text = image_to_text(image)
        except Exception as e:
            text = f"An error occurred: {e}"
        st.write(text)

with audio_tab:
    st.write("Record audio and convert it to text.")
    record = st.button("Record")
    if record:
        text = audio_to_text()
        st.write(text)
        
st.title("Multimodality App")