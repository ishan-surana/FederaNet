import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Path to the audio file
audio_file_path = r'Improvements/multimodality/speech_to_text/Test.wav'

# Load audio file
with sr.AudioFile(audio_file_path) as source:
    audio_data = recognizer.record(source)  # Read the entire audio file

# Recognize (convert from speech to text)
try:
    # Using Google Web Speech API
    text = recognizer.recognize_google(audio_data)
    print("Transcription:", text)
except sr.UnknownValueError:
    print("Could not understand the audio.")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
