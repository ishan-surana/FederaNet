import vosk
import pyaudio
import wave
import json
import os
from pydub import AudioSegment
import speech_recognition as sr

# test the function
audio_path = "Improvements/multimodality/speech_to_text/Test1.wav"

def extract_text_from_audio(audio_path):
    if not os.path.isfile(audio_path):
        return f"File not found: {audio_path}"
    try:
        model = vosk.Model("Improvements/multimodality/vosk-model-en-in-0.5")
        recognizer = vosk.KaldiRecognizer(model, 16000)
        audio = AudioSegment.from_file(audio_path)
        # 2 channels audio file. work wit both 1 and 2 channels
        audio = audio.set_channels(1)
        audio.export(audio_path, format="wav")
        wf = wave.open(audio_path, "rb")
        print(wf.readframes(4000))
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            return f"Audio file must be WAV format mono PCM."
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return result['text']
        return "No text found in the audio."
    except Exception as e:
        return f"An error occurred: {e}"
    
# test the function
print(extract_text_from_audio(audio_path))