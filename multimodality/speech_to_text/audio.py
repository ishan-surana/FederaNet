import speech_recognition as sr

# Initialize the recognizer
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        print("Recording stopped.")
        text = recognizer.recognize_google(audio)
    return text
        # return audio.get_wav_data()

print(record_audio())

# with open("Improvements/multimodality/speech_to_text/audio.wav", "wb") as f:
#     f.write(record_audio())


"""
        # Get the audio through microphone
        print("Recognizing...")
        print("Audio: ", audio.frame_data)
        # get the audio data
        if audio:
        #     audio_data = audio.get_wav_data()
        #     # recognize the audio data
            recognizer.AcceptWaveform(audio)
            result = json.loads(recognizer.FinalResult())
            return result['text']
        else:
            return "No audio found."
    except Exception as e:
        return f"An error occurred: {e}"
"""

"""
    import vosk
import pyaudio
import wave
import json
import os
from pydub import AudioSegment
import speech_recognition as sr

# def record_audio():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         audio = recognizer.listen(source)
#     print("Recording stopped.")
#     print("Audio: ", audio.frame_data)
#     return audio

# recorded_audio = record_audio()


def extract_text_from_audio(audio_path):
    # if not os.path.isfile(audio_path):
    #     return f"File not found: {audio_path}"
    try:
        model = vosk.Model("Improvements/multimodality/vosk-model-en-in-0.5")
        recognizer = vosk.KaldiRecognizer(model, 16000)
        # audio = AudioSegment.from_wav(audio_path)
        # audio = audio.set_channels(1)
        # audio.export(audio_path, format="wav")
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            return f"Audio file must be WAV format mono PCM."
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        while True:
            data = wf.readframes(8000)
            print(data)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                break
        stream.stop_stream()
        stream.close()
        result = json.loads(recognizer.FinalResult())
        return result["text"]
    except Exception as e:
        return f"Error: {e}"

    
# test the function
audio_path = "Improvements/multimodality/speech_to_text/audio.wav"
print(extract_text_from_audio(audio_path))
    """