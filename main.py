import speech_recognition as sr
import librosa
import numpy as np
from nltk.corpus import cmudict
import Levenshtein
import os
import pyttsx3
import nltk
from gtts import gTTS
import time  # Add this import at the top of the file

# Initialize CMU Pronouncing Dictionary
try:
    pron_dict = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    pron_dict = cmudict.dict()

recognizer = sr.Recognizer()

def extract_acoustic_features(audio_file):
    """Extract MFCC features from audio"""
    y, sr_val = librosa.load(audio_file)  # Store the sample rate in sr_val
    mfcc = librosa.feature.mfcc(y=y, sr=sr_val, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def get_reference_phonemes(word):
    """Get reference pronunciation from CMU dict"""
    phones = pron_dict.get(word.lower(), [['']])
    return [ph for ph in phones[0] if ph[-1].isdigit()]

def dynamic_time_warping(user_mfcc, ref_mfcc):
    """Calculate DTW distance between MFCC sequences"""
    n = len(user_mfcc)
    m = len(ref_mfcc)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(user_mfcc[i-1] - ref_mfcc[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]

def text_to_speech(text):
    """Convert text to speech and play"""
    engine = pyttsx3.init()
    
    # Set volume to maximum
    engine.setProperty('volume', 1.0)
    
    engine.say(text)
    engine.runAndWait()

def verify_pronunciation(audio_file, reference_word):
    """Verify pronunciation using acoustic and phonetic comparison"""
    try:
        # Acoustic feature comparison
        user_features = extract_acoustic_features(audio_file)
        reference_audio = f"reference_{reference_word}.mp3"

        # Generate reference audio using TTS
        tts = gTTS(text=reference_word, lang='en')
        tts.save(reference_audio)
        ref_features = extract_acoustic_features(reference_audio)
        dtw_score = dynamic_time_warping(user_features, ref_features)

        # Phonetic sequence comparison
        with sr.AudioFile(audio_file) as source:
            user_audio = recognizer.record(source)
        
        spoken_text = recognizer.recognize_google(user_audio).lower()
        user_phonemes = get_reference_phonemes(spoken_text)
        ref_phonemes = get_reference_phonemes(reference_word)
        lev_score = Levenshtein.distance(''.join(user_phonemes), ''.join(ref_phonemes))

        os.remove(reference_audio)  # Clean up reference audio file
        return dtw_score < 15 and lev_score < 3, ref_phonemes

    except Exception as e:
        print(f"Error: {e}")
        return False, []

# **Step 1: Ask user to input a word**
while True:
    reference_word = input("Please enter a word: ").strip().lower()
    if reference_word in pron_dict:
        break
    else:
        print("The word is not in the CMU Pronouncing Dictionary. Please try again.")

print(f"Word chosen: {reference_word}")
#text_to_speech("Repeat the word")

try:
    # **Step 2: Capture pronunciation attempt**
    with sr.Microphone() as source:
        text_to_speech("Please say the word now.")
        recognizer.adjust_for_ambient_noise(source, duration=5)  # Adjust for ambient noise
        audio_attempt = recognizer.listen(source, timeout=15)

    # Save user pronunciation attempt
    user_audio_file = "user_attempt.wav"
    with open(user_audio_file, "wb") as f:
        f.write(audio_attempt.get_wav_data())

    # **Step 3: Verify pronunciation**
    is_correct, correct_phonemes = verify_pronunciation(user_audio_file, reference_word)
    if is_correct:
        text_to_speech("Great job! Your pronunciation is correct.")
    else:
        correct_phon = ' '.join(correct_phonemes)
        text_to_speech(f"Incorrect pronunciation. It should sound like: {correct_phon}")
        text_to_speech(reference_word)  # Announce the correct pronunciation
        time.sleep(7)  # Hold the corrected pronunciation for 7 seconds

except sr.UnknownValueError:
    text_to_speech("I couldn't understand what you said.")
except sr.RequestError as e:
    text_to_speech(f"Could not request results from Google Speech Recognition; {e}")