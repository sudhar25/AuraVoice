import speech_recognition as sr
import librosa
import numpy as np
from nltk.corpus import cmudict
import Levenshtein
import os
import pyttsx3
import nltk
from gtts import gTTS
import time
from datetime import datetime
from attempts import create_database, save_attempt, view_attempts

# Initialize CMU Pronouncing Dictionary
try:
    pron_dict = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    pron_dict = cmudict.dict()

recognizer = sr.Recognizer()

# Createing the database
create_database()

def extract_acoustic_features(audio_file):
    """Extract MFCC features from audio"""
    y, sr_val = librosa.load(audio_file)                 # Store the sample rate in sr_val
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
    """Verify pronunciation with flexible matching for Indian accent."""
    try:
        # Step 1: Extract acoustic features
        user_features = extract_acoustic_features(audio_file)
        reference_audio = f"reference_{reference_word}.mp3"

        # Generate reference audio using TTS
        tts = gTTS(text=reference_word, lang='en-IN')
        tts.save(reference_audio)
        ref_features = extract_acoustic_features(reference_audio)
        dtw_score = dynamic_time_warping(user_features, ref_features)

        # Step 2: Extract phonemes from CMU dictionary
        ref_phonemes_list = pron_dict.get(reference_word.lower(), [['']])  # Multiple phoneme versions

        # Step 3: Speech Recognition (for reference)
        with sr.AudioFile(audio_file) as source:
            user_audio = recognizer.record(source)
        
        try:
            spoken_text = recognizer.recognize_google(user_audio, language="en-IN").lower()

            print(f"Recognized word: {spoken_text}")
        except sr.UnknownValueError:
            print("Speech not recognized, skipping phoneme comparison.")
            return dtw_score, None, False, ref_phonemes_list[0]

        # Step 4: Extract phonemes from spoken text
        user_phonemes = get_reference_phonemes(spoken_text)
        if not user_phonemes:
            print("No phonemes found, skipping phoneme comparison.")
            return dtw_score, None, False, ref_phonemes_list[0]

        # Step 5: Compare phonemes with multiple reference versions
        best_lev_score = min([Levenshtein.distance(''.join(user_phonemes), ''.join(ref_phonemes)) for ref_phonemes in ref_phonemes_list])

        # Step 6: Adjust threshold dynamically for Indian accents
        phoneme_length = len(ref_phonemes_list[0])
        max_allowed_distance = max(2, int(phoneme_length * 0.75))  # 40% phoneme match allowed

        print(f"DTW Score: {dtw_score}, Levenshtein Score: {best_lev_score}, Allowed Max Distance: {max_allowed_distance}")

        # Step 7: Final Decision (More Relaxed Matching)
        is_correct = (dtw_score < 400) and (best_lev_score <= max_allowed_distance)

        os.remove(reference_audio)  # Clean up
        return dtw_score, best_lev_score, is_correct, ref_phonemes_list[0]

    except Exception as e:
        print(f"Error: {e}")
        return None, None, False, []

# **Step 1: Asking user to input a word**
while True:
    reference_word = input("Please enter a word: ").strip().lower()
    if reference_word in pron_dict:
        break
    else:
        print("The word is not in the CMU Pronouncing Dictionary. Please try again.")

print(f"Word chosen: {reference_word}")

try:
    # **Step 2: Capturing pronunciation attempt**
    with sr.Microphone() as source:
        text_to_speech("Please say the word now.")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        audio_attempt = recognizer.listen(source, timeout=20)

    # Saveing user pronunciation attempt
    user_audio_file = "user_attempt.wav"
    with open(user_audio_file, "wb") as f:
        f.write(audio_attempt.get_wav_data())

    # **Step 3: Verifing pronunciation**
    dtw_score, lev_score, is_correct, correct_phonemes = verify_pronunciation(user_audio_file, reference_word)
    result = "correct" if is_correct else "incorrect"

    # Save the data to the database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_attempt(reference_word, timestamp, user_audio_file, dtw_score, lev_score, result)

    if is_correct:
        text_to_speech("Great job! Your pronunciation is correct.")
    else:
        correct_phon = ' '.join(correct_phonemes)
        text_to_speech(f"Incorrect pronunciation. It should sound like: {correct_phon}")
        text_to_speech(reference_word)  # announcing the correct pronunciation
        time.sleep(5)  # Hold the corrected pronunciation for 7 seconds

except sr.UnknownValueError:
    text_to_speech("I couldn't understand what you said.")
except sr.RequestError as e:
    text_to_speech(f"Could not request results from Google Speech Recognition; {e}")

# **Step 4: Asking the if the user wants to view all attempts**
view_data = input("Do you want to view all attempts? (yes/no): ").strip().lower()
if view_data == "yes":
    print("\n attempts:")
    view_attempts()
else:
    print("Exiting without viewing attempts.")