from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import speech_recognition as sr
import librosa
import os
import nltk
from nltk.corpus import cmudict
import numpy as np
from Levenshtein import distance as levenshtein_distance
from dotenv import load_dotenv
#start
load_dotenv()
try:
    pron_dict = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    pron_dict = cmudict.dict()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Database configuration 
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

recognizer = sr.Recognizer()

# SQLAlchemy Model for Attempts
class Attempt(db.Model):
    __tablename__ = 'attempts'
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(100), nullable=False)
    recognized_text = db.Column(db.String(500), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

def save_attempt(word, recognized_text, similarity_score, is_correct):
    new_attempt = Attempt(
        word=word,
        recognized_text=recognized_text,
        similarity_score=similarity_score,
        is_correct=is_correct
    )
    db.session.add(new_attempt)
    db.session.commit()

def extract_acoustic_features(audio_file):
    """Extract MFCC features from audio"""
    try:
        y, sr_val = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr_val, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        return f"Error extracting MFCC: {str(e)}"

def get_reference_phonemes(word):
    """Get reference pronunciation from CMU dict"""
    phones = pron_dict.get(word.lower(), [['']])
    return [ph for ph in phones[0] if ph[-1].isdigit()]

def calculate_similarity(word1, word2):
    """Compute Levenshtein similarity score"""
    return 1 - (levenshtein_distance(word1.lower(), word2.lower()) / max(len(word1), len(word2)))

@app.route('/analyze_speech/', methods=['POST'])
def analyze_speech():
    if 'audio_file' not in request.files or 'word' not in request.form:
        return jsonify({'error': 'Missing audio file or word'}), 400

    audio_file = request.files['audio_file']
    word = request.form['word'].strip()

    # Save audio file permanently in database
    audio_dir = "saved_audio" 
    os.makedirs(audio_dir, exist_ok=True)
    file_path = os.path.join(audio_dir, f"{word}.wav")
    audio_file.save(file_path)

    try:
        # Speech Recognition main logic
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            try:
                recognized_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return jsonify({'error': 'Could not understand the audio'}), 400
            except sr.RequestError:
                return jsonify({'error': 'Speech recognition service unavailable'}), 500

        # Calculate Pronunciation Accuracy
        similarity_score = calculate_similarity(word, recognized_text)
        is_correct = similarity_score >= 0.4  # 80% accuracy threshold

        # Save attempt to database for viewing
        save_attempt(word, recognized_text, similarity_score, is_correct)

        return jsonify({
            'recognized_text': recognized_text,
            'feedback': 'Correct pronunciation' if is_correct else 'Try again',
            'similarity_score': round(similarity_score, 2),
            'is_correct': is_correct
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  
    app.run(host='0.0.0.0', port=8000, debug=True)
