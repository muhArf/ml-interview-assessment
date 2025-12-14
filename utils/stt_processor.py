import os
import re
import itertools
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import torch
from faster_whisper import WhisperModel
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from pydub import AudioSegment

# Konfigurasi
WHISPER_MODEL_NAME = "small" 
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SR_RATE = 16000 

# Daftar istilah ML/AI
ML_TERMS = [
    "tensorflow", "keras", "vgc16", "vgc19", "mobilenet",
    "efficientnet", "cnn", "relu", "dropout", "model",
    "layer normalization", "batch normalization", "attention",
    "embedding", "deep learning", "dataset", "submission",
    "machine learning", "artificial intelligence", "neural network",
    "convolutional", "pooling", "activation", "optimizer",
    "loss function", "training", "validation", "testing"
]

# Mapping frasa yang sering salah
PHRASE_MAP = {
    "celiac": "cellular", "script": "skripsi", "i mentioned": "submission",
    "time short flow": "tensorflow", "eras": "keras", "vic": "vgc16",
    "vic": "vgc19", "va": "vgc16", "va": "vgc19", "mobile net": "mobilenet",
    "data set": "dataset", "violation laws": "validation loss",
    "tense of flow": "tensorflow", "transfer learning": "transfer learning",
    "convolutional neural": "convolutional neural", "image classification": "image classification"
}

# Filler words
FILLERS = ["umm", "uh", "uhh", "erm", "hmm", "eee", "emmm", "yeah", "ah", "okay", "like", "you know", "so"]

# --- MODEL CACHING ---
def load_stt_model():
    """Memuat Faster Whisper model tanpa torch GPU."""
    try:
        # Force CPU
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        from faster_whisper import WhisperModel
        return WhisperModel("small", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Error loading WhisperModel: {e}")
        return None

# --- AUDIO UTILITIES ---
def video_to_wav(input_video_path, output_wav_path, sr=SR_RATE):
    """Mengkonversi video ke WAV mono pada 16kHz menggunakan pydub."""
    try:
        audio = AudioSegment.from_file(input_video_path)
        audio = audio.set_channels(1).set_frame_rate(sr)
        audio.export(output_wav_path, format="wav")
        return True
    except Exception as e:
        raise RuntimeError(f"Video to WAV conversion failed: {e}")

def noise_reduction(in_wav, out_wav, prop_decrease=0.6):
    """Menerapkan Noise Reduction."""
    try:
        y, sr = librosa.load(in_wav, sr=SR_RATE)
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
        sf.write(out_wav, y_clean, sr)
        return True
    except Exception as e:
        raise RuntimeError(f"Noise reduction failed: {e}")

# --- TEXT CLEANING LOGIC ---
def correct_ml_terms(word, spell, english_words):
    """Mengoreksi istilah ML/AI."""
    w = word.lower()
    if w in english_words:
        return word

    match, score, _ = process.extractOne(w, ML_TERMS)
    dist = Levenshtein.distance(w, match.lower())

    if dist <= 3 or score >= 65:
        return match
    return word

def remove_duplicate_words(text):
    """Menghapus kata duplikat berurutan."""
    return " ".join([k for k, g in itertools.groupby(text.split())])

def clean_text(text, spell, english_words):
    """Membersihkan teks transkripsi."""
    # 1. Hapus filler words
    pattern = r"\b(" + "|".join(FILLERS) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 2. Hapus tanda baca berlebihan
    text = re.sub(r"\.{2,}", "", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    
    # 3. Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()
    
    # 4. Koreksi frasa
    for wrong, correct in PHRASE_MAP.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.IGNORECASE)
    
    # 5. Koreksi per kata
    words = []
    for w in text.split():
        sp = spell.correction(w)
        if sp:
            w = sp
        w = correct_ml_terms(w, spell, english_words)
        words.append(w)
    
    text = " ".join(words)
    
    # 6. Hilangkan kata duplikat berurutan
    text = remove_duplicate_words(text)
    
    # 7. Kapitalisasi awal kalimat
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.capitalize() for s in sentences if s]
    text = ' '.join(sentences)
    
    return text

# --- FUNGSI UTAMA TRANSKRIPSI ---
def transcribe_and_clean(audio_path, whisper_model, spell_checker, english_words):
    """Melakukan transkripsi dan membersihkan teks."""
    try:
        segments, _ = whisper_model.transcribe(
            audio_path, 
            language="en", 
            task="transcribe", 
            beam_size=4, 
            vad_filter=True
        )
        raw_text = " ".join([seg.text for seg in segments])
        
        cleaned_text = clean_text(raw_text, spell_checker, english_words)
        return cleaned_text
    except Exception as e:
        raise RuntimeError(f"Transcription error: {e}")

# ==== TAMBAHKAN DI SINI ====
def process_audio_for_streamlit(uploaded_file, temp_dir):
    """Optimized audio processing for Streamlit Cloud"""
    try:
        # Simpan file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = uploaded_file.name.split('.')[-1]
        
        # Batasi format
        if file_ext.lower() not in ['mp3', 'wav', 'm4a']:
            # Convert ke wav jika format tidak support
            file_ext = 'wav'
            
        filename = f"response_{timestamp}.{file_ext}"
        file_path = temp_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Batasi durasi (max 3 menit)
        import librosa
        y, sr = librosa.load(file_path, sr=16000)
        duration = len(y) / sr
        
        if duration > 180:  # 3 menit
            # Potong audio
            y = y[:180 * sr]  # Ambil 3 menit pertama
            import soundfile as sf
            sf.write(file_path, y, sr)
        
        return file_path
        
    except Exception as e:
        raise RuntimeError(f"Audio processing error: {str(e)}")