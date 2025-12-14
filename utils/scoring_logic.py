import json
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import numpy as np

# --- Thresholds ---
NON_RELEVANT_SIM_THRESHOLD = 0.2
MIN_LENGTH_FOR_SCORE = 5

# --- MODEL CACHING ---
def load_embedder_model():
    """Memuat model SentenceTransformer untuk scoring."""
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        return None

# --- FUNGSI RELEVANSI ---
def is_non_relevant(text: str) -> bool:
    """Mengecek apakah transkrip cenderung tidak relevan/kosong."""
    t = text.strip().lower()
    if len(t) == 0:
        return True

    if len(t.split()) <= 2:
        return True

    # Frasa umum yang menunjukkan ketidakmampuan menjawab
    non_answers = [
        "i don't know", "i dont know", "no idea",
        "i have no idea", "not sure", "i can't answer",
        "i cannot answer", "i don't understand",
        "i dont understand", "sorry", "i'm not sure",
        "i am not sure"
    ]
    if any(na in t for na in non_answers):
        return True
    
    return False

# --- FUNGSI CONFIDENCE SCORE ---
def compute_confidence_score(transcript: str, text_confidence: float = 0.0) -> float:
    """
    Menghitung skor kepercayaan.
    """
    if not transcript or is_non_relevant(transcript):
        return 0.1 
    
    # Skala ulang (simplified)
    word_count = len(transcript.split())
    if word_count < 10:
        scaled_confidence = 0.5
    elif word_count < 20:
        scaled_confidence = 0.7
    else:
        scaled_confidence = 0.85
    
    # Tambahkan penalti jika sangat pendek
    if len(transcript.split()) < 5:
        scaled_confidence *= 0.8
        
    return scaled_confidence

# --- FUNGSI SCORING SEMANTIK ---
def score_with_rubric(question_id, question_text, answer, rubric_data, model_embedder):
    """
    Menghitung skor berdasarkan perbandingan semantik dengan rubrik.
    """
    if model_embedder is None:
        return 0, "Error: Embedding model failed to load."

    rubric_entry = rubric_data.get(question_id, {})
    rubric = rubric_entry.get("ideal_points", {})
    a = answer.strip()

    if is_non_relevant(a) or len(a.split()) < MIN_LENGTH_FOR_SCORE:
        return 0, rubric.get("0", ["Unanswered"])[0]

    embedding_a = model_embedder.encode(a.lower())

    # Fungsi untuk menghitung kecocokan
    def count_matches(indicators, threshold=0.40):
        if not indicators:
            return 0
        hits = 0
        embeddings_indicators = model_embedder.encode([ind.lower() for ind in indicators])
        similarities = util.cos_sim(embedding_a, embeddings_indicators).flatten()
        
        for sim in similarities:
            if sim.item() >= threshold:
                hits += 1
        return hits

    # Iterasi dari skor tertinggi ke terendah
    for point_str in ["4", "3", "2", "1"]:
        point = int(point_str)
        indicators = rubric.get(point_str)
        
        if not indicators: 
            continue

        hits = count_matches(indicators)
        
        # Logika Min hits:
        if point == 4:
            min_hits = max(1, int(len(indicators) * 0.6))
        elif point == 3:
            min_hits = max(1, int(len(indicators) * 0.5))
        else:
            min_hits = 1

        if hits >= min_hits:
            return point, rubric.get(point_str, [f"Score {point} achieved"])[0]

    # Jika tidak ada yang cocok
    return 1, rubric.get("1", ["Minimal or Vague Response"])[0]