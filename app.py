import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import sys
import tempfile
from pathlib import Path
import time

# Tambahkan path ke utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import modul
from utils.stt_processor import load_stt_model, load_text_models, transcribe_and_clean
from utils.nonverbal_analysis import analyze_non_verbal
from utils.scoring_logic import load_embedder_model, score_with_rubric, compute_confidence_score

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="AI Interview Assessment",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback CSS
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ==================== SESSION STATE ====================
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1  # 1: Landing, 2: Registration, 3: Interview, 4: Report
if 'candidate_info' not in st.session_state:
    st.session_state.candidate_info = {}
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'current_question' not in st.session_state:
    st.session_state.current_question = 1
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False

# ==================== LOAD DATA ====================
@st.cache_resource
def load_questions():
    with open('data/questions.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_rubric():
    with open('data/rubric_data.json', 'r') as f:
        return json.load(f)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_all_models():
    """Load semua model AI secara cached"""
    try:
        whisper_model = load_stt_model()
        spell_checker, english_words = load_text_models()
        embedder_model = load_embedder_model()
        
        st.session_state.models_loaded = True
        return whisper_model, spell_checker, embedder_model, english_words
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# ==================== FUNGSI HELPER ====================
def create_temp_dir():
    """Membuat direktori untuk file sementara"""
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def save_uploaded_file(uploaded_file, temp_dir):
    """Menyimpan file yang diupload"""
    # Buat nama file unik
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = uploaded_file.name.split('.')[-1]
    filename = f"response_{timestamp}.{file_ext}"
    file_path = temp_dir / filename
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def calculate_final_score():
    """Menghitung skor akhir"""
    if not st.session_state.scores:
        return 0
    
    total_score = sum(score['score'] for score in st.session_state.scores.values())
    max_possible = len(st.session_state.scores) * 4
    return (total_score / max_possible) * 100 if max_possible > 0 else 0

# ==================== KOMPONEN UI ====================
def show_landing_page():
    """Landing page dengan HTML/CSS styling"""
    st.markdown("""
    <div class="main-container fade-in">
        <div class="landing-container">
            <div style="text-align: center; margin-bottom: 40px;">
                <h1 style="font-size: 3.5rem; color: #1E3A8A; margin-bottom: 20px;">🎙️ AI Interview Assessment</h1>
                <p style="font-size: 1.2rem; color: #666; max-width: 700px; margin: 0 auto 30px;">
                    Practice your interview skills with AI-powered feedback. Get detailed analysis on your responses, 
                    speech patterns, and confidence level for Machine Learning & AI positions.
                </p>
            </div>
            
            <div style="display: flex; justify-content: center; margin: 40px 0;">
                <button id="startBtn" style="
                    background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
                    color: white;
                    border: none;
                    padding: 18px 50px;
                    font-size: 1.3rem;
                    font-weight: 600;
                    border-radius: 50px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: inline-flex;
                    align-items: center;
                    gap: 15px;
                    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
                ">
                    <span>Start Free Interview Session</span>
                    <span style="font-size: 1.5rem;">→</span>
                </button>
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <div style="font-size: 2.5rem; margin-bottom: 20px;">🤖</div>
                    <h3 style="color: #1E3A8A; margin-bottom: 15px;">AI-Powered Analysis</h3>
                    <p>Advanced AI analyzes your speech content, tone, and delivery patterns</p>
                </div>
                
                <div class="feature-card">
                    <div style="font-size: 2.5rem; margin-bottom: 20px;">📊</div>
                    <h3 style="color: #1E3A8A; margin-bottom: 15px;">Detailed Scoring</h3>
                    <p>Get comprehensive scores based on industry-standard rubrics</p>
                </div>
                
                <div class="feature-card">
                    <div style="font-size: 2.5rem; margin-bottom: 20px;">⚡</div>
                    <h3 style="color: #1E3A8A; margin-bottom: 15px;">Instant Feedback</h3>
                    <p>Receive immediate feedback after each response to improve quickly</p>
                </div>
                
                <div class="feature-card">
                    <div style="font-size: 2.5rem; margin-bottom: 20px;">🎯</div>
                    <h3 style="color: #1E3A8A; margin-bottom: 15px;">ML/AI Focused</h3>
                    <p>Specialized for Machine Learning and Artificial Intelligence interviews</p>
                </div>
            </div>
            
            <div style="margin-top: 50px; padding: 30px; background: #F8FAFC; border-radius: 15px;">
                <h3 style="color: #1E3A8A; margin-bottom: 20px; text-align: center;">How It Works</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 10px;">1</div>
                        <p><strong>Register</strong><br>Enter your details</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 10px;">2</div>
                        <p><strong>Answer Questions</strong><br>5 ML/AI technical questions</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 10px;">3</div>
                        <p><strong>Get Feedback</strong><br>Detailed analysis and scores</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript untuk tombol
    js_code = """
    <script>
    document.getElementById('startBtn').addEventListener('click', function() {
        // Simulasi klik Streamlit button
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'start_interview'}, '*');
    });
    </script>
    """
    
    st.components.v1.html(js_code, height=0)
    
    # Tombol Streamlit yang tersembunyi
    if st.button("Start Interview", key="hidden_start", type="primary", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()

def candidate_registration():
    """Form registrasi kandidat"""
    st.markdown("""
    <div class="main-container">
        <div class="interview-header fade-in">
            <h1 style="margin:0; font-size: 2.5rem;">👤 Candidate Registration</h1>
            <p style="margin:0; opacity: 0.9; font-size: 1.1rem;">
                Please provide your information to start the interview
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("candidate_form"):
        st.markdown("### Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="John Doe")
            email = st.text_input("Email Address *", placeholder="john@example.com")
        
        with col2:
            phone = st.text_input("Phone Number *", placeholder="+62 812-3456-7890")
            position = st.text_input("Target Position", placeholder="ML Engineer / AI Researcher")
        
        st.markdown("---")
        
        # Terms agreement
        agree = st.checkbox("I agree to the terms and conditions")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Start Interview Session", use_container_width=True)
        
        if submitted:
            if name and email and phone and agree:
                st.session_state.candidate_info = {
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'position': position,
                    'start_time': datetime.now().isoformat()
                }
                st.session_state.current_step = 3
                st.rerun()
            else:
                if not agree:
                    st.warning("Please agree to the terms and conditions")
                else:
                    st.warning("Please fill all required fields (*)")

def show_question_ui(question_num, total_questions):
    """UI untuk pertanyaan interview"""
    # Load data
    questions = load_questions()
    question_data = questions[str(question_num)]
    
    # Progress
    progress = question_num / total_questions
    
    st.markdown(f"""
    <div class="main-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h3 style="color: white; margin: 0;">Question {question_num} of {total_questions}</h3>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">ML/AI Technical Interview</p>
            </div>
            <div style="background: white; padding: 10px 20px; border-radius: 20px; color: #1E3A8A; font-weight: bold;">
                {question_num}/{total_questions}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    st.progress(progress)
    
    # Question card
    st.markdown(f"""
    <div class="interview-card fade-in">
        <h3 style="color: #1E3A8A; margin-bottom: 20px;">📝 Question</h3>
        <p style="font-size: 1.2rem; line-height: 1.6; color: #374151;">{question_data['question']}</p>
        
        <div style="margin-top: 30px; padding: 15px; background: #F8FAFC; border-radius: 10px;">
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                💡 <strong>Tips:</strong> Be specific, provide examples, and explain your thought process clearly.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Response section
    st.markdown("### 🎤 Upload Your Response")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose audio or video file",
            type=['mp3', 'wav', 'm4a', 'mp4', 'mov'],
            key=f"uploader_{question_num}",
            help="Upload your recorded response (max 5 minutes)"
        )
    
    with col2:
        st.markdown("""
        <div style="background: #F0F9FF; padding: 15px; border-radius: 10px; border-left: 4px solid #0EA5E9;">
            <p style="margin: 0; color: #0369A1; font-size: 0.9rem;">
                ✅ Supported formats: MP3, WAV, M4A, MP4, MOV<br>
                ⏱️ Recommended: 1-3 minutes per question<br>
                🎯 Focus on clarity and examples
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file:
        # Display audio player
        st.audio(uploaded_file, format=uploaded_file.type)
        
        # Process button
        if st.button(f"✅ Process Question {question_num}", 
                    type="primary", 
                    use_container_width=True,
                    disabled=st.session_state.processing):
            
            st.session_state.processing = True
            
            try:
                # Create temp directory
                temp_dir = create_temp_dir()
                
                # Save uploaded file
                audio_path = save_uploaded_file(uploaded_file, temp_dir)
                
                # Show processing status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load models
                status_text.text("🔄 Loading AI models...")
                whisper_model, spell_checker, embedder_model, english_words = load_all_models()
                progress_bar.progress(20)
                
                # Step 2: Analyze non-verbal
                status_text.text("📊 Analyzing speech patterns...")
                nonverbal_result = analyze_non_verbal(str(audio_path))
                progress_bar.progress(40)
                
                # Step 3: Transcribe
                status_text.text("🗣️ Transcribing your response...")
                transcript = transcribe_and_clean(
                    str(audio_path), 
                    whisper_model, 
                    spell_checker, 
                    english_words
                )
                progress_bar.progress(70)
                
                # Step 4: Score response
                status_text.text("📝 Evaluating your answer...")
                rubric = load_rubric()
                question_key = question_data['key']
                question_text = question_data['question']
                
                # Calculate confidence score
                confidence = compute_confidence_score(transcript)
                
                # Calculate semantic score
                score, feedback = score_with_rubric(
                    question_key, question_text, transcript, rubric, embedder_model
                )
                
                progress_bar.progress(100)
                
                # Save results
                st.session_state.responses[question_num] = {
                    'transcript': transcript,
                    'nonverbal': nonverbal_result,
                    'audio_path': str(audio_path),
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.scores[question_num] = {
                    'score': score,
                    'confidence': confidence,
                    'feedback': feedback,
                    'question': question_text
                }
                
                # Success message
                st.success(f"✅ Question {question_num} processed successfully!")
                
                # Show quick results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Content Score", f"{score}/4")
                with col2:
                    st.metric("Confidence", f"{confidence:.0%}")
                with col3:
                    if 'qualitative_summary' in nonverbal_result:
                        st.metric("Delivery", nonverbal_result['qualitative_summary'])
                
                # Auto-advance after 2 seconds
                time.sleep(2)
                
                # Move to next question or report
                if question_num < total_questions:
                    st.session_state.current_question += 1
                else:
                    st.session_state.current_step = 4
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing response: {str(e)}")
            
            finally:
                st.session_state.processing = False
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        if question_num > 1:
            if st.button("← Previous Question", use_container_width=True):
                st.session_state.current_question -= 1
                st.rerun()
    
    with col3:
        if question_num < total_questions:
            if st.button("Skip Question →", use_container_width=True):
                st.session_state.current_question += 1
                st.rerun()
        else:
            if st.session_state.scores and len(st.session_state.scores) == total_questions:
                if st.button("View Final Report →", type="primary", use_container_width=True):
                    st.session_state.current_step = 4
                    st.rerun()
            else:
                st.info("Complete all questions to view final report")

def show_final_report():
    """Tampilkan laporan akhir"""
    st.markdown("""
    <div class="main-container">
        <div class="interview-header fade-in">
            <h1 style="margin:0; font-size: 2.5rem;">📋 Interview Report</h1>
            <p style="margin:0; opacity: 0.9; font-size: 1.1rem;">
                Comprehensive evaluation of your interview performance
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Candidate info
    st.markdown("### Candidate Details")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Name:** {st.session_state.candidate_info.get('name', 'N/A')}")
        st.info(f"**Email:** {st.session_state.candidate_info.get('email', 'N/A')}")
    with col2:
        st.info(f"**Phone:** {st.session_state.candidate_info.get('phone', 'N/A')}")
        if 'position' in st.session_state.candidate_info:
            st.info(f"**Target Position:** {st.session_state.candidate_info.get('position', 'N/A')}")
    
    # Overall score
    final_score = calculate_final_score()
    st.markdown("### 📈 Overall Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <h1 style="margin:0; font-size: 4rem; text-align: center;">{final_score:.1f}%</h1>
            <p style="margin:0; opacity: 0.9; text-align: center;">Overall Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.scores:
            scores = [s['score'] for s in st.session_state.scores.values()]
            avg_score = sum(scores) / len(scores)
            
            st.metric("Average Score", f"{avg_score:.1f}/4.0")
            st.metric("Questions Completed", len(st.session_state.scores))
            st.metric("Recommendation", "Strong Candidate" if final_score >= 70 else "Needs Improvement")
    
    # Detailed breakdown
    st.markdown("### 📊 Detailed Evaluation")
    
    if st.session_state.scores:
        for q_num in sorted(st.session_state.scores.keys()):
            score_data = st.session_state.scores[q_num]
            
            with st.expander(f"Question {q_num}: {score_data['question'][:70]}..."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Content Score", f"{score_data['score']}/4")
                with col2:
                    st.metric("Confidence", f"{score_data['confidence']:.0%}")
                with col3:
                    # Get nonverbal data if available
                    if q_num in st.session_state.responses:
                        nonverbal = st.session_state.responses[q_num].get('nonverbal', {})
                        if 'qualitative_summary' in nonverbal:
                            st.metric("Delivery", nonverbal['qualitative_summary'])
                
                st.markdown("**Feedback:**")
                st.success(score_data['feedback'])
                
                # Show transcript if available
                if q_num in st.session_state.responses:
                    with st.expander("View Transcript"):
                        st.write(st.session_state.responses[q_num]['transcript'])
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Start New Interview", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        # Create downloadable report (simplified)
        report_data = {
            'Candidate': st.session_state.candidate_info.get('name', ''),
            'Email': st.session_state.candidate_info.get('email', ''),
            'Overall Score': f"{final_score:.1f}%",
            'Evaluation Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if st.button("📥 Download Report (PDF)", use_container_width=True):
            # In production, generate actual PDF
            st.success("Report generation would be implemented with proper PDF library")
    
    with col3:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

# ==================== APLIKASI UTAMA ====================
def main():
    # Hide Streamlit default elements
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Routing berdasarkan step
    if st.session_state.current_step == 1:
        show_landing_page()
    
    elif st.session_state.current_step == 2:
        candidate_registration()
    
    elif st.session_state.current_step == 3:
        questions = load_questions()
        total_questions = len(questions)
        show_question_ui(st.session_state.current_question, total_questions)
    
    elif st.session_state.current_step == 4:
        show_final_report()
    
    # Sidebar info (optional)
    with st.sidebar:
        st.markdown("### Interview Status")
        if st.session_state.current_step >= 2:
            if 'name' in st.session_state.candidate_info:
                st.info(f"**Candidate:** {st.session_state.candidate_info['name']}")
        
        if st.session_state.current_step == 3:
            questions = load_questions()
            total = len(questions)
            current = st.session_state.current_question
            st.progress(current / total)
            st.caption(f"Question {current} of {total}")
        
        if st.session_state.current_step == 4:
            final_score = calculate_final_score()
            st.metric("Final Score", f"{final_score:.1f}%")
        
        st.markdown("---")
        st.caption("AI Interview Assessment v1.0")

# ==================== RUN APLIKASI ====================
if __name__ == "__main__":
    # Ensure temp directory exists
    create_temp_dir()
    
    # Run main app
    main()