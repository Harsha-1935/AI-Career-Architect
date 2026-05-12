import streamlit as st
import pdfplumber
import json
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime
from groq import Groq
from gtts import gTTS
import io
import base64
import streamlit.components.v1 as components
import threading
import time
import cv2
import numpy as np
import os
import atexit

# Global list for cleanup
_ACTIVE_CAPS = []

def cleanup_all_cameras():
    for cap in _ACTIVE_CAPS:
        if cap.isOpened():
            cap.release()

atexit.register(cleanup_all_cameras)

# 📸 BACKGROUND PROCTORING MANAGER (PURE PYTHON)
class ProctorManager:
    def __init__(self, folder="captured_frames"):
        self.folder = folder
        self.active = False
        self.thread = None
        self.latest_frame_path = None
        self.live_frame = None  
        self.capture_history = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.last_ping = time.time() # Watchdog timer
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def ping(self):
        """Heartbeat to keep the thread alive."""
        with self.lock:
            self.last_ping = time.time()

    def _capture_loop(self):
        # Settle delay for Windows
        time.sleep(2.0)
        
        cap = None
        for b in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
            for idx in [0, 1]:
                try:
                    c = cv2.VideoCapture(idx, b) if b != 0 else cv2.VideoCapture(idx)
                    if c.isOpened():
                        ret, _ = c.read()
                        if ret:
                            cap = c
                            break
                    c.release()
                except: continue
            if cap: break

        if cap is None or not cap.isOpened():
            print("CRITICAL: Camera acquisition failed in ackup.py.")
            with self.lock:
                self.active = False
            return
        
        _ACTIVE_CAPS.append(cap)

        try:
            last_save_time = 0
            while not self.stop_event.is_set():
                # Watchdog check: If no ping for 10 seconds, auto-stop
                with self.lock:
                    if time.time() - self.last_ping > 10:
                        break

                ret, frame = cap.read()
                if ret:
                    # Store latest frame for live display
                    with self.lock:
                        self.live_frame = frame.copy()
                    
                    # Periodic snapshot and analysis (every 5s)
                    now_time = time.time()
                    if now_time - last_save_time >= 5:
                        now = datetime.now()
                        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                        display_time = now.strftime("%H:%M:%S")
                        filename = f"user_{timestamp}.jpg"
                        path = os.path.join(self.folder, filename)
                        
                        # Use RGB for saving/displaying in Streamlit
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                        
                        # Real-time confidence analysis
                        conf_data = analyze_camera_image(frame, is_frame=True)
                        
                        with self.lock:
                            self.latest_frame_path = path
                            self.capture_history.append((display_time, path, conf_data))
                            if len(self.capture_history) > 5:
                                self.capture_history.pop(0)
                        last_save_time = now_time
                time.sleep(0.05) # ~20 FPS is plenty for background monitoring
        finally:
            if cap in _ACTIVE_CAPS:
                _ACTIVE_CAPS.remove(cap)
            cap.release()
            with self.lock:
                self.live_frame = None

    def capture_baseline(self):
        """Captures the current live frame as the baseline identity."""
        with self.lock:
            if self.live_frame is not None:
                baseline_path = os.path.join(self.folder, "baseline_identity.jpg")
                # Save the current frame as baseline
                cv2.imwrite(baseline_path, self.live_frame)
                # Analyze it to get the expected person count
                analysis = analyze_camera_image(self.live_frame, is_frame=True)
                return baseline_path, analysis.get('faces_detected', 1)
        return None, 1

    def get_live_frame(self):
        with self.lock:
            if self.live_frame is not None:
                # Convert BGR to RGB for Streamlit
                return cv2.cvtColor(self.live_frame, cv2.COLOR_BGR2RGB)
        return None

    def get_latest_analysis(self):
        with self.lock:
            if self.capture_history:
                return self.capture_history[-1][2]
        return None

    def start(self):
        if not self.active:
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.active = True
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.active = False
        self.latest_frame_path = None
        self.capture_history = []

    def __del__(self):
        """Ensure cleanup if the object is garbage collected."""
        self.stop()


# Singleton-like instance in session state
def get_proctor_manager():
    if "proctor_manager" not in st.session_state:
        st.session_state.proctor_manager = ProctorManager()
    return st.session_state.proctor_manager


# -----------------------------
# 🚀 CONFIGURATION
# -----------------------------

# Hardcoded key as fallback, prioritizing environment variable
API_KEY = os.getenv("GROQ_API_KEY", "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").strip()
MODEL_NAME = "llama-3.1-8b-instant"

# Initialize Groq client (cached at module level)
client = Groq(api_key=API_KEY) if API_KEY else None

# Cache the OpenCV face cascade so it loads once, not on every frame
@st.cache_resource
def _load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

FACE_CASCADE = _load_face_cascade()

# 🎙️ ADVANCED VOICE SYSTEM — plays in background, no visible player
@st.cache_data(show_spinner=False)
def _generate_tts_audio(text):
    """Generates TTS audio bytes (cached per unique text to avoid repeated network calls)."""
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp.getvalue()

def speak_now(text, key_suffix, force=False):
    """Plays audio in the background via a hidden HTML audio element — no visible player."""
    if f"played_{key_suffix}" not in st.session_state or force:
        try:
            audio_bytes = _generate_tts_audio(text)
            audio_b64 = base64.b64encode(audio_bytes).decode()
            st.html(
                f'<audio autoplay="true" style="display:none;">'
                f'<source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">'
                f'</audio>'
            )
            st.session_state[f"played_{key_suffix}"] = True
        except Exception as e:
            st.error(f"Voice Error: {e}")

def stop_speaking():
    # gTTS/st.audio doesn't have a simple 'stop' from Python, 
    # but we can prevent new audio from playing.
    pass


def transcribe_audio(audio_bytes):
    """Transcribes audio bytes using Groq's Whisper API."""
    try:
        # Write audio to a temporary file-like object for the API
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language="en",
        )
        return transcription.text.strip()
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


# 📂 DATABASE LOGIC

@st.cache_resource
def init_db():
    """Initialize DB tables once per app lifecycle."""
    conn = sqlite3.connect('career_app.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, user_id INTEGER, date TEXT, role TEXT, score REAL, report TEXT)')
    conn.commit()
    conn.close()
    return True


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def clean_ai_list(data, keys):
    if not data:
        return []
    for k in keys:
        if k in data and data[k]:
            raw = data[k]
            # Handle list of strings or list of dicts, convert to string
            items = [str(i) if not isinstance(i, dict) else str(list(i.values())[0]) for i in raw]
            # Filter out purely numeric noise but keep short skills like 'R' or 'C'
            return [i.strip() for i in items if i.strip() and not i.strip().isdigit()]
    return []


def analyze_camera_image(data, is_frame=False):
    try:
        if is_frame:
            img = data
        else:
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: return {'confidence_score': 0, 'cheating_detected': False, 'faces_detected': 0, 'multiple_faces': False}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        confidence_score = 0
        cheating_flag = False

        if len(faces) == 0:
            cheating_flag = True
            confidence_score = 0
        elif len(faces) > 1:
            cheating_flag = True
            confidence_score = 20
        else:
            (x, y, w, h) = faces[0]
            face_area = w * h
            img_area = img.shape[0] * img.shape[1]
            face_ratio = face_area / img_area
            if face_ratio > 0.15:
                confidence_score = 90
            elif face_ratio > 0.1:
                confidence_score = 70
            elif face_ratio > 0.05:
                confidence_score = 50
            else:
                confidence_score = 30
            center_x = x + w / 2
            center_y = y + h / 2
            img_center_x = img.shape[1] / 2
            img_center_y = img.shape[0] / 2
            distance_from_center = np.sqrt((center_x - img_center_x) ** 2 + (center_y - img_center_y) ** 2)
            max_distance = np.sqrt(img_center_x ** 2 + img_center_y ** 2)
            center_ratio = 1 - (distance_from_center / max_distance)
            confidence_score += center_ratio * 10

        return {
            'confidence_score': min(100, max(0, int(confidence_score))),
            'cheating_detected': cheating_flag,
            'faces_detected': len(faces),
            'multiple_faces': len(faces) > 1
        }
    except Exception:
        return {
            'confidence_score': 50,
            'cheating_detected': False,
            'faces_detected': 1,
            'multiple_faces': False
        }




def initialize_state():
    defaults = {
        'user_id': None,
        'username': None,
        'started': False,
        'questions': [],
        'answers': [],
        'results': [],
        'q_index': 0,
        'career_analysis': None,
        'final_report': None,
        'selected_role': None,
        'interview_paused': False,
        'interview_terminated': False,
        'initial_faces': 0,
        'last_spoken_q_index': -1,
        'current_confidence_data': {'confidence_score': 50, 'cheating_detected': False, 'faces_detected': 0, 'multiple_faces': False},
        'proctor_image_b64': None,
        'monitoring_enabled': True,
        'termination_reason': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    



def reset_interview_state():
    st.session_state.started = False
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.results = []
    st.session_state.q_index = 0
    st.session_state.career_analysis = None
    st.session_state.final_report = None
    st.session_state.selected_role = None
    st.session_state.interview_paused = False
    st.session_state.interview_terminated = False
    st.session_state.initial_faces = 0
    st.session_state.last_spoken_q_index = -1
    st.session_state.termination_reason = None
    if "proctor_manager" in st.session_state:
        st.session_state.proctor_manager.stop()


def reset_user_session():
    if "proctor_manager" in st.session_state:
        st.session_state.proctor_manager.stop()
        
    preserve = {'user_id': None, 'username': None}
    for key in list(st.session_state.keys()):
        if key not in preserve:
            del st.session_state[key]
    st.session_state.user_id = None
    st.session_state.username = None


def render_login_register():
    st.title("AI Career Architect")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        login_form = st.form(key="login_form")
        with login_form:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Login")
            if login_submit:
                conn = sqlite3.connect('career_app.db')
                c = conn.cursor()
                c.execute('SELECT id, username, password FROM users WHERE username=?', (login_username,))
                res = c.fetchone()
                conn.close()
                if res and make_hashes(login_password) == res[2]:
                    st.session_state.user_id = res[0]
                    st.session_state.username = res[1]
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with register_tab:
        register_form = st.form(key="register_form")
        with register_form:
            register_username = st.text_input("New Username", key="register_username")
            register_password = st.text_input("New Password", type="password", key="register_password")
            register_confirm = st.text_input("Confirm Password", type="password", key="register_confirm")
            register_submit = st.form_submit_button("Register")
            if register_submit:
                if not register_username or not register_password or not register_confirm:
                    st.error("Please fill in all fields.")
                elif register_password != register_confirm:
                    st.error("Passwords do not match.")
                else:
                    conn = sqlite3.connect('career_app.db')
                    c = conn.cursor()
                    c.execute('SELECT id FROM users WHERE username=?', (register_username,))
                    exists = c.fetchone()
                    if exists:
                        conn.close()
                        st.error("Username already exists. Please choose another.")
                    else:
                        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (register_username, make_hashes(register_password)))
                        conn.commit()
                        conn.close()
                        st.success("Registration successful. You can now log in.")
                        st.info("Switch to the Login tab and sign in with your new account.")


def render_resume_audit():
    # Ensure camera is off during audit/setup phase
    if "proctor_manager" in st.session_state:
        st.session_state.proctor_manager.stop()

    section = st.container()
    with section:
        st.header("Resume Intelligence Audit")
        resume_file = st.file_uploader("Upload PDF", type=["pdf"], key="resume_upload")

        if resume_file and not st.session_state.career_analysis:
            with st.spinner("Analyzing resume..."):
                with pdfplumber.open(resume_file) as pdf:
                    txt = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])[:2000]
                prompt = f"Analyze resume: {txt}. Return JSON: {{'skills':[], 'strengths':[], 'technical_gaps':[], 'roles':[]}}"
                res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                st.session_state.career_analysis = json.loads(res.choices[0].message.content)
                st.rerun()

        if st.session_state.career_analysis:
            data = st.session_state.career_analysis
            st.success(f"**Skills:** {', '.join(clean_ai_list(data, ['skills']))}")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.info("**Strengths**")
                for s in clean_ai_list(data, ['strengths']):
                    st.write(f"- {s}")
                st.warning("**Technical Gaps Identified**")
                for g in clean_ai_list(data, ['technical_gaps', 'gaps']):
                    st.write(f"- {g}")
                selected_role = st.selectbox("Path:", clean_ai_list(data, ['roles']), key="selected_role")
            with c2:
                st.subheader("System Status")
                st.success("✅ Python Engine Ready")
                st.info("The AI Proctor will automatically monitor your session via the system camera.")
                st.caption("No browser permissions required.")
            
            if st.button("Start Interview", key="start_interview"):
                pm = get_proctor_manager()
                pm.start()
                
                # Give the camera a moment to initialize and capture baseline
                time.sleep(1.0) 
                baseline_path, baseline_count = pm.capture_baseline()
                
                st.session_state.started = True
                st.session_state.interview_paused = False
                st.session_state.interview_terminated = False
                st.session_state.q_index = 0
                st.session_state.questions = []
                st.session_state.answers = []
                st.session_state.results = []
                st.session_state.last_spoken_q_index = -1
                st.session_state.baseline_image = baseline_path
                st.session_state.initial_faces = baseline_count
                
                st.rerun()


@st.fragment(run_every="0.5s")
def render_live_monitoring_panel():
    """Dual-feed monitoring panel: Pure Python Live Stream + Verification Snapshot."""
    pm = get_proctor_manager()
    pm.ping() # Heartbeat: Keep camera alive while tab is open
    
    if not st.session_state.monitoring_enabled:
        st.warning("🔒 Monitoring Disabled")
        return

    # Header Card
    st.markdown("""
        <div style="background:#111; padding:10px; border-radius:5px; border-left:4px solid #4CAF50; margin-bottom:10px;">
            <span style="color:#4CAF50; font-weight:bold;">🛡️ PYTHON PROCTOR ACTIVE</span><br>
            <span style="color:#888; font-size:11px;">📸 Background monitoring via OpenCV</span>
        </div>
    """, unsafe_allow_html=True)

    # --- SECTION 1: LIVE CAMERA FEED ---
    st.caption("🎥 LIVE CAMERA FEED")
    frame = pm.get_live_frame()
    if frame is not None:
        st.image(frame, use_container_width=True)
    else:
        st.error("Camera Hardware Locked or Not Found")
    
    # --- SECTION 2: LATEST SNAPSHOT ---
    st.divider()
    col_base, col_snap = st.columns(2)
    with col_base:
        st.caption("👤 IDENTITY BASELINE")
        if "baseline_image" in st.session_state and st.session_state.baseline_image:
            st.image(st.session_state.baseline_image, use_container_width=True)
        else:
            st.info("No baseline")
            
    with col_snap:
        st.caption("📸 LATEST CHECK")
        if pm.latest_frame_path and os.path.exists(pm.latest_frame_path):
            st.image(pm.latest_frame_path, use_container_width=True)
        else:
            st.info("Waiting...")

    # Update confidence data in session state for UI display
    analysis = pm.get_latest_analysis()
    if analysis:
        st.session_state.current_confidence_data = analysis
        
        # 🚨 IDENTITY INTEGRITY CHECK
        expected = st.session_state.initial_faces
        detected = analysis.get('faces_detected', 0)
        
        if detected != expected and expected > 0:
            if detected > expected:
                st.session_state.interview_terminated = True
                st.session_state.termination_reason = f"Security Alert: Multiple persons detected ({detected} found, expected {expected})"
            elif detected < expected:
                # Just a warning or lower confidence, maybe the user moved away
                pass


def render_interview_stage():
    # Ensure Python Proctor is running
    pm = get_proctor_manager()
    
    # 🚨 SECURITY CHECK: TERMINATE IF VIOLATION DETECTED
    if st.session_state.interview_terminated and st.session_state.termination_reason:
        st.error(f"🚨 SECURITY VIOLATION: {st.session_state.termination_reason}. Interview terminated.")
        pm.stop()
        if st.button("Restart Session", key="restart_after_violation"):
            reset_interview_state()
            st.rerun()
        return

    if not pm.active:
        pm.start()

    if len(st.session_state.questions) <= st.session_state.q_index:
        q_prompt = f"You are an interviewer. Ask ONE short, focused interview question (max 2-3 sentences) for the role: {st.session_state.selected_role}. Do NOT include preamble, context, or instructions. Just the question."
        q_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": q_prompt}])
        st.session_state.questions.append(q_res.choices[0].message.content.strip())
    curr_q = st.session_state.questions[st.session_state.q_index]
    
    # --- LIVE MONITORING SIDEBAR ---
    with st.sidebar:
        st.subheader("🛡️ Live Proctoring")
        st.session_state.monitoring_enabled = st.toggle("Enable Monitor", value=True)
        
        if st.session_state.monitoring_enabled:
            # Pure Python monitoring panel
            render_live_monitoring_panel()
            
            st.divider()
    
            conf_data = st.session_state.current_confidence_data
            faces = conf_data.get('faces_detected', 0)
            st.metric("👥 Persons Detected", faces)
            
            if conf_data.get('cheating_detected') or faces > 1:
                st.error(f"⚠️ Face Error: {faces} detected")
            else:
                st.success(f"✅ Secure Session: {conf_data.get('confidence_score', 0)}%")
            st.caption("Tracking presence and environment via Python/OpenCV.")
        else:
            st.warning("🔒 Monitoring Disabled")
            pm.stop()
    


        
    confidence_data = st.session_state.current_confidence_data
    
    # Main interview content
    question_holder = st.empty()
    answer_holder = st.empty()
    submit_holder = st.empty()

    with question_holder.container():
        st.subheader(f"Question {st.session_state.q_index + 1} of 5")
        st.info(curr_q)

    answer_key = f"ans_{st.session_state.q_index}"
    speech_key = f"speech_transcript_{st.session_state.q_index}"

    # Initialize speech transcript in session state
    if speech_key not in st.session_state:
        st.session_state[speech_key] = ""

    with answer_holder.container():
        # 🎤 Audio recording for speech-to-text
        if not st.session_state.interview_terminated:
            audio_data = st.audio_input("🎤 Record your answer", key=f"mic_{st.session_state.q_index}")
            
            last_audio_bytes_key = f"last_audio_bytes_{st.session_state.q_index}"
            if audio_data:
                audio_bytes = audio_data.getvalue()
                # If this is a new recording (different bytes than last time)
                if last_audio_bytes_key not in st.session_state or st.session_state[last_audio_bytes_key] != audio_bytes:
                    with st.spinner("🎙️ Transcribing..."):
                        transcript = transcribe_audio(audio_bytes)
                        if transcript:
                            st.session_state[speech_key] = transcript
                            st.session_state[answer_key] = transcript
                            st.session_state[last_audio_bytes_key] = audio_bytes
                            st.rerun()

        # Text area purely controlled by its session state key
        if st.session_state.interview_terminated:
            ans = st.text_area("Your Answer:", key=answer_key, height=150, disabled=True)
        else:
            ans = st.text_area("Your Answer:", key=answer_key, height=150,
                               help="Type your answer or record audio above — it will be transcribed automatically")

    with submit_holder.container():
        col_audio, col_submit = st.columns([1, 4])
        with col_audio:
            if st.button("🔊 Re-play"):
                speak_now(curr_q, f"manual_{st.session_state.q_index}", force=True)
        submit_button = col_submit.button("Submit Answer", key="submit_answer", disabled=st.session_state.interview_terminated)


    if submit_button and not ans.strip():
        st.warning("⚠️ Your answer is empty. Please type your answer or record audio using the 🎤 mic above.")

    if submit_button and ans.strip():
        question_holder.empty()
        answer_holder.empty()
        submit_holder.empty()

        ev_prompt = (
            f"Score this answer from 0 to 10. Q:{curr_q} | A:{ans}. "
            f"Confidence Score: {confidence_data['confidence_score']}/100. "
            f"Cheating Detected: {confidence_data['cheating_detected']}. "
            "Return JSON object with only 'score' and 'feedback', where 'score' is a number from 0 to 10. "
            "Example: {'score': 7, 'feedback': 'Good explanation'}"
        )
        ev_res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": ev_prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(ev_res.choices[0].message.content)
        score = result.get('score', 0)
        try:
            score = float(score)
        except Exception:
            score = 0
        if score > 10:
            if score <= 100:
                score = score / 10
            else:
                score = 10
        score = max(0, min(10, score))
        if float(score).is_integer():
            score = int(score)
        result['score'] = score
        result['confidence_score'] = confidence_data['confidence_score']
        result['cheating_detected'] = confidence_data['cheating_detected']
        if confidence_data['cheating_detected']:
            result['confidence_feedback'] = "Potential cheating detected - please keep only one face visible and remain focused on the camera."
        elif confidence_data['confidence_score'] >= 80:
            result['confidence_feedback'] = "High confidence detected from your camera pose and presence."
        elif confidence_data['confidence_score'] >= 50:
            result['confidence_feedback'] = "Moderate confidence detected. Try to keep your face centered and clearly visible."
        else:
            result['confidence_feedback'] = "Low confidence detected. Make sure you are visible and stay in frame."
        st.session_state.results.append(result)
        st.session_state.answers.append(ans)
        st.session_state.q_index += 1
        st.rerun()
    
    # Speak the current question (no rerun — audio plays inline)
    if st.session_state.last_spoken_q_index != st.session_state.q_index:
        stop_speaking()
        speak_now(curr_q, f"auto_{st.session_state.q_index}", force=True)
        st.session_state.last_spoken_q_index = st.session_state.q_index


def render_report_stage():
    # Stop camera immediately on report stage
    pm = get_proctor_manager()
    pm.stop()
        
    section = st.container()
    with section:
        rep = st.session_state.final_report
        if rep is None:
            with st.spinner("Generating Final Career Insights..."):
                # Use actual length of results to avoid division by zero or incorrect averages
                results_len = len(st.session_state.results)
                total_score = sum([r.get('score', 0) for r in st.session_state.results])
                avg = round(total_score / results_len if results_len > 0 else 0, 2)
                
                # Analyze interview weaknesses to add to technical gaps
                weakness_prompt = (
                    f"Based on these interview results: {st.session_state.results}, "
                    "identify the top 3-5 SPECIFIC technical skill gaps. "
                    "Return JSON: {'interview_gaps':['skill name 1', 'skill name 2']}"
                )
                w_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": weakness_prompt}], response_format={"type": "json_object"})
                raw_w_data = json.loads(w_res.choices[0].message.content)
                interview_gaps = raw_w_data.get('interview_gaps', [])
                
                # Combine resume gaps and interview gaps safely
                resume_gaps = clean_ai_list(st.session_state.career_analysis, ['technical_gaps', 'gaps'])
                # Clean interview gaps too in case AI returned dicts
                cleaned_interview_gaps = [str(i) if not isinstance(i, dict) else str(list(i.values())[0]) for i in interview_gaps]
                
                all_gaps = list(set(resume_gaps + cleaned_interview_gaps))
                if "Not identified" in all_gaps: all_gaps.remove("Not identified")
                
                # Generate Roadmap based on ALL gaps
                road_prompt = (
                    f"Act as a Career Coach. Create a 7-day technical roadmap for a {st.session_state.selected_role} to fix these gaps: {all_gaps}. "
                    "You MUST return a JSON object with exactly 7 keys: 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7'. "
                    "The value for each key must be a single string describing the task for that day. "
                    "Example: {'day1': 'Study Bayesian fundamentals', 'day2': 'Practice SQL joins', ...}"
                )
                try:
                    road_res = client.chat.completions.create(
                        model=MODEL_NAME, 
                        messages=[{"role": "user", "content": road_prompt}], 
                        response_format={"type": "json_object"}
                    )
                    roadmap_data = json.loads(road_res.choices[0].message.content)
                except Exception as e:
                    # Fallback roadmap if AI fails to generate valid JSON
                    roadmap_data = {f"day{i}": f"Deep dive into {all_gaps[i % len(all_gaps)] if all_gaps else 'core concepts'} and practice related interview questions." for i in range(1, 8)}
                
                st.session_state.final_report = {
                    "avg": avg, 
                    "gaps": all_gaps,
                    "roadmap": roadmap_data
                }
                
                # Save to DB
                conn = sqlite3.connect('career_app.db')
                c = conn.cursor()
                c.execute('INSERT INTO history (user_id, date, role, score, report) VALUES (?,?,?,?,?)', 
                          (st.session_state.user_id, datetime.now().strftime("%Y-%m-%d %H:%M"), 
                           st.session_state.selected_role, avg, json.dumps(st.session_state.results)))
                conn.commit()
                conn.close()
                rep = st.session_state.final_report
        
        st.title(f"🏆 Final Performance Rating: {rep['avg']}/10")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Key Strengths**")
            for s in clean_ai_list(st.session_state.career_analysis, ['strengths']):
                st.write(f"✅ {s}")
        with c2:
            st.warning("**Priority Technical Gaps**")
            for g in rep.get('gaps', ["No critical gaps identified"]):
                st.write(f"🎯 {g}")
        st.subheader("🗓️ 7-Day Growth Roadmap")
        roadmap = rep.get('roadmap', {})
        # Safety: un-nest if AI wrapped it in a 'roadmap' key
        if isinstance(roadmap, dict) and 'roadmap' in roadmap:
            roadmap = roadmap['roadmap']
        
        for i in range(1, 8):
            day_key = f"day{i}"
            task = roadmap.get(day_key, roadmap.get(f"Day {i}", "Continue focusing on core technical concepts and practice coding problems."))
            st.success(f"**Day {i}:** {task}")
        st.divider()
        for i in range(5):
            with st.expander(f"Question {i+1} Breakdown"):
                st.write(f"**Q:** {st.session_state.questions[i]}")
                st.write(f"**A:** {st.session_state.answers[i]}")
                st.success(f"Score: {st.session_state.results[i]['score']}/10")
                st.write(f"Feedback: {st.session_state.results[i]['feedback']}")
                st.write(f"Confidence Score: {st.session_state.results[i]['confidence_score']}/100")
                if st.session_state.results[i].get('cheating_detected'):
                    st.warning("Cheating analysis flagged a possible issue with camera presence.")
                if st.session_state.results[i].get('confidence_feedback'):
                    st.info(st.session_state.results[i]['confidence_feedback'])
        if st.button("New Audit", key="new_audit"):
            reset_interview_state()
            st.rerun()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_rankings():
    """Cached DB query for rankings — refreshes every 30 seconds."""
    conn = sqlite3.connect('career_app.db')
    df = pd.read_sql_query("SELECT u.username, ROUND(AVG(h.score), 2) as avg FROM users u JOIN history h ON u.id = h.user_id GROUP BY u.username ORDER BY avg DESC", conn)
    conn.close()
    return df

def render_rankings():
    section = st.container()
    with section:
        st.header("Global Rankings")
        df = _fetch_rankings()
        st.table(df)


def render_application():
    if client is None:
        st.error("Groq API key is missing. Set GROQ_API_KEY in environment variables.")
        return
    if st.session_state.user_id is None:
        render_login_register()
        return
    with st.sidebar:
        st.header(f"{st.session_state.username}")
        if st.button("Logout", key="logout"):
            reset_user_session()
            initialize_state()
            st.rerun()
    tab1, tab2 = st.tabs(["Interview Engine", "Rankings"])
    with tab1:
        if not st.session_state.started and not st.session_state.final_report:
            render_resume_audit()
        elif st.session_state.started and st.session_state.q_index < 5:
            render_interview_stage()
        else:
            render_report_stage()
    with tab2:
        render_rankings()


def main():
    init_db()
    st.set_page_config(page_title="Ultimate AI Career Architect", layout="wide")
    initialize_state()
    render_application()


if __name__ == '__main__':
    main()
