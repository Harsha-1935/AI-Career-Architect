import streamlit as st
from loader_utils import (
    inject_master_css, show_loader,
    show_overlay_loader, hide_loader, show_skeleton,
    glass_card, animated_title, progress_score_ring,
    transition_placeholder, LoaderContext
)
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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import urllib.request

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
        # Startup delay to ensure Streamlit is settled
        time.sleep(2.0)
        
        cap = None
        # Use the exact same logic that worked in diagnostics
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
            print("CRITICAL: Camera acquisition failed after settling.")
            with self.lock:
                self.active = False
            return
        
        _ACTIVE_CAPS.append(cap)

        # Initialize thread-local detectors
        try:
            mp_landmarker = _get_mp_face_landmarker()
            mp_legacy = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        except Exception as e:
            print(f"MediaPipe Init Error: {e}")
            mp_landmarker = None
            mp_legacy = None

        try:
            last_save_time = 0
            while not self.stop_event.is_set():
                try:
                    with self.lock:
                        if time.time() - self.last_ping > 10:
                            break

                    ret, frame = cap.read()
                    if ret:
                        # --- REAL-TIME FEED WITH VISUAL FEEDBACK ---
                        display_frame = frame.copy()
                        
                        # Use Haar Cascade for fast real-time feedback rectangles
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        with self.lock:
                            self.live_frame = display_frame
                        
                        # Periodic deep analysis
                        now_time = time.time()
                        if now_time - last_save_time >= 5:
                            now = datetime.now()
                            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                            display_time = now.strftime("%H:%M:%S")
                            filename = f"user_{timestamp}.jpg"
                            path = os.path.join(self.folder, filename)
                            
                            # Save the RAW frame
                            cv2.imwrite(path, frame)
                            
                            # Deep analysis
                            conf_data = self._analyze_in_thread(frame, mp_landmarker, mp_legacy)
                            
                            with self.lock:
                                self.latest_frame_path = path
                                self.capture_history.append((display_time, path, conf_data))
                                if conf_data.get('looking_away'):
                                    self.look_away_count = getattr(self, 'look_away_count', 0) + 1
                                else:
                                    self.look_away_count = 0
                                if len(self.capture_history) > 5:
                                    self.capture_history.pop(0)
                            last_save_time = now_time
                except Exception as e:
                    print(f"ERROR in ProctorManager Loop: {e}")
                time.sleep(0.05)
        except Exception as e:
            print(f"CRITICAL ERROR in ProctorManager Thread: {e}")
        finally:
            if cap in _ACTIVE_CAPS:
                _ACTIVE_CAPS.remove(cap)
            if cap:
                cap.release()
            try:
                mp_legacy.close()
            except: pass
            with self.lock:
                self.live_frame = None
                self.active = False

    def _analyze_in_thread(self, frame, landmarker, legacy):
        """Thread-safe analysis using provided detector instances."""
        # 1. Try Landmarker
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            num_f = len(res.face_landmarks) if res.face_landmarks else 0
            if num_f > 0:
                # Basic gaze logic...
                return {'confidence_score': 100 if num_f == 1 else 20, 'cheating_detected': num_f != 1, 'faces_detected': num_f, 'multiple_faces': num_f > 1, 'looking_away': False}
        except: pass

        # 2. Try Legacy
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = legacy.process(rgb)
            if res.detections:
                num_f = len(res.detections)
                return {'confidence_score': 90 if num_f == 1 else 20, 'cheating_detected': num_f != 1, 'faces_detected': num_f, 'multiple_faces': num_f > 1, 'looking_away': False}
        except: pass

        # 3. Try Haar
        return analyze_camera_image(frame, is_frame=True)

    def capture_baseline(self):
        """Captures the current live frame as the baseline identity. Retries if camera is warming up."""
        for _ in range(10): # Try for up to 5 seconds
            with self.lock:
                if self.live_frame is not None:
                    baseline_path = os.path.join(self.folder, "baseline_identity.jpg")
                    cv2.imwrite(baseline_path, self.live_frame)
                    return baseline_path
            time.sleep(0.5)
        return None

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
        with self.lock:
            if not self.active or self.thread is None or not self.thread.is_alive():
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

# 👁️ MEDIAPIPE TASKS (NEW API) SETUP
MODEL_PATH = "face_landmarker.task"

def _download_mediapipe_model():
    """Downloads the required MediaPipe Face Landmarker model if missing."""
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        try:
            # Use a slightly longer timeout for downloads
            with urllib.request.urlopen(url, timeout=30) as response:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.read())
        except Exception as e:
            st.error(f"Failed to download MediaPipe model: {e}")

@st.cache_resource
def _get_mp_face_landmarker():
    _download_mediapipe_model()
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=4  # Detect multiple faces for cheating detection
        )
        return vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        st.warning(f"MediaPipe Tasks Initialization Failed: {e}")
        return None

# We'll initialize this per-thread or cautiously due to MediaPipe Tasks threading rules
FACE_DETECTOR = _get_mp_face_landmarker()

@st.cache_resource
def _get_legacy_face_detector():
    """Legacy MediaPipe face detector — extremely robust fallback."""
    try:
        return mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    except:
        return None

LEGACY_DETECTOR = _get_legacy_face_detector()

def detect_eye_gaze(frame):
    """Detects if the user is looking away and counts faces using MediaPipe."""
    if FACE_DETECTOR is None:
        return False, 0, 0
        
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = FACE_DETECTOR.detect(mp_image)
        
        num_faces = len(result.face_landmarks) if result.face_landmarks else 0
        if num_faces == 0:
            return False, 0, 0
            
        # Analyze the first face for gaze
        landmarks = result.face_landmarks[0]
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        dist_l = math.sqrt((nose_tip.x - left_eye.x)**2 + (nose_tip.y - left_eye.y)**2)
        dist_r = math.sqrt((nose_tip.x - right_eye.x)**2 + (nose_tip.y - right_eye.y)**2)
        yaw_ratio = dist_l / dist_r if dist_r != 0 else 1
        
        # Iris landmarks 468, 473
        l_iris = landmarks[468]
        r_iris = landmarks[473]
        l_outer = landmarks[33]
        l_inner = landmarks[133]
        r_inner = landmarks[362]
        r_outer = landmarks[263]
        
        def get_ratio(iris, inner, outer):
            total = abs(inner.x - outer.x)
            return abs(iris.x - outer.x) / total if total != 0 else 0.5
            
        avg_eye_ratio = (get_ratio(l_iris, l_inner, l_outer) + get_ratio(r_iris, r_inner, r_outer)) / 2
        is_looking_away = (yaw_ratio < 0.6 or yaw_ratio > 1.6) or (avg_eye_ratio < 0.35 or avg_eye_ratio > 0.65)
        
        return is_looking_away, avg_eye_ratio, num_faces
        
    except Exception:
        return False, 0, 0

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
        # More sensitive Haar parameters (1.1, 3 instead of 1.3, 5)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 3)

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
            'multiple_faces': len(faces) > 1,
            'looking_away': False # Default
        }

    except Exception:
        return {
            'confidence_score': 50,
            'cheating_detected': False,
            'faces_detected': 1,
            'multiple_faces': False,
            'looking_away': False
        }


def analyze_security_violation(frame):
    """Combined analysis for face presence, multiple faces, and eye tracking.
    Uses a 3-layer detection strategy: Landmarker -> Legacy MP -> Haar."""
    
    # LAYER 1: MediaPipe Landmarker (Best for gaze + presence)
    looking_away, eye_ratio, mp_faces = detect_eye_gaze(frame)
    
    if mp_faces > 0:
        base_analysis = {
            'confidence_score': 100 if mp_faces == 1 else 20,
            'cheating_detected': mp_faces != 1,
            'faces_detected': mp_faces,
            'multiple_faces': mp_faces > 1,
            'looking_away': looking_away
        }
        if looking_away:
            base_analysis['confidence_score'] = max(0, base_analysis['confidence_score'] - 30)
        return base_analysis

    # LAYER 2: Legacy MediaPipe Face Detection (Extremely robust for presence)
    if LEGACY_DETECTOR:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = LEGACY_DETECTOR.process(rgb_frame)
            if results.detections:
                num_faces = len(results.detections)
                return {
                    'confidence_score': 90 if num_faces == 1 else 20,
                    'cheating_detected': num_faces != 1,
                    'faces_detected': num_faces,
                    'multiple_faces': num_faces > 1,
                    'looking_away': False
                }
        except:
            pass
            
    # LAYER 3: Haar Cascade (Final fallback)
    return analyze_camera_image(frame, is_frame=True)




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
        'security_warnings': 0,
        'last_violation_check': "", 
        'security_warning_active': False,
        'violation_message': "",
        'waiting_for_baseline': False,
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
    st.session_state.security_warnings = 0
    st.session_state.last_violation_check = ""
    st.session_state.security_warning_active = False
    st.session_state.violation_message = ""
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
    st.caption("AI-Powered Interview Coaching & Career Intelligence")
    login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

    with login_tab:
        with st.form(key="login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Login")
        if login_submit:
            with st.spinner("Authenticating…"):
                conn = sqlite3.connect('career_app.db')
                c = conn.cursor()
                c.execute('SELECT id, username, password FROM users WHERE username=?', (login_username,))
                res = c.fetchone()
                conn.close()
            if res and make_hashes(login_password) == res[2]:
                st.session_state.user_id = res[0]
                st.session_state.username = res[1]
                st.rerun()
            else:
                st.error("Invalid username or password")

    with register_tab:
        with st.form(key="register_form"):
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
                with st.spinner("Creating account…"):
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
        st.header("📄 Resume Intelligence Audit")
        resume_file = st.file_uploader("Upload PDF", type=["pdf"], key="resume_upload")

        if resume_file and not st.session_state.career_analysis:
            analysis_ph = st.empty()
            with analysis_ph.container():
                show_loader("Analyzing your resume…", style="hologram", submessage="AI is extracting skills, strengths, and career paths")
                show_skeleton(rows=5)
            with pdfplumber.open(resume_file) as pdf:
                txt = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])[:2000]
            prompt = (
                f"Analyze this resume: {txt}. \n"
                "Return a STRICTLY VALID JSON object with these keys: 'skills', 'strengths', 'technical_gaps', 'roles'. \n"
                "IMPORTANT: Use JSON arrays (square brackets []) for all values. Do not use curly braces for lists."
            )
            try:
                res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
                st.session_state.career_analysis = json.loads(res.choices[0].message.content)
                analysis_ph.empty()
                st.rerun()
            except Exception as e:
                analysis_ph.empty()
                st.error(f"AI Analysis Error: {e}. Please ensure your resume text is clear and try again.")

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
                st.info("Identity verification is required to start the interview.")
            
            if not st.session_state.waiting_for_baseline:
                if st.button("Proceed to Identity Check", key="start_identity_check", width="stretch"):
                    st.session_state.waiting_for_baseline = True
                    st.rerun()
            else:
                st.divider()
                st.subheader("👤 Step 2: Identity Verification")
                st.info("Please look at the camera and take a clear photo of yourself. This will be used as your identity baseline for the session.")
                
                # Use browser-native camera input for the baseline to ensure user clicks "capture"
                captured_image = st.camera_input("Capture Baseline Photo", key="baseline_capture_input")
                
                if captured_image:
                    # 1. Save the captured image as the baseline
                    pm = get_proctor_manager()
                    baseline_path = os.path.join(pm.folder, "baseline_identity.jpg")
                    with open(baseline_path, "wb") as f:
                        f.write(captured_image.getbuffer())
                    
                    # 2. Initialize the Interview Session
                    st.session_state.started = True
                    st.session_state.waiting_for_baseline = False
                    st.session_state.interview_paused = False
                    st.session_state.interview_terminated = False
                    st.session_state.q_index = 0
                    st.session_state.questions = []
                    st.session_state.answers = []
                    st.session_state.results = []
                    st.session_state.last_spoken_q_index = -1
                    st.session_state.baseline_image = baseline_path
                    st.session_state.initial_faces = 1 
                    
                    # 3. Start the background Python Proctor
                    pm.start()
                    
                    st.rerun()

                if st.button("Cancel", key="cancel_baseline"):
                    st.session_state.waiting_for_baseline = False
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

    # --- NEW: PROMINENT FACE COUNT METRIC ---
    analysis = pm.get_latest_analysis()
    faces = analysis.get('faces_detected', 0) if analysis else 0
    st.metric("👥 Persons Detected", faces)
    st.divider()

    # --- SECTION 1: LIVE CAMERA FEED ---
    st.caption("🎥 LIVE CAMERA FEED")
    frame = pm.get_live_frame()
    if frame is not None:
        st.image(frame, width="stretch")
    else:
        st.error("Camera Hardware Locked or Not Found")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Quick Retry"):
                pm.stop()
                pm.start()
                st.rerun()
        with c2:
            if st.button("⚠️ Hard Reset"):
                # Use a specific session state flag to trigger a system-level reset on the next run
                st.session_state.needs_hard_reset = True
                st.rerun()

    if st.session_state.get('needs_hard_reset'):
        st.warning("Performing system-level hardware reset... Please wait.")
        # We can't easily kill ourselves from within, but we can try to release everything
        pm.stop()
        st.session_state.needs_hard_reset = False
        st.info("Reset requested. If camera still fails, please manually close and reopen your browser.")
    
    # --- SECTION 2: LATEST SNAPSHOT ---
    st.divider()
    col_base, col_snap = st.columns(2)
    with col_base:
        st.caption("👤 IDENTITY BASELINE")
        if "baseline_image" in st.session_state and st.session_state.baseline_image:
            st.image(st.session_state.baseline_image, width="stretch")
        else:
            st.info("No baseline")
            
    with col_snap:
        st.caption("📸 LATEST CHECK")
        if pm.latest_frame_path and os.path.exists(pm.latest_frame_path):
            st.image(pm.latest_frame_path, width="stretch")
        else:
            st.info("Waiting...")

    # Update confidence data in session state for UI display
    analysis = pm.get_latest_analysis()
    if analysis:
        st.session_state.current_confidence_data = analysis
        
        # Unique ID for the current analysis frame to avoid double counting
        analysis_id = analysis.get('timestamp', str(time.time()))
        
        # 🚨 IDENTITY INTEGRITY CHECK
        detected = analysis.get('faces_detected', 0)
        looking_away = getattr(get_proctor_manager(), 'look_away_count', 0) >= 3
        
        is_violation = (detected > 1) or looking_away
        
        if is_violation and st.session_state.last_violation_check != analysis_id:
            st.session_state.security_warnings += 1
            st.session_state.last_violation_check = analysis_id
            
            if st.session_state.security_warnings >= 3:
                st.session_state.interview_terminated = True
                reason = "Multiple persons detected" if detected > 1 else "Looking away from screen"
                st.session_state.termination_reason = f"Security Alert: Repeated violations ({reason})"
                st.rerun()
            else:
                st.session_state.security_warning_active = True
                st.session_state.violation_message = "Multiple persons detected!" if detected > 1 else "Please stay focused on the screen!"
                st.rerun()


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
        
    # 🚨 INTERACTIVE SECURITY WARNING
    if st.session_state.get('security_warning_active'):
        st.html(f"""
            <div style="background:#331111; padding:20px; border-radius:10px; border:2px solid #ff4b4b; margin-bottom:20px;">
                <h2 style="color:#ff4b4b; margin-top:0;">⚠️ SECURITY WARNING {st.session_state.security_warnings}/3</h2>
                <div style="font-size:18px; color:#e8e8f0;">{st.session_state.violation_message}</div>
                <div style="color:#888; margin-top:8px;">The interview has been paused for security reasons. Please ensure you are alone and facing the camera before resuming.</div>
            </div>
        """)
        if st.button("✅ I Understand - Resume Interview"):
            st.session_state.security_warning_active = False
            # Also reset the proctor look-away count to give them a fresh start for the next attempt
            pm_instance = get_proctor_manager()
            with pm_instance.lock:
                pm_instance.look_away_count = 0
            st.rerun()
        return

    if len(st.session_state.questions) <= st.session_state.q_index:
        q_gen_ph = st.empty()
        with q_gen_ph.container():
            show_loader(f"Generating Question {st.session_state.q_index + 1} of 5…", style="neural", submessage="AI interviewer is preparing your next challenge")
        q_prompt = f"You are an interviewer. Ask ONE short, focused interview question (max 2-3 sentences) for the role: {st.session_state.selected_role}. Do NOT include preamble, context, or instructions. Just the question."
        q_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": q_prompt}])
        st.session_state.questions.append(q_res.choices[0].message.content.strip())
        q_gen_ph.empty()
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
            
            if conf_data.get('cheating_detected') or faces > 1:
                st.error(f"⚠️ Face Error: {conf_data.get('faces_detected', 0)} detected")
            elif conf_data.get('looking_away'):
                st.warning("👀 Warning: Looking away from screen")
            else:
                st.success(f"✅ Secure Session: {conf_data.get('confidence_score', 0)}%")
            st.caption("Tracking presence, gaze, and environment via Python/OpenCV.")
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
                    transcribe_ph = st.empty()
                    with transcribe_ph.container():
                        show_loader("Transcribing audio…", style="mic", submessage="Converting speech to text")
                    transcript = transcribe_audio(audio_bytes)
                    transcribe_ph.empty()
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

        eval_ph = st.empty()
        with eval_ph.container():
            show_loader("Evaluating your answer…", style="scan", submessage="AI is scoring your response against industry standards")

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
        eval_ph.empty()
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
            report_ph = st.empty()
            
            # Phase 1: Analyzing gaps
            with report_ph.container():
                show_loader("Analyzing interview performance…", style="hologram", submessage="Identifying skill gaps from your responses")
                show_skeleton(rows=6)

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
            
            # Phase 2: Building roadmap
            report_ph.empty()
            with report_ph.container():
                show_loader("Building your growth roadmap…", style="neural", submessage="Creating a personalized 7-day action plan")

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
            report_ph.empty()
            rep = st.session_state.final_report
        
        st.title(f"🏆 Final Performance Rating")
        progress_score_ring(rep['avg'], 10)
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
        if st.button("🔄 New Audit", key="new_audit", width="stretch"):
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
        st.header("🏅 Global Rankings")
        rank_ph = st.empty()
        with rank_ph.container():
            show_loader("Loading leaderboard…", style="dots")
        df = _fetch_rankings()
        rank_ph.empty()
        if df.empty:
            st.info("No rankings yet. Complete an interview to appear on the leaderboard!")
        else:
            st.table(df)


def render_application():
    if client is None:
        st.error("Groq API key is missing. Set GROQ_API_KEY in environment variables.")
        return
    if st.session_state.user_id is None:
        render_login_register()
        return
    with st.sidebar:
        st.header(f"👤 {st.session_state.username}")
        if st.button("🚪 Logout", key="logout"):
            reset_user_session()
            initialize_state()
            st.rerun()
    tab1, tab2 = st.tabs(["🎯 Interview Engine", "🏅 Rankings"])
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
    inject_master_css()
    initialize_state()
    render_application()


if __name__ == '__main__':
    main()
