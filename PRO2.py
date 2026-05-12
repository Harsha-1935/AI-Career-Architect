import streamlit as st
from loader_utils import (
    inject_master_css, show_loader,
    show_overlay_loader, hide_loader, show_skeleton,
    glass_card, animated_title, progress_score_ring,
    transition_placeholder, LoaderContext
)
from emotion_analyzer import (
    run_full_analysis, analyze_facial_emotion,
    analyze_voice_tone, analyze_hesitation,
    analyze_speaking_speed, compute_overall_confidence
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
import ast
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
                        # Increased minNeighbors to 7 and minSize to (80, 80) to eliminate shirt collar phantom faces
                        faces = FACE_CASCADE.detectMultiScale(gray, 1.15, 7, minSize=(80, 80))
                        for (x, y, w, h) in faces:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        now_time = time.time()
                        with self.lock:
                            self.live_frame = display_frame
                            if len(faces) > 0:
                                self.last_face_seen = now_time
                            elif not hasattr(self, 'last_face_seen'):
                                self.last_face_seen = now_time
                            
                            self.is_missing_face = (now_time - self.last_face_seen) >= 5.0
                        
                        # Periodic deep analysis
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
                            conf_data['timestamp'] = timestamp
                            
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
        h, w = frame.shape[:2]
        min_face_area = (h * w) * 0.03  # Reduced to 3% to support candidates sitting naturally back from webcams

        # 1. Try Landmarker
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if res.face_landmarks:
                # Filter out small/phantom faces and remove overlapping detections
                face_boxes = []
                valid_faces = []
                for lm in res.face_landmarks:
                    xs = [p.x for p in lm]
                    ys = [p.y for p in lm]
                    fx, fy = min(xs) * w, min(ys) * h
                    fw = (max(xs) - min(xs)) * w
                    fh = (max(ys) - min(ys)) * h
                    if fw * fh < min_face_area:
                        continue
                    # Check overlap with already accepted faces
                    is_duplicate = False
                    for (bx, by, bw, bh) in face_boxes:
                        overlap_x = max(0, min(fx+fw, bx+bw) - max(fx, bx))
                        overlap_y = max(0, min(fy+fh, by+bh) - max(fy, by))
                        overlap_area = overlap_x * overlap_y
                        if overlap_area > 0.3 * min(fw*fh, bw*bh):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        valid_faces.append(lm)
                        face_boxes.append((fx, fy, fw, fh))
                num_f = len(valid_faces)
                if num_f > 0:
                    emotion_data = analyze_facial_emotion(valid_faces[0])
                    return {
                        'confidence_score': 100 if num_f == 1 else 20,
                        'cheating_detected': num_f != 1,
                        'faces_detected': num_f,
                        'multiple_faces': num_f > 1,
                        'looking_away': False,
                        'facial_emotion': emotion_data
                    }
        except: pass

        # 2. Try Legacy — also filter by face size
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = legacy.process(rgb)
            if res.detections:
                valid_count = 0
                for det in res.detections:
                    bbox = det.location_data.relative_bounding_box
                    bw = bbox.width * w
                    bh = bbox.height * h
                    if bw * bh >= min_face_area:
                        valid_count += 1
                if valid_count > 0:
                    return {'confidence_score': 90 if valid_count == 1 else 20, 'cheating_detected': valid_count != 1, 'faces_detected': valid_count, 'multiple_faces': valid_count > 1, 'looking_away': False, 'facial_emotion': {'emotion': 'Unknown', 'emotion_score': 50}}
        except: pass

        # 3. Try Haar
        haar_result = analyze_camera_image(frame, is_frame=True)
        haar_result['facial_emotion'] = {'emotion': 'Unknown', 'emotion_score': 50}
        return haar_result

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
                res = dict(self.capture_history[-1][2])
                res['is_missing_face'] = getattr(self, 'is_missing_face', False)
                return res
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
        
        if not result.face_landmarks:
            return False, 0, 0
        
        # Filter out small/phantom faces (must be at least 8% of frame)
        h, w = frame.shape[:2]
        min_face_area = (h * w) * 0.08
        face_boxes = []
        valid_faces = []
        for lm in result.face_landmarks:
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            fx, fy = min(xs) * w, min(ys) * h
            fw = (max(xs) - min(xs)) * w
            fh = (max(ys) - min(ys)) * h
            if fw * fh < min_face_area:
                continue
            # Check overlap with already accepted faces
            is_duplicate = False
            for (bx, by, bw, bh) in face_boxes:
                overlap_x = max(0, min(fx+fw, bx+bw) - max(fx, bx))
                overlap_y = max(0, min(fy+fh, by+bh) - max(fy, by))
                overlap_area = overlap_x * overlap_y
                if overlap_area > 0.3 * min(fw*fh, bw*bh):
                    is_duplicate = True
                    break
            if not is_duplicate:
                valid_faces.append(lm)
                face_boxes.append((fx, fy, fw, fh))
        
        num_faces = len(valid_faces)
        if num_faces == 0:
            return False, 0, 0
            
        # Analyze the first valid face for gaze
        landmarks = valid_faces[0]
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
        # Widened thresholds for head-pose (yaw) and eye-gaze to reduce false positives
        # Yaw: 0.4 - 2.5 (more generous head rotation)
        # Eye Ratio: 0.25 - 0.75 (accommodates glasses and subtle eye shifts better)
        is_looking_away = (yaw_ratio < 0.4 or yaw_ratio > 2.5) or (avg_eye_ratio < 0.25 or avg_eye_ratio > 0.75)
        
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


def transcribe_audio(audio_bytes, filename="recording.wav"):
    """Transcribes audio bytes using Groq's Whisper API."""
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(filename, audio_bytes),
            language="en",
        )
        return transcription.text.strip()
    except Exception as e:
        print(f"Groq Transcription Error: {e}")
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
        
    # Build a normalized mapping of lowercased keys to original keys
    lower_map = {str(k).strip().lower(): k for k in data.keys()}
    
    # Expand keys to check common case variations or substrings
    expanded_keys = []
    for k in keys:
        expanded_keys.append(k.lower())
        if k == 'roles':
            expanded_keys.extend(['role', 'career_paths', 'paths', 'suggested_roles', 'suggested_paths'])
        elif k == 'strengths':
            expanded_keys.extend(['strength', 'key_strengths'])
        elif k == 'skills':
            expanded_keys.extend(['skill', 'core_skills'])
        elif k == 'technical_gaps' or k == 'gaps':
            expanded_keys.extend(['technical_gap', 'gap', 'areas_for_improvement', 'weaknesses', 'improvements', 'technical_gaps'])
            
    # Remove duplicates preserving order
    seen = set()
    unique_expanded = [x for x in expanded_keys if not (x in seen or seen.add(x))]
            
    for lk in unique_expanded:
        # Find if any lowercased data key matches lk or contains lk
        matched_orig_key = None
        if lk in lower_map:
            matched_orig_key = lower_map[lk]
        else:
            # Substring match fallback
            for d_lower, orig in lower_map.items():
                if lk in d_lower or d_lower in lk:
                    matched_orig_key = orig
                    break
                    
        if matched_orig_key and data[matched_orig_key]:
            raw = data[matched_orig_key]
            
            if isinstance(raw, dict):
                sub_list = []
                for v in raw.values():
                    if isinstance(v, list):
                        sub_list.extend(v)
                    elif isinstance(v, str):
                        sub_list.append(v)
                raw = sub_list
                
            if isinstance(raw, str):
                try:
                    if raw.strip().startswith('['):
                        raw = ast.literal_eval(raw.strip())
                    else:
                        raw = [i.strip() for i in raw.split(',') if i.strip()]
                except:
                    raw = [raw]
            
            if isinstance(raw, list):
                flat_items = []
                for item in raw:
                    if isinstance(item, dict):
                        vals = list(item.values())
                        if vals and isinstance(vals[0], list):
                            flat_items.extend([str(vi) for vi in vals[0]])
                        elif vals:
                            flat_items.append(str(vals[0]))
                    elif isinstance(item, str) and item.strip().startswith('['):
                        try:
                            parsed = ast.literal_eval(item.strip())
                            if isinstance(parsed, list):
                                flat_items.extend([str(p) for p in parsed])
                            else:
                                flat_items.append(item)
                        except:
                            flat_items.append(item)
                    else:
                        flat_items.append(str(item))
                
                res = [i.strip() for i in flat_items if i.strip() and not i.strip().isdigit()]
                if res:
                    return res
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
        # Haar parameters: scaleFactor=1.15, minNeighbors=7, robust minSize filters out shirt folds/shadows
        min_dim = max(80, int(min(img.shape[0], img.shape[1]) * 0.12))
        faces = FACE_CASCADE.detectMultiScale(gray, 1.15, 7, minSize=(min_dim, min_dim))

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
            h, w = frame.shape[:2]
            min_face_area = (h * w) * 0.08
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = LEGACY_DETECTOR.process(rgb_frame)
            if results.detections:
                valid_count = 0
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    bw = bbox.width * w
                    bh = bbox.height * h
                    if bw * bh >= min_face_area:
                        valid_count += 1
                if valid_count > 0:
                    return {
                        'confidence_score': 90 if valid_count == 1 else 20,
                        'cheating_detected': valid_count != 1,
                        'faces_detected': valid_count,
                        'multiple_faces': valid_count > 1,
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
        'violation_detail': "",
        'violation_evidence_path': None,
        'waiting_for_baseline': False,
        'emotion_analyses': [],  # per-question emotion/confidence analysis
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
    st.session_state.emotion_analyses = []
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

        # Track file changes to reset cached analysis when user changes or clears the file
        current_filename = resume_file.name if resume_file else None
        if current_filename != st.session_state.get('last_uploaded_filename'):
            st.session_state.career_analysis = None
            st.session_state.last_uploaded_filename = current_filename
            if not resume_file:
                st.rerun()

        if resume_file and not st.session_state.career_analysis:
            analysis_ph = st.empty()
            with analysis_ph.container():
                show_loader("Analyzing your resume…", style="hologram", submessage="AI is extracting skills, strengths, and career paths")
                show_skeleton(rows=5)
            with pdfplumber.open(resume_file) as pdf:
                txt = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])[:2000]
            
            # Real resumes have substantial text. If under 50 characters, reject as blank/scanned.
            if len(txt.strip()) < 50:
                analysis_ph.empty()
                st.error("⚠️ No meaningful readable text could be extracted from this PDF. Please ensure the file is not blank or a scanned image.")
            else:
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
                strengths_list = clean_ai_list(data, ['strengths'])
                if not strengths_list:
                    strengths_list = ["Solid academic foundation in core engineering principles", "Hands-on exposure to circuit design and embedded software", "Demonstrated problem-solving capabilities"]
                for s in strengths_list:
                    st.write(f"- {s}")
                st.warning("**Technical Gaps Identified**")
                gaps_list = clean_ai_list(data, ['technical_gaps', 'gaps'])
                if not gaps_list:
                    gaps_list = ["Requires deeper production-grade firmware optimization experience", "Limited exposure to high-speed PCB layouts and advanced signal integrity"]
                for g in gaps_list:
                    st.write(f"- {g}")
                roles_list = clean_ai_list(data, ['roles'])
                if not roles_list:
                    roles_list = ["Embedded Systems Engineer", "Hardware Design Engineer", "Firmware Developer", "Systems Architect"]
                selected_role = st.selectbox("Path:", roles_list, key="selected_role")
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
    if frame is not None and frame.size > 0:
        # Use HTML data URI for maximum stability across different Streamlit versions/environments
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode()
        st.html(f'<img src="data:image/jpeg;base64,{img_base64}" style="width:100%; border-radius:5px; border:1px solid #333;">')
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
        # Increased look-away streak to 5 for higher accuracy (less sensitive to momentary blinks/glitches)
        looking_away = getattr(get_proctor_manager(), 'look_away_count', 0) >= 5
        
        # Debounce multi-face: require 5+ consecutive detections to avoid false positives (e.g. background objects)
        if detected > 1:
            st.session_state['_multi_face_streak'] = st.session_state.get('_multi_face_streak', 0) + 1
        else:
            st.session_state['_multi_face_streak'] = 0
        
        is_multi_face = st.session_state.get('_multi_face_streak', 0) >= 5
        is_missing_face = analysis.get('is_missing_face', False)
        
        is_violation = is_multi_face or looking_away or is_missing_face
        
        if is_violation and st.session_state.last_violation_check != analysis_id:
            st.session_state.security_warnings += 1
            st.session_state.last_violation_check = analysis_id
            
            # Build detailed reason
            if detected > 1:
                reason = f"Multiple persons detected ({detected} faces found)"
                detail = "Our camera detected more than one person in the frame. Only the registered candidate should be visible during the interview."
            elif detected == 0:
                reason = "No face detected"
                detail = "The camera could not detect your face. Please make sure you are clearly visible and facing the camera."
            else:
                reason = "Looking away from screen"
                detail = "You were detected looking away from the screen for an extended period. Please maintain focus on the interview."
            
            # Capture evidence frame
            evidence_path = None
            _pm = get_proctor_manager()
            with _pm.lock:
                if _pm.live_frame is not None:
                    evidence_path = os.path.join(_pm.folder, f"violation_{st.session_state.security_warnings}_{int(time.time())}.jpg")
                    cv2.imwrite(evidence_path, _pm.live_frame)
                elif _pm.latest_frame_path and os.path.exists(_pm.latest_frame_path):
                    evidence_path = _pm.latest_frame_path
            
            if st.session_state.security_warnings >= 3:
                st.session_state.interview_terminated = True
                st.session_state.termination_reason = f"Security Alert: Repeated violations ({reason})"
                st.session_state.violation_evidence_path = evidence_path
                st.rerun()
            else:
                st.session_state.security_warning_active = True
                st.session_state.violation_message = reason
                st.session_state.violation_detail = detail
                st.session_state.violation_evidence_path = evidence_path
                st.rerun()


def render_interview_stage():
    # Ensure Python Proctor is running
    pm = get_proctor_manager()
    
    # 🚨 SECURITY CHECK: TERMINATE IF VIOLATION DETECTED
    if st.session_state.interview_terminated and st.session_state.termination_reason:
        st.error(f"🚨 SECURITY VIOLATION: {st.session_state.termination_reason}. Interview terminated.")
        evidence = st.session_state.get('violation_evidence_path')
        if evidence and os.path.exists(evidence):
            st.image(evidence, caption="📸 Evidence: Frame captured at time of violation", width=400)
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
                <div style="font-size:20px; color:#ff6b6b; font-weight:600; margin:12px 0;">Reason: {st.session_state.violation_message}</div>
                <div style="font-size:15px; color:#ccc;">{st.session_state.get('violation_detail', '')}</div>
                <div style="color:#888; margin-top:12px; font-size:13px;">The interview has been paused for security reasons. Please ensure you are alone and facing the camera before resuming.</div>
            </div>
        """)
        # Show evidence frame
        evidence = st.session_state.get('violation_evidence_path')
        if evidence and os.path.exists(evidence):
            st.image(evidence, caption=f"📸 Captured at time of violation — Warning {st.session_state.security_warnings}/3", width=400)
        
        if st.button("✅ I Understand - Resume Interview"):
            st.session_state.security_warning_active = False
            # Also reset the proctor look-away count to give them a fresh start for the next attempt
            pm_instance = get_proctor_manager()
            with pm_instance.lock:
                pm_instance.look_away_count = 0
                pm_instance.last_face_seen = time.time()
                pm_instance.is_missing_face = False
            st.rerun()
        return

    if len(st.session_state.questions) <= st.session_state.q_index:
        q_gen_ph = st.empty()
        with q_gen_ph.container():
            show_loader(f"Generating Question {st.session_state.q_index + 1} of 5…", style="neural", submessage="AI interviewer is preparing your next challenge")
        history_ctx = ""
        if st.session_state.q_index > 0:
            parts = [f"Q: {st.session_state.questions[i]} | A: {st.session_state.answers[i]}" for i in range(st.session_state.q_index)]
            history_ctx = "\n\nContext of previous answers to adapt difficulty and follow-ups:\n" + "\n".join(parts)

        q_prompt = (
            f"You are a real senior interviewer conducting a realistic live interview "
            f"for the role: {st.session_state.selected_role}. "
            "Ask ONLY ONE question at a time. "
            "Be conversational, professional, and slightly challenging. "
            "Questions should feel like real company interviews. "
            "Mix technical, behavioral, and real-world scenario questions. "
            "Adapt difficulty based on previous answers. "
            "Ask natural follow-up questions when needed. "
            "Avoid robotic wording, explanations, or multiple questions. "
            "Keep questions concise but deep. "
            "Return ONLY the interview question."
            f"{history_ctx}"
        )

        q_res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": q_prompt}]
        )
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
            
            # Debounce error display to avoid flickering on frame drops
            if faces == 0:
                st.session_state['_zero_face_streak'] = st.session_state.get('_zero_face_streak', 0) + 1
            else:
                st.session_state['_zero_face_streak'] = 0
                
            if faces > 1 or st.session_state.get('_zero_face_streak', 0) >= 3:
                st.error(f"⚠️ Face Error: {faces} detected")
            elif conf_data.get('looking_away'):
                st.warning("👀 Warning: Looking away from screen")
            else:
                st.success(f"✅ Secure Session: {conf_data.get('confidence_score', 0)}%")
            
            # 🎭 Real-time Facial Emotion
            face_emo = conf_data.get('facial_emotion', {})
            if face_emo and face_emo.get('emotion', 'Unknown') != 'Unknown':
                emo_label = face_emo.get('emotion', 'Neutral')
                emo_score = face_emo.get('emotion_score', 50)
                emo_icons = {
                    'Happy / Confident': '😊', 'Focused / Serious': '🧐',
                    'Surprised': '😮', 'Speaking': '🗣️',
                    'Tired / Disengaged': '😴', 'Neutral': '😐'
                }
                icon = emo_icons.get(emo_label, '🎭')
                st.metric(f"{icon} Expression", emo_label, f"{emo_score}/100")

            st.caption("Tracking presence, gaze, emotion & environment via Python/OpenCV.")
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

    def handle_audio_change():
        q_idx = st.session_state.q_index
        mic_key = f"mic_{q_idx}"
        upload_key = f"upload_mic_{q_idx}"
        last_bytes_key = f"last_audio_bytes_{q_idx}"
        speech_k = f"speech_transcript_{q_idx}"
        ans_k = f"ans_{q_idx}"
        
        active_a = st.session_state.get(mic_key) or st.session_state.get(upload_key)
        if active_a:
            bytes_data = active_a.getvalue()
            if last_bytes_key not in st.session_state or st.session_state[last_bytes_key] != bytes_data:
                st.session_state[last_bytes_key] = bytes_data
                fname = getattr(active_a, 'name', 'recording.wav')
                transcript = transcribe_audio(bytes_data, filename=fname)
                if transcript:
                    st.session_state[speech_k] = transcript
                    st.session_state[ans_k] = transcript
                    r = st.session_state.get(f"ans_rev_{q_idx}", 0) + 1
                    st.session_state[f"ans_rev_{q_idx}"] = r
                    st.session_state[f"{ans_k}_rev_{r}"] = transcript
                    st.session_state[f"status_{q_idx}"] = f"✅ Transcribed successfully:\n\n**{transcript}**"
                else:
                    st.session_state[f"status_{q_idx}"] = f"⚠️ Could not transcribe audio. Please make sure the audio contains speech."

    with answer_holder.container():
        # 🎤 Audio recording for speech-to-text
        if not st.session_state.interview_terminated:
            audio_data = st.audio_input("🎤 Record your answer", key=f"mic_{st.session_state.q_index}", on_change=handle_audio_change)
            audio_file_upload = st.file_uploader("📂 Or upload audio file (if browser recorder fails)", type=["wav", "mp3", "m4a"], key=f"upload_mic_{st.session_state.q_index}", on_change=handle_audio_change)

        # Text area purely controlled by its session state key
        status_msg = st.session_state.get(f"status_{st.session_state.q_index}")
        if status_msg:
            if "✅" in status_msg:
                st.success(status_msg)
            else:
                st.warning(status_msg)
                
        rev = st.session_state.get(f"ans_rev_{st.session_state.q_index}", 0)
        widget_dynamic_key = f"{answer_key}_rev_{rev}"
        if widget_dynamic_key not in st.session_state:
            st.session_state[widget_dynamic_key] = st.session_state.get(answer_key, "")
            
        if st.session_state.interview_terminated:
            ans = st.text_area("Your Answer:", key=widget_dynamic_key, height=150, disabled=True)
        else:
            ans = st.text_area("Your Answer:", key=widget_dynamic_key, height=150,
                               help="Type your answer or record audio above — it will be transcribed automatically")
        st.session_state[answer_key] = ans

    with submit_holder.container():
        col_audio, col_submit = st.columns([1, 4])
        with col_audio:
            if st.button("🔊 Re-play"):
                speak_now(curr_q, f"manual_{st.session_state.q_index}", force=True)
        submit_button = col_submit.button("Submit Answer", key="submit_answer", disabled=st.session_state.interview_terminated)


    if submit_button and not ans.strip():
        st.warning("⚠️ Your answer is empty. Please type your answer or record audio using the 🎤 mic above.")

    if submit_button and ans.strip():
        # CRITICAL: Capture question and answer immediately to ensure they are saved
        current_question = st.session_state.questions[st.session_state.q_index]
        current_answer = ans
        
        question_holder.empty()
        answer_holder.empty()
        submit_holder.empty()

        eval_ph = st.empty()
        with eval_ph.container():
            show_loader("Analyzing emotion & evaluating…", style="scan", submessage="AI is scoring your response, voice, and confidence")

        # ── Pre-check: Detect non-answers BEFORE emotion analysis ──
        NON_ANSWER_PATTERNS = [
            'i dont know', 'i don\'t know', 'idk', 'no idea', 'not sure',
            'i have no idea', 'pass', 'skip', 'next', 'no answer',
            'i am not sure', 'i\'m not sure', 'can\'t answer', 'dont know',
            'no clue', 'beats me', 'i give up',
        ]
        ans_stripped = ans.strip().lower().rstrip('.!?,')
        ans_words = ans_stripped.split()
        
        is_non_answer = (
            len(ans_words) <= 5 and
            any(ans_stripped == p or ans_stripped.startswith(p) for p in NON_ANSWER_PATTERNS)
        )
        
        # Check for gibberish / random character strings
        if not is_non_answer:
            import re as _re
            alpha_only = _re.sub(r'[^a-z]', '', ans_stripped)
            if len(alpha_only) > 0:
                vowel_ratio = sum(1 for c in alpha_only if c in 'aeiou') / len(alpha_only)
                unique_chars = len(set(alpha_only))
                is_single_char_spam = unique_chars <= 2 and len(alpha_only) >= 3
                is_no_vowels = vowel_ratio < 0.08 and len(alpha_only) >= 4
                is_repetitive = unique_chars <= 3 and len(alpha_only) >= 6
                is_short_gibberish = (
                    len(ans_words) <= 2 and len(alpha_only) >= 3 and
                    vowel_ratio < 0.15
                )
                if is_single_char_spam or is_no_vowels or is_repetitive or is_short_gibberish:
                    is_non_answer = True

        # ── Emotion & Confidence Analysis ──
        if is_non_answer:
            # Non-answer: skip all analysis, zero everything out
            emotion_result = {
                'voice_tone': {'available': False, 'tone_label': 'Non-Answer', 'tone_score': None},
                'hesitation': {'available': False, 'hesitation_label': 'Non-Answer', 'hesitation_score': None, 'filler_count': 0, 'filler_words_found': []},
                'speaking_speed': {'available': False, 'speed_label': 'Non-Answer', 'speed_score': None, 'wpm': 0},
                'facial_emotion': {'available': False, 'emotion': 'Non-Answer', 'emotion_score': None},
                'overall': {'overall_score': None, 'overall_label': 'Non-Answer', 'available_metrics': 0, 'total_metrics': 4, 'breakdown': {}}
            }
        else:
            audio_key = f"last_audio_bytes_{st.session_state.q_index}"
            audio_bytes_for_analysis = st.session_state.get(audio_key)
            face_emo_data = confidence_data.get('facial_emotion')

            emotion_result = run_full_analysis(
                audio_bytes=audio_bytes_for_analysis,
                transcript=ans,
                face_landmarks=None
            )
            # Override emotion from live proctor data if available
            if face_emo_data and face_emo_data.get('emotion', 'Unknown') != 'Unknown':
                emotion_result['facial_emotion'] = face_emo_data
                emotion_result['overall'] = compute_overall_confidence(
                    emotion_result['voice_tone'],
                    emotion_result['hesitation'],
                    emotion_result['speaking_speed'],
                    face_emo_data
                )

        st.session_state.emotion_analyses.append(emotion_result)

        if is_non_answer:
            result = {
                'score': 0,
                'feedback': 'No substantive answer provided. You must attempt to answer the question to receive a score.'
            }
        else:
            # Build enriched prompt with emotion context — only include available metrics
            emo_parts = []
            vt = emotion_result['voice_tone']
            if vt.get('available'):
                emo_parts.append(f"Voice Tone: {vt['tone_label']} (score {vt['tone_score']}/100)")
            hes = emotion_result['hesitation']
            if hes.get('available'):
                emo_parts.append(f"Hesitation: {hes['hesitation_label']} ({hes['filler_count']} fillers)")
            spd = emotion_result['speaking_speed']
            if spd.get('available'):
                emo_parts.append(f"Speaking Speed: {spd['wpm']} WPM ({spd['speed_label']})")
            fe = emotion_result['facial_emotion']
            if fe.get('available'):
                emo_parts.append(f"Facial Expression: {fe['emotion']}")
            ov = emotion_result['overall']
            if ov.get('available_metrics', 0) > 0:
                emo_parts.append(f"Overall Confidence: {ov['overall_score']}/100 ({ov['overall_label']})")
            emo_ctx = ". ".join(emo_parts) + ". " if emo_parts else "No behavioral analysis data available. "

            ev_prompt = (
                f"Score this answer from 0 to 10. Q:{curr_q} | A:{ans}. "
                f"Camera Confidence Score: {confidence_data['confidence_score']}/100. "
                f"Cheating Detected: {confidence_data['cheating_detected']}. "
                f"Behavioral Analysis: {emo_ctx}"
                "Return JSON object with only 'score' and 'feedback', where 'score' is a number from 0 to 10. "
                "STRICT SCORING RULES: "
                "- If the answer is 'I don't know', empty, irrelevant, or a non-answer, score MUST be 0. "
                "- Vague one-liner answers with no substance must score 0-2. "
                "- Only answers demonstrating real knowledge deserve 3+. "
                "Factor in the candidate's voice confidence, hesitation, speaking pace, and facial expression when giving feedback. "
                "Example: {'score': 7, 'feedback': 'Good explanation with confident delivery'}"
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
        result['question'] = current_question
        result['answer'] = current_answer
        result['confidence_score'] = confidence_data['confidence_score']
        result['cheating_detected'] = confidence_data['cheating_detected']
        result['emotion_analysis'] = emotion_result
        if confidence_data['cheating_detected']:
            result['confidence_feedback'] = "Potential cheating detected - please keep only one face visible and remain focused on the camera."
        elif confidence_data['confidence_score'] >= 80:
            result['confidence_feedback'] = "High confidence detected from your camera pose and presence."
        elif confidence_data['confidence_score'] >= 50:
            result['confidence_feedback'] = "Moderate confidence detected. Try to keep your face centered and clearly visible."
        else:
            result['confidence_feedback'] = "Low confidence detected. Make sure you are visible and stay in frame."
        st.session_state.results.append(result)
        st.session_state.answers.append(current_answer)
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
            
            # Save to DB - include both per-question results and final report (roadmap/gaps)
            # We also save the full questions/answers lists for redundancy
            full_db_report = {
                "results": st.session_state.results,
                "final_report": st.session_state.final_report,
                "questions": st.session_state.questions,
                "answers": st.session_state.answers
            }
            conn = sqlite3.connect('career_app.db')
            c = conn.cursor()
            c.execute('INSERT INTO history (user_id, date, role, score, report) VALUES (?,?,?,?,?)', 
                      (st.session_state.user_id, datetime.now().strftime("%Y-%m-%d %H:%M"), 
                       st.session_state.selected_role or "Career Audit", avg, json.dumps(full_db_report)))
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
        
        # ── 🎭 EMOTION & CONFIDENCE ANALYSIS SUMMARY ──
        st.divider()
        st.subheader("🎭 Emotion & Confidence Analysis")
        
        emotion_data_list = st.session_state.get('emotion_analyses', [])
        if emotion_data_list:
            # Helper: safely average only available scores (ignore None)
            def _avg_available(data_list, path_fn):
                vals = []
                for e in data_list:
                    try:
                        v = path_fn(e)
                        if v is not None and v != 0: # Also ignore 0s for behavioral averages if they mean 'not captured'
                            vals.append(v)
                    except: continue
                return round(np.mean(vals), 1) if vals else None

            avg_voice = _avg_available(emotion_data_list, lambda e: e['voice_tone'].get('tone_score'))
            avg_hesitation = _avg_available(emotion_data_list, lambda e: e['hesitation'].get('hesitation_score'))
            avg_speed = _avg_available(emotion_data_list, lambda e: e['speaking_speed'].get('speed_score'))
            avg_emotion = _avg_available(emotion_data_list, lambda e: e['facial_emotion'].get('emotion_score'))
            avg_overall = _avg_available(emotion_data_list, lambda e: e['overall'].get('overall_score'))
            
            available_count = sum(1 for v in [avg_voice, avg_hesitation, avg_speed, avg_emotion] if v is not None)
            
            # Overall confidence ring
            display_overall = avg_overall if avg_overall is not None else 0
            st.html(f"""
                <div style="text-align:center; padding:20px; background:linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-radius:16px; margin-bottom:20px; border:1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:48px; font-weight:800; background:linear-gradient(135deg, #00d2ff, #7b2ff7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{display_overall}/100</div>
                    <div style="font-size:16px; color:#aaa; margin-top:4px;">Overall Behavioral Confidence Score ({available_count}/4 metrics available)</div>
                </div>
            """)
            
            # Four metric cards — show N/A when unavailable
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("🎤 Voice Tone", f"{avg_voice}/100" if avg_voice is not None else "N/A — no mic used")
            with m2:
                st.metric("💬 Fluency", f"{avg_hesitation}/100" if avg_hesitation is not None else "N/A")
            with m3:
                st.metric("⏱️ Speaking Speed", f"{avg_speed}/100" if avg_speed is not None else "N/A — no mic used")
            with m4:
                st.metric("🎭 Expression", f"{avg_emotion}/100" if avg_emotion is not None else "N/A")
            
            # Detailed per-question breakdown bar
            st.caption("Per-question breakdown:")
            for idx, emo in enumerate(emotion_data_list):
                ov = emo['overall']
                score = ov.get('overall_score')
                if score is not None and score > 0:
                    bar_val = max(0.01, score / 100)
                    st.progress(bar_val, text=f"Q{idx+1}: {ov.get('overall_label', 'N/A')} ({score}/100)")
                else:
                    st.progress(0.01, text=f"Q{idx+1}: No Data")
        else:
            st.info("No emotion analysis data available for this session.")
        
        # ── PER-QUESTION DETAILED BREAKDOWN ──
        st.divider()
        for i in range(5):
            with st.expander(f"📝 Question {i+1} Breakdown"):
                st.write(f"**Q:** {st.session_state.questions[i]}")
                st.write(f"**A:** {st.session_state.answers[i]}")
                st.success(f"Score: {st.session_state.results[i]['score']}/10")
                st.write(f"Feedback: {st.session_state.results[i]['feedback']}")
                st.write(f"Confidence Score: {st.session_state.results[i]['confidence_score']}/100")
                if st.session_state.results[i].get('cheating_detected'):
                    st.warning("Cheating analysis flagged a possible issue with camera presence.")
                if st.session_state.results[i].get('confidence_feedback'):
                    st.info(st.session_state.results[i]['confidence_feedback'])
                
                # Emotion details for this question
                emo = st.session_state.results[i].get('emotion_analysis')
                if emo:
                    st.divider()
                    st.caption("🎭 Behavioral Analysis")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        vt = emo.get('voice_tone', {})
                        if vt.get('available'):
                            st.markdown(f"**🎤 Voice Tone:** {vt.get('tone_label')} ({vt.get('tone_score')}/100)")
                        else:
                            st.markdown("**🎤 Voice Tone:** N/A — no mic used")
                        
                        hes = emo.get('hesitation', {})
                        if hes.get('available'):
                            st.markdown(f"**💬 Hesitation:** {hes.get('hesitation_label')} ({hes.get('hesitation_score')}/100)")
                            fillers = hes.get('filler_words_found', [])
                            if fillers:
                                st.caption(f"Fillers: {', '.join(fillers)}")
                        else:
                            st.markdown("**💬 Hesitation:** N/A")
                    with ec2:
                        spd = emo.get('speaking_speed', {})
                        if spd.get('available'):
                            st.markdown(f"**⏱️ Speed:** {spd.get('wpm')} WPM ({spd.get('speed_label')})")
                        else:
                            st.markdown("**⏱️ Speed:** N/A — no mic used")
                        
                        fe = emo.get('facial_emotion', {})
                        if fe.get('available'):
                            st.markdown(f"**🎭 Expression:** {fe.get('emotion')} ({fe.get('emotion_score')}/100)")
                        else:
                            st.markdown("**🎭 Expression:** N/A")
                        
                        ov = emo.get('overall', {})
                        avail_count = ov.get('available_metrics', 0)
                        if avail_count > 0:
                            st.markdown(f"**📊 Overall:** {ov.get('overall_score')}/100 ({ov.get('overall_label')}) — {avail_count}/4 metrics")
                        else:
                            st.markdown("**📊 Overall:** N/A")

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

def _fetch_user_history(user_id):
    """Fetch detailed interview history for the current user."""
    conn = sqlite3.connect('career_app.db')
    df = pd.read_sql_query(
        "SELECT date, role, score, report FROM history WHERE user_id = ? ORDER BY date DESC", 
        conn, params=(user_id,)
    )
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
            
        st.divider()
        st.header("🕒 My Interview History")
        if st.session_state.user_id:
            history_df = _fetch_user_history(st.session_state.user_id)
            if history_df.empty:
                st.info("You haven't completed any interviews yet.")
            else:
                for idx, row in history_df.iterrows():
                    role_display = row['role'] if row['role'] and str(row['role']) != 'nan' else "General Interview"
                    with st.expander(f"📅 {row['date']} - {role_display} (Score: {row['score']}/10)"):
                        try:
                            data = json.loads(row['report'])
                            # Handle both formats (new: dict with 'results'/'final_report', old: direct list)
                            if isinstance(data, dict):
                                results = data.get('results', [])
                                final_rep = data.get('final_report', {})
                            else:
                                results = data
                                final_rep = {}

                            # 1. Show Technical Gaps from that session
                            gaps = final_rep.get('gaps', [])
                            if gaps:
                                st.warning("**Technical Gaps Identified**")
                                for g in gaps:
                                    st.write(f"- {g}")
                            
                            # 2. Show Roadmap from that session
                            roadmap = final_rep.get('roadmap', {})
                            if roadmap:
                                with st.expander("📅 View 7-Day Growth Roadmap"):
                                    for day, task in roadmap.items():
                                        day_label = day.replace('day', 'Day ')
                                        st.write(f"**{day_label}:** {task}")
                            
                            st.divider()
                            st.subheader("📝 Question Feedback")
                            # Fallback lists if individual results are missing Q/A (only for newer dict-based records)
                            fallback_qs = []
                            fallback_as = []
                            if isinstance(data, dict):
                                fallback_qs = data.get('questions', [])
                                fallback_as = data.get('answers', [])
                            
                            for i, res in enumerate(results):
                                # Try to get Q/A from the result object first
                                q_text = res.get('question')
                                a_text = res.get('answer')
                                
                                # Fallback to top-level lists if missing
                                if not q_text and i < len(fallback_qs): q_text = fallback_qs[i]
                                if not a_text and i < len(fallback_as): a_text = fallback_as[i]
                                
                                st.markdown(f"**Question {i+1}:**")
                                st.write(f"**Q:** {q_text or '*Not captured for this older session*'}")
                                st.write(f"**A:** {a_text or '*Not captured for this older session*'}")
                                st.success(f"**Score:** {res.get('score', 'N/A')}/10")
                                st.write(f"**Feedback:** {res.get('feedback', 'No feedback provided.')}")
                                if i < len(results) - 1:
                                    st.divider()
                        except Exception:
                            st.error("Could not load detailed feedback for this session.")
        else:
            st.warning("Please log in to see your interview history.")


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
