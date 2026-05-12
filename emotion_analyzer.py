"""
Emotion & Confidence Analysis Module
=====================================
Provides real-time analysis of:
  1. Voice Tone (pitch variance, energy levels)
  2. Facial Emotion (from MediaPipe landmarks)
  3. Hesitation Analysis (filler words, repeated phrases)
  4. Speaking Speed (words per minute)

Each analyzer returns an 'available' flag — False means no real data
was provided, so the score should be ignored in aggregation and shown
as "N/A" in the UI.
"""

import numpy as np
import io
import wave
import re
import time

# ── Voice Tone Analysis ──────────────────────────────────────────────

def _read_wav_bytes(audio_bytes):
    """Decode WAV/WebM audio bytes into a numpy array + sample rate."""
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                samples = samples[::n_channels]
            return samples, sr
    except Exception:
        return None, None


def analyze_voice_tone(audio_bytes):
    """
    Analyze voice tone from audio bytes.
    Returns dict with pitch_mean, pitch_variance, energy, tone_label.
    """
    if not audio_bytes:
        return {
            'available': False,
            'pitch_mean': 0, 'pitch_variance': 0,
            'energy': 0, 'tone_label': 'No Audio',
            'tone_score': None
        }

    samples, sr = _read_wav_bytes(audio_bytes)
    if samples is None or len(samples) < sr * 0.5:
        return {
            'available': False,
            'pitch_mean': 0, 'pitch_variance': 0,
            'energy': 0, 'tone_label': 'No Audio',
            'tone_score': None
        }

    # Normalize
    samples = samples / (np.max(np.abs(samples)) + 1e-9)

    # RMS Energy
    rms = float(np.sqrt(np.mean(samples ** 2)))

    # Simple autocorrelation pitch estimation
    frame_len = int(0.03 * sr)
    hop = int(0.01 * sr)
    pitches = []
    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len]
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        # Find first peak after the initial decay
        min_lag = int(sr / 500)  # 500 Hz max
        max_lag = int(sr / 75)   # 75 Hz min
        if max_lag > len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            continue
        peak_idx = np.argmax(segment) + min_lag
        if corr[peak_idx] > 0.3 * corr[0]:
            pitches.append(sr / peak_idx)

    pitch_mean = float(np.mean(pitches)) if pitches else 0
    pitch_var = float(np.std(pitches)) if pitches else 0

    # Classify tone
    if rms < 0.05:
        tone_label = "Very Quiet"
        tone_score = 25
    elif rms < 0.12:
        tone_label = "Soft / Reserved"
        tone_score = 45
    elif rms < 0.25:
        if pitch_var > 30:
            tone_label = "Expressive & Confident"
            tone_score = 90
        else:
            tone_label = "Calm & Steady"
            tone_score = 75
    else:
        if pitch_var > 40:
            tone_label = "Energetic & Dynamic"
            tone_score = 85
        else:
            tone_label = "Loud & Assertive"
            tone_score = 70

    return {
        'available': True,
        'pitch_mean': round(pitch_mean, 1),
        'pitch_variance': round(pitch_var, 1),
        'energy': round(rms * 100, 1),
        'tone_label': tone_label,
        'tone_score': int(tone_score)
    }


# ── Hesitation Analysis ──────────────────────────────────────────────

FILLER_WORDS = [
    'um', 'uh', 'umm', 'uhh', 'er', 'err', 'ah', 'ahh',
    'like', 'you know', 'basically', 'actually', 'so',
    'i mean', 'kind of', 'sort of', 'well', 'right',
    'okay so', 'let me think', 'hmm', 'hm'
]

REPEAT_PATTERN = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)


def analyze_hesitation(transcript):
    """
    Analyze hesitation patterns in a transcript.
    Returns dict with filler_count, filler_words_found, repeat_count, hesitation_score.
    Hesitation is always available when there's a transcript (typed or spoken).
    """
    if not transcript or not transcript.strip():
        return {
            'available': False,
            'filler_count': 0, 'filler_words_found': [],
            'repeat_count': 0, 'hesitation_score': None,
            'hesitation_label': 'No Data'
        }

    text_lower = transcript.lower().strip()
    word_count = len(text_lower.split())

    # Count fillers
    found_fillers = []
    filler_count = 0
    for fw in FILLER_WORDS:
        occurrences = text_lower.count(fw)
        if occurrences > 0:
            filler_count += occurrences
            found_fillers.append(f"{fw} (×{occurrences})")

    # Count repeated words (stuttering)
    repeats = REPEAT_PATTERN.findall(text_lower)
    repeat_count = len(repeats)

    # Score: lower fillers = better
    filler_ratio = filler_count / max(word_count, 1)
    if filler_ratio < 0.02:
        score = 95
        label = "Very Fluent"
    elif filler_ratio < 0.05:
        score = 80
        label = "Fluent"
    elif filler_ratio < 0.10:
        score = 60
        label = "Moderate Hesitation"
    elif filler_ratio < 0.20:
        score = 40
        label = "Noticeable Hesitation"
    else:
        score = 20
        label = "High Hesitation"

    # Penalize repeated words
    score = max(10, score - repeat_count * 5)

    return {
        'available': True,
        'filler_count': filler_count,
        'filler_words_found': found_fillers[:5],
        'repeat_count': repeat_count,
        'hesitation_score': int(score),
        'hesitation_label': label
    }


# ── Speaking Speed ────────────────────────────────────────────────────

def analyze_speaking_speed(transcript, audio_bytes):
    """
    Calculate words per minute from transcript length and audio duration.
    Requires BOTH a transcript AND audio bytes to be meaningful.
    """
    if not audio_bytes or not transcript or not transcript.strip():
        return {
            'available': False,
            'wpm': 0, 'speed_label': 'No Audio',
            'speed_score': None
        }

    word_count = len(transcript.strip().split())

    # Get audio duration
    duration_sec = 0
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, 'rb') as wf:
            duration_sec = wf.getnframes() / wf.getframerate()
    except Exception:
        return {
            'available': False,
            'wpm': 0, 'speed_label': 'No Audio',
            'speed_score': None
        }

    if duration_sec < 1:
        return {
            'available': False,
            'wpm': 0, 'speed_label': 'Too Short', 'speed_score': None,
            'duration_sec': round(duration_sec, 1), 'word_count': word_count
        }

    wpm = (word_count / duration_sec) * 60

    if wpm < 80:
        label = "Very Slow"
        score = 40
    elif wpm < 110:
        label = "Slow"
        score = 60
    elif wpm < 160:
        label = "Optimal"
        score = 95
    elif wpm < 200:
        label = "Fast"
        score = 70
    else:
        label = "Very Fast"
        score = 45

    return {
        'available': True,
        'wpm': int(wpm),
        'speed_label': label,
        'speed_score': int(score),
        'duration_sec': round(duration_sec, 1),
        'word_count': word_count
    }


# ── Facial Emotion Detection ─────────────────────────────────────────

def analyze_facial_emotion(landmarks):
    """
    Analyze facial emotion from MediaPipe face landmarks (468+ points).
    Uses geometric ratios to estimate expression.
    Returns dict with emotion, confidence.
    """
    if landmarks is None or len(landmarks) < 468:
        return {'available': False, 'emotion': 'Unknown', 'emotion_score': None, 'details': {}}

    try:
        # Key landmark indices (MediaPipe Face Mesh)
        # Mouth
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        # Eyes
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        # Eyebrows
        left_brow = landmarks[70]
        right_brow = landmarks[300]
        left_eye_ref = landmarks[33]
        right_eye_ref = landmarks[263]

        # Nose
        nose_tip = landmarks[1]

        # Calculate ratios
        mouth_open = abs(upper_lip.y - lower_lip.y)
        mouth_width = abs(left_mouth.x - right_mouth.x)
        mouth_ratio = mouth_open / (mouth_width + 1e-6)

        left_eye_open = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_open = abs(right_eye_top.y - right_eye_bottom.y)
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        # Brow raise = distance from brow to eye
        left_brow_raise = abs(left_brow.y - left_eye_ref.y)
        right_brow_raise = abs(right_brow.y - right_eye_ref.y)
        avg_brow_raise = (left_brow_raise + right_brow_raise) / 2

        # Mouth corner upturn (smile detection)
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2
        corner_avg_y = (left_mouth.y + right_mouth.y) / 2
        smile_ratio = mouth_center_y - corner_avg_y  # positive = smile

        details = {
            'mouth_ratio': round(mouth_ratio, 4),
            'eye_openness': round(avg_eye_open, 4),
            'brow_raise': round(avg_brow_raise, 4),
            'smile_ratio': round(smile_ratio, 4),
        }

        # Classification
        if mouth_ratio > 0.08 and avg_brow_raise > 0.04:
            emotion = "Surprised"
            score = 60
        elif smile_ratio > 0.005 and mouth_ratio < 0.05:
            emotion = "Happy / Confident"
            score = 90
        elif avg_brow_raise < 0.025 and mouth_ratio < 0.02:
            emotion = "Focused / Serious"
            score = 75
        elif avg_eye_open < 0.01:
            emotion = "Tired / Disengaged"
            score = 35
        elif mouth_ratio > 0.06:
            emotion = "Speaking"
            score = 70
        else:
            emotion = "Neutral"
            score = 65

        return {'available': True, 'emotion': emotion, 'emotion_score': score, 'details': details}

    except Exception:
        return {'available': False, 'emotion': 'Unknown', 'emotion_score': None, 'details': {}}


# ── Combined Analysis ─────────────────────────────────────────────────

def compute_overall_confidence(voice, hesitation, speed, emotion):
    """
    Weighted combination of AVAILABLE analysis scores into a single confidence metric.
    Metrics that are unavailable (no audio, no face data) are excluded from the average.
    """
    all_metrics = {
        'voice':      (voice.get('tone_score'),       0.20, voice.get('available', False)),
        'hesitation': (hesitation.get('hesitation_score'), 0.30, hesitation.get('available', False)),
        'speed':      (speed.get('speed_score'),      0.20, speed.get('available', False)),
        'emotion':    (emotion.get('emotion_score'),   0.30, emotion.get('available', False)),
    }

    # Only include metrics that are available (have real data)
    active = {k: (score, weight) for k, (score, weight, avail) in all_metrics.items() if avail and score is not None}

    if not active:
        return {
            'overall_score': 0,
            'overall_label': 'No Data',
            'breakdown': {k: None for k in all_metrics},
            'available_metrics': 0,
            'total_metrics': 4
        }

    # Re-normalize weights so they sum to 1.0
    total_weight = sum(w for _, w in active.values())
    overall = sum((score * weight / total_weight) for score, weight in active.values())

    if overall >= 85:
        label = "Excellent"
    elif overall >= 70:
        label = "Strong"
    elif overall >= 55:
        label = "Moderate"
    elif overall >= 40:
        label = "Needs Improvement"
    else:
        label = "Low Confidence"

    breakdown = {}
    for k in all_metrics:
        score, _, avail = all_metrics[k]
        breakdown[k] = score if avail and score is not None else None

    return {
        'overall_score': round(overall, 1),
        'overall_label': label,
        'breakdown': breakdown,
        'available_metrics': len(active),
        'total_metrics': 4
    }


def run_full_analysis(audio_bytes=None, transcript=None, face_landmarks=None):
    """
    Run all four analyses and return a combined result dict.
    Any input can be None — that module will mark itself unavailable.
    """
    voice = analyze_voice_tone(audio_bytes)
    hesitation = analyze_hesitation(transcript)
    speed = analyze_speaking_speed(transcript, audio_bytes)
    emotion = analyze_facial_emotion(face_landmarks)
    overall = compute_overall_confidence(voice, hesitation, speed, emotion)

    return {
        'voice_tone': voice,
        'hesitation': hesitation,
        'speaking_speed': speed,
        'facial_emotion': emotion,
        'overall': overall
    }
