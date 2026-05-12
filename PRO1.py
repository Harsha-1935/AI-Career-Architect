import streamlit as st
import pdfplumber
import io
import json
import os
import time
import pandas as pd
from datetime import datetime
from gtts import gTTS
from google import genai

# -----------------------------
# 🔑 AI CONFIGURATION
# -----------------------------
# Your verified API Key
API_KEY = "AIzaSyAF2FAmqHDREQdBFXjzELPoUrGf5a0wcfI" 

def get_ai_data(prompt, retries=3):
    """Retrieves JSON data with rate-limit protection."""
    try:
        client = genai.Client(api_key=API_KEY)
        # Using the confirmed gemini-2.5-flash model from your list
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        if not response or not response.text:
            return None
            
        # Strip potential markdown code blocks
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)
    
    except Exception as e:
        if "429" in str(e) and retries > 0:
            wait_time = (4 - retries) * 15
            st.warning(f"⏳ Rate limit reached. Waiting {wait_time}s...")
            time.sleep(wait_time)
            return get_ai_data(prompt, retries - 1)
        st.error(f"🤖 AI Error: {e}")
        return None

# -----------------------------
# 📁 STORAGE & ANALYTICS LOGIC
# -----------------------------
HISTORY_FILE = "interview_history.json"

def save_to_history(avg_score):
    """Appends the session score to the permanent JSON file."""
    data = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": round(avg_score, 2)}
    history = get_history()
    history.append(data)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def get_history():
    """Reads all historical attempts from storage."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def speak(text):
    """Voice output for questions."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp, format='audio/mp3', autoplay=True)
    except Exception as e:
        st.error(f"Audio Error: {e}")

# -----------------------------
# 🧠 AI INTELLIGENCE TASKS
# -----------------------------
def generate_questions_ai(resume_text):
    """AI analyzes CV to build personalized questions."""
    prompt = f"Analyze this resume: {resume_text}\nGenerate 5 specific interview questions. Return ONLY a JSON list of strings."
    return get_ai_data(prompt)

def analyze_answer_ai(question, answer):
    """AI performs semantic STAR analysis."""
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Evaluate using STAR. Return ONLY JSON:
    {{
        "Score": 1-10, 
        "Situation": "✅/❌", "Task": "✅/❌", "Action": "✅/❌", "Result": "✅/❌",
        "Tip": "Brief feedback."
    }}
    """
    return get_ai_data(prompt)

# -----------------------------
# 🖥️ STREAMLIT INTERFACE
# -----------------------------
st.set_page_config(page_title="AI Coach 2.5", layout="wide")
st.title("🧠 Intelligent CV-Based Interview Coach")

if "started" not in st.session_state:
    st.session_state.update({
        "started": False, "q_idx": 0, "questions": [], 
        "results": [], "spoken": False, "history_saved": False
    })

# --- SIDEBAR: ALL-TIME AVERAGE ---
with st.sidebar:
    st.header("📈 Progress Tracker")
    history = get_history()
    if history:
        df = pd.DataFrame(history)
        st.line_chart(df.set_index("date"))
        
        # Latest Attempt Score
        st.metric("Latest Score", f"{history[-1]['score']}/10")
        
        # CUMULATIVE AVERAGE CALCULATION
        all_scores = [attempt['score'] for attempt in history]
        cumulative_avg = sum(all_scores) / len(all_scores)
        st.metric("All-Time Average", f"{cumulative_avg:.2f}/10")
        
        st.write(f"Total Attempts: {len(history)}")
    else:
        st.info("No data yet. Complete an interview to see your all-time average.")

# --- STEP 1: UPLOAD & ANALYZE ---
if not st.session_state.started:
    file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    if file and st.button("🚀 Start Interview"):
        with st.spinner("AI is analyzing your skills..."):
            with pdfplumber.open(file) as pdf:
                text = "".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            questions = generate_questions_ai(text)
            if questions:
                st.session_state.questions = questions
                st.session_state.started = True
                st.rerun()

# --- STEP 2: INTERVIEW LOOP ---
if st.session_state.started and st.session_state.q_idx < 5:
    current_q = st.session_state.questions[st.session_state.q_idx]
    st.subheader(f"Question {st.session_state.q_idx + 1}")
    st.info(current_q)
    
    if not st.session_state.spoken:
        speak(current_q)
        st.session_state.spoken = True

    with st.form(key=f"q_{st.session_state.q_idx}", clear_on_submit=True):
        ans = st.text_area("Your Response:", height=150)
        if st.form_submit_button("Submit Answer"):
            with st.spinner("AI is evaluating..."):
                analysis = analyze_answer_ai(current_q, ans)
                if analysis:
                    st.session_state.results.append({"q": current_q, "a": ans, "analysis": analysis})
                    st.session_state.q_idx += 1
                    st.session_state.spoken = False
                    st.rerun()

# --- STEP 3: FINAL SESSION REPORT ---
elif st.session_state.q_idx >= 5:
    st.header("📊 Session Performance Analysis")
    
    # Calculate Session Average
    scores = [r['analysis'].get('Score', 0) for r in st.session_state.results]
    session_avg = sum(scores) / 5
    st.metric("This Session Average", f"{session_avg:.1f}/10")
    
    # Save to history once
    if not st.session_state.history_saved:
        save_to_history(session_avg)
        st.session_state.history_saved = True

    st.divider()
    for i, res in enumerate(st.session_state.results):
        ana = res['analysis']
        with st.expander(f"Q{i+1}: {res['q']} (Score: {ana.get('Score', 0)}/10)"):
            st.write(f"**Your Answer:** {res['a']}")
            st.info(f"💡 **AI Tip:** {ana.get('Tip')}")
            cols = st.columns(4)
            cols[0].write(f"**Sit:** {ana.get('Situation')}")
            cols[1].write(f"**Task:** {ana.get('Task')}")
            cols[2].write(f"**Act:** {ana.get('Action')}")
            cols[3].write(f"**Res:** {ana.get('Result')}")

    if st.button("New Session"):
        st.session_state.clear()
        st.rerun()
