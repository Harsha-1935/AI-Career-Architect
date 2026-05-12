import streamlit as st
import pdfplumber
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
from groq import Groq
from gtts import gTTS
import io

# -----------------------------
# 🔑 CONFIGURATION
# -----------------------------
API_KEY = "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"
client = Groq(api_key=API_KEY)

MAX_QUESTIONS = 5
HISTORY_FILE = "interview_history.json"
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="AI Career Architect", layout="wide")
st.title("🚀 AI Resume Intelligence & Career Architect")

# -----------------------------
# 📁 DATA & VOICE LOGIC
# -----------------------------
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_to_history(score):
    history = get_history()
    history.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": round(score, 2)})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp, format='audio/mp3', autoplay=True)
    except: pass

# -----------------------------
# 📄 DAY 31: CAREER ARCHITECT ENGINE
# -----------------------------
def analyze_career_fit_ai(resume_text):
    """Suggests roles, identifies gaps, and recommends improvements."""
    prompt = f"""
    Analyze this candidate's resume text and act as a Career Strategist.
    Resume: {resume_text}

    Return ONLY a JSON object:
    {{
        "suggested_roles": ["Role 1", "Role 2", "Role 3"],
        "primary_skills": ["Skill 1", "Skill 2"],
        "skill_gaps": ["Missing Skill A", "Missing Skill B"],
        "improvement_plan": ["Recommendation 1", "Recommendation 2"],
        "market_readiness": (0-100)
    }}
    """
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"suggested_roles": ["Generalist"], "primary_skills": [], "skill_gaps": ["Could not analyze"], "improvement_plan": [], "market_readiness": 50}

# -----------------------------
# 🔄 SIDEBAR & SESSION
# -----------------------------
with st.sidebar:
    st.header("📈 Career Progress")
    history_data = get_history()
    if history_data:
        all_scores = [item['score'] for item in history_data]
        st.metric("Avg Interview Score", f"{sum(all_scores)/len(all_scores):.2f}/10")
        st.dataframe(pd.DataFrame(history_data[::-1]), hide_index=True)

# -----------------------------
# 📤 UPLOAD & ANALYSIS FLOW
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF) to Begin Analysis", type=["pdf"])

if uploaded_file and "career_analysis" not in st.session_state:
    with st.spinner("🚀 Architecting your career path..."):
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        
        # New Day 31 Feature
        analysis = analyze_career_fit_ai(text)
        st.session_state.career_analysis = analysis
        st.session_state.skills = analysis["primary_skills"]
        st.session_state.raw_text = text

# -----------------------------
# 🎯 DAY 31: CAREER DASHBOARD
# -----------------------------
if "career_analysis" in st.session_state:
    data = st.session_state.career_analysis
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏁 Market Readiness")
        st.metric("Readiness Score", f"{data['market_readiness']}%")
        st.write("**Top Job Matches:**")
        for role in data['suggested_roles']:
            st.success(f"🎯 {role}")
            
    with col2:
        st.subheader("⚠️ Skill Gaps")
        for gap in data['skill_gaps']:
            st.error(f"❌ Missing: {gap}")
        
        st.subheader("💡 Improvement Plan")
        for tip in data['improvement_plan']:
            st.info(tip)

    # -----------------------------
    # 🎤 INTERVIEW SECTION
    # -----------------------------
    st.divider()
    st.subheader("🎤 Ready to Practice?")
    practice_role = st.selectbox("Practice Interview For:", data['suggested_roles'])
    
    if "started" not in st.session_state:
        st.session_state.update({"started": False, "questions": [], "answers": [], "results": [], "q_index": 0, "spoken": False, "saved": False})

    if st.button("Start AI Interview"):
        st.session_state.update({"started": True, "questions": [], "answers": [], "results": [], "q_index": 0, "spoken": False, "saved": False})

    if st.session_state.started:
        # (Standard Interview Loop Logic)
        if st.session_state.q_index < MAX_QUESTIONS:
            if len(st.session_state.questions) <= st.session_state.q_index:
                prompt = f"Professional interviewer for {practice_role}. Candidate Skills: {st.session_state.skills}. Generate ONE unique question."
                res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
                st.session_state.questions.append(res.choices[0].message.content.strip())

            current_q = st.session_state.questions[st.session_state.q_index]
            st.info(f"**Q{st.session_state.q_index+1}:** {current_q}")
            
            if not st.session_state.spoken:
                speak(current_q)
                st.session_state.spoken = True

            with st.form(key=f"q_{st.session_state.q_index}", clear_on_submit=True):
                ans = st.text_area("Your Response:")
                if st.form_submit_button("Submit"):
                    eval_prompt = f"Evaluate this answer for {practice_role}. Question: {current_q} | Answer: {ans}. Return JSON score 1-10."
                    eval_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": eval_prompt}], response_format={"type": "json_object"})
                    st.session_state.results.append(json.loads(eval_res.choices[0].message.content))
                    st.session_state.q_index += 1
                    st.session_state.spoken = False
                    st.rerun()
        else:
            st.success("Practice Complete! Check sidebar for progress.")
            if not st.session_state.saved:
                scores = [r.get('score', 0) for r in st.session_state.results]
                save_to_history(sum(scores)/len(scores))
                st.session_state.saved = True
                st.rerun()
