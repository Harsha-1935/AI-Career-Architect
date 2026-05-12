"""
AI Career Architect — Centralized Loader & Transition Manager
=============================================================
Provides animated overlays, fade-in containers, skeleton placeholders,
and AI-themed loading effects to eliminate Streamlit rerender flicker.

All loaders use INLINE styles to avoid Streamlit's HTML sanitization
stripping CSS class-based styling from inner elements.
"""

import streamlit as st
import streamlit.components.v1 as components
import time

# ─────────────────────────────────────────────
# 🎨 MASTER CSS — injected once per session
# ─────────────────────────────────────────────

_MASTER_CSS = """
<style>
/* ===== GLOBAL THEME POLISH ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables for the theme */
:root {
    --accent-primary: #6C63FF;
    --accent-secondary: #00D4AA;
    --accent-warning: #FF6B6B;
    --bg-dark: #0a0a0f;
    --bg-card: #12121a;
    --bg-card-hover: #1a1a2e;
    --text-primary: #e8e8f0;
    --text-secondary: #8888aa;
    --glow-primary: rgba(108,99,255,0.3);
    --glow-secondary: rgba(0,212,170,0.3);
    --border-subtle: rgba(108,99,255,0.15);
    --transition-smooth: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Smooth-in for all Streamlit elements */
.stApp {
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif !important;
}

/* Smooth transitions for tab panels only */
div[data-baseweb="tab-panel"] {
    animation: fadeSlideIn 0.4s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Page entry animation */
.main .block-container {
    animation: pageEntry 0.7s ease-out both;
}
@keyframes pageEntry {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ===== FULL-SCREEN OVERLAY ===== */
.ai-overlay {
    position: fixed;
    inset: 0;
    z-index: 999999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(6,6,14,0.92);
    backdrop-filter: blur(18px);
    animation: overlayIn 0.35s ease-out;
}
@keyframes overlayIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.ai-overlay.fade-out {
    animation: overlayOut 0.5s ease-in forwards;
}
@keyframes overlayOut {
    from { opacity: 1; }
    to   { opacity: 0; pointer-events: none; }
}

/* ===== HOLOGRAM LOADER ===== */
@keyframes holoSpin { to { transform: rotate(360deg); } }
@keyframes corePulse {
    0%, 100% { opacity: 0.4; transform: scale(0.8); }
    50%      { opacity: 1;   transform: scale(1.1); }
}

/* ===== NEURAL NETWORK PULSE ===== */
@keyframes nodePulse {
    0%, 100% { transform: scale(0.5); opacity: 0.3; }
    50%      { transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px rgba(108,99,255,0.3); }
}

/* ===== AI SCANNING EFFECT ===== */
@keyframes scanSlide {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
}

/* ===== MIC WAVE ANIMATION ===== */
@keyframes micBounce {
    0%   { height: 8px;  opacity: 0.5; }
    100% { height: 40px; opacity: 1; }
}

/* ===== PROCESSING DOTS ===== */
@keyframes dotFade {
    0%, 100% { opacity: 0.2; transform: scale(0.8); }
    50%      { opacity: 1;   transform: scale(1.2); }
}

/* ===== SKELETON SHIMMER ===== */
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ===== CAMERA PULSE ===== */
@keyframes cameraPulse {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50%      { transform: scale(1.15); opacity: 1; filter: drop-shadow(0 0 10px rgba(108,99,255,0.3)); }
}

/* ===== TITLE GRADIENT ===== */
.ai-title-gradient {
    background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 50%, #ff6bff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 2.4rem;
    margin-bottom: 4px;
}

/* ===== TAB TRANSITION ===== */
div[data-baseweb="tab-panel"] {
    animation: tabSlide 0.4s ease-out;
}
@keyframes tabSlide {
    from { opacity: 0; transform: translateX(12px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ===== METRIC CARDS ===== */
div[data-testid="stMetric"] {
    background: #12121a;
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 12px;
    padding: 16px !important;
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(108,99,255,0.3);
    box-shadow: 0 4px 20px rgba(108,99,255,0.08);
}

/* ===== EXPANDER POLISH ===== */
details[data-testid="stExpander"] {
    border: 1px solid rgba(108,99,255,0.15) !important;
    border-radius: 12px !important;
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
details[data-testid="stExpander"]:hover {
    border-color: rgba(108,99,255,0.3) !important;
}

/* ===== INLINE SPINNER ENHANCEMENT ===== */
div[data-testid="stSpinner"] {
    background: #12121a !important;
    border: 1px solid rgba(108,99,255,0.15) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* ===== FORM POLISH ===== */
div[data-testid="stForm"] {
    border: 1px solid rgba(108,99,255,0.15) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    background: #12121a !important;
}

/* ===== BUTTON GLOW ENHANCEMENT ===== */
div.stButton > button {
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    border-radius: 10px !important;
}
div.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(108,99,255,0.3) !important;
    transform: translateY(-1px) !important;
}
</style>
"""


# ─────────────────────────────────────────────
# 🔧 INJECTION HELPERS
# ─────────────────────────────────────────────

def inject_master_css():
    """Inject the master CSS once per Streamlit session run."""
    st.markdown(_MASTER_CSS, unsafe_allow_html=True)


def inject_page_entry_animation():
    """Adds a one-time full-page fade-in on initial load to hide first-paint jank."""
    if 'page_anim_injected' not in st.session_state:
        st.session_state.page_anim_injected = True
        st.markdown("""
            <style>
                .main .block-container {
                    animation: pageEntry 0.7s ease-out both;
                }
                @keyframes pageEntry {
                    from { opacity: 0; transform: translateY(24px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
            </style>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 🌀 LOADER HTML BUILDERS (ALL INLINE STYLES)
# ─────────────────────────────────────────────

def _build_hologram():
    return """
    <div style="width:90px;height:90px;position:relative;margin:0 auto 28px auto;">
        <div style="position:absolute;inset:0;border-radius:50%;border:3px solid transparent;border-top-color:#6C63FF;animation:holoSpin 1.2s linear infinite;"></div>
        <div style="position:absolute;inset:8px;border-radius:50%;border:3px solid transparent;border-right-color:#00D4AA;animation:holoSpin 1.6s linear infinite reverse;"></div>
        <div style="position:absolute;inset:16px;border-radius:50%;border:3px solid transparent;border-bottom-color:#ff6bff;animation:holoSpin 2s linear infinite;"></div>
        <div style="position:absolute;inset:22px;border-radius:50%;background:radial-gradient(circle,#6C63FF,transparent);animation:corePulse 1.5s ease-in-out infinite;"></div>
    </div>
    """

def _build_neural():
    colors = ["#6C63FF","#7c75ff","#00D4AA","#00e5bb","#ff6bff","#6C63FF","#00D4AA"]
    delays = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    nodes = ""
    for c, d in zip(colors, delays):
        nodes += f'<div style="width:12px;height:12px;border-radius:50%;background:{c};animation:nodePulse 1.4s ease-in-out infinite;animation-delay:{d}s;"></div>'
    return f'<div style="display:flex;gap:6px;margin:0 auto 24px auto;justify-content:center;">{nodes}</div>'

def _build_scan():
    return """
    <div style="width:200px;height:4px;background:rgba(108,99,255,0.15);border-radius:4px;overflow:hidden;margin:0 auto 20px auto;">
        <div style="width:40%;height:100%;background:linear-gradient(90deg,transparent,#6C63FF,#00D4AA,transparent);border-radius:4px;animation:scanSlide 1.8s ease-in-out infinite;"></div>
    </div>
    """

def _build_mic():
    heights = [10,20,35,25,40,15,30,20,38]
    delays = [0, 0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.2]
    bars = ""
    for h, d in zip(heights, delays):
        bars += f'<div style="width:5px;height:{h}px;background:linear-gradient(to top,#6C63FF,#00D4AA);border-radius:3px;animation:micBounce 0.8s ease-in-out infinite alternate;animation-delay:{d}s;"></div>'
    return f'<div style="display:flex;align-items:flex-end;gap:3px;height:40px;margin:0 auto 20px auto;justify-content:center;">{bars}</div>'

def _build_dots():
    dots = ""
    for i, d in enumerate([0, 0.2, 0.4]):
        dots += f'<div style="width:8px;height:8px;border-radius:50%;background:#8888aa;animation:dotFade 1.2s ease-in-out infinite;animation-delay:{d}s;"></div>'
    return f'<div style="display:flex;gap:8px;margin:10px auto 0 auto;justify-content:center;">{dots}</div>'

def _build_camera():
    return '<div style="text-align:center;padding:30px;font-size:3rem;animation:cameraPulse 1.5s ease-in-out infinite;">📸</div>'


_LOADER_BUILDERS = {
    "hologram": _build_hologram,
    "neural": _build_neural,
    "scan": _build_scan,
    "mic": _build_mic,
    "dots": _build_dots,
    "camera": _build_camera,
}


# ─────────────────────────────────────────────
# 🎛️ PUBLIC API
# ─────────────────────────────────────────────

def show_loader(message: str, *, style: str = "hologram", submessage: str = "", overlay: bool = False):
    """
    Display an AI-themed loader with fully inline-styled elements.
    Uses st.html() to bypass Streamlit's markdown HTML sanitizer.
    """
    builder = _LOADER_BUILDERS.get(style, _build_hologram)
    loader_html = builder()

    msg_html = f'<div style="color:#e8e8f0;font-size:1.15rem;font-weight:500;letter-spacing:0.3px;text-align:center;line-height:1.6;margin-top:8px;font-family:Inter,sans-serif;">{message}</div>'
    sub_html = f'<div style="color:#8888aa;font-size:0.82rem;text-align:center;margin-top:6px;font-family:Inter,sans-serif;">{submessage}</div>' if submessage else ""

    # Embedded keyframes so animations work inside st.html iframe
    keyframes_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');
    body { margin:0; background:transparent; }
    @keyframes holoSpin { to { transform:rotate(360deg); } }
    @keyframes corePulse { 0%,100%{opacity:.4;transform:scale(.8)} 50%{opacity:1;transform:scale(1.1)} }
    @keyframes nodePulse { 0%,100%{transform:scale(.5);opacity:.3} 50%{transform:scale(1.3);opacity:1;box-shadow:0 0 12px rgba(108,99,255,.3)} }
    @keyframes scanSlide { 0%{transform:translateX(-100%)} 100%{transform:translateX(350%)} }
    @keyframes micBounce { 0%{height:8px;opacity:.5} 100%{height:40px;opacity:1} }
    @keyframes dotFade { 0%,100%{opacity:.2;transform:scale(.8)} 50%{opacity:1;transform:scale(1.2)} }
    @keyframes cameraPulse { 0%,100%{transform:scale(1);opacity:.6} 50%{transform:scale(1.15);opacity:1} }
    </style>
    """

    content = f"""
    {keyframes_css}
    <div style="display:flex;flex-direction:column;align-items:center;padding:24px 0;">
        {loader_html}
        {msg_html}
        {sub_html}
    </div>
    """

    # Calculate height based on content
    height = 180 if submessage else 160
    if style == "hologram":
        height = 200 if submessage else 180
    
    st.html(content)


def show_overlay_loader(message: str, *, style: str = "hologram", submessage: str = ""):
    """Convenience shortcut for a full-screen overlay loader."""
    show_loader(message, style=style, submessage=submessage, overlay=True)


def hide_loader():
    """Inject a tiny script to fade-out any active overlay (best-effort)."""
    components.html("""
        <script>
            const el = window.parent.document.getElementById('aiOverlay');
            if (el) {
                el.classList.add('fade-out');
                setTimeout(() => el.remove(), 600);
            }
        </script>
    """, height=0)


def show_skeleton(rows: int = 4, *, card: bool = True):
    """Render shimmer placeholder skeleton to mask slow content."""
    shimmer_bg = "background:linear-gradient(90deg,#1a1a2e 25%,#252540 50%,#1a1a2e 75%);background-size:200% 100%;animation:shimmer 1.5s infinite;border-radius:8px;"
    
    inner = f'<div style="height:24px;width:45%;margin-bottom:16px;{shimmer_bg}"></div>'
    for i in range(rows):
        w = "width:60%;" if i % 3 == 2 else "width:100%;"
        h = "height:10px;" if i % 3 == 2 else "height:14px;"
        inner += f'<div style="{h}{w}margin-bottom:10px;{shimmer_bg}"></div>'

    if card:
        html = f'<div style="padding:20px;border-radius:12px;border:1px solid rgba(108,99,255,0.15);background:#12121a;margin-bottom:16px;">{inner}</div>'
    else:
        html = inner
    st.html(f'<style>@keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}</style>{html}')


def glass_card(content_html: str):
    """Render HTML inside a glassmorphism card."""
    st.markdown(f"""
        <div style="background:rgba(18,18,26,0.7);backdrop-filter:blur(16px);border:1px solid rgba(108,99,255,0.15);border-radius:16px;padding:20px 24px;transition:all 0.4s ease;">
            {content_html}
        </div>
    """, unsafe_allow_html=True)


def animated_title(title: str, subtitle: str = ""):
    """Render a gradient-animated app title."""
    sub_html = f'<p style="color:#8888aa;font-size:1rem;font-weight:400;margin:0 0 24px 0;font-family:Inter,sans-serif;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
        <div style="text-align:center;margin-bottom:12px;animation:fadeSlideIn 0.6s ease-out both;">
            <div class="ai-title-gradient">{title}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def progress_score_ring(score: float, max_score: float = 10):
    """Render an animated SVG progress ring for a score."""
    pct = min(score / max_score, 1) if max_score > 0 else 0
    circumference = 2 * 3.14159 * 52  # radius 52
    offset = circumference * (1 - pct)
    st.markdown(f"""
        <div style="position:relative;width:120px;height:120px;margin:0 auto 20px auto;">
            <svg width="120" height="120" viewBox="0 0 120 120">
                <defs>
                    <linearGradient id="progressGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" style="stop-color:#6C63FF"/>
                        <stop offset="100%" style="stop-color:#00D4AA"/>
                    </linearGradient>
                </defs>
                <circle fill="none" stroke="rgba(108,99,255,0.15)" stroke-width="8" cx="60" cy="60" r="52"/>
                <circle fill="none" stroke="url(#progressGrad)" stroke-width="8" stroke-linecap="round"
                    cx="60" cy="60" r="52"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    style="transform:rotate(-90deg);transform-origin:50% 50%;transition:stroke-dashoffset 1.5s ease-out;"/>
            </svg>
            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:700;color:#6C63FF;font-family:Inter,sans-serif;">
                {score}
            </div>
        </div>
    """, unsafe_allow_html=True)


def transition_placeholder(placeholder, message: str, style: str = "hologram", duration: float = 0.0):
    """
    Show a loader inside a st.empty() placeholder, optionally sleeping.
    
    Usage:
        ph = st.empty()
        transition_placeholder(ph, "Loading…")
        # do work
        ph.empty()
    """
    with placeholder.container():
        show_loader(message, style=style)
    if duration > 0:
        time.sleep(duration)


class LoaderContext:
    """
    Context manager for showing a loader during a block of work.
    
    Usage:
        with LoaderContext("Evaluating answer…", style="neural"):
            result = some_slow_call()
    """
    def __init__(self, message: str, *, style: str = "hologram", submessage: str = "", overlay: bool = False):
        self.message = message
        self.style = style
        self.submessage = submessage
        self.overlay = overlay
        self._placeholder = None

    def __enter__(self):
        self._placeholder = st.empty()
        with self._placeholder.container():
            show_loader(self.message, style=self.style, submessage=self.submessage, overlay=self.overlay)
        return self

    def __exit__(self, *args):
        if self._placeholder:
            self._placeholder.empty()
