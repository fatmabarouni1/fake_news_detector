import html

import joblib
import numpy as np
import scipy.sparse as sp
import streamlit as st

from fact_check import run_fact_check
from utils import (
    MIN_CHARS,
    THR_FAKE,
    THR_REAL,
    clean_text,
    sensational_features,
    sensational_vector,
    verdict_from_prob,
)

st.set_page_config(
    layout="centered",
    page_title="Fake News Checker",
    page_icon="\U0001F4F1",
)


@st.cache_resource
def load_artifacts():
    model_obj = joblib.load("model.joblib")
    vectorizer_obj = joblib.load("vectorizer.joblib")
    return model_obj, vectorizer_obj


model, vectorizer = load_artifacts()


def build_matrix(texts_raw: list[str]):
    cleaned = [clean_text(t) for t in texts_raw]
    tfidf = vectorizer.transform(cleaned)
    sens_mat = sp.csr_matrix(
        np.array([sensational_vector(t) for t in texts_raw], dtype=np.float32)
    )
    return sp.hstack([tfidf, sens_mat], format="csr")


def predict(text: str):
    mat = build_matrix([text])
    proba = model.predict_proba(mat)[0]
    p_fake = float(proba[1])
    return verdict_from_prob(p_fake), p_fake


def explain_tokens(text: str, top_k: int = 6):
    try:
        if not hasattr(model, "coef_"):
            return [], []

        cleaned = clean_text(text)
        if not cleaned:
            return [], []

        vec = vectorizer.transform([cleaned])
        row = vec[0]
        if row.nnz == 0:
            return [], []

        n_tfidf = len(vectorizer.get_feature_names_out())
        coef = model.coef_[0][:n_tfidf]
        names = vectorizer.get_feature_names_out()

        idx = row.indices
        vals = row.data
        contrib = vals * coef[idx]

        fake_items = []
        real_items = []
        for i, score in zip(idx, contrib):
            token = str(names[i])
            s = float(score)
            if s > 0:
                fake_items.append((token, s))
            elif s < 0:
                real_items.append((token, s))

        fake_items.sort(key=lambda x: x[1], reverse=True)
        real_items.sort(key=lambda x: x[1])
        return fake_items[:top_k], real_items[:top_k]
    except Exception:
        return [], []


def chips_html(items, chip_class: str, empty_text: str) -> str:
    if not items:
        return f"<div class='chip-empty'>{html.escape(empty_text)}</div>"
    rendered = []
    for token, score in items:
        rendered.append(
            f"<span class='chip {chip_class}'>{html.escape(token)}"
            f" <span class='chip-score'>{abs(score):.3f}</span></span>"
        )
    return "".join(rendered)


def tiny_bar(label: str, value: float, max_value: float) -> str:
    safe_max = max(max_value, 1e-6)
    pct = max(0.0, min(100.0, value / safe_max * 100.0))
    return (
        "<div class='tiny-row'>"
        f"<div class='tiny-label'>{html.escape(label)}</div>"
        f"<div class='tiny-value'>{value:.2f}</div>"
        "</div>"
        "<div class='tiny-track'>"
        f"<div class='tiny-fill' style='width:{pct:.1f}%'></div>"
        "</div>"
    )


def rating_tone(rating_text: str) -> str:
    r = (rating_text or "").lower()
    false_words = [
        "false",
        "fake",
        "misleading",
        "incorrect",
        "wrong",
        "debunked",
        "inaccurate",
        "faux",
        "trompeur",
        "errone",
    ]
    true_words = [
        "true",
        "correct",
        "accurate",
        "verified",
        "confirmed",
        "vrai",
        "exact",
        "verifie",
        "confirme",
    ]
    if any(w in r for w in false_words):
        return "fake"
    if any(w in r for w in true_words):
        return "real"
    return "uncertain"


st.markdown(
    """
<style>
:root {
  --bg-a: #fbfbff;
  --bg-b: #f3f7ff;
  --bg-c: #fdf4fb;
  --card: rgba(255, 255, 255, 0.92);
  --card-border: #e7e9f4;
  --text: #1f2430;
  --muted: #687086;
  --accent-a: #abd7ff;
  --accent-b: #f5c5e3;
  --accent-c: #d8e8ff;
  --fake-bg: #ffe1de;
  --fake-border: #ffb7af;
  --real-bg: #ddf7ea;
  --real-border: #9ddfc3;
  --uncertain-bg: #ffeecf;
  --uncertain-border: #f6d38a;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 520px at 0% -10%, #f5f7ff 0%, transparent 70%),
    radial-gradient(900px 520px at 100% -5%, #fef3fb 0%, transparent 68%),
    linear-gradient(180deg, var(--bg-a), var(--bg-b) 55%, var(--bg-c));
  color: var(--text);
}

.main .block-container {
  max-width: 520px;
  padding-top: 0.8rem;
  padding-bottom: 1rem;
}

[data-testid="stSidebar"] { display: none; }

.card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 20px;
  padding: 12px;
  margin-bottom: 10px;
  box-shadow: 0 8px 24px rgba(69, 77, 104, 0.08);
}

.hero-title {
  text-align: center;
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: 0.2px;
  margin-bottom: 2px;
}

.hero-sub {
  text-align: center;
  font-size: 0.88rem;
  color: var(--muted);
}

.section-title {
  font-size: 0.93rem;
  font-weight: 700;
  margin-bottom: 7px;
  color: var(--text);
}

.stTextArea textarea {
  border-radius: 16px !important;
  border: 1px solid #d8deef !important;
  background: #f9fbff !important;
  color: var(--text) !important;
  padding: 12px !important;
  min-height: 120px !important;
}

.stTextArea textarea::placeholder {
  color: #7f88a3 !important;
}

.stButton > button[kind="primary"] {
  width: 100%;
  border: 0;
  border-radius: 999px;
  padding: 0.68rem 1rem;
  font-weight: 700;
  color: #1f2430;
  background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
}

.stButton > button[kind="secondary"] {
  width: 100%;
  border-radius: 999px;
  border: 1px solid #d9deef;
  padding: 0.44rem 0.55rem;
  font-size: 0.76rem;
  font-weight: 600;
  color: #374058;
  background: var(--accent-c);
}

.result-wrap {
  text-align: center;
}

.verdict-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 8px 14px;
  font-size: 0.95rem;
  font-weight: 800;
  margin-bottom: 8px;
}

.verdict-badge.fake {
  background: var(--fake-bg);
  border: 1px solid var(--fake-border);
}

.verdict-badge.real {
  background: var(--real-bg);
  border: 1px solid var(--real-border);
}

.verdict-badge.uncertain {
  background: var(--uncertain-bg);
  border: 1px solid var(--uncertain-border);
}

.prob-label {
  font-size: 0.79rem;
  color: var(--muted);
}

.prob-value {
  font-size: 2.25rem;
  line-height: 1.05;
  font-weight: 800;
  color: var(--text);
}

.progress-track {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #e8ecf8;
  overflow: hidden;
  margin-top: 8px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
}

.threshold-note {
  margin-top: 7px;
  font-size: 0.77rem;
  color: var(--muted);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 8px;
}

.kpi {
  border-radius: 14px;
  border: 1px solid #e2e7f5;
  background: #fbfcff;
  padding: 9px;
}

.kpi-label {
  font-size: 0.72rem;
  color: var(--muted);
}

.kpi-value {
  font-size: 1rem;
  font-weight: 700;
  margin-top: 2px;
}

.tiny-wrap {
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
}

.tiny-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 3px;
}

.tiny-label {
  font-size: 0.77rem;
  color: var(--muted);
}

.tiny-value {
  font-size: 0.77rem;
  font-weight: 700;
  color: var(--text);
}

.tiny-track {
  width: 100%;
  height: 7px;
  border-radius: 999px;
  overflow: hidden;
  background: #e9edf8;
  margin-bottom: 4px;
}

.tiny-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
}

.explain-col-title {
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--muted);
  margin-bottom: 4px;
}

.chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chip {
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 0.76rem;
  border: 1px solid #e1e6f4;
  background: #f8faff;
}

.chip-fake { background: #ffeceb; }
.chip-real { background: #e9f9f1; }
.chip-score { opacity: 0.85; font-weight: 700; }
.chip-empty { font-size: 0.78rem; color: var(--muted); }

.preview {
  border-radius: 14px;
  border: 1px solid #e1e6f4;
  background: #fafcff;
  padding: 10px;
  line-height: 1.4;
  font-size: 0.88rem;
  white-space: pre-wrap;
}

.alert {
  border-radius: 12px;
  border: 1px solid #f0cf8f;
  background: #fff5df;
  color: #7d5b1f;
  padding: 10px;
  font-size: 0.84rem;
  margin-bottom: 10px;
}

.footer {
  text-align: center;
  font-size: 0.73rem;
  color: var(--muted);
  margin-top: 2px;
}

.fact-meta {
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 7px;
}

.fact-summary {
  font-size: 0.86rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 7px;
}

.fact-source-card {
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid #e2e7f5;
  border-left-width: 5px;
  padding: 10px;
  margin-bottom: 8px;
}

.fact-source-card.fake {
  border-left-color: #ff8f85;
}

.fact-source-card.real {
  border-left-color: #7ed0ac;
}

.fact-source-card.uncertain {
  border-left-color: #f0bf62;
}

.fact-source-name {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--muted);
  margin-bottom: 2px;
}

.fact-source-rating {
  font-size: 0.93rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
}

.fact-link-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  text-decoration: none;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.78rem;
  font-weight: 700;
  color: #2f3f64;
  background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
}
</style>
""",
    unsafe_allow_html=True,
)

if "post_text" not in st.session_state:
    st.session_state.post_text = ""

st.markdown(
    """
<div class="card">
  <div class="hero-title">Fake News Checker</div>
  <div class="hero-sub">Pastel AI demo for style-based misinformation signals</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<div class='section-title'>Input</div>", unsafe_allow_html=True)

ex1, ex2, ex3 = st.columns(3)
if ex1.button("Sensational", type="secondary"):
    st.session_state.post_text = (
        "BREAKING!!! Secret miracle cure exposed!!! Share before it is deleted!!!"
    )
    st.rerun()
if ex2.button("Institutional", type="secondary"):
    st.session_state.post_text = (
        "The public health authority published an annual report with methods and references."
    )
    st.rerun()
if ex3.button("Ambiguous", type="secondary"):
    st.session_state.post_text = (
        "Users disagree about this update; some call it major while others call it routine."
    )
    st.rerun()

text = st.text_area(
    "",
    key="post_text",
    label_visibility="collapsed",
    placeholder="Paste caption, tweet, or post text...",
)

check_clicked = st.button("Check Post", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if check_clicked:
    stripped = text.strip()

    if not stripped:
        st.markdown("<div class='alert'>Please paste a post first.</div>", unsafe_allow_html=True)

    elif len(stripped) < MIN_CHARS:
        st.markdown(
            f"<div class='alert'>Text too short ({len(stripped)} chars). Minimum required: {MIN_CHARS}.</div>",
            unsafe_allow_html=True,
        )

    else:
        p_fake_ml = predict(text)[1]

        with st.spinner("Checking external sources..."):
            fc = run_fact_check(text)

        if fc.get("count", 0) > 0:
            fc_score_raw = float(fc.get("score", 0.0))
            fc_score_norm = (fc_score_raw + 1.0) / 2.0
            fc_score_norm = max(0.0, min(1.0, fc_score_norm))
            p_fake_combined = 0.6 * p_fake_ml + 0.4 * fc_score_norm
        else:
            p_fake_combined = p_fake_ml

        p_fake_combined = max(0.0, min(1.0, p_fake_combined))
        verdict = verdict_from_prob(p_fake_combined)
        p_fake = p_fake_combined

        metrics = sensational_features(text)
        fake_tokens, real_tokens = explain_tokens(text, top_k=6)

        badge_map = {
            "FAKE": ("fake", "FAKE"),
            "REAL": ("real", "REAL"),
            "UNCERTAIN": ("uncertain", "UNCERTAIN"),
        }
        badge_class, badge_text = badge_map.get(verdict, ("uncertain", verdict))

        st.markdown('<div class="card result-wrap">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='verdict-badge {badge_class}'>{badge_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='prob-label'>FAKE probability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prob-value'>{p_fake * 100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(
            (
                "<div class='progress-track'>"
                f"<div class='progress-fill' style='width:{max(0.0, min(100.0, p_fake * 100)):.1f}%'></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='threshold-note'>REAL <= {int(THR_REAL * 100)}% | FAKE >= {int(THR_FAKE * 100)}%</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Vibe</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">sensational_score</div><div class="kpi-value">{metrics['sensational_score']:.2f}</div></div>
  <div class="kpi"><div class="kpi-label">uppercase ratio</div><div class="kpi-value">{metrics['upper_ratio']:.2f}</div></div>
  <div class="kpi"><div class="kpi-label">exclamation count</div><div class="kpi-value">{metrics['exclam']}</div></div>
  <div class="kpi"><div class="kpi-label">emotional words</div><div class="kpi-value">{metrics['emo_hits']}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        bar_data = [
            ("Exclamations", float(metrics["exclam"])),
            ("Questions", float(metrics["question"])),
            ("Upper x10", float(metrics["upper_ratio"] * 10)),
            ("Emotional words", float(metrics["emo_hits"])),
        ]
        max_bar = max([v for _, v in bar_data] + [1.0])
        bars_html = "".join(tiny_bar(name, value, max_bar) for name, value in bar_data)
        st.markdown(f"<div class='tiny-wrap'>{bars_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Explanation</div>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='explain-col-title'>Pushes FAKE</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='chip-wrap'>{chips_html(fake_tokens, 'chip-fake', 'No strong FAKE tokens')}</div>",
                unsafe_allow_html=True,
            )
        with col_r:
            st.markdown("<div class='explain-col-title'>Pushes REAL</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='chip-wrap'>{chips_html(real_tokens, 'chip-real', 'No strong REAL tokens')}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>External Fact Check</div>", unsafe_allow_html=True)
        if not fc.get("enabled", False):
            st.markdown(
                "<div class='fact-meta'>Google Fact Check API is not configured.</div>",
                unsafe_allow_html=True,
            )
        elif fc.get("count", 0) == 0:
            st.markdown(
                f"<div class='fact-meta'>Query: <b>{html.escape(fc.get('claim', ''))}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='fact-meta'>No matching fact-check sources found.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='fact-meta'>Query: <b>{html.escape(fc.get('claim', ''))}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='fact-summary'>{html.escape(fc.get('verdict', ''))} - {int(fc.get('count', 0))} source(s)</div>",
                unsafe_allow_html=True,
            )
            for source in fc.get("sources", []):
                src_name = html.escape(str(source.get("source", "Unknown")))
                src_rating = html.escape(str(source.get("rating", "N/A")))
                src_url = html.escape(str(source.get("url", "")))
                tone = rating_tone(src_rating)
                st.markdown(
                    (
                        f"<div class='fact-source-card {tone}'>"
                        f"<div class='fact-source-name'>{src_name}</div>"
                        f"<div class='fact-source-rating'>{src_rating}</div>"
                        f"<a class='fact-link-btn' href='{src_url}' target='_blank' rel='noopener noreferrer'>View source ↗</a>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Post Preview</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='preview'>{html.escape(text)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>Disclaimer: This tool predicts writing-style patterns, not factual truth.</div>",
    unsafe_allow_html=True,
)
