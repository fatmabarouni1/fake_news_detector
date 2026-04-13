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
    page_title="TRUTHX",
    page_icon="\U0001F5DE",
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


def combine_fake_scores(p_fake_ml: float, fc: dict) -> tuple[float, float, float | None]:
    ml_weight = 0.6
    fact_weight = 0.4

    if fc.get("count", 0) <= 0:
        return p_fake_ml, ml_weight, None

    fc_score_raw = float(fc.get("score", 0.0))
    fc_score_norm = max(0.0, min(1.0, (fc_score_raw + 1.0) / 2.0))
    combined = (ml_weight * p_fake_ml) + (fact_weight * fc_score_norm)
    return max(0.0, min(1.0, combined)), ml_weight, fc_score_norm


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

:root {
  --paper: #f5f0e8;
  --ink: #1a1a1a;
  --accent: #e8341c;
  --muted: #8a8070;
  --panel: #efe8dc;
  --line: #1a1a1a;
  --fake: #e8341c;
  --real: #2f7d4a;
  --uncertain: #b17400;
}

@media (prefers-color-scheme: dark) {
  :root {
    --paper: #171411;
    --ink: #f2ebdf;
    --muted: #b2a796;
    --panel: #211d19;
    --line: #f2ebdf;
    --real: #5bb878;
    --uncertain: #e0a93d;
  }
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper);
  color: var(--ink);
  font-family: "Space Mono", monospace;
}

.main .block-container {
  max-width: 980px;
  padding-top: 1rem;
  padding-bottom: 1.5rem;
}

[data-testid="stSidebar"] { display: none; }

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 0;
  padding: 16px;
  margin-bottom: 12px;
}

.masthead {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-end;
  border-bottom: 3px solid var(--line);
  padding: 0 0 14px;
  margin-bottom: 18px;
}

.brand-wrap {
  display: flex;
  align-items: baseline;
  gap: 16px;
  flex-wrap: wrap;
}

.hero-title {
  margin: 0;
  font-family: "Syne", sans-serif;
  font-size: clamp(2.2rem, 6vw, 4.4rem);
  line-height: 0.95;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.hero-title .accent {
  color: var(--accent);
}

.hero-sub {
  margin: 0;
  color: var(--muted);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.beta-stamp {
  border: 2px solid var(--accent);
  color: var(--accent);
  padding: 6px 10px;
  font-family: "Syne", sans-serif;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.section-title {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  margin-bottom: 10px;
  color: var(--muted);
}

.counter {
  text-align: right;
  color: var(--muted);
  font-size: 0.78rem;
  margin-top: 8px;
}

.stTextArea textarea {
  border-radius: 0 !important;
  border: 1px solid var(--line) !important;
  background: transparent !important;
  color: var(--ink) !important;
  padding: 14px !important;
  min-height: 150px !important;
  font-family: "Space Mono", monospace !important;
  line-height: 1.55 !important;
}

.stTextArea textarea::placeholder {
  color: var(--muted) !important;
}

.stButton > button[kind="primary"] {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 0;
  padding: 0.9rem 1rem;
  color: var(--paper);
  background: var(--ink);
  font-family: "Syne", sans-serif;
  font-size: 1rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.stButton > button[kind="secondary"] {
  width: 100%;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 0.44rem 0.55rem;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--ink);
  background: transparent;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.stButton > button:hover {
  color: var(--paper);
  background: var(--accent);
  border-color: var(--accent);
}

.result-wrap {
  text-align: left;
}

.verdict-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 7px 12px;
  font-family: "Syne", sans-serif;
  font-size: 0.82rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  color: #fff;
  margin-bottom: 8px;
  text-transform: uppercase;
}

.verdict-badge.fake {
  background: var(--fake);
  border: 1px solid var(--fake);
}

.verdict-badge.real {
  background: var(--real);
  border: 1px solid var(--real);
}

.verdict-badge.uncertain {
  background: var(--uncertain);
  border: 1px solid var(--uncertain);
}

.prob-label {
  font-size: 0.79rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.14em;
}

.prob-value {
  font-family: "Syne", sans-serif;
  font-size: 2.8rem;
  line-height: 1.05;
  font-weight: 800;
  color: var(--ink);
}

.progress-track {
  width: 100%;
  height: 14px;
  border-radius: 0;
  border: 1px solid var(--line);
  background: transparent;
  overflow: hidden;
  margin-top: 8px;
}

.progress-fill {
  height: 100%;
  background: var(--fake);
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.threshold-note {
  margin-top: 7px;
  font-size: 0.77rem;
  color: var(--muted);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.kpi {
  border: 1px solid var(--line);
  background: transparent;
  padding: 14px 10px;
  min-height: 110px;
}

.kpi-label {
  font-size: 0.68rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.kpi-value {
  font-family: "Syne", sans-serif;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 18px;
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
  border-radius: 0;
  overflow: hidden;
  background: transparent;
  border: 1px solid var(--line);
  margin-bottom: 4px;
}

.tiny-fill {
  height: 100%;
  background: var(--accent);
}

.explain-col-title {
  font-family: "Syne", sans-serif;
  font-size: 1rem;
  font-weight: 800;
  color: var(--muted);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chip {
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.76rem;
  border: 1px solid var(--line);
  background: transparent;
}

.chip-fake { border-color: var(--fake); color: var(--fake); }
.chip-real { border-color: var(--real); color: var(--real); }
.chip-score { opacity: 0.85; font-weight: 700; }
.chip-empty { font-size: 0.78rem; color: var(--muted); }

.preview {
  border: 1px solid var(--line);
  background: transparent;
  padding: 14px;
  line-height: 1.4;
  font-size: 0.88rem;
  white-space: pre-wrap;
}

.alert {
  border: 1px solid var(--accent);
  background: transparent;
  color: var(--accent);
  padding: 12px;
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
  color: var(--ink);
  margin-bottom: 7px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.fact-source-card {
  background: transparent;
  border: 1px solid var(--line);
  padding: 12px;
  margin-bottom: 8px;
}

.fact-source-card.fake {
  border-color: var(--fake);
}

.fact-source-card.real {
  border-color: var(--real);
}

.fact-source-card.uncertain {
  border-color: var(--uncertain);
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
  color: var(--ink);
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
  color: var(--paper);
  background: var(--ink);
}

@media (max-width: 720px) {
  .masthead {
    display: block;
  }

  .beta-stamp {
    display: inline-block;
    margin-top: 12px;
  }

  .kpi-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
""",
    unsafe_allow_html=True,
)

if "post_text" not in st.session_state:
    st.session_state.post_text = ""
if "has_analyzed" not in st.session_state:
    st.session_state.has_analyzed = False

st.markdown(
    """
<div class="masthead">
  <div class="brand-wrap">
    <div class="hero-title">TRUTH<span class="accent">X</span></div>
    <div class="hero-sub">Misinformation Signal Detector</div>
  </div>
  <div class="beta-stamp">Beta</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<div class='section-title'>Claim To Analyze</div>", unsafe_allow_html=True)

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
    placeholder="Paste a claim, headline, post, or article excerpt...",
)
char_count = len(text)
st.markdown(
    f"<div class='counter'>{char_count} character{'s' if char_count != 1 else ''}</div>",
    unsafe_allow_html=True,
)

button_label = "→ RE-ANALYZE" if st.session_state.has_analyzed else "→ ANALYZE"
check_clicked = st.button(button_label, type="primary")
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
        st.session_state.has_analyzed = True
        p_fake_ml = predict(text)[1]

        with st.spinner("Checking external sources..."):
            fc = run_fact_check(text)

        p_fake_combined, ml_weight, fc_score_norm = combine_fake_scores(p_fake_ml, fc)
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
        st.markdown("<div class='section-title'>Verdict</div>", unsafe_allow_html=True)
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
        if fc_score_norm is None:
            score_note = f"Combined score = ML only ({p_fake_ml * 100:.1f}%)"
        else:
            score_note = (
                f"Combined score = ML {int(ml_weight * 100)}% + external fact check {int((1.0 - ml_weight) * 100)}%"
                f" ({fc_score_norm * 100:.1f}%)"
            )
        st.markdown(
            f"<div class='threshold-note'>{html.escape(score_note)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Signal Metrics</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-value">{metrics['sensational_score']:.2f}</div><div class="kpi-label">Sensational Score</div></div>
  <div class="kpi"><div class="kpi-value">{metrics['upper_ratio']:.2f}</div><div class="kpi-label">Uppercase Ratio</div></div>
  <div class="kpi"><div class="kpi-value">{metrics['exclam']}</div><div class="kpi-label">Exclaim! Count</div></div>
  <div class="kpi"><div class="kpi-value">{metrics['emo_hits']}</div><div class="kpi-label">Emotional Words</div></div>
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
                f"<div class='fact-meta'>{html.escape(fc.get('message', 'Google Fact Check API is not configured.'))}</div>",
                unsafe_allow_html=True,
            )
        elif not fc.get("query"):
            st.markdown(
                f"<div class='fact-meta'>{html.escape(fc.get('message', 'No strong factual claim detected for external verification.'))}</div>",
                unsafe_allow_html=True,
            )
        elif fc.get("count", 0) == 0:
            st.markdown(
                f"<div class='fact-meta'>Query used: <b>{html.escape(fc.get('query', ''))}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='fact-meta'>Sources found: {int(fc.get('count', 0))}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='fact-meta'>{html.escape(fc.get('message', 'No matching fact-check sources found.'))}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='fact-meta'>Query used: <b>{html.escape(fc.get('query', ''))}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='fact-meta'>Sources found: {int(fc.get('count', 0))}</div>",
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
    "<div class='footer'>Analyzes writing-style signals + external fact check sources · Not a substitute for editorial judgment</div>",
    unsafe_allow_html=True,
)
