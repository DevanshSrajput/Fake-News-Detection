import os
import pickle
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fake_news_detector import FakeNewsDetector

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


SAMPLE_ARTICLES = {
    "Market Update": (
        "The stock market experienced significant volatility today as investors reacted "
        "to new economic data released by the Federal Reserve. Trading volume was higher "
        "than average as major indices fluctuated throughout the session."
    ),
    "Suspicious Headline": (
        "SHOCKING: Scientists discover that eating pizza cures all diseases instantly! "
        "Doctors hate this one trick that pharmaceutical companies do not want you to know!"
    ),
    "City Policy Brief": (
        "The mayor announced new infrastructure improvements including road repairs and "
        "public transportation upgrades, with construction expected to begin next month "
        "after city council approval."
    ),
}

SENSATIONAL_TERMS = [
    "shocking",
    "miracle",
    "secret",
    "doctors hate",
    "you won't believe",
    "viral",
    "click here",
    "breaking",
    "exclusive",
    "guaranteed",
]

PAGES = ["🏠 Home", "📊 Train Models", "🔍 Detect News", "📈 Model Analytics", "ℹ️ About"]


def initialize_session_state():
    defaults = {
        "detector": None,
        "model_trained": False,
        "training_data": None,
        "model_metrics": None,
        "current_page": "🏠 Home",
        "analysis_history": [],
        "url_article_text": "",
        "url_article_meta": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Sans+3:wght@400;600&display=swap');

            :root {
                --bg-main: #f7f4ec;
                --bg-card: rgba(255, 252, 247, 0.92);
                --bg-soft: #f1e6d2;
                --text-main: #4f4036;
                --text-muted: #5f5147;
                --accent: #d9643a;
                --accent-dark: #8d4e2f;
                --accent-soft: #fff1ea;
                --success: #1d8f6d;
                --danger: #c23b3b;
                --border: rgba(141, 78, 47, 0.12);
                --shadow: 0 14px 35px rgba(141, 78, 47, 0.10);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(217, 100, 58, 0.12), transparent 28%),
                    radial-gradient(circle at top left, rgba(206, 163, 114, 0.14), transparent 26%),
                    var(--bg-main);
                color: var(--text-main);
            }

            html, body, [class*="css"] {
                font-family: "Source Sans 3", sans-serif;
            }

            p, li, label, .stMarkdown, .stCaption, .stAlert, .stCheckbox, .stRadio,
            .stSelectbox, .stTextInput, .stTextArea, .stNumberInput, .stFileUploader,
            div[data-testid="stMarkdownContainer"], div[data-testid="stText"], small {
                color: var(--text-main) !important;
            }

            h1, h2, h3, h4 {
                font-family: "Space Grotesk", sans-serif;
                letter-spacing: -0.02em;
                color: var(--accent-dark);
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #fff8f0 0%, #f1e4d4 100%);
                border-right: 1px solid rgba(141, 78, 47, 0.12);
            }

            section[data-testid="stSidebar"] * {
                color: var(--text-main);
            }

            .hero-shell {
                background: linear-gradient(135deg, rgba(255, 248, 240, 0.98), rgba(248, 233, 214, 0.96));
                border-radius: 24px;
                padding: 2.2rem;
                color: var(--text-main);
                box-shadow: var(--shadow);
                border: 1px solid rgba(141, 78, 47, 0.12);
                margin-bottom: 1.25rem;
            }

            .hero-kicker {
                text-transform: uppercase;
                font-size: 0.82rem;
                letter-spacing: 0.16em;
                color: rgba(79, 64, 54, 0.72);
                margin-bottom: 0.8rem;
            }

            .hero-title {
                font-size: clamp(2.1rem, 4vw, 3.4rem);
                font-weight: 700;
                line-height: 1.05;
                margin-bottom: 0.8rem;
            }

            .hero-copy {
                font-size: 1.05rem;
                color: rgba(79, 64, 54, 0.86);
                max-width: 760px;
                margin-bottom: 0;
            }

            .panel-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1.2rem;
                box-shadow: var(--shadow);
                backdrop-filter: blur(8px);
                margin-bottom: 1rem;
            }

            .panel-title {
                font-family: "Space Grotesk", sans-serif;
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--accent-dark);
                margin-bottom: 0.35rem;
            }

            .panel-copy {
                color: var(--text-muted);
                margin-bottom: 0;
            }

            .stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.9rem;
                margin: 1rem 0 0.25rem 0;
            }

            .stat-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,244,235,0.98));
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
                min-height: 112px;
            }

            .stat-label {
                font-size: 0.86rem;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
            }

            .stat-value {
                font-family: "Space Grotesk", sans-serif;
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--accent-dark);
                margin-bottom: 0.15rem;
            }

            .stat-subtext {
                font-size: 0.94rem;
                color: var(--text-muted);
            }

            .status-pill {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                font-size: 0.84rem;
                font-weight: 600;
                margin-right: 0.45rem;
                margin-bottom: 0.45rem;
            }

            .status-pill.soft {
                background: rgba(217, 100, 58, 0.10);
                color: var(--accent-dark);
            }

            .status-pill.real {
                background: rgba(29, 143, 109, 0.12);
                color: var(--success);
            }

            .status-pill.fake {
                background: rgba(194, 59, 59, 0.12);
                color: var(--danger);
            }

            .result-banner {
                border-radius: 22px;
                padding: 1.4rem;
                margin: 1rem 0;
                border: 1px solid transparent;
                box-shadow: var(--shadow);
            }

            .result-banner.real {
                background: linear-gradient(135deg, rgba(29, 143, 109, 0.12), rgba(255,255,255,0.92));
                border-color: rgba(29, 143, 109, 0.18);
            }

            .result-banner.fake {
                background: linear-gradient(135deg, rgba(194, 59, 59, 0.12), rgba(255,255,255,0.92));
                border-color: rgba(194, 59, 59, 0.18);
            }

            .result-title {
                font-family: "Space Grotesk", sans-serif;
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                color: var(--accent-dark);
            }

            .section-label {
                font-family: "Space Grotesk", sans-serif;
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--accent-dark);
                margin: 1.1rem 0 0.6rem 0;
            }

            .micro-note {
                color: var(--text-muted);
                font-size: 0.94rem;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.7rem;
                box-shadow: var(--shadow);
            }

            .stButton > button {
                background: linear-gradient(180deg, #fffaf4, #f2e5d6);
                color: var(--accent-dark);
                border: 1px solid rgba(141, 78, 47, 0.16);
                border-radius: 14px;
                font-weight: 600;
                box-shadow: 0 8px 18px rgba(141, 78, 47, 0.08);
            }

            .stButton > button:hover {
                border-color: rgba(141, 78, 47, 0.28);
                color: var(--accent-dark);
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #d9643a, #c7512d);
                color: var(--text-main);
                border: 1px solid rgba(199, 81, 45, 0.35);
            }

            .stButton > button[kind="primary"]:hover {
                color: var(--text-main);
                border-color: rgba(199, 81, 45, 0.55);
            }

            div[data-baseweb="input"] > div,
            div[data-baseweb="base-input"] > div,
            div[data-baseweb="select"] > div,
            .stTextInput input,
            .stTextArea textarea,
            .stNumberInput input,
            .stSelectbox [data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="select"] > div {
                background: linear-gradient(180deg, #fffaf4, #f4e8da) !important;
                color: var(--text-main) !important;
                border: 1px solid rgba(141, 78, 47, 0.18) !important;
                border-radius: 14px !important;
                box-shadow: 0 6px 14px rgba(141, 78, 47, 0.06);
            }

            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder,
            .stNumberInput input::placeholder {
                color: rgba(79, 64, 54, 0.58) !important;
            }

            div[data-baseweb="select"] span,
            div[data-baseweb="select"] input,
            .stTextInput input,
            .stTextArea textarea,
            .stNumberInput input {
                color: var(--text-main) !important;
            }

            div[data-baseweb="popover"],
            div[data-baseweb="menu"],
            ul[role="listbox"] {
                background: #fff8f0 !important;
                color: var(--text-main) !important;
                border: 1px solid rgba(141, 78, 47, 0.16) !important;
                box-shadow: 0 12px 24px rgba(141, 78, 47, 0.10) !important;
            }

            li[role="option"] {
                background: #fff8f0 !important;
                color: var(--text-main) !important;
            }

            li[role="option"]:hover {
                background: #f6e8d7 !important;
            }

            .stFileUploader > div {
                background: linear-gradient(180deg, #fffaf4, #f4e8da) !important;
                border: 1px dashed rgba(141, 78, 47, 0.30) !important;
                border-radius: 16px !important;
            }

            .stRadio > div {
                background: transparent !important;
                color: var(--text-main) !important;
            }

            .stRadio label,
            .stSelectbox label,
            .stTextInput label,
            .stTextArea label,
            .stNumberInput label,
            .stFileUploader label {
                color: var(--text-main) !important;
            }

            div[data-testid="stSidebar"] div[data-baseweb="select"] > div,
            div[data-testid="stSidebar"] div[data-baseweb="input"] > div,
            div[data-testid="stSidebar"] .stTextInput input {
                background: linear-gradient(180deg, #fffaf4, #f4e8da) !important;
                color: var(--text-main) !important;
                border: 1px solid rgba(141, 78, 47, 0.18) !important;
            }

            .history-table {
                border-radius: 18px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_model(detector, filename="trained_model.pkl"):
    try:
        with open(filename, "wb") as file_handle:
            pickle.dump(detector, file_handle)
        return True
    except Exception as exc:
        st.error(f"Error saving model: {exc}")
        return False


def load_model(filename="trained_model.pkl"):
    try:
        if os.path.exists(filename):
            with open(filename, "rb") as file_handle:
                return pickle.load(file_handle)
        return None
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


def activate_detector(detector):
    st.session_state.detector = detector
    st.session_state.model_trained = detector is not None


def render_hero(title, copy, pills=None):
    pill_markup = ""
    if pills:
        pill_markup = "".join(
            f'<span class="status-pill soft">{pill}</span>' for pill in pills
        )
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Fake News Detection Workspace</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{copy}</p>
            <div style="margin-top: 1rem;">{pill_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title, copy):
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{title}</div>
            <p class="panel-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(stats):
    card_markup = "".join(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-subtext">{subtext}</div>
        </div>
        """
        for label, value, subtext in stats
    )
    st.markdown(f'<div class="stat-grid">{card_markup}</div>', unsafe_allow_html=True)


def create_comparison_chart(metrics):
    if not metrics:
        return None

    pac_metrics, svm_metrics = metrics
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Accuracy", "F1 Score"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Bar(
            x=["Passive Aggressive", "SVM"],
            y=[pac_metrics[0], svm_metrics[0]],
            marker_color=["#c58a5c", "#d9643a"],
            text=[f"{pac_metrics[0]:.3f}", f"{svm_metrics[0]:.3f}"],
            textposition="outside",
            name="Accuracy",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["Passive Aggressive", "SVM"],
            y=[pac_metrics[1], svm_metrics[1]],
            marker_color=["#2f7d65", "#ef8d5a"],
            text=[f"{pac_metrics[1]:.3f}", f"{svm_metrics[1]:.3f}"],
            textposition="outside",
            name="F1 Score",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Model Performance Comparison",
        height=420,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_yaxes(range=[0, 1], gridcolor="rgba(141,78,47,0.12)")
    return fig


def create_confusion_matrix_chart(confusion_matrix, model_name):
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["REAL", "FAKE"],
            y=["REAL", "FAKE"],
            colorscale=[
                [0.0, "#fff1ea"],
                [0.5, "#f7b28d"],
                [1.0, "#c58a5c"],
            ],
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def create_confidence_chart(results):
    confidence_frame = pd.DataFrame(
        {
            "Model": ["Passive Aggressive", "SVM"],
            "Confidence": [
                round(results["pac_confidence"] * 100, 2),
                round(results["svm_confidence"] * 100, 2),
            ],
            "Prediction": [results["pac_prediction"], results["svm_prediction"]],
        }
    )
    fig = px.bar(
        confidence_frame,
        x="Model",
        y="Confidence",
        color="Prediction",
        color_discrete_map={"REAL": "#1d8f6d", "FAKE": "#c23b3b"},
        text="Confidence",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        title="Per-Model Confidence",
        yaxis_title="Confidence (%)",
        xaxis_title="",
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(range=[0, 105], gridcolor="rgba(141,78,47,0.12)")
    return fig


def create_agreement_gauge(results):
    agreement_delta = 8 if results["agreement"] else -12
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=results["combined_confidence"] * 100,
            number={"suffix": "%"},
            delta={"reference": 75 + agreement_delta},
            title={"text": "Consensus Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c58a5c"},
                "steps": [
                    {"range": [0, 50], "color": "#fbe0d4"},
                    {"range": [50, 80], "color": "#f0b79b"},
                    {"range": [80, 100], "color": "#d9643a"},
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_radar_chart(pac_metrics, svm_metrics):
    categories = ["Accuracy", "F1 Score"]
    pac_values = [pac_metrics[0], pac_metrics[1]]
    svm_values = [svm_metrics[0], svm_metrics[1]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=pac_values + [pac_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Passive Aggressive",
            line_color="#c58a5c",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=svm_values + [svm_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="SVM",
            line_color="#d9643a",
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        title="Model Shape Comparison",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def extract_text_from_html(html):
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "form"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled article"

        candidate_groups = []
        for selector in ["article p", "main p", "section p", "p"]:
            paragraphs = [
                paragraph.get_text(" ", strip=True)
                for paragraph in soup.select(selector)
                if paragraph.get_text(" ", strip=True)
            ]
            combined = " ".join(paragraphs)
            candidate_groups.append((len(combined.split()), combined))

        candidate_groups.sort(key=lambda item: item[0], reverse=True)
        best_text = candidate_groups[0][1] if candidate_groups else ""
    else:
        title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Untitled article"
        stripped = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<style.*?>.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL)
        best_text = re.sub(r"<[^>]+>", " ", stripped)

    clean_text = re.sub(r"\s+", " ", best_text).strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title, clean_text


def fetch_article_from_url(url):
    if not url.strip():
        return None, "Enter a news article URL to fetch."

    try:
        request = urllib.request.Request(
            url.strip(),
            headers={"User-Agent": "Mozilla/5.0 FakeNewsDetector/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_html = response.read()

        if "text/html" not in content_type:
            return None, "The URL did not return an HTML page that looks like an article."

        html = raw_html.decode("utf-8", errors="ignore")
        title, text = extract_text_from_html(html)
        if len(text.split()) < 40:
            return None, "I could not extract enough article text from that page."

        parsed_url = urlparse(url)
        return {
            "title": title,
            "domain": parsed_url.netloc.replace("www.", ""),
            "text": text[:20000],
        }, None
    except urllib.error.HTTPError as exc:
        return None, f"Could not fetch the article. The server returned HTTP {exc.code}."
    except urllib.error.URLError:
        return None, "Could not reach that URL. Check the link and your network connection."
    except Exception as exc:
        return None, f"Unexpected error while fetching the article: {exc}"


def compute_article_signals(article_text):
    words = re.findall(r"\b[\w'-]+\b", article_text)
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", article_text) if segment.strip()]
    letters = [character for character in article_text if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters) if letters else 0
    )
    sensational_hits = [
        term for term in SENSATIONAL_TERMS if term in article_text.lower()
    ]

    flags = []
    if len(words) < 60:
        flags.append("Short text gives the model less context, so the prediction is less reliable.")
    if uppercase_ratio > 0.18:
        flags.append("The article uses a high amount of all-caps text, which can be a clickbait signal.")
    if article_text.count("!") >= 3:
        flags.append("Frequent exclamation marks can indicate sensational framing.")
    if sensational_hits:
        flags.append(
            "Sensational phrases detected: " + ", ".join(sorted(set(sensational_hits[:4])))
        )

    return {
        "word_count": len(words),
        "character_count": len(article_text),
        "sentence_count": max(len(sentences), 1),
        "reading_time": max(1, round(len(words) / 200)),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "exclamation_count": article_text.count("!"),
        "question_count": article_text.count("?"),
        "uppercase_ratio": round(uppercase_ratio * 100, 2),
        "sensational_terms": sensational_hits,
        "flags": flags,
    }


def add_analysis_to_history(article_text, source_label, results, signals):
    history_entry = {
        "time": time.strftime("%H:%M:%S"),
        "source": source_label,
        "consensus": results["consensus"],
        "confidence": round(results["combined_confidence"] * 100, 2),
        "agreement": "Yes" if results["agreement"] else "No",
        "words": signals["word_count"],
        "preview": re.sub(r"\s+", " ", article_text).strip()[:120],
    }
    st.session_state.analysis_history.insert(0, history_entry)
    st.session_state.analysis_history = st.session_state.analysis_history[:8]


def render_history():
    if not st.session_state.analysis_history:
        st.info("No analyses yet. Run a prediction and your recent results will appear here.")
        return

    history_frame = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(
        history_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "time": "Time",
            "source": "Source",
            "consensus": "Consensus",
            "confidence": st.column_config.NumberColumn("Confidence (%)", format="%.2f"),
            "agreement": "Agreement",
            "words": "Words",
            "preview": "Preview",
        },
    )


def main():
    initialize_session_state()
    inject_styles()

    render_hero(
        "Spot questionable stories with a cleaner workflow and sharper insight.",
        (
            "Train on your own dataset, inspect model behavior, analyze pasted text or fetched "
            "articles, and keep a lightweight prediction history without changing the core project."
        ),
        pills=[
            "Dual-model classifier",
            "Interactive Streamlit UI",
            "URL article extraction",
            "Prediction history",
        ],
    )

    with st.sidebar:
        st.title("Control Center")
        st.caption("Keep the project flow the same, just smoother.")

        model_file_exists = os.path.exists("trained_model.pkl")
        st.metric("Saved model", "Available" if model_file_exists else "Missing")
        st.metric("Session model", "Ready" if st.session_state.model_trained else "Not loaded")

        if model_file_exists and not st.session_state.model_trained:
            if st.button("Load Saved Model", use_container_width=True):
                detector = load_model()
                if detector:
                    activate_detector(detector)
                    st.success("Saved model loaded into the session.")
                    st.rerun()

        page = st.selectbox(
            "Navigate to",
            PAGES,
            index=PAGES.index(st.session_state.current_page),
        )
        st.session_state.current_page = page

        st.markdown("---")
        st.markdown("**Supported labels**")
        st.caption("`REAL`, `TRUE`, `0` and `FAKE`, `FALSE`, `1`")

    if page == "🏠 Home":
        home_page()
    elif page == "📊 Train Models":
        train_models_page()
    elif page == "🔍 Detect News":
        detect_news_page()
    elif page == "📈 Model Analytics":
        analytics_page()
    elif page == "ℹ️ About":
        about_page()


def home_page():
    stats = [
        ("Model status", "Ready" if st.session_state.model_trained else "Waiting", "Load or train a detector"),
        ("Saved model", "Yes" if os.path.exists("trained_model.pkl") else "No", "Persistence is enabled"),
        ("Recent analyses", len(st.session_state.analysis_history), "Session-only history"),
        (
            "Analytics",
            "Live" if st.session_state.model_metrics else "Pending",
            "Available after training in-session",
        ),
    ]
    render_stat_cards(stats)

    col1, col2 = st.columns([1.25, 1])
    with col1:
        render_panel(
            "What is better now",
            (
                "The app keeps the same train and predict flow, but the interface is more polished, "
                "the article analysis is clearer, URL fetching is usable, and predictions now show "
                "confidence from both models instead of a single raw score."
            ),
        )
    with col2:
        render_panel(
            "Recommended path",
            (
                "If you already have `trained_model.pkl`, load it from the sidebar and jump to detection. "
                "If not, start with the training page and either use the bundled dataset file or upload your own CSV."
            ),
        )

    st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Open Training", use_container_width=True):
            st.session_state.current_page = "📊 Train Models"
            st.rerun()
    with col2:
        if st.button("Open Detection", use_container_width=True):
            st.session_state.current_page = "🔍 Detect News"
            st.rerun()
    with col3:
        if st.button("Open Analytics", use_container_width=True):
            st.session_state.current_page = "📈 Model Analytics"
            st.rerun()

    st.markdown('<div class="section-label">Workflow</div>', unsafe_allow_html=True)
    workflow_columns = st.columns(3)
    workflow_cards = [
        ("1. Prepare data", "Use a CSV with `text` and `label` columns. Sample size limiting is built in for faster experiments."),
        ("2. Train and compare", "The app trains Passive Aggressive and SVM models, then compares accuracy and F1 side by side."),
        ("3. Analyze articles", "Paste text, upload a `.txt` file, or pull an article from a URL and review the model agreement."),
    ]
    for column, (title, copy) in zip(workflow_columns, workflow_cards):
        with column:
            render_panel(title, copy)

    st.markdown('<div class="section-label">Recent Session Activity</div>', unsafe_allow_html=True)
    render_history()


def train_models_page():
    st.markdown('<div class="section-label">Train Detection Models</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="micro-note">Use the existing dataset file or upload a CSV. '
        "The training flow is unchanged, but the page now previews your data and surfaces clearer feedback.</p>",
        unsafe_allow_html=True,
    )

    if os.path.exists("trained_model.pkl"):
        st.info("A saved model file is available. You can load it instantly or train a fresh one.")
        if st.button("Load Existing Trained Model"):
            detector = load_model()
            if detector:
                activate_detector(detector)
                st.success("Saved model loaded successfully.")

    dataset_path = "WELFake_Dataset.csv"
    dataset_exists = os.path.exists(dataset_path)

    config_col, preview_col = st.columns([1.05, 1])

    with config_col:
        st.markdown("### Dataset Setup")
        use_existing = dataset_exists and st.checkbox("Use `WELFake_Dataset.csv` from the project root", value=True)

        uploaded_file = None
        if not use_existing:
            uploaded_file = st.file_uploader(
                "Upload a CSV dataset",
                type=["csv"],
                help="The file should contain a text column and a label column.",
            )

        text_column = st.text_input("Text column", value="text")
        label_column = st.text_input("Label column", value="label")
        sample_size = st.number_input(
            "Sample size (0 uses the full dataset)",
            min_value=0,
            max_value=200000,
            value=10000,
            step=500,
        )
        sample_size = None if sample_size == 0 else int(sample_size)

        st.caption("Supported labels: `REAL`, `TRUE`, `0`, `FAKE`, `FALSE`, `1`")
        st.caption("Training now uses an optimized linear SVM so larger text datasets finish much faster.")

        if st.button("Start Training", type="primary", use_container_width=True):
            if use_existing:
                train_models(dataset_path, text_column, label_column, sample_size)
            elif uploaded_file is not None:
                train_models(uploaded_file, text_column, label_column, sample_size)
            else:
                st.error("Upload a CSV dataset or use the existing file first.")

    with preview_col:
        st.markdown("### Data Preview")
        preview_frame = None
        try:
            if use_existing:
                if dataset_exists:
                    preview_frame = pd.read_csv(dataset_path, nrows=5)
                else:
                    st.warning("`WELFake_Dataset.csv` is not present in the project root.")
            elif uploaded_file is not None:
                uploaded_file.seek(0)
                preview_frame = pd.read_csv(uploaded_file, nrows=5)
                uploaded_file.seek(0)
        except Exception as exc:
            st.error(f"Could not preview dataset: {exc}")

        if preview_frame is not None:
            st.dataframe(preview_frame, use_container_width=True, hide_index=True)
            st.caption(f"Preview columns: {', '.join(preview_frame.columns.astype(str).tolist())}")
        else:
            render_panel(
                "Preview will appear here",
                "Once a CSV source is available, the first few rows and detected columns will show up in this panel.",
            )

    if st.session_state.model_trained and st.session_state.model_metrics:
        display_training_results()


def train_models(dataset_source, text_col, label_col, sample_size):
    try:
        detector = st.session_state.detector or FakeNewsDetector()
        st.session_state.detector = detector

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Loading dataset...")
        progress_bar.progress(10)

        if isinstance(dataset_source, str):
            df = detector.load_dataset_from_csv(dataset_source, text_col, label_col, sample_size)
        else:
            dataset_source.seek(0)
            df = pd.read_csv(dataset_source)
            required_columns = {text_col, label_col}
            missing_columns = required_columns.difference(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            df = df.dropna(subset=[text_col, label_col])
            df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Training data must contain text and label fields after loading.")

        st.session_state.training_data = df.copy()
        progress_bar.progress(25)

        status_text.text("Preprocessing text...")
        X_train, X_test, y_train, y_test = detector.prepare_data(df)
        progress_bar.progress(55)

        def update_training_progress(message, progress_value):
            status_text.text(message)
            progress_bar.progress(progress_value)

        training_stats = detector.train_models(
            X_train,
            y_train,
            progress_callback=update_training_progress,
        )
        progress_bar.progress(82)

        status_text.text("Evaluating models...")
        pac_accuracy, pac_f1, pac_cm, _ = detector.evaluate_model(
            detector.pac_model,
            X_test,
            y_test,
            "Passive Aggressive Classifier",
        )
        svm_accuracy, svm_f1, svm_cm, _ = detector.evaluate_model(
            detector.svm_model,
            X_test,
            y_test,
            "Support Vector Machine",
        )

        st.session_state.model_metrics = {
            "pac_metrics": (pac_accuracy, pac_f1, pac_cm),
            "svm_metrics": (svm_accuracy, svm_f1, svm_cm),
            "test_data": (X_test, y_test),
            "training_stats": training_stats,
        }

        status_text.text("Saving model...")
        save_model(detector)
        progress_bar.progress(100)

        st.session_state.model_trained = True
        st.success("Training completed successfully.")

        time.sleep(0.6)
        progress_bar.empty()
        status_text.empty()
    except Exception as exc:
        st.error(f"Training failed: {exc}")


def display_training_results():
    metrics = st.session_state.model_metrics
    pac_metrics = metrics["pac_metrics"]
    svm_metrics = metrics["svm_metrics"]
    training_stats = metrics.get("training_stats", {})
    best_model = "Passive Aggressive" if pac_metrics[0] >= svm_metrics[0] else "SVM"
    accuracy_gap = abs(pac_metrics[0] - svm_metrics[0])

    st.markdown('<div class="section-label">Training Results</div>', unsafe_allow_html=True)
    render_stat_cards(
        [
            ("PAC accuracy", f"{pac_metrics[0]:.3f}", "Passive Aggressive classifier"),
            ("PAC F1", f"{pac_metrics[1]:.3f}", "Balanced classification quality"),
            ("SVM accuracy", f"{svm_metrics[0]:.3f}", "Support Vector Machine"),
            ("Best model", best_model, f"Accuracy gap: {accuracy_gap:.3f}"),
        ]
    )

    if training_stats:
        st.caption(
            "Training times: "
            f"PAC {training_stats.get('pac_training_seconds', 0):.2f}s, "
            f"Linear SVM {training_stats.get('svm_training_seconds', 0):.2f}s"
        )

    chart_col, radar_col = st.columns([1.25, 1])
    with chart_col:
        st.plotly_chart(
            create_comparison_chart((pac_metrics[:2], svm_metrics[:2])),
            use_container_width=True,
        )
    with radar_col:
        st.plotly_chart(create_radar_chart(pac_metrics, svm_metrics), use_container_width=True)

    matrix_col1, matrix_col2 = st.columns(2)
    with matrix_col1:
        st.plotly_chart(
            create_confusion_matrix_chart(pac_metrics[2], "Passive Aggressive"),
            use_container_width=True,
        )
    with matrix_col2:
        st.plotly_chart(
            create_confusion_matrix_chart(svm_metrics[2], "SVM"),
            use_container_width=True,
        )


def detect_news_page():
    st.markdown('<div class="section-label">Analyze News Content</div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("Train a model first or load the saved model from the sidebar before analyzing articles.")
        return

    source_label = "Typed text"
    article_text = ""
    input_method = st.radio(
        "Input source",
        ["✍️ Paste text", "📁 Upload text file", "🔗 Fetch from URL"],
        horizontal=True,
    )

    if input_method == "✍️ Paste text":
        source_label = "Typed text"
        article_text = st.text_area(
            "Article text",
            height=220,
            placeholder="Paste a headline, article excerpt, or full story here...",
        )

    elif input_method == "📁 Upload text file":
        source_label = "Uploaded text file"
        uploaded_text = st.file_uploader("Upload a `.txt` article file", type=["txt"])
        if uploaded_text is not None:
            raw_bytes = uploaded_text.read()
            try:
                article_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                article_text = raw_bytes.decode("latin-1")
            st.text_area("Loaded content", value=article_text, height=220, disabled=True)

    else:
        source_label = "Fetched URL"
        url = st.text_input("Article URL", placeholder="https://example.com/news-story")
        if st.button("Fetch Article Text"):
            fetched_article, error_message = fetch_article_from_url(url)
            if error_message:
                st.error(error_message)
            else:
                st.session_state.url_article_text = fetched_article["text"]
                st.session_state.url_article_meta = fetched_article
                st.success(f"Fetched article from {fetched_article['domain']}")

        if st.session_state.url_article_text:
            article_text = st.session_state.url_article_text
            meta = st.session_state.url_article_meta
            st.text_input("Fetched title", value=meta.get("title", "Untitled article"), disabled=True)
            st.text_area("Fetched article text", value=article_text, height=220, disabled=True)

    st.markdown("### Try a sample article")
    selected_sample = st.selectbox("Sample article", [""] + list(SAMPLE_ARTICLES.keys()))
    if selected_sample:
        source_label = f"Sample: {selected_sample}"
        article_text = SAMPLE_ARTICLES[selected_sample]
        st.text_area("Sample text", value=article_text, height=180, disabled=True)

    if article_text.strip():
        signals = compute_article_signals(article_text)
        render_stat_cards(
            [
                ("Words", signals["word_count"], "Input size"),
                ("Sentences", signals["sentence_count"], "Basic structure"),
                ("Read time", f"{signals['reading_time']} min", "Estimated reading time"),
                ("All caps", f"{signals['uppercase_ratio']:.2f}%", "Uppercase ratio"),
            ]
        )

    if st.button("Analyze Article", type="primary", use_container_width=True, disabled=not article_text.strip()):
        analyze_article(article_text, source_label)

    st.markdown('<div class="section-label">Recent Analyses</div>', unsafe_allow_html=True)
    render_history()


def analyze_article(article_text, source_label):
    with st.spinner("Analyzing article..."):
        detector = st.session_state.detector
        results = detector.predict_article(article_text)

    if "error" in results:
        st.error(results["error"])
        return

    signals = compute_article_signals(article_text)
    add_analysis_to_history(article_text, source_label, results, signals)

    result_class = "fake" if results["consensus"] == "FAKE" else "real"
    result_copy = (
        "This content shows patterns that the models associate with misleading or fabricated reporting."
        if result_class == "fake"
        else "This content looks more consistent with legitimate reporting based on the current model."
    )
    agreement_copy = "Both models agree on this verdict." if results["agreement"] else "The two models disagree, so treat this call with extra caution."

    st.markdown(
        f"""
        <div class="result-banner {result_class}">
            <div class="result-title">{results["consensus"]} news signal</div>
            <p class="panel-copy">{result_copy}</p>
            <p class="panel-copy"><strong>Confidence:</strong> {results["combined_confidence"] * 100:.2f}% |
            <strong>Agreement:</strong> {agreement_copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_stat_cards(
        [
            ("Consensus", results["consensus"], "Final combined verdict"),
            ("Confidence", f"{results['combined_confidence'] * 100:.2f}%", "Average normalized certainty"),
            ("Agreement", "Yes" if results["agreement"] else "No", "Whether both models match"),
            ("Read time", f"{signals['reading_time']} min", "Estimated from word count"),
        ]
    )

    chart_col, gauge_col = st.columns([1.25, 1])
    with chart_col:
        st.plotly_chart(create_confidence_chart(results), use_container_width=True)
    with gauge_col:
        st.plotly_chart(create_agreement_gauge(results), use_container_width=True)

    breakdown_frame = pd.DataFrame(
        {
            "Model": ["Passive Aggressive", "SVM"],
            "Prediction": [results["pac_prediction"], results["svm_prediction"]],
            "Confidence (%)": [
                round(results["pac_confidence"] * 100, 2),
                round(results["svm_confidence"] * 100, 2),
            ],
            "Raw score": [round(results["pac_score"], 4), round(results["svm_score"], 4)],
        }
    )
    st.markdown("### Model breakdown")
    st.dataframe(breakdown_frame, use_container_width=True, hide_index=True)

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.markdown("### Article profile")
        st.write(
            pd.DataFrame(
                {
                    "Metric": [
                        "Word count",
                        "Character count",
                        "Sentence count",
                        "Average sentence length",
                        "Exclamation marks",
                        "Question marks",
                    ],
                    "Value": [
                        signals["word_count"],
                        signals["character_count"],
                        signals["sentence_count"],
                        signals["avg_sentence_length"],
                        signals["exclamation_count"],
                        signals["question_count"],
                    ],
                }
            )
        )
    with stats_col2:
        st.markdown("### Heuristic review")
        if signals["flags"]:
            for flag in signals["flags"]:
                st.warning(flag)
        else:
            st.success("No strong sensational-style warning flags were detected in the raw text.")

    with st.expander("Show analyzed text"):
        st.write(article_text)


def analytics_page():
    st.markdown('<div class="section-label">Model Analytics</div>', unsafe_allow_html=True)
    if not st.session_state.model_trained or not st.session_state.model_metrics:
        st.warning("Train a model in this session to unlock analytics and dataset insights.")
        return

    pac_metrics = st.session_state.model_metrics["pac_metrics"]
    svm_metrics = st.session_state.model_metrics["svm_metrics"]
    best_model = "Passive Aggressive" if pac_metrics[0] >= svm_metrics[0] else "SVM"

    render_stat_cards(
        [
            ("Best accuracy", f"{max(pac_metrics[0], svm_metrics[0]):.3f}", "Higher is better"),
            ("Best F1", f"{max(pac_metrics[1], svm_metrics[1]):.3f}", "Balance of precision and recall"),
            ("Winning model", best_model, "Based on accuracy"),
            ("Model gap", f"{abs(pac_metrics[0] - svm_metrics[0]):.3f}", "Accuracy difference"),
        ]
    )

    compare_col, radar_col = st.columns([1.25, 1])
    with compare_col:
        st.plotly_chart(
            create_comparison_chart((pac_metrics[:2], svm_metrics[:2])),
            use_container_width=True,
        )
    with radar_col:
        st.plotly_chart(create_radar_chart(pac_metrics, svm_metrics), use_container_width=True)

    matrix_col1, matrix_col2 = st.columns(2)
    with matrix_col1:
        st.plotly_chart(
            create_confusion_matrix_chart(pac_metrics[2], "Passive Aggressive"),
            use_container_width=True,
        )
    with matrix_col2:
        st.plotly_chart(
            create_confusion_matrix_chart(svm_metrics[2], "SVM"),
            use_container_width=True,
        )

    if st.session_state.training_data is not None:
        st.markdown("### Dataset insight")
        data_frame = st.session_state.training_data.copy()
        normalized_labels = data_frame["label"].astype(str).str.upper().replace(
            {"TRUE": "REAL", "FALSE": "FAKE", "0": "REAL", "1": "FAKE"}
        )
        data_frame["normalized_label"] = normalized_labels
        data_frame["word_count"] = data_frame["text"].astype(str).str.split().str.len()

        dataset_col1, dataset_col2, dataset_col3 = st.columns(3)
        with dataset_col1:
            st.metric("Articles", len(data_frame))
        with dataset_col2:
            st.metric("Real samples", int((normalized_labels == "REAL").sum()))
        with dataset_col3:
            st.metric("Fake samples", int((normalized_labels == "FAKE").sum()))

        distribution_col, length_col = st.columns(2)
        with distribution_col:
            label_counts = data_frame["normalized_label"].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Label distribution",
                color=label_counts.index,
                color_discrete_map={"REAL": "#1d8f6d", "FAKE": "#c23b3b"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        with length_col:
            fig = px.histogram(
                data_frame,
                x="word_count",
                color="normalized_label",
                nbins=30,
                title="Article length distribution",
                color_discrete_map={"REAL": "#1d8f6d", "FAKE": "#c23b3b"},
            )
            fig.update_layout(
                xaxis_title="Words per article",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def about_page():
    st.markdown('<div class="section-label">About This Project</div>', unsafe_allow_html=True)
    render_panel(
        "What the project does",
        (
            "This app trains two traditional machine learning models on labeled news text and then uses "
            "their predictions together to flag content as likely real or fake. It is a practical NLP demo "
            "with a clearer interface now, not a fact-checking authority."
        ),
    )

    render_stat_cards(
        [
            ("Models", "PAC + SVM", "Two complementary linear classifiers"),
            ("Features", "TF-IDF", "Sparse text representation"),
            ("Preprocessing", "NLTK + regex", "Cleaning, stopwords, stemming"),
            ("Frontend", "Streamlit", "Interactive single-file web app"),
        ]
    )

    st.markdown("### What changed in this enhancement")
    st.write(
        """
        - The UI was refreshed with a more deliberate visual system and stronger information hierarchy.
        - URL-based article fetching now works for many standard article pages.
        - Predictions show confidence from both models plus agreement and simple text heuristics.
        - Recent analyses are stored in-session so you can compare a few runs quickly.
        """
    )

    st.markdown("### Practical notes")
    st.write(
        """
        - Longer, well-formed articles usually produce more stable predictions.
        - If the two models disagree, treat the result as a soft signal rather than a final answer.
        - Analytics are available only after training during the current session because they depend on fresh evaluation output.
        """
    )


if __name__ == "__main__":
    main()
