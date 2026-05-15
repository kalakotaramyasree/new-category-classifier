import streamlit as st
import pickle
import re
import os

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered",
)

# ── Basic stopwords (no NLTK download needed) ────────────────────────────────
STOPWORDS = {
    'the','a','an','in','on','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','shall','should',
    'may','might','must','can','could','to','of','and','or','but','if',
    'then','than','so','as','at','by','for','with','about','against',
    'into','through','during','before','after','above','below','from',
    'up','down','out','off','over','under','again','further','once',
    'here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only',
    'own','same','too','very','i','my','me','we','our','you','your',
    'he','his','him','she','her','it','its','they','their','this','that',
    'these','those','what','which','who','whom','just','because'
}

# ── Text cleaning (mirrors your notebook) ───────────────────────────────────
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(base, "mnb_model.pkl"), "rb") as f:
        mnb = pickle.load(f)
    return tfidf, mnb

tfidf, mnb = load_models()

# ── Category metadata ────────────────────────────────────────────────────────
CATEGORY_META = {
    "sport":         {"emoji": "⚽", "color": "#22c55e", "desc": "Sports & Athletics"},
    "politics":      {"emoji": "🏛️",  "color": "#3b82f6", "desc": "Politics & Government"},
    "business":      {"emoji": "💼", "color": "#f59e0b", "desc": "Business & Economy"},
    "tech":          {"emoji": "💻", "color": "#8b5cf6", "desc": "Technology & Science"},
    "entertainment": {"emoji": "🎬", "color": "#ec4899", "desc": "Entertainment & Media"},
}

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("📰 BBC News Text Classifier")
st.markdown("**Multinomial Naive Bayes** · TF-IDF features · 5 categories")
st.divider()

# Sample articles for quick testing
SAMPLES = {
    "🏏 Sport":          "India won the cricket world cup final against Australia in a thrilling match. The team scored 287 runs with brilliant batting performance from the top order. Bowlers defended the total successfully to win the championship.",
    "🏛️ Politics":       "The prime minister announced a new economic reform policy in parliament today. The government aims to reduce inflation and create employment through budget allocations for infrastructure and education sectors.",
    "💼 Business":       "Stock markets rallied today as quarterly earnings from major technology companies exceeded analyst expectations. The S&P 500 rose two percent while investors reacted positively to strong revenue growth and profit margins.",
    "💻 Technology":     "OpenAI released a new large language model with improved reasoning capabilities. The model outperforms previous versions on coding benchmarks and shows enhanced performance on complex mathematical problems.",
    "🎬 Entertainment":  "The blockbuster film broke opening weekend box office records globally. The director received critical acclaim for the stunning visual effects and the lead actor is expected to receive an Oscar nomination.",
}

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Enter News Article")
with col2:
    sample_choice = st.selectbox("Load sample →", ["(type your own)"] + list(SAMPLES.keys()), label_visibility="collapsed")

default_text = SAMPLES.get(sample_choice, "") if sample_choice != "(type your own)" else ""
user_text = st.text_area(
    "Paste a news article or headline below:",
    value=default_text,
    height=180,
    placeholder="e.g. The government announced new budget reforms targeting inflation…",
)

predict_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        cleaned = clean(user_text)
        if not cleaned.strip():
            st.error("Text is empty after cleaning. Try a longer article.")
        else:
            vec = tfidf.transform([cleaned]).toarray()
            prediction = mnb.predict(vec)[0]
            proba = mnb.predict_proba(vec)[0]
            classes = mnb.classes_

            meta = CATEGORY_META.get(prediction, {"emoji": "📄", "color": "#6b7280", "desc": prediction.title()})

            st.divider()
            st.subheader("Result")

            # Big result card
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {meta['color']}22, {meta['color']}11);
                    border: 2px solid {meta['color']};
                    border-radius: 12px;
                    padding: 24px;
                    text-align: center;
                    margin-bottom: 20px;
                ">
                    <div style="font-size: 56px;">{meta['emoji']}</div>
                    <div style="font-size: 28px; font-weight: 700; color: {meta['color']};">{prediction.upper()}</div>
                    <div style="font-size: 16px; color: #6b7280; margin-top: 4px;">{meta['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence bars
            st.markdown("**Confidence scores across all categories:**")
            sorted_pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
            for cls, prob in sorted_pairs:
                m = CATEGORY_META.get(cls, {"emoji": "📄", "color": "#6b7280"})
                bar_pct = int(prob * 100)
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; margin-bottom:8px; gap:10px;">
                        <span style="width:90px; font-size:14px;">{m['emoji']} {cls}</span>
                        <div style="flex:1; background:#e5e7eb; border-radius:6px; height:18px; overflow:hidden;">
                            <div style="width:{bar_pct}%; background:{m['color']}; height:100%; border-radius:6px;"></div>
                        </div>
                        <span style="width:46px; text-align:right; font-size:14px; font-weight:600;">{prob:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            with st.expander("🔎 See cleaned text"):
                st.code(cleaned)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:13px;'>"
    "Built with Streamlit · Multinomial Naive Bayes · TF-IDF Vectorizer"
    "</p>",
    unsafe_allow_html=True,
)
