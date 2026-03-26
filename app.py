import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import requests
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = os.path.join(BASE_DIR, "vectorizer.jb")
MODEL_PATH = os.path.join(BASE_DIR, "lr_model.jb")

MAX_ARTICLE_CHARS = 8000  # Prevent huge inputs to the vectorizer
NEWSAPI_BASE_URL = "https://newsapi.org/v2"


@st.cache_resource
def load_artifacts() -> Tuple[Any, Any]:
    if not os.path.exists(VECTOR_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Missing model artifacts. Expected `vectorizer.jb` and `lr_model.jb` next to app.py."
        )
    vectorizer = joblib.load(VECTOR_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def get_newsapi_key() -> Optional[str]:
    # Prefer Streamlit secrets; fall back to environment variable.
    try:
        return st.secrets["NEWSAPI_KEY"]  # type: ignore[index]
    except Exception:
        return os.environ.get("NEWSAPI_KEY")


def classify_text(vectorizer: Any, model: Any, text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {"verdict": None, "confidence_real": None}

    transform_input = vectorizer.transform([cleaned])
    prediction = model.predict(transform_input)
    pred_label = int(prediction[0])
    verdict = "Real" if pred_label == 1 else "Fake"

    confidence_real: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(transform_input)[0]
        # Map probability of class `1` to "Real"
        if hasattr(model, "classes_") and 1 in list(model.classes_):
            real_idx = list(model.classes_).index(1)
            confidence_real = float(proba[real_idx])
        else:
            # Best-effort fallback if we can't identify which column means class=1
            confidence_real = float(proba[-1])

    return {"verdict": verdict, "confidence_real": confidence_real}


def article_to_text(article: Dict[str, Any]) -> str:
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    content = (article.get("content") or "").strip()

    text = " ".join([title, description, content]).strip()
    if not text:
        return ""
    return text[:MAX_ARTICLE_CHARS]


def fetch_newsapi_top_headlines(
    api_key: str,
    country: str,
    language: str,
    page_size: int,
) -> List[Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "country": country,
        "language": language,
        "pageSize": page_size,
    }
    r = requests.get(f"{NEWSAPI_BASE_URL}/top-headlines", params=params, timeout=20)
    if r.status_code != 200:
        # Include NewsAPI's error payload (often has useful `message`).
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text[:500]}
        raise RuntimeError(f"NewsAPI error HTTP {r.status_code}: {payload}")
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data}")
    return data.get("articles", []) or []


def fetch_newsapi_everything(
    api_key: str,
    query: str,
    language: str,
    page_size: int,
) -> List[Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt",
    }
    r = requests.get(f"{NEWSAPI_BASE_URL}/everything", params=params, timeout=20)
    if r.status_code != 200:
        # Include NewsAPI's error payload (often has useful `message`).
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text[:500]}
        raise RuntimeError(f"NewsAPI error HTTP {r.status_code}: {payload}")
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data}")
    return data.get("articles", []) or []


st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

vectorizer, model = (None, None)
try:
    vectorizer, model = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# -------------------------
# Claim input (existing flow)
# -------------------------
news_input = st.text_area("News Article:", "")
if st.button("Check News"):
    if news_input.strip():
        result = classify_text(vectorizer, model, news_input)
        if result["verdict"] == "Real":
            st.success("The news is Real")
        else:
            st.error("The news is FAKE")
        if result.get("confidence_real") is not None:
            st.info(f"Confidence (Real): {result['confidence_real']:.2%}")
    else:
        st.warning("Please enter some text to analyze.")


# -------------------------
# Real-time headline input
# -------------------------
st.divider()
st.header("Real-time Headline Check (NewsAPI)")

api_key_from_env = get_newsapi_key()
api_key_input = st.text_input(
    "NewsAPI key (optional override)",
    value="",
    type="password",
    help="If provided, this key will be used instead of environment/secrets for the real-time fetch."
)
api_key = (api_key_input or api_key_from_env or "").strip()

rt_mode = st.selectbox("Fetch mode", ["Top headlines", "Search query (everything)"])
col1, col2 = st.columns(2)
with col1:
    country = st.text_input("Country code", "us")
with col2:
    language = st.text_input("Language", "en")

page_size = st.slider("Number of articles", min_value=1, max_value=20, value=10)
query = ""
if rt_mode == "Search query (everything)":
    query = st.text_input("Query", "election")

if not api_key:
    st.warning("Missing NewsAPI key. Paste a key above or set `NEWSAPI_KEY` in Streamlit secrets / environment variable.")

if st.button("Fetch latest headlines"):
    try:
        if not api_key:
            st.stop()

        if rt_mode == "Top headlines":
            articles = fetch_newsapi_top_headlines(
                api_key=api_key,
                country=(country or "us").lower(),
                language=(language or "en").lower(),
                page_size=page_size,
            )
        else:
            if not query.strip():
                st.warning("Please provide a search `Query`.")
                st.stop()
            articles = fetch_newsapi_everything(
                api_key=api_key,
                query=query.strip(),
                language=(language or "en").lower(),
                page_size=page_size,
            )

        if not articles:
            st.info("No articles returned by NewsAPI.")
            st.stop()

        results = []
        for a in articles[:page_size]:
            text = article_to_text(a)
            if not text:
                continue

            res = classify_text(vectorizer, model, text)
            results.append(
                {
                    "title": a.get("title") or "",
                    "source": (a.get("source") or {}).get("name") or "",
                    "publishedAt": a.get("publishedAt") or "",
                    "url": a.get("url") or "",
                    "verdict": res.get("verdict"),
                    "confidence_real": res.get("confidence_real"),
                }
            )

        if not results:
            st.info("No article text was available to classify.")
            st.stop()

        for r in results:
            if r["confidence_real"] is not None:
                r["confidence_real"] = f"{float(r['confidence_real']):.2%}"

        st.subheader("Results")
        st.dataframe(results, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to fetch/classify headlines: {e}")