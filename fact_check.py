import json
import os
import re
import urllib.parse
import urllib.request


GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()


def extract_claim(text: str, max_words: int = 8) -> str:
    """
    Extract simple keywords from the input text for API lookup.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "this", "that", "these",
        "those", "it", "its", "we", "they", "he", "she", "you", "i", "my", "our",
        "le", "la", "les", "un", "une", "des", "est", "sont", "et", "ou", "en",
        "du", "de", "que", "qui", "se", "ce", "il", "elle", "ils", "elles",
        "just", "also", "very", "more", "some", "about", "after", "before",
        "now", "new", "say", "said", "says", "know", "want", "get", "make",
    }

    text = re.sub(r"[!?#@]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)

    freq = {}
    for word in text.lower().split():
        word = re.sub(r"[^a-z]", "", word)
        if word and word not in stop_words and len(word) > 3:
            freq[word] = freq.get(word, 0) + 1

    keywords = sorted(freq, key=freq.get, reverse=True)
    return " ".join(keywords[:max_words])


def query_fact_check_api(
    query: str, api_key: str = GOOGLE_API_KEY, max_results: int = 5
):
    """
    Query the Google Fact Check Tools API.
    Returns:
    - None when the API key is not configured
    - [] on request failures or when there are no claims
    """
    if not api_key:
        return None

    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = urllib.parse.urlencode(
        {
            "query": query,
            "key": api_key,
            "pageSize": max_results,
        }
    )
    url = f"{base_url}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
    except Exception:
        return []

    results = []
    for claim in data.get("claims", []):
        for review in claim.get("claimReview", []):
            results.append(
                {
                    "claim": claim.get("text", ""),
                    "source": review.get("publisher", {}).get("name", "Unknown"),
                    "rating": review.get("textualRating", ""),
                    "url": review.get("url", ""),
                    "title": review.get("title", ""),
                }
            )

    return results


FALSE_KEYWORDS = {
    "false",
    "fake",
    "incorrect",
    "misleading",
    "wrong",
    "debunked",
    "misinformation",
    "inaccurate",
    "faux",
    "trompeur",
    "errone",
}

TRUE_KEYWORDS = {
    "true",
    "correct",
    "accurate",
    "verified",
    "confirmed",
    "vrai",
    "exact",
    "verifie",
    "confirme",
}


def external_credibility_score(results: list) -> dict:
    """
    Summarize fact-check reviews into a coarse external credibility score.
    """
    if not results:
        return {
            "score": 0.0,
            "verdict": "NOT FOUND",
            "sources": [],
            "count": 0,
        }

    score = 0.0
    sources = []

    for result in results:
        rating = result.get("rating", "").lower()
        if any(keyword in rating for keyword in FALSE_KEYWORDS):
            score += 1.0
        elif any(keyword in rating for keyword in TRUE_KEYWORDS):
            score -= 1.0

        sources.append(
            {
                "source": result["source"],
                "rating": result["rating"],
                "url": result["url"],
            }
        )

    score = score / max(1, len(results))

    if score > 0.3:
        verdict = "LIKELY FAKE"
    elif score < -0.3:
        verdict = "LIKELY REAL"
    else:
        verdict = "MIXED / UNCERTAIN"

    return {
        "score": round(score, 3),
        "verdict": verdict,
        "sources": sources,
        "count": len(results),
    }


def run_fact_check(text: str) -> dict:
    """
    End-to-end lookup used by the Streamlit app.
    """
    claim = extract_claim(text)
    results = query_fact_check_api(claim)

    if results is None:
        return {
            "enabled": False,
            "claim": claim,
            "score": 0.0,
            "verdict": "API NOT CONFIGURED",
            "sources": [],
            "count": 0,
        }

    credibility = external_credibility_score(results)
    return {
        "enabled": True,
        "claim": claim,
        **credibility,
    }
