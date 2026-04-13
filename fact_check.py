import json
import os
import re
import urllib.parse
import urllib.request


GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()

MAX_QUERY_CHARS = 180
MIN_QUERY_WORDS = 5
MIN_QUERY_CHARS = 24

HYPE_WORDS = {
    "breaking",
    "shocking",
    "urgent",
    "exclusive",
    "miracle",
    "secret",
    "must-see",
    "share",
    "viral",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "this", "that", "these",
    "those", "it", "its", "we", "they", "he", "she", "you", "i", "my", "our",
    "your", "their", "them", "his", "her", "as", "if", "then", "than", "into",
    "about", "after", "before", "just", "also", "very", "more", "some", "such",
    "now", "new", "say", "said", "says", "know", "want", "get", "make", "made",
    "call", "called", "calls", "calling", "claim", "claims", "claimed",
    "report", "reported", "reports", "post", "posted", "posts", "video",
    "watch", "read", "check", "please", "people", "users", "everyone",
    "le", "la", "les", "un", "une", "des", "est", "sont", "et", "ou", "en",
    "du", "de", "que", "qui", "se", "ce", "cet", "cette", "il", "elle", "ils",
    "elles", "sur", "dans", "pour", "avec", "sans", "plus", "moins", "tres",
    "trop", "mais", "donc", "car", "pas", "comme", "aux", "par", "leur",
}

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


def limit_query_length(text: str, max_chars: int = MAX_QUERY_CHARS) -> str:
    text = " ".join(text.split()).strip(" ,;:-")
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0].strip(" ,;:-")
    return truncated or text[:max_chars].strip(" ,;:-")


def normalize_claim_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[_*~`\"“”‘’]+", " ", text)
    text = re.sub(r"\b(?:%s)\b" % "|".join(re.escape(word) for word in HYPE_WORDS), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[!?]{2,}", ". ", text)
    text = re.sub(r"\.{2,}", ". ", text)
    text = re.sub(r"\s*[-–—]\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;-")


def tokenize_informative_words(text: str) -> list[str]:
    words = []
    for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower()):
        if len(token) <= 2:
            continue
        if token in STOP_WORDS:
            continue
        words.append(token)
    return words


def sentence_score(sentence: str) -> tuple[int, int, int]:
    informative = tokenize_informative_words(sentence)
    unique_count = len(set(informative))
    informative_count = len(informative)
    length_score = min(len(sentence), MAX_QUERY_CHARS)
    return (unique_count, informative_count, length_score)


def extract_best_sentence(text: str) -> str:
    normalized = normalize_claim_text(text)
    if not normalized:
        return ""

    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    candidates = []
    for part in parts:
        sentence = part.strip(" .,:;-")
        if not sentence:
            continue
        informative_count = len(tokenize_informative_words(sentence))
        if informative_count < MIN_QUERY_WORDS:
            continue
        candidates.append(limit_query_length(sentence))

    if not candidates:
        return ""

    return max(candidates, key=sentence_score)


def compress_keywords(text: str, max_words: int = 14) -> str:
    normalized = normalize_claim_text(text)
    if not normalized:
        return ""

    seen = set()
    compressed = []
    for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", normalized):
        lowered = token.lower()
        if len(lowered) <= 2 or lowered in STOP_WORDS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        compressed.append(token)
        if len(compressed) >= max_words:
            break

    return limit_query_length(" ".join(compressed))


def build_fact_check_query(text: str) -> str:
    best_sentence = extract_best_sentence(text)
    if best_sentence:
        return best_sentence

    return compress_keywords(text, max_words=16)


def build_fact_check_candidates(text: str) -> list[tuple[str, str]]:
    candidates = []
    for strategy, query in [
        ("best_sentence", build_fact_check_query(text)),
        ("keyword_compressed", compress_keywords(text, max_words=14)),
    ]:
        query = limit_query_length(query)
        if len(query) < MIN_QUERY_CHARS:
            continue
        if len(tokenize_informative_words(query)) < MIN_QUERY_WORDS:
            continue
        if any(existing_query == query for _, existing_query in candidates):
            continue
        candidates.append((strategy, query))
        if len(candidates) == 2:
            break
    return candidates


def query_fact_check_api(
    query: str, api_key: str = GOOGLE_API_KEY, max_results: int = 5
):
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


def external_credibility_score(results: list) -> dict:
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
    if not GOOGLE_API_KEY:
        return {
            "enabled": False,
            "query": "",
            "query_strategy": "",
            "attempted_queries": [],
            "score": 0.0,
            "verdict": "API NOT CONFIGURED",
            "sources": [],
            "count": 0,
            "message": "Google Fact Check API is not configured.",
        }

    candidates = build_fact_check_candidates(text)
    if not candidates:
        return {
            "enabled": True,
            "query": "",
            "query_strategy": "",
            "attempted_queries": [],
            "score": 0.0,
            "verdict": "NO QUERY",
            "sources": [],
            "count": 0,
            "message": "No strong factual claim detected for external verification.",
        }

    attempted_queries = []
    for strategy, query in candidates:
        attempted_queries.append(query)
        results = query_fact_check_api(query)
        if results is None:
            return {
                "enabled": False,
                "query": "",
                "query_strategy": "",
                "attempted_queries": attempted_queries,
                "score": 0.0,
                "verdict": "API NOT CONFIGURED",
                "sources": [],
                "count": 0,
                "message": "Google Fact Check API is not configured.",
            }
        if results:
            credibility = external_credibility_score(results)
            return {
                "enabled": True,
                "query": query,
                "query_strategy": strategy,
                "attempted_queries": attempted_queries,
                **credibility,
                "message": "",
            }

    return {
        "enabled": True,
        "query": candidates[0][1],
        "query_strategy": candidates[0][0],
        "attempted_queries": attempted_queries,
        "score": 0.0,
        "verdict": "NOT FOUND",
        "sources": [],
        "count": 0,
        "message": "No matching fact-check sources found.",
    }
