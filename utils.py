import re

# ─────────────────────────────────────────
# Seuils unifiés (utilisés dans train.py ET app.py)
# ─────────────────────────────────────────
THR_FAKE = 0.65   # au-dessus → FAKE
THR_REAL = 0.35   # en-dessous → REAL
MIN_CHARS = 30    # texte minimum pour analyser

# ─────────────────────────────────────────
# Mots émotionnels / sensationnels
# ─────────────────────────────────────────
EMO_WORDS = {
    # Anglais
    "breaking", "shocking", "urgent", "secret", "miracle",
    "scandal", "revelation", "exclusive", "attention", "impossible",
    "censored", "banned", "deleted", "exposed", "leaked",
    # Français
    "scandale", "incroyable", "alerte", "danger", "révélation",
    "choc", "interdit", "censuré", "supprimé", "exclusif",
}

# ─────────────────────────────────────────
# Nettoyage texte
# ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # liens
    text = re.sub(r"@\w+", " ", text)                     # mentions
    text = re.sub(r"#\w+", " ", text)                     # hashtags
    # On garde les chiffres (signal important dans les fake news)
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────
# Features sensationnelles
# ─────────────────────────────────────────
def sensational_features(raw: str) -> dict:
    """
    Retourne un dict de features numériques.
    Ces features sont utilisées DANS le modèle (pas juste affichées).
    """
    raw = str(raw)
    letters = [c for c in raw if c.isalpha()]
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))

    exclam      = raw.count("!")
    question    = raw.count("?")
    words       = clean_text(raw).split()
    emo_hits    = sum(1 for w in words if w in EMO_WORDS)
    word_count  = max(1, len(words))
    avg_word_len = sum(len(w) for w in words) / word_count

    # Score composite (pour affichage UI)
    sensational_score = (
        exclam * 0.5 +
        question * 0.2 +
        upper_ratio * 5.0 +
        emo_hits * 0.8
    )

    return {
        "upper_ratio":       round(float(upper_ratio), 4),
        "exclam":            int(exclam),
        "question":          int(question),
        "emo_hits":          int(emo_hits),
        "word_count":        int(word_count),
        "avg_word_len":      round(float(avg_word_len), 4),
        "sensational_score": round(float(sensational_score), 4),
    }

def sensational_vector(raw: str) -> list:
    """
    Retourne une liste de floats à concaténer avec TF-IDF.
    Ordre fixe — ne pas changer sans re-entraîner le modèle.
    """
    f = sensational_features(raw)
    return [
        f["upper_ratio"],
        f["exclam"],
        f["question"],
        f["emo_hits"],
        f["word_count"],
        f["avg_word_len"],
    ]

# ─────────────────────────────────────────
# Normalisation des labels
# ─────────────────────────────────────────
def to_binary_label_series(series):
    """Map labels vers 0/1. Convention : 1 = FAKE, 0 = REAL."""
    import pandas as pd
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype(int)
        uniq = sorted(pd.unique(s.dropna()))
        if set(uniq) == {0, 1}:
            return s
        if set(uniq) == {1, 2}:
            return s.map({1: 0, 2: 1})
        return s
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "fake": 1, "false": 1, "faux": 1, "1": 1,
        "real": 0, "true": 0, "vrai": 0, "legit": 0, "genuine": 0, "0": 0,
    }
    return s.map(mapping)

# ─────────────────────────────────────────
# Verdict à partir de la probabilité
# ─────────────────────────────────────────
def verdict_from_prob(p_fake: float) -> str:
    if p_fake >= THR_FAKE:
        return "FAKE"
    if p_fake <= THR_REAL:
        return "REAL"
    return "UNCERTAIN"
