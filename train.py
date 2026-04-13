import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ← Import depuis utils.py (plus de duplication)
from utils import clean_text, sensational_vector, to_binary_label_series

# ─────────────────────────────────────────
# Helpers colonnes
# ─────────────────────────────────────────
TEXT_CANDIDATES  = ["text", "content", "article", "body", "statement", "news", "headline", "title"]
LABEL_CANDIDATES = ["label", "labels", "target", "class", "y", "is_fake", "fake"]

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None

# ─────────────────────────────────────────
# Explicabilité
# ─────────────────────────────────────────
def explain_text(vectorizer, model, text: str, top_k=8):
    feat   = np.array(vectorizer.get_feature_names_out())
    coef   = model.coef_[0][:len(feat)]   # seulement la partie TF-IDF
    v      = vectorizer.transform([clean_text(text)])
    contrib = v.toarray()[0] * coef

    idx_pos  = np.argsort(contrib)[::-1]
    top_fake = [(feat[i], float(contrib[i])) for i in idx_pos[:top_k] if contrib[i] > 0]

    idx_neg  = np.argsort(contrib)
    top_real = [(feat[i], float(contrib[i])) for i in idx_neg[:top_k] if contrib[i] < 0]

    return top_fake, top_real

# ─────────────────────────────────────────
# Construction de la matrice finale
# ─────────────────────────────────────────
def build_matrix(texts_raw, vectorizer, fit=False):
    """
    Combine TF-IDF + features sensationnelles en une seule matrice.
    fit=True → appelle fit_transform, False → transform seulement.
    """
    cleaned = [clean_text(t) for t in texts_raw]

    if fit:
        tfidf_mat = vectorizer.fit_transform(cleaned)
    else:
        tfidf_mat = vectorizer.transform(cleaned)

    # Features sensationnelles (6 colonnes numériques)
    sens_mat = sp.csr_matrix(
        np.array([sensational_vector(t) for t in texts_raw], dtype=np.float32)
    )

    return sp.hstack([tfidf_mat, sens_mat], format="csr")

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    # 1) Charger
    df = pd.read_csv("news.csv", encoding_errors="ignore")
    print("✅ CSV chargé. Colonnes:", list(df.columns))

    # 2) Trouver texte + label
    text_col  = find_col(df, TEXT_CANDIDATES)
    label_col = find_col(df, LABEL_CANDIDATES)

    if text_col is None:
        title_col = find_col(df, ["title", "headline"])
        body_col  = find_col(df, ["text", "content", "body", "article"])
        if title_col and body_col:
            df["text"] = (df[title_col].fillna("") + " " + df[body_col].fillna("")).str.strip()
            text_col = "text"
            print(f"ℹ️ Texte combiné: {title_col} + {body_col}")
        else:
            obj_cols = [c for c in df.columns if df[c].dtype == "object"]
            if not obj_cols:
                raise ValueError("Impossible de trouver une colonne texte.")
            text_col = obj_cols[0]
            print(f"⚠️ Fallback colonne texte: {text_col}")

    if label_col is None:
        raise ValueError("Colonne label introuvable. Renomme ta colonne cible en 'label'.")

    # 3) Labels
    df[label_col] = to_binary_label_series(df[label_col])
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(int)

    print(f"📊 Distribution: {df[label_col].value_counts().to_dict()}")

    # 4) Split
    X_raw = df[text_col].astype(str).tolist()
    y     = df[label_col].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    # 6) Matrices combinées (TF-IDF + features sensationnelles)
    print("⚙️  Construction des matrices...")
    X_train = build_matrix(X_train_raw, vectorizer, fit=True)
    X_test  = build_matrix(X_test_raw,  vectorizer, fit=False)
    print(f"   Shape finale: {X_train.shape} (TF-IDF + 6 features sensationnelles)")

    # 7) Modèle
    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0)
    model.fit(X_train, y_train)

    # 8) Validation croisée (5 folds) sur les données d'entraînement
    print("\n⏳ Validation croisée (5 folds)...")
    X_all = build_matrix(X_raw, vectorizer, fit=False)
    cv_scores = cross_val_score(model, X_all, y, cv=5, scoring="f1_weighted", n_jobs=-1)
    print(f"   F1 scores: {np.round(cv_scores, 3)}")
    print(f"   Moyenne: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 9) Évaluation sur le test set
    pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, pred))

    # 10) Sauvegarde
    joblib.dump(model,      "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("\n✅ Sauvegardé: model.joblib + vectorizer.joblib")

    # 11) Demo rapide
    from utils import verdict_from_prob

    def predict(text: str):
        mat   = build_matrix([text], vectorizer, fit=False)
        proba = model.predict_proba(mat)[0]
        p_fake = float(proba[1])
        return verdict_from_prob(p_fake), p_fake

    examples = [
        "BREAKING!!! Scientists confirm drinking lemon water cures cancer instantly!!! Share before deleted!!!",
        "The European Central Bank announced an increase in interest rates following inflation reports.",
        "NASA successfully launched a new satellite to monitor climate change patterns.",
    ]

    print("\n=== Quick Demo ===")
    for t in examples:
        verdict, p = predict(t)
        top_fake, top_real = explain_text(vectorizer, model, t, top_k=3)
        print(f"\nTEXT   : {t[:80]}...")
        print(f"VERDICT: {verdict} | p(fake)={p:.3f}")
        print(f"→ FAKE words : {[w for w, _ in top_fake]}")
        print(f"→ REAL words : {[w for w, _ in top_real]}")

if __name__ == "__main__":
    main()