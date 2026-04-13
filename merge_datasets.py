import pandas as pd

# ------------------------
# 1. Charger TON dataset actuel
# ------------------------
df_old = pd.read_csv("news.csv")

# ------------------------
# 2. Charger dataset Kaggle
# ------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# ------------------------
# 3. Ajouter labels
# ------------------------
fake["label"] = 1
true["label"] = 0

# ------------------------
# 4. Garder uniquement texte + label
# ------------------------
fake = fake[["text", "label"]]
true = true[["text", "label"]]

# ------------------------
# 5. Fusionner tout
# ------------------------
df = pd.concat([df_old, fake, true], ignore_index=True)

# ------------------------
# 6. Nettoyage rapide
# ------------------------
df = df.dropna()
df = df.drop_duplicates()

# ------------------------
# 7. Sauvegarder
# ------------------------
df.to_csv("news_merged.csv", index=False)

print("✅ Dataset fusionné créé :", len(df), "lignes")