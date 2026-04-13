# TRUTHX — Misinformation Signal Detector

## 🚀 Overview

TRUTHX is a lightweight machine learning project designed to explore how misinformation can be detected using a combination of **text analysis, stylistic signals, and external fact-checking APIs**.

This project is intended for **educational and research purposes**, not as a production-ready moderation system.

---

## 🧠 How It Works

TRUTHX uses a hybrid pipeline:

1. **Text Processing**

   * TF-IDF vectorization of input claims

2. **Machine Learning Model**

   * Logistic Regression classifier trained on labeled news data

3. **Handcrafted Features**

   * Sensationalism score
   * Uppercase ratio
   * Exclamation/question counts
   * Emotional word detection

4. **External Fact Checking (Optional)**

   * Google Fact Check Tools API
   * Retrieves real-world verification sources

5. **Final Output**

   * Fake probability score
   * Verdict (REAL / FAKE / UNCERTAIN)
   * Signal metrics
   * External evidence (if available)

---

## 📊 Example Use Cases

| Claim                                        | Result                              |
| -------------------------------------------- | ----------------------------------- |
| "BREAKING!!! Secret miracle cure exposed!!!" | 🚨 Fake (100%)                      |
| "COVID-19 vaccines contain microchips"       | ❌ Likely Fake (verified by sources) |
| "Public health authority released a report"  | ✅ Likely Real (~9%)                 |

---

## 🛠️ Tech Stack

* Python
* Streamlit (UI)
* scikit-learn
* TF-IDF (TfidfVectorizer)
* Logistic Regression
* pandas / NumPy / SciPy
* joblib (model persistence)
* Google Fact Check Tools API

---

## 📁 Project Structure

```
.
├── app.py              # Streamlit web app
├── train.py            # Model training pipeline
├── fact_check.py       # External API logic
├── utils.py            # Feature engineering
├── model.joblib        # Trained model
├── vectorizer.joblib   # TF-IDF vectorizer
├── news.csv            # Demo dataset
├── Fake.csv / True.csv # Extended datasets
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/truthx.git
cd truthx
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file or use your terminal:

```
GOOGLE_FACT_CHECK_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```
streamlit run app.py
```

---

## 📦 Model Training

To retrain the model:

```
python train.py
```

---

## ⚠️ Limitations

* Small dataset → limited generalization
* Relies heavily on textual/style signals
* External API may return noisy or broad results
* Not a fact-checking authority

---

## 🔮 Future Improvements

* Use transformer models (BERT / RoBERTa)
* Semantic search instead of keyword-based queries
* Larger and more diverse datasets
* Better source ranking for fact-check results
* Multilingual support

---

## 📌 Disclaimer

This project is a prototype for experimentation and learning. Results should **not** be considered definitive truth.

---

## 🤝 Contributing

Contributions, ideas, and feedback are welcome!

---

## ⭐ Acknowledgments

* Google Fact Check Tools API
* scikit-learn community

---



