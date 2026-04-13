# Fake News Checker

Academic machine learning project for exploring misinformation detection with a lightweight web app.

The project combines:
- TF-IDF text features
- Logistic Regression classification
- handcrafted "sensationalism" features
- optional Google Fact Check Tools API lookup
- a Streamlit interface for quick testing

## Project Scope

This repository is a prototype, not a production moderation system.

Current limitations:
- the included dataset is very small
- predictions are based mainly on textual and style signals
- external fact-check lookup is optional and depends on API availability
- results should be interpreted as educational output, not verified truth

## Files

- `app.py`: Streamlit app
- `train.py`: model training pipeline
- `fact_check.py`: external fact-check lookup logic
- `utils.py`: shared preprocessing and feature engineering
- `news.csv`: demo dataset

## Installation

```bash
pip install -r requirements.txt
```

## Optional API Setup

To enable Google Fact Check lookup, set an environment variable before running the app.

PowerShell:

```powershell
$env:GOOGLE_FACT_CHECK_API_KEY="your_api_key_here"
```

If the variable is missing, the app still runs and disables the external fact-check step.

## Run The App

```bash
streamlit run app.py
```

## Train The Model

```bash
python train.py
```


