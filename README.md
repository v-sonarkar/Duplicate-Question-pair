
# Duplicate Question Pair — Streamlit App

A lightweight Streamlit front-end for a trained model that detects duplicate question pairs (Quora-style).

## Overview
- Simple UI to enter two questions and predict whether they are duplicates.
- Uses traditional feature engineering (length, token, fuzzy, BoW) and a persisted ML model (`model.pkl`).

## Required files
Place the following files in the `streamlit-app/` directory before running the app:
- `model.pkl` — trained scikit-learn estimator (supports `predict` on the input vector)
- `cv.pkl` — scikit-learn CountVectorizer (or similar) used to create BoW features
- `stopwords.pkl` — Python pickle of the stopwords set used by feature functions

## Install (recommended virtualenv)

Windows PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r streamlit-app/requirements.txt
```

## Run locally

From the repository root:

```powershell
cd streamlit-app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` by default.

## Deployment notes
- Streamlit Cloud: create a new app pointing to this GitHub repository and set the working directory to `streamlit-app/`.
- Heroku: use the provided `Procfile`; include `gunicorn` if deploying as a web process wrapper.

## Files of interest
- `app.py` — Streamlit application entrypoint (page config, caching, input validation, SEO metadata injection)
- `helper.py` — preprocessing and feature extraction utilities used by the model
- `requirements.txt` — pinned dependencies for the Streamlit app

## Security & privacy
- Model and pickled objects may contain sensitive or proprietary information. Do not commit `model.pkl`, `cv.pkl`, or `stopwords.pkl` to public repositories unless you intend to share them.

## License
This repository does not include a license by default. Add a `LICENSE` file if you wish to open-source the code under a specific license.

---

If you want, I can:
- update the root `README.md` to include a short section about the Streamlit app, or
- create a small `deploy.md` with step-by-step Streamlit Cloud / Heroku instructions.
