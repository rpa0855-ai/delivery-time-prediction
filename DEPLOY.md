# Deploying the Delivery Time Predictor

## Step 1 — Generate the model files (in Colab)

Run the full notebook (`notebooks/1.0-delivery-time-mlr.ipynb`) through
**Cell 41 (Save model for the Streamlit app)**. This creates two files:

- `best_model.pkl`
- `feature_columns.pkl`

Download both from Colab's file browser (left sidebar → right-click each
file → Download) and place them in this `app/` folder, next to `app.py`.

## Step 2 — Test locally (optional but recommended)

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Step 3 — Deploy to Streamlit Community Cloud (free)

1. Push this whole `delivery-time-prediction/` folder to a GitHub repo
   (make sure `best_model.pkl` and `feature_columns.pkl` are inside `app/`
   and are NOT excluded by `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click "New app", select your repo
4. Set **Main file path** to `app/app.py`
5. Deploy — you'll get a public URL to put on your resume/portfolio

## Notes

- `best_model.pkl` is whichever model (XGBoost/CatBoost/LightGBM/etc.) scored
  best in the notebook's comparison table — the app doesn't assume a specific
  one, it just loads what you saved.
- If model files are large (CatBoost/XGBoost models can be a few MB), GitHub
  handles this fine; if it ever exceeds 100MB, use Git LFS.
- The app rebuilds the SHAP explainer from the loaded model at startup
  (`shap.TreeExplainer(model)`) rather than saving the explainer object
  itself — simpler and avoids pickle-compatibility issues across environments.
