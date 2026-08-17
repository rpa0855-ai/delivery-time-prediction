# Delivery Time Prediction

Predicting food delivery time (minutes) from order, delivery-partner, and
environmental features — two parts:

**Part 1 — Classical MLR:** outlier handling (Cook's Distance, DFFITS),
multicollinearity resolution (VIF + PCA), model selection (backward
elimination via AIC/BIC/Mallow's Cp/PRESS), and MLR assumption validation.

**Part 2 — ML/DL comparison + explanation agent:** Linear/Ridge/Lasso,
Random Forest, Gradient Boosting, XGBoost, CatBoost, LightGBM, and a simple
feedforward Neural Network — compared against the MLR model on R²/MAE/RMSE.
The best model is explained with SHAP, wrapped in a rule-based **Delay
Explanation Agent** that says which features drove a specific prediction
(with an optional step to rephrase that explanation naturally using a small
local LLM, Qwen2.5-3B-Instruct). The best model is then deployed as a
**Streamlit app**.

## Data

Real Swiggy delivery dataset, 45,593 rows, India (`data/raw/swiggy.csv`).
Target: `Time_taken(min)`. Distance computed from restaurant/delivery lat-long
via the Haversine formula, after filtering to India's valid coordinate bounds.

## Project Organization

```
├── README.md            <- This file
├── requirements.txt     <- Python dependencies (notebook)
├── data
│   └── raw              <- Original, immutable data (swiggy.csv)
├── notebooks             <- Full analysis notebook (Colab-compatible),
│                            85 cells: Part 1 (MLR) + Part 2 (ML/DL/agent)
├── reports
│   ├── why_sheet.md      <- Reasoning behind every design decision
│   └── figures           <- Exported plots (Cook's D, VIF, residuals, SHAP)
└── app                   <- Streamlit deployment
    ├── app.py             <- The app itself
    ├── requirements.txt   <- Python dependencies (app)
    ├── DEPLOY.md          <- Step-by-step deployment guide
    ├── best_model.pkl     <- Saved best model (generate via notebook Cell 41)
    └── feature_columns.pkl <- Saved feature column order (same cell)
```

## How to run the notebook

1. Open `notebooks/1.0-delivery-time-mlr.ipynb` in Google Colab (GPU runtime
   recommended for Part 2 — XGBoost/CatBoost/LightGBM/Neural Network/agent)
2. Run the setup cell, then upload `data/raw/swiggy.csv` when prompted
3. Run all cells top to bottom
4. Cell 41 saves `best_model.pkl` and `feature_columns.pkl` — download both
   and place them in `app/` to run the Streamlit app

## How to run the app

See `app/DEPLOY.md` for local testing and Streamlit Community Cloud
deployment steps.

## Results

See the notebook's results cells for:
- Adj R² progression (baseline → after outlier removal → after VIF/PCA +
  backward elimination) for the MLR model
- The MLR model comparison table (AIC/BIC/Mallow's Cp/PRESS)
- The final MLR vs ML vs DL comparison table (R²/MAE/RMSE)
- SHAP feature importance for the best model

## Note on structure

This follows the standard Cookiecutter Data Science layout, trimmed down: no
`src/` module scripts, `Makefile`, or `Sphinx` docs, since the entire
analysis lives in one self-contained notebook by design — easier to walk
through end-to-end in an interview than a multi-file pipeline would be. The
one exception is `app/`, which is a real deployable Streamlit app.
