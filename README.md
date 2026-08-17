# Delivery Time Prediction (MLR)

Predicting food delivery time (minutes) from order, delivery-partner, and
environmental features using classical Multiple Linear Regression — outlier
handling (Cook's Distance, DFFITS), multicollinearity resolution (VIF + PCA),
model selection (backward elimination via AIC/BIC/Mallow's Cp/PRESS), and
MLR assumption validation.

## Data

Real Swiggy delivery dataset, 45,593 rows, India (`data/raw/swiggy.csv`).
Target: `Time_taken(min)`. Distance computed from restaurant/delivery lat-long
via the Haversine formula.

## Project Organization

```
├── README.md          <- This file
├── requirements.txt    <- Python dependencies
├── data
│   └── raw            <- Original, immutable data (swiggy.csv)
├── notebooks           <- Full analysis notebook (Colab-compatible)
├── reports
│   ├── why_sheet.md    <- Reasoning behind every design decision
│   └── figures         <- Exported plots (Cook's D, VIF, residuals, etc.)
└── models              <- (empty — this project keeps the fitted model inside
                            the notebook rather than serializing it, since the
                            deliverable is the analysis, not a deployable model)
```

## How to run

1. Open `notebooks/1.0-delivery-time-mlr.ipynb` in Google Colab
2. Run the setup cell, then upload `data/raw/swiggy.csv` when prompted
3. Run all cells top to bottom

## Results

See the notebook's final cells for the Adj R² progression (baseline → after
outlier removal → after VIF/PCA + backward elimination) and the model
comparison table (AIC/BIC/Mallow's Cp/PRESS).

## Note on structure

This follows the standard Cookiecutter Data Science layout, trimmed down: no
`src/` module scripts, `Makefile`, or `Sphinx` docs, since the entire analysis
lives in one self-contained notebook by design — easier to walk through
end-to-end in an interview than a multi-file pipeline would be.
