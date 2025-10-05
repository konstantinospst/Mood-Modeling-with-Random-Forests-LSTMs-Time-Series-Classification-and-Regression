# Mood Modeling: RF Classification & LSTM (Classification + Regression)

This repo contains a complete workflow for modeling mood from mobile/behavioral signals using both classical ML and deep learning:

- **EDA & Feature Engineering** (separate notebooks)
- **Classification**
  - Baseline: RandomForestClassifier on all features with **TimeSeriesSplit**
  - Post-baseline **feature selection** (SelectKBest) → retrain
  - **Hyperparameter tuning** (RandomizedSearchCV) on the selected feature set
  - Optional: LSTM classification on a chronological train/val/test split
- **Regression**
  - LSTM regression (chronological train/val/test)
  - **Keras-Tuner** (RandomSearch) for depth/units/dropout/lr
  - Final retrain with best hyperparameters + evaluation & plots
