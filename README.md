---
title: Student Grade Predictor
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.16.0
app_file: app_gradio.py
pinned: false
---

# 🎓 Student Grade Predictor — MLOps Mini Project

Predicts whether a student will **PASS or FAIL** their final Math exam
based on study habits, family background, and lifestyle.

## Tech Stack
- **Model**: Random Forest (Scikit-learn)
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **UI**: Gradio
- **CI/CD**: GitHub Actions → Hugging Face Spaces

## Dataset
UCI Student Performance Dataset (Math course) — 395 students, 27 features.

## Project Structure
```
├── train.py          # Model training + MLflow logging
├── app.py            # FastAPI REST API
├── app_gradio.py     # Gradio web UI (deployed to HF Spaces)
├── Data/             # Dataset
├── Model/            # Saved model.pkl
├── Results/          # Confusion matrix plots
├── tests/            # pytest test suite
└── .github/workflows # CI/CD pipelines
```