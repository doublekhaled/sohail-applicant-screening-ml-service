# Sohail Applicant Screening ML Service

## Overview

This project is a Machine Learning applicant-screening system developed as part of the AI Foundation Internship Program.

The goal of the project is to predict whether an internship applicant should be:

- Shortlisted
- Review Later

using a supervised Machine Learning model trained on applicant data.

---

## Team Structure

### Khaled – Data & Model Training Layer

Responsibilities:

- Dataset preparation
- Data cleaning
- Feature selection
- Train/test split
- Model training
- Model serialization using joblib
- Data schema documentation

### Omar – Evaluation & Reporting Layer

Responsibilities:

- Model evaluation
- Accuracy analysis
- Confusion matrix
- Classification report
- Business interpretation

### Easa – API & Deployment Layer

Responsibilities:

- FastAPI deployment
- Prediction endpoint
- Health endpoint
- API documentation

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| gpa | Applicant GPA |
| skills_count | Number of technical skills |
| prior_projects | Number of completed projects |
| track | Internship track (AI, Data, Web) |

### Target

| Value | Meaning |
|---------|---------|
| 1 | Shortlisted |
| 0 | Review Later |

---

## Machine Learning Model

Algorithm Used:

- Logistic Regression

Library:

- scikit-learn

---

## Project Files

### Data & Training

- applicants.csv
- train_model.py
- model.joblib
- encoder.joblib
- DATA_SCHEMA.md

---

## Reproducibility

All train/test splits use:

```python
random_state = 42
```

This ensures that all team members work with the exact same dataset split and obtain consistent results.

---

## How to Load the Model

```python
import joblib

model = joblib.load("model.joblib")
```

### Load Encoder

```python
encoder = joblib.load("encoder.joblib")
```

---

## Technologies Used

- Python
- pandas
- scikit-learn
- joblib

---

## Internship Project

AI Foundation Internship Program

Sohail Applicant Screening ML Service
