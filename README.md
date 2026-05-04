# 💳 Loan Default Prediction

A Machine Learning project that predicts whether a loan applicant is likely to default or not based on financial and personal data.

---

## 📌 Overview

This project applies multiple Machine Learning models to classify loan applicants into:

- ✅ **Non-Defaulter**
- ❌ **Defaulter**

It compares different algorithms and selects the best-performing model based on accuracy.

---

## 🚀 Features

- Data preprocessing (handling missing values, encoding, scaling)
- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
- Model comparison based on accuracy
- Saved trained models for reuse
- Simple Python app for predictions

---

## 📂 Project Structure
Loan-Default-prediction/
│
├── app.py
├── trainmodel.ipynb
├── train_u6lujuX_CVtuz9i.csv
│
├── logreg_model.pkl
├── rf_model.pkl
├── kn_model.pkl
│
├── scaler.pkl
├── label_encoders.pkl
├── model_accuracies.pkl
│
└── .gitignore


---

## 🧠 Machine Learning Models Used

| Model                | Description |
|---------------------|------------|
| Logistic Regression | Baseline linear model |
| Random Forest       | Ensemble learning model |
| KNN                 | Distance-based model |

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Loan-Default-prediction.git
cd Loan-Default-prediction
