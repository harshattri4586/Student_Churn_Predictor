# 🎓 Student Churn Predictor

An end-to-end Machine Learning project that predicts whether a student is likely to **drop out** based on academic, financial, and demographic features. Built with a production-grade pipeline architecture including data ingestion, transformation, model training, and a Flask web interface.

---

## 📌 Problem Statement

Student churn (dropout) is a critical challenge for educational institutions. Early identification of at-risk students allows timely interventions. This project builds a binary classification model to predict dropout probability using the **Student Dropout and Academic Success** dataset.

---

## 🗂️ Project Structure

```
student_churn_predictor/
│
├── artifacts/                  → Saved datasets & model files
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── src/                        → Core source code (installable package)
│   ├── __init__.py
│   ├── exception.py            → Custom exception handler
│   ├── logger.py               → Logging configuration
│   ├── utils.py                → Shared utility functions
│   │
│   └── components/             → ML pipeline components
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_transformation.py
│       └── model_trainer.py
│
├── pipeline/                   → Train & predict pipelines
│   ├── __init__.py
│   ├── train_pipeline.py
│   └── predict_pipeline.py
│
├── notebooks/                  → EDA & experimentation
│   └── EDA_and_ModelBuilding.ipynb
│
├── templates/                  → HTML for Flask frontend
│   └── index.html
│
├── logs/                       → Auto-generated log files
├── app.py                      → Flask application entry point
├── setup.py                    → Package setup
└── requirements.txt            → Project dependencies
```

---

## 🚀 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12+ |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Serialization | Dill |
| Experiment Tracking | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## 📊 Dataset

**Source:** [Student Dropout and Academic Success Dataset](https://www.kaggle.com/datasets/missionjee/students-dropout-and-academic-success-dataset)

**Key Features Used:**

| Feature | Description |
|---|---|
| Age at enrollment | Student age when joining |
| Admission grade | Entry qualification grade |
| Curricular units credited | Academic credit performance |
| Scholarship holder | Whether student has scholarship |
| Debtor | Whether student is in financial debt |
| Attendance | Day or Evening attendance |
| Curricular units approved | Units passed per semester |

**Target Variable:**
```
Dropout  → 1 (Churned)
Graduate → 0 (Not Churned)
Enrolled → 0 (Not Churned)
```

---

## ⚙️ ML Pipeline Overview

```
Raw Data
   ↓
Data Ingestion     → Reads CSV, splits into train/test, saves to artifacts/
   ↓
Data Transformation → Handles missing values, encoding, scaling, SMOTE
   ↓
Model Training     → Trains multiple models, selects best by F1-Score
   ↓
Model Saved        → preprocessor.pkl + model.pkl saved to artifacts/
   ↓
Prediction Pipeline → Loads pkl files, transforms input, returns prediction
   ↓
Flask Frontend     → User fills form → prediction displayed
```

---

## 📈 Models Evaluated

| Model | Metric Used |
|---|---|
| Logistic Regression | F1-Score, AUC-ROC |
| Decision Tree | F1-Score, AUC-ROC |
| Random Forest | F1-Score, AUC-ROC |
| Gradient Boosting | F1-Score, AUC-ROC |
| XGBoost | F1-Score, AUC-ROC |

> ⚠️ Accuracy was **not** used as the primary metric due to class imbalance (~33% dropout rate). F1-Score and AUC-ROC were prioritized.

---


## 👤 Author

 ** HARSHVARDHAN **
- GitHub: [@harshattri4586](https://github.com/harshattri4586)
- Linkedin: [HarshAttri](https://www.linkedin.com/in/harsh-attri-b72ab2248/)

- Live Demo: [StudentChurnPredictor](https://studentchurnpredictor-wwsvahbueeapi9jgjlxev4.streamlit.app/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).