# 🏦 Loan Approval Prediction

> **An end-to-end Machine Learning pipeline** that predicts whether a loan application will be approved or rejected — covering the complete data science lifecycle from raw data collection to model deployment.

---

## 📌 Project Overview

Banks and financial institutions process thousands of loan applications every day. Manual review is slow, inconsistent, and prone to bias. This project builds a **supervised binary classification model** that automatically predicts loan approval decisions based on applicant attributes such as income, credit history, loan amount, and employment status.

Developed as part of the **HopeAI Learning program**, the project follows a structured, production-aware ML workflow across 7 clearly separated phases.

---

## 🎯 Problem Statement

> *Given an applicant's financial and personal profile, predict whether their loan will be Approved ✅ or Rejected ❌.*

This kind of model helps financial institutions:
- **Automate** initial screening of loan applications
- **Reduce** manual review time and human inconsistency
- **Improve** fairness and traceability of approval decisions

---

## 🗂️ Project Structure

```
Loan-Approval-Prediction/
│
├── Data Collection/              # Raw dataset sourcing and loading
├── Data Analysis/                # Exploratory Data Analysis (EDA)
├── Data Preprocessing/           # Cleaning, encoding, scaling
├── Feature Selection/            # Identifying the most predictive features
├── Model Creation and Training/  # Model building, training & evaluation
├── Deployment Phase/             # Model serialisation & deployment code
└── Documentation/                # Reports, findings, references
```

---

## 🔬 Technical Workflow

### 1. Data Collection
- Sourced a real-world loan application dataset containing applicant details such as gender, marital status, education, number of dependants, income, co-applicant income, loan amount, loan term, credit history, and property area.
- Loaded and inspected the raw data to understand its shape, types, class distribution, and initial data quality.

### 2. Exploratory Data Analysis (EDA)
- Analysed the distribution of the target variable (Loan Status: Y/N) to assess class balance.
- Visualised relationships between applicant features and loan approval outcomes.
- Key insight: **Credit history** was by far the strongest single predictor — applicants with a clean credit history had significantly higher approval rates.
- Identified that **income, loan amount, and property area** also correlated meaningfully with approval outcomes.

### 3. Data Preprocessing
- Handled missing values using mode imputation for categorical fields and median imputation for numerical fields.
- Encoded categorical variables (Gender, Married, Education, Self Employed, Property Area) using Label Encoding.
- Applied log transformation on skewed features (LoanAmount, ApplicantIncome) to reduce the effect of outliers.
- Scaled numerical features for model stability where needed.

### 4. Feature Selection
- Used correlation matrices and feature importance scores to rank input variables.
- Retained high-signal features and dropped redundant ones to reduce noise and improve generalisation.
- Final selected features balanced predictive power with model simplicity.

### 5. Model Creation & Training
- Trained and benchmarked multiple classification algorithms:
  - Logistic Regression (baseline)
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting / XGBoost
- Evaluated models using **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **ROC-AUC**.
- Applied cross-validation to ensure results generalise beyond the training split.
- Selected the best model based on balanced F1 Score — important given real-world consequences of false positives and false negatives in lending.

### 6. Deployment Phase
- Serialised the final trained model using `pickle` / `joblib`.
- Built a prediction interface (Flask / Streamlit) where a user inputs applicant details and receives an instant approval prediction.
- Designed for lightweight hosting and easy demonstration.

---

## 📊 Model Performance Summary

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| Logistic Regression | ~78% | ~0.76 | Strong baseline for binary classification |
| Decision Tree | ~80% | ~0.79 | Interpretable but overfits |
| **Random Forest** | **~85%** | **~0.84** | ✅ Best overall — robust and generalises well |
| XGBoost | ~84% | ~0.83 | Close second, faster inference |

> *Random Forest Classifier delivered the best balance of precision and recall, making it the chosen production model.*

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Serialisation | Pickle / Joblib |
| Deployment | Flask / Streamlit |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Charu305/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit

# 3. Run notebooks in order
# Data Collection → Data Analysis → Data Preprocessing
# → Feature Selection → Model Creation and Training

# 4. Launch the deployment app
cd "Deployment Phase"
streamlit run app.py
```

---

## 💡 Key Learnings & Takeaways

- **Class imbalance matters in classification** — more approved loans than rejected ones in the dataset required careful evaluation using F1 Score and AUC rather than raw accuracy.
- **Credit history dominates** — a single binary feature (credit history: yes/no) had more predictive power than several combined features, reinforcing that domain knowledge should drive feature prioritisation.
- **Log transformations help** — applying log scaling to heavily skewed income and loan amount columns noticeably improved model performance.
- Followed the same structured pipeline discipline as production ML systems — each phase is independently reviewable and reproducible.

---

## 📁 Dataset

Publicly available loan application dataset with ~600 records.

**Key features:**

| Feature | Description |
|---|---|
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependants |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Employment type |
| `ApplicantIncome` | Monthly income of applicant |
| `CoapplicantIncome` | Monthly income of co-applicant |
| `LoanAmount` | Requested loan amount (thousands) |
| `Loan_Amount_Term` | Repayment term in months |
| `Credit_History` | Whether credit history meets guidelines (1/0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | ✅ Target — Y (Approved) / N (Rejected) |

---

## 👩‍💻 Author

**Charunya**
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

This project was developed for educational and internship purposes under the HopeAI program.
