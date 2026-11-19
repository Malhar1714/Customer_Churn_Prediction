# Customer_Churn_Prediction


# 📊 ChurnIQ: Customer Churn Analysis & Prediction

**Find out why customers leave, and spot the ones who are about to.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-SQL%20Analysis-4479A1?logo=mysql&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)


---

## Table of Contents

- [Overview](#overview)
- [The Business Problem](#the-business-problem)
- [Key Findings](#key-findings)
- [Machine Learning Model](#machine-learning-model)
- [The ChurnIQ Dashboard](#the-churniq-dashboard)
- [SQL Analysis](#sql-analysis)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Cleaning Notes](#data-cleaning-notes)
- [Limitations & Next Steps](#limitations--next-steps)
- [Tech Stack](#tech-stack)
- [About the Dataset](#about-the-dataset)

---

## Overview

Winning a new customer usually costs far more than keeping an existing one, so knowing *who is likely to leave* and *why* is valuable for any subscription business.

This project works through that question end to end using a telecom customer dataset of **7,043 customers**:

1. **Clean** the data and fix quality issues
2. **Explore** it to find what separates customers who stay from those who leave
3. **Test** the main pattern statistically
4. **Query** it with SQL, the way a business analyst would
5. **Predict** churn with a machine learning model
6. **Share** everything through an interactive Streamlit dashboard

---

## The Business Problem

Roughly **1 in 4 customers churned** (1,869 of 7,043). Those customers were paying about **$139,000 per month** in total, which is nearly 30% of the company's monthly revenue ($456,000).

The goals of this project are to:

- understand which customer groups churn the most,
- identify the factors linked to churn, and
- build a model that flags at-risk customers early enough for the retention team to act.

---

## Key Findings

| | |
|---|---|
| **Overall churn rate** | **26.5%** |
| **Month-to-month contracts** | **42.7%** churn (vs. 11.3% for one-year and 2.8% for two-year) |
| **Fiber optic internet** | **41.9%** churn (vs. 19.0% for DSL and 7.4% for no internet) |
| **Electronic check payers** | **45.3%** churn (vs. roughly 15 to 19% for other payment methods) |
| **New customers (first year)** | **47.4%** churn, falling to 9.5% for customers with 4+ years |
| **Senior citizens** | **41.7%** churn (vs. 23.6% for non-seniors) |
| **No tech support / online security** | About **42%** churn (vs. 15% for customers who have them) |

A few things stand out:

- **Contract type is the strongest single signal.** A chi-square test confirms the link between contract and churn is statistically significant (χ² ≈ 1,184.6, p ≈ 6e-258).
- **Churn happens early.** Churned customers had a median tenure of just 10 months, compared with 38 months for customers who stayed.
- **Churners pay more.** They averaged $74.44 per month versus $61.27 for retained customers, which fits with the high churn among fiber optic users.
- **Gender makes no real difference** (26.9% for women vs. 26.2% for men).

**What this suggests:** the most promising retention levers are moving month-to-month customers onto longer contracts, supporting new customers during their first year, bundling tech support and security for internet customers, and looking into why fiber optic and electronic check customers leave so often.

> These are patterns in the data, not proof of cause and effect.

---

## Machine Learning Model

Three classifiers were trained on an 80/20 train/test split (stratified, so both sets keep the same churn ratio) and evaluated on the **held-out test set of 1,409 customers**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| **Logistic Regression** ✅ | 0.738 | 0.504 | **0.783** | 0.614 | **0.842** |
| Random Forest | 0.777 | 0.566 | 0.687 | 0.621 | 0.838 |
| Decision Tree | 0.740 | 0.507 | 0.789 | 0.617 | 0.828 |

**Why Logistic Regression?** It had the best ROC-AUC, so it ranks at-risk customers best. It also catches **about 78% of actual churners**. That matters because missing a customer who is about to leave usually costs more than contacting one who would have stayed anyway. It is also easy to interpret.

**The trade-off:** precision for the churn class is about 50%, meaning roughly half the customers flagged as "likely to churn" would have stayed. The model works best as a tool for *prioritising* who to look at first, not for making final decisions.

**What the model leans on:** looking at the model's coefficients, the factors pushing predictions toward churn include fiber optic internet, month-to-month contracts, paying by electronic check, streaming services, and having no tech support or online security. Factors pushing toward staying include longer tenure, two-year contracts and DSL internet.

**How it's built:**

- A single scikit-learn `Pipeline`, so the saved model accepts raw customer data directly
- Numeric features: median imputation and standard scaling
- Categorical features: most-frequent imputation and one-hot encoding
- `class_weight="balanced"` on all models, because churners are the minority class
- Saved with `joblib` as `churn_model.joblib`

---

## The ChurnIQ Dashboard

An interactive Streamlit app with four pages:

| Page | What you can do |
|------|-----------------|
| **Overview** | See total customers, churned customers, retention rate and monthly charges, plus charts for churn split, churn by contract, tenure and monthly charges |
| **Customer Insights** | Filter by contract, gender and payment method, view churn by internet service and payment method, and download the filtered data as CSV |
| **Churn Prediction** | Enter a customer's details and get a churn probability with a likely-to-churn / likely-to-stay result |
| **Model Performance** | See a summary of the loaded model (full metrics, confusion matrix and ROC curve are in the notebook) |


## SQL Analysis

The SQL scripts are written for **MySQL**.

1. Run `create_database.sql` to create the `customer_churn_db` database and the `customers` table.
2. Import `customer_churn_cleaned.csv` into the table (the Table Data Import Wizard in MySQL Workbench works well), then check the row count and look for nulls.
3. Run `business_analysis.sql`, which answers four questions:

| # | Business question | SQL techniques used |
|---|-------------------|---------------------|
| 1 | What is the overall churn rate? | Aggregation, `CASE WHEN` |
| 2 | How does churn differ by contract type? | `GROUP BY`, conditional aggregation |
| 3 | How do monthly charges compare for churned vs. retained customers? | `GROUP BY`, `SUM`, `AVG` |
| 4 | Who are the top 10 longest-tenured customers in each contract type? | CTE, `ROW_NUMBER()` window function |

---

## Project Structure

The dashboard expects the files to be organised like this:

```
customer_churn_analysis/
├── app.py                           # Streamlit dashboard
├── customer_churn_prediction.ipynb  # Cleaning, EDA, statistics and modelling
├── README.md
├── data/
│   ├── Customer_Churn.csv           # Original dataset
│   └── customer_churn_cleaned.csv   # Cleaned dataset used by the app
├── models/
│   └── churn_model.joblib           # Trained pipeline
└── sql/
    ├── create_database.sql          # Database and table setup
    └── business_analysis.sql        # Business queries
```


## Getting Started

**1. Clone the project and open the folder**

```bash
git clone <your-repo-url>
cd customer_churn_analysis
```

**2. Install the requirements**

```bash
pip install streamlit pandas numpy scikit-learn==1.7.2 joblib plotly matplotlib seaborn scipy
```

The model was saved with scikit-learn 1.7.2. Using the same version avoids loading warnings or errors.

**3. Run the dashboard**

```bash
streamlit run app.py
```

**4. (Optional) Re-run the analysis**

Open `customer_churn_prediction.ipynb` in Jupyter. The notebook currently reads data from a Windows path (`C:\customer_churn_analysis\...`), so update the file paths near the top to match your own machine first.

---

## Data Cleaning Notes

The raw data was already quite tidy, with no duplicate rows or customer IDs. The one issue was `TotalCharges`, which was stored as text. Converting it to numbers revealed **11 blank values**. All 11 belong to customers with a **tenure of 0 months**, meaning they hadn't been billed yet, so the values were set to **0** instead of dropping those customers.

---

## Limitations & Next Steps

**Limitations**

- Precision for the churn class is around 50%, so some flagged customers will not actually leave.
- The prediction page estimates `TotalCharges` as monthly charges × tenure, which is an approximation of the real value.
- The data is a single snapshot, so it can show what is *associated* with churn but not what *causes* it.

**Ideas for improvement**

- [ ] Tune hyperparameters and try gradient boosting models (XGBoost, LightGBM)
- [ ] Adjust the decision threshold to balance precision and recall
- [ ] Add feature importance and the real test metrics to the dashboard
- [ ] Estimate the revenue at risk for each flagged customer
- [ ] Deploy the app online (for example, Streamlit Community Cloud)

---

## Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Data & analysis | pandas, NumPy, SciPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine learning | scikit-learn |
| Database | MySQL |
| App | Streamlit |

---

## About the Dataset

The data is the publicly available **Telco Customer Churn** dataset, commonly used for churn modelling practice. It has **7,043 customers and 21 columns**, covering demographics, the services each customer has, billing details, and whether they left.

---


**Built as a data science portfolio project.**
If you found it useful, feel free to ⭐ the repo.

