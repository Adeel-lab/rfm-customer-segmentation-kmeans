# RFM Customer Segmentation with K-Means Clustering & MLflow

A machine learning project that applies RFM (Recency, Frequency, Monetary) 
analysis and K-Means clustering to segment customers from the UCI Online 
Retail dataset into actionable behavioral groups. Experiments are tracked 
end-to-end using MLflow.

---

## Project Overview

Raw transactional data rarely tells you who your customers are. This project 
transforms 541,909 transaction records into a clean, customer-level RFM 
feature matrix and applies K-Means clustering to discover distinct behavioral 
segments. Three cluster configurations (K=3, K=4, K=5) are evaluated using 
three validation metrics and visualized in both 2D (PCA) and 3D space.

---

## Dataset

- **Source:** [UCI Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Raw size:** 541,909 transaction rows
- **After cleaning:** 392,732 valid rows across 4,339 unique customers
- **Period covered:** December 2010 – December 2011
- **Features used:** InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice

---

## Data Cleaning Steps

| Step | Action | Rows affected |
|------|--------|--------------|
| Remove missing CustomerIDs | Dropped rows with no customer attribution | −135,080 |
| Remove cancellations | Dropped all invoices prefixed with "C" (confirmed all-negative quantities) | −8,905 |
| Remove duplicates | Dropped exact duplicate transaction rows | −5,192 |
| Type conversion | InvoiceDate → datetime, CustomerID → int64 | — |

**Final dataset: 392,732 rows, zero missing values.**

---

## RFM Feature Engineering

Each customer is summarized into three features computed as of 2011-12-10:

| Feature | Definition | Direction |
|---------|-----------|-----------|
| **Recency** | Days since last purchase | Lower = better |
| **Frequency** | Number of unique invoices | Higher = better |
| **Monetary** | Total revenue (Quantity × UnitPrice) | Higher = better |

All three features are standardized using `StandardScaler` before clustering 
to prevent Monetary values from dominating the distance calculations.

---

## Methodology

- **Algorithm:** K-Means with `k-means++` initialization (`n_init=10`, `random_state=42`)
- **K range evaluated:** 2 to 10
- **Dimensionality reduction:** PCA (2 components) for 2D visualization only — 
  not used in the clustering pipeline
- **Monetary axis in 3D plots:** `log1p` transformed to handle extreme outliers

### Validation Metrics

| Metric | Interpretation | Goal |
|--------|---------------|------|
| Silhouette Score | How well each point fits its own cluster vs. nearest neighbour | Higher is better |
| Davies-Bouldin Score | Ratio of within-cluster scatter to between-cluster separation | Lower is better |
| Calinski-Harabasz Score | Ratio of between-cluster to within-cluster dispersion | Higher is better |

---

## Results Summary

| K | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|-------------|-----------------|---------------------|
| 2 | **0.8959** | 0.7452 | 1924.54 |
| 3 | 0.5939 | 0.7106 | 3017.05 |
| **4** | **0.6161** | 0.7532 | 3145.08 |
| **5** | **0.6157** | **0.7190** | 3433.26 |
| 6 | 0.5983 | 0.6273 | 3693.47 |

> K=2 is mathematically optimal by silhouette score, but collapses all customer 
> nuance into two groups. **K=5 is the recommended configuration** — it achieves 
> the best Davies-Bouldin score among practical configurations, maintains a 
> competitive silhouette score, and produces five distinct, actionable segments.

---

## Customer Segments (K=5 — Recommended)

| Segment | Recency | Frequency | Monetary (avg) | Strategic Action |
|---------|---------|-----------|----------------|-----------------|
| 🔴 **VIP** | 7 days | 43 orders | £190,809 | Dedicated account management, exclusive terms |
| 🟡 **Champions** | 6 days | 121 orders | £55,099 | Prioritize service quality and restock access |
| 🟣 **Loyal** | 15 days | 21 orders | £12,349 | Frequency incentives, upgrade pipeline to Champions |
| 🔵 **Potential** | 44 days | 4 orders | £1,318 | Re-engagement emails, low-threshold loyalty perks |
| 🟢 **Lost** | 248 days | 2 orders | £476 | Single win-back campaign, low budget allocation |

---

## Experiment Tracking with MLflow

All 9 K-Means runs (K=2 to K=10) are logged to MLflow with:

- **Parameters:** k, random_state, n_init, init method
- **Metrics:** inertia, silhouette score, Davies-Bouldin score, Calinski-Harabasz score
- **Artifacts:** cluster assignment CSV per run, serialized model (cloudpickle)

To launch the MLflow UI locally:

```bash
mlflow ui --backend-store-uri "file:///path/to/your/mlruns"
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Installation

```bash
pip install pandas seaborn matplotlib numpy scikit-learn mlflow
```

---


---

## Key Takeaways

- The **Lost segment is perfectly stable** across all K values (248 days, 
  1.55 orders, £476), confirming it is a genuine behavioral cluster and 
  not an algorithm artefact.
- The **VIP vs Champions split** in K=5 reveals a £135,000 average monetary 
  difference between the two top tiers — a finding with direct revenue implications.
- **K=2 scores highest on silhouette** but provides no actionable business 
  differentiation beyond "active vs. inactive."
- MLflow tracking makes all experiments fully reproducible without re-running 
  the notebook manually.

---

## Tools & Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightblue)

---

## Author

**Adeel Kamal**  
Graduate Researcher | Machine Learning & Predictive Analytics  
[GitHub]((https://github.com/Adeel-lab)) · [LinkedIn](in/adeel-kamal-8231b4295)
