# 📊 Sales Forecasting & Demand Intelligence System

An end-to-end data science pipeline that predicts future product demand, detects unusual sales patterns, segments products by demand behavior, and presents everything through a live interactive dashboard — built on four years of retail sales data.

**🔗 Live Dashboard:** [sales-forecasting-dashboard.streamlit.app](https://sales-forecasting-dashboard-irgw6k8q5us4lxhktu65hp.streamlit.app/)

---

## 🎯 Problem Statement

Retail and e-commerce companies live and die by one question: *how much of each product will we sell next month, and will we have enough stock to meet that demand?* This project builds a working system to answer that — combining time-series forecasting, anomaly detection, and customer/product segmentation into a single decision-support tool a business manager could actually use.

## 📦 Dataset

- **Superstore Sales Dataset** ([Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)) — 9,800 orders across 4 years (2015–2018), spanning 3 categories, 17 sub-categories, and 4 US regions.

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Data & Analysis | Python, Pandas, NumPy |
| Time Series | Statsmodels (SARIMA, decomposition, ADF test), pmdarima (auto_arima), Prophet |
| Machine Learning | XGBoost, Scikit-learn (Isolation Forest, K-Means, PCA) |
| Visualization | Matplotlib, Seaborn |
| Dashboard & Deployment | Streamlit, Streamlit Community Cloud |

## 📁 Repository Structure

```
sales-forecasting-dashboard/
├── Analysis.ipynb            # Full analysis notebook (all 8 tasks)
├── train.csv                 # Superstore sales dataset
├── app.py                    # Streamlit dashboard (4 pages)
├── requirements.txt          # Python dependencies
├── summary.docx              # 2-page executive business report
├── cleaned_sales.csv         # Processed dataset (dashboard input)
├── monthly_sales.csv         # Monthly aggregated time series
├── model_comparison.csv      # Forecasting model results
├── segment_forecasts.csv     # Category/region-level forecasts
├── anomaly_data.csv          # Weekly sales with anomaly flags
├── cluster_data.csv          # Product sub-category clusters
└── sarima_model.pkl          # Trained SARIMA model (production model)
```

## 🔍 What's Inside

1. **Data Exploration** — category, region, and shipping-time analysis; seasonality patterns
2. **Time Series Decomposition** — trend/seasonal/residual breakdown, stationarity testing (ADF)
3. **Forecasting — 3 Models Compared**
4. **Segment-Level Forecasting** — repeated for each product category and region
5. **Anomaly Detection** — Isolation Forest + Z-score, cross-validated against each other
6. **Product Demand Segmentation** — K-Means clustering (4 demand groups) with PCA visualization
7. **Interactive Dashboard** — Streamlit app with 4 pages (Overview, Forecast Explorer, Anomaly Report, Segments)
8. **Executive Business Report** — plain-language summary for non-technical stakeholders

## 📈 Key Results

### Model Comparison (3-month holdout test)

| Model | MAE | RMSE | MAPE | Recommended |
|---|---|---|---|---|
| **SARIMA** | 16,825 | 19,208 | **17.7%** | ✅ Production |
| XGBoost | 18,910 | 21,009 | 19.4% | |
| Prophet | 20,251 | 22,318 | 21.9% | |

SARIMA — `ARIMA(2,1,0)(1,0,0)[12]`, selected via `auto_arima` — was the most accurate and is used in the deployed dashboard.

### Highlights

- **Technology** is the top revenue category ($827K), ahead of Furniture ($729K) and Office Supplies ($705K)
- **East region** shows the most consistent year-over-year growth of any region, and the strongest forecasted growth next quarter (+69.7%)
- **11 anomalous weeks** flagged by Isolation Forest, **6** by Z-score, with holiday-season spikes (Nov–Dec) among the strongest signals
- Products cluster into **4 demand groups** — from "Explosive Growth" (Copiers, +480% YoY) to "Declining Demand" (Machines, -30% YoY) — each with a distinct recommended stocking strategy

## 🚀 Running Locally

```bash
git clone https://github.com/AkashSahu33-code/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Known Limitations

- Forecasting is trained on 45 months of aggregated monthly data — a relatively small sample for capturing rare events like exact holiday-spike magnitude (all three models underestimated the November peak in testing).
- Segment-level (Task 4) forecasts reuse the overall best SARIMA parameters rather than re-tuning per segment, as a time-scoped simplification.
- Two of the four demand clusters (Copiers, Machines) contain a single sub-category each — statistically small, but retained separately as business-relevant outliers rather than merged into larger groups.

## 👤 Author

**Akash Sahu**
B.Tech Biomedical Engineering, NIT Raipur

---

*Built as part of a Data Science internship project — Week 3 & 4: End-to-End Sales Forecasting & Demand Intelligence System.*
