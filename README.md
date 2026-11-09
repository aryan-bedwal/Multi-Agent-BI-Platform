# Autonomous Business Intelligence Platform using Multi-Agent AI

An end-to-end analytics pipeline made of **10 specialised agents**, each
responsible for one stage of the BI lifecycle. Every agent subclasses a common
`BaseAgent`, consumes the previous agent's `DatasetManifest`, and writes its own,
so stages are independently runnable, testable and resumable.

**Pipeline:** Data Ingestion → Data Cleaning → EDA → ML → Forecasting →
Anomaly Detection → Explainability → Insight & Recommendation → Dashboard →
Report Generation

## Highlights
- **Multi-table ingestion:** joins 9 relational Olist CSVs into an order-item-grain fact table
- **Fair model comparison:** Linear Regression, Random Forest and XGBoost share one preprocessing pipeline, with leakage columns excluded explicitly
- **Forecasting:** Prophet vs XGBoost vs LSTM, evaluated on a chronological hold-out
- **Anomaly detection:** Isolation Forest, LOF, One-Class SVM and an Autoencoder, with consensus voting, plus rolling z-score for time series
- **Explainability:** SHAP and LIME for predictions, deviation ranking for anomaly flags
- **Rule-based recommendations:** structured insights mapped to prioritised actions, with no external LLM API needed
- **Outputs:** an interactive Streamlit dashboard and DOCX / PDF / XLSX business reports, both built from one shared data layer
- **Tested:** 149 automated tests, plus end-to-end verification of the dashboard and reports by actually launching and rendering them

## Tech stack
Python · pandas · scikit-learn · XGBoost · Prophet · PyTorch · SHAP · LIME ·
Streamlit · Plotly · python-docx · ReportLab · openpyxl

## Quick start
    pip install -r requirements.txt
    python main.py
    streamlit run dashboard_app.py
