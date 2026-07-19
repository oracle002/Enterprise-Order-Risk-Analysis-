# E-Commerce Order Risk Data Aggregation & Reporting

A risk-reporting analytics project built on 12,000 synthetic e-commerce orders, designed to mirror
the risk data aggregation, reporting, and framework-adherence work of an enterprise risk function
(aligned with **BCBS 239** risk data aggregation principles and **OCC Heightened Standards** on
risk governance).

**Python, Pandas, NumPy, scikit-learn, Matplotlib/Seaborn, Excel Reporting**

## About Dataset

[E-Commerce Order Risk Classification Dataset](https://www.kaggle.com/datasets/emirhanakku/e-commerce-order-risk-classification-dataset) (Kaggle) — a synthetic dataset for return prediction, fraud detection, and multi-class order risk classification.

This dataset is a fully synthetic e-commerce order risk and return prediction dataset created for machine learning, data analysis, and portfolio projects. It contains 12,000 synthetic online order records with 23 features related to customer behavior, order details, payment method, product category, delivery risk, suspicious activity indicators, customer support interactions, and review scores.

All data is artificially generated and does not contain any real customer, company, payment, address, or personal information.

## What this project does

- Ingests and validates 12,000 raw order-level records (data quality checks: duplicates, nulls,
  out-of-range values, negative amounts) to ensure data integrity before aggregation.
- Aggregates transaction-level data into an enterprise risk scorecard across five dimensions
  (country, payment method, product category, device type, traffic source), reporting fraud rate,
  return rate, average order value, and review score per segment.
- Applies a defined risk appetite threshold (2x the enterprise average fraud rate) to flag segment-level
  breaches — e.g., **Gift Card payments flagged at 8.21% fraud rate vs. a 3.72% enterprise average**.
- Builds a monthly risk trend report (fraud rate / return rate over time) to support ongoing
  monitoring and emerging-risk identification.
- Trains a multi-class classification model (Random Forest) to predict order risk category
  (Normal / Return Risk / Fraud Risk) and a binary fraud-detection model (ROC-AUC ≈ 0.71),
  surfacing the top quantitative risk drivers (review score, prior order history, shipping
  distance, high-risk IP flag, address mismatch) to support reporting narratives.
- Exports a consolidated, multi-tab **risk reporting pack** (Excel) with enterprise summary,
  monthly trend, appetite breach log, and dimension-level scorecards — structured the way a
  periodic risk report would be packaged for stakeholders.

- Runs a deeper exploratory analysis: distribution and outlier checks (IQR method) on order
  value, shipping distance, and delivery time; bivariate breakdowns of order value and review
  score by risk label; and risk-rate comparisons across device type.
- Forecasts the next 3 months of order volume, fraud rate, and return rate using Holt-Winters
  exponential smoothing with seasonal decomposition, so the reporting pack isn't just
  backward-looking.
- Mines association rules across order attributes (country, payment method, category, channel,
  device, and risk flags) — the standard adaptation of market-basket / Apriori analysis for
  fraud-pattern detection when there's no multi-item basket or repeat-customer ID to group
  transactions by. Surfaces the specific attribute combinations most strongly linked to fraud
  (`high_risk_ip` → 4.86x lift, `address_mismatch` → 3.91x lift) and to returns
  (`product_category = Fashion` → ~1.9–2.0x lift, `late_delivery_risk` → 1.88x lift).

## Repo structure
```
data/
  synthetic_ecommerce_order_risk_dataset.csv   # 12,000-row synthetic source dataset
notebooks/
  risk_reporting_analysis.ipynb                # notebook version of the core pipeline, outputs included
  eda_forecasting_basket.ipynb                 # notebook version of EDA/forecast/basket analysis
src/
  risk_reporting_analysis.py      # end-to-end aggregation, scorecards, model, plots
  eda_forecasting_basket.py       # deeper EDA, 3-month forecast, association-rule mining
reports/
  risk_reporting_pack.xlsx        # consolidated multi-tab risk report
  risk_appetite_breach_log.csv    # segments exceeding defined risk appetite threshold
  monthly_risk_trend.csv          # monthly fraud/return rate trend
  feature_importance.csv          # top risk drivers from the classification model
  model_classification_report.txt
  eda_numeric_summary.csv         # descriptive statistics for numeric fields
  eda_outlier_report.csv          # IQR-based outlier counts
  forecast_next_3_months.csv      # 3-month forward forecast of orders, fraud rate, return rate
  market_basket_fraud_rules.csv   # association rules ranked by lift, fraud outcome
  market_basket_return_rules.csv  # association rules ranked by lift, return outcome
  plots/                          # risk distribution, trend, scorecard, EDA, forecast, and rule visuals
requirements.txt
```

## Setup & how to run

```bash
git clone https://github.com/<your-username>/ecommerce-order-risk-reporting.git
cd ecommerce-order-risk-reporting
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Run the pipeline as scripts (writes into reports/)
cd src
python risk_reporting_analysis.py
python eda_forecasting_basket.py

# Or explore interactively (outputs are already saved in the notebooks)
cd ../notebooks
jupyter notebook risk_reporting_analysis.ipynb
```

Both scripts/notebooks read `../data/synthetic_ecommerce_order_risk_dataset.csv` and write results
into `../reports/` (tables) and `../reports/plots/` (visuals), so they must be run from inside `src/`
or `notebooks/` respectively — paths are relative to keep the repo portable.

## Key findings
- Enterprise-wide fraud rate: **3.72%**, return rate: **9.27%**, combined high-risk rate: **12.65%**.
- Gift Card is the only payment method breaching the defined risk appetite threshold.
- Paid Ads traffic carries the highest fraud rate (5.97%) among acquisition channels, nearly
  double that of Marketplace or Direct traffic — a candidate finding for channel-level risk review.
- Review score, order history depth, and shipping distance are the leading quantitative
  predictors of order risk classification.
- 3-month forward forecast (Jan–Mar 2026) projects order volume holding steady with fraud and
  return rates staying within recent historical ranges — no material trend shift detected.
- `high_risk_ip` and `address_mismatch` are the strongest individual fraud signals (4.86x and
  3.91x lift respectively), far outweighing channel- or category-level effects — a finding that
  supports prioritizing those two flags in any real-time fraud-scoring rule.
