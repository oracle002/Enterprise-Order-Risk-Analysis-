# E-Commerce Order Risk Data Aggregation & Reporting

A risk-reporting analytics project built on 12,000 synthetic e-commerce orders, designed to mirror
the risk data aggregation, reporting, and framework-adherence work of an enterprise risk function
(aligned with **BCBS 239** risk data aggregation principles and **OCC Heightened Standards** on
risk governance).

**Python, Pandas, NumPy, scikit-learn, Matplotlib/Seaborn, Excel Reporting**

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

## Repo structure
```
risk_reporting_analysis.py      # end-to-end aggregation, scorecards, model, plots
risk_reporting_pack.xlsx        # consolidated multi-tab risk report
risk_appetite_breach_log.csv    # segments exceeding defined risk appetite threshold
monthly_risk_trend.csv          # monthly fraud/return rate trend
feature_importance.csv          # top risk drivers from the classification model
model_classification_report.txt
plots/                          # risk distribution, trend, scorecard, and model visuals
```

## Key findings
- Enterprise-wide fraud rate: **3.72%**, return rate: **9.27%**, combined high-risk rate: **12.65%**.
- Gift Card is the only payment method breaching the defined risk appetite threshold.
- Paid Ads traffic carries the highest fraud rate (5.97%) among acquisition channels, nearly
  double that of Marketplace or Direct traffic — a candidate finding for channel-level risk review.
- Review score, order history depth, and shipping distance are the leading quantitative
  predictors of order risk classification.
